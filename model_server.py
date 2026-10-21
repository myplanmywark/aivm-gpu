"""
AI Video Maker — Vast.ai Serverless MODEL SERVER
================================================
MODEL_FAMILY env:
  "ltx1" (अभी active) — LTX-Video 2B Distilled: तेज़, 24GB-class GPUs के लिए
  "ltx2"              — LTX-2.3: full quality + audio (बड़े GPUs के लिए)
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import torch
from fastapi import FastAPI
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "ltx1").strip().lower()
MODEL_ID = os.environ.get(
    "LTX_MODEL_ID",
    "Lightricks/LTX-Video" if MODEL_FAMILY == "ltx1"
    else "diffusers/LTX-2.3-Diffusers",
)
VARIANT = os.environ.get("LTX_VARIANT", "").strip() or None
STEPS = int(os.environ.get("LTX_STEPS", "20"))
GUIDANCE = float(os.environ.get("LTX_GUIDANCE", "3.0"))

WIDTH, HEIGHT, FPS = 768, 480, 24
BASE_FRAMES = 97
EXTEND_FRAMES = 113 if MODEL_FAMILY == "ltx2" else 121
CROSSFADE_SECONDS = 0.3

NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery motion, inconsistent "
    "characters, distorted faces, extra limbs, watermark, text overlay"
)

TMP_ROOT = Path("/tmp/aivm")

# Optional shared-secret: WORKER_KEY set hone par X-AIVM-KEY header zaroori
WORKER_KEY = os.environ.get("WORKER_KEY", "").strip()

app = FastAPI(title="aivm-model-server")
_state: dict = {}


@app.middleware("http")
async def key_guard(request, call_next):
    if WORKER_KEY and request.url.path not in ("/health",) and \
            request.headers.get("X-AIVM-KEY") != WORKER_KEY:
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "bad key"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def report(job_id: str, **fields) -> None:
    try:
        import redis

        client = redis.Redis.from_url(
            os.environ["REDIS_URL"], socket_timeout=5, socket_connect_timeout=5
        )
        client.hset(f"job:{job_id}", mapping={k: str(v) for k, v in fields.items()})
        client.expire(f"job:{job_id}", 72 * 3600)
    except Exception:
        pass


def r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def r2_upload(path: Path, key: str) -> None:
    r2_client().upload_file(
        str(path), os.environ["R2_BUCKET"], key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def r2_download(key: str, path: Path) -> None:
    r2_client().download_file(os.environ["R2_BUCKET"], key, str(path))


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {proc.stderr[-2000:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-300:]}")
    return float(proc.stdout.strip())


def extract_last_frame(video: Path, out_png: Path) -> None:
    _run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(video),
          "-update", "1", "-frames:v", "1", str(out_png)])


def crossfade_stitch(a: Path, b: Path, out: Path) -> None:
    offset = max(probe_duration(a) - CROSSFADE_SECONDS, 0.0)
    _run(
        ["ffmpeg", "-y", "-i", str(a), "-i", str(b),
         "-filter_complex",
         f"[0:v][1:v]xfade=transition=fade:duration={CROSSFADE_SECONDS:.2f}:offset={offset:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    )


# ---------------------------------------------------------------------------
# model load (once)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    print(f"Model load starting: family={MODEL_FAMILY} id={MODEL_ID}", flush=True)

    if MODEL_FAMILY == "ltx1":
        from diffusers import LTXImageToVideoPipeline, LTXPipeline
        from transformers import T5EncoderModel

        # shared T5 encoder — dono pipelines ka total VRAM ~half rehta hai
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        load_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "text_encoder": text_encoder,
        }
        if VARIANT:
            load_kwargs["variant"] = VARIANT

        t2v = LTXPipeline.from_pretrained(MODEL_ID, **load_kwargs)
        i2v = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, **load_kwargs)
        t2v.to("cuda")
        i2v.to("cuda")
        audio_sr = None
    else:
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline

        t2v = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        i2v = LTX2ImageToVideoPipeline.from_pipe(t2v)
        t2v.to("cuda")
        i2v.to("cuda")
        try:
            audio_sr = int(t2v.vocoder.config.output_sampling_rate)
        except Exception:
            audio_sr = 24000

    for pipe in (t2v, i2v):
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    _state.update(t2v=t2v, i2v=i2v, audio_sr=audio_sr)
    print("Model ready — accepting jobs", flush=True)  # PyWorker on_load marker


# ---------------------------------------------------------------------------
# generation core
# ---------------------------------------------------------------------------
def generate_clip(prompt: str, num_frames: int, image=None):
    seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe = _state["i2v"] if image is not None else _state["t2v"]
    common = dict(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    )
    if MODEL_FAMILY == "ltx2":
        out = pipe(
            num_frames=num_frames,
            frame_rate=FPS,
            **common,
            **({"image": image} if image is not None else {}),
        )
        audio = getattr(out, "audio", None)
        if audio is not None:
            try:
                audio = audio[0]
            except Exception:
                audio = None
    else:
        out = pipe(
            num_frames=num_frames,
            **common,
            **({"image": image} if image is not None else {}),
        )
        audio = None

    frames = out.frames
    if frames and isinstance(frames[0], (list, tuple)):
        frames = frames[0]
    if not frames:
        raise RuntimeError("Pipeline returned no frames")
    return frames, audio


def export_clip(frames, audio, path: Path) -> None:
    if MODEL_FAMILY == "ltx2" and audio is not None:
        try:
            from diffusers.utils.export_utils import encode_video

            encode_video(
                frames, fps=FPS, output_path=str(path),
                audio=audio, audio_sample_rate=_state["audio_sr"],
            )
            return
        except Exception:
            pass
    from diffusers.utils import export_to_video

    export_to_video(frames, str(path), fps=FPS)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.post("/generate")
def route_generate(payload: dict):
    job_id, prompt = payload["job_id"], payload["prompt"]
    workdir = TMP_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    clip = workdir / "clip.mp4"
    try:
        report(job_id, status="running", stage="generating", progress=10)
        frames, audio = generate_clip(prompt, BASE_FRAMES)
        export_clip(frames, audio, clip)

        report(job_id, stage="uploading", progress=85)
        key = f"videos/{job_id}/final.mp4"
        r2_upload(clip, key)
        duration = probe_duration(clip)
        report(job_id, status="done", stage="done", progress=100,
               r2_key=key, duration=f"{duration:.2f}")
        return {"r2_key": key, "duration": duration}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/extend")
def route_extend(payload: dict):
    job_id, prompt = payload["job_id"], payload["prompt"]
    prev_key = payload["prev_r2_key"]
    workdir = TMP_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    prev, nxt, merged, last = (
        workdir / "prev.mp4", workdir / "next.mp4",
        workdir / "merged.mp4", workdir / "last_frame.png",
    )
    try:
        report(job_id, status="running", stage="downloading", progress=5)
        r2_download(prev_key, prev)

        report(job_id, stage="generating", progress=20)
        extract_last_frame(prev, last)
        image = Image.open(last).convert("RGB").resize((WIDTH, HEIGHT))
        frames, audio = generate_clip(prompt, EXTEND_FRAMES, image=image)
        export_clip(frames, audio, nxt)

        report(job_id, stage="stitching", progress=70)
        crossfade_stitch(prev, nxt, merged)

        report(job_id, stage="uploading", progress=92)
        key = f"videos/{job_id}/final.mp4"
        r2_upload(merged, key)
        duration = probe_duration(merged)
        report(job_id, status="done", stage="done", progress=100,
               r2_key=key, duration=f"{duration:.2f}")
        return {"r2_key": key, "duration": duration}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/story")
def route_story(payload: dict):
    job_id = payload["job_id"]
    scenes = [s.strip() for s in payload["scenes"] if s.strip()][:6]
    if len(scenes) < 2:
        return {"error": "A story needs at least 2 scenes"}
    workdir = TMP_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    acc = workdir / "scene_000.mp4"
    last = workdir / "last_frame.png"
    try:
        report(job_id, status="running", stage="scene_1", progress=5)
        frames, audio = generate_clip(scenes[0], BASE_FRAMES)
        export_clip(frames, audio, acc)

        for i, scene in enumerate(scenes[1:], start=2):
            report(job_id, stage=f"scene_{i}",
                   progress=5 + int(85 * (i - 1) / len(scenes)))
            extract_last_frame(acc, last)
            image = Image.open(last).convert("RGB").resize((WIDTH, HEIGHT))
            nxt = workdir / f"scene_{i:03d}.mp4"
            frames, audio = generate_clip(scene, BASE_FRAMES, image=image)
            export_clip(frames, audio, nxt)

            report(job_id, stage=f"stitch_{i}",
                   progress=5 + int(85 * i / len(scenes)))
            merged = workdir / f"acc_{i:03d}.mp4"
            crossfade_stitch(acc, nxt, merged)
            acc.unlink(missing_ok=True)
            acc = merged

        report(job_id, stage="uploading", progress=93)
        key = f"videos/{job_id}/final.mp4"
        r2_upload(acc, key)
        duration = probe_duration(acc)
        report(job_id, status="done", stage="done", progress=100,
               r2_key=key, duration=f"{duration:.2f}")
        return {"r2_key": key, "duration": duration}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health():
    return {"ok": bool(_state)}
