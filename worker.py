"""
AI Video Maker — Modal Serverless GPU Worker
============================================
LTX-2.3 audio+video generation on an NVIDIA L4 (24GB), with the older
LTX-Video 2B kept as a switchable fallback family.

MODEL_FAMILY (env):
  "ltx2" (default) — Lightricks LTX-2.3: video + synchronized AUDIO,
                     runs on 24GB, i2v via from_pipe (shared components)
  "ltx1"           — LTX-Video 2B (silent, cheapest, distilled-capable)

Capabilities
------------
1. generate_base   : prompt -> ~5s clip (768x480 @ 24fps)
2. extend_clip     : previous clip's LAST frame -> image-to-video
                     continuation -> FFmpeg 0.3s crossfade stitch
3. generate_story  : 2-6 scene prompts -> chained clips -> one stitched video

Every result is uploaded to Cloudflare R2; /tmp files are ALWAYS removed.
Progress is reported to Upstash Redis (best-effort).

Deploy (see DEPLOYMENT.md)
--------------------------
    modal volume create ltx-models
    modal secret create r2-credentials ...
    modal secret create upstash-redis REDIS_URL=...
    modal deploy worker.py
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import uuid
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "ltx2").strip().lower()
MODEL_ID = os.environ.get(
    "LTX_MODEL_ID",
    "diffusers/LTX-2.3-Diffusers" if MODEL_FAMILY == "ltx2"
    else "Lightricks/LTX-Video",
)
VARIANT = os.environ.get("LTX_VARIANT", "").strip() or None
STEPS = int(os.environ.get("LTX_STEPS", "20"))
GUIDANCE = float(os.environ.get("LTX_GUIDANCE", "3.0"))

WIDTH, HEIGHT, FPS = 768, 480, 24

# LTX frame counts must satisfy (k*n)+1 divisibility:
#   97  = 6*16+1 = 12*8+1  (base, ~4.0s)
#   113 = 7*16+1 = 14*8+1  (extension, ~4.7s -> ~10s total after stitch)
BASE_FRAMES = 97
EXTEND_FRAMES = 113 if MODEL_FAMILY == "ltx2" else 121
CROSSFADE_SECONDS = 0.3

NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery motion, inconsistent "
    "characters, distorted faces, extra limbs, watermark, text overlay"
)

MODEL_DIR = "/models"
TMP_ROOT = Path("/tmp/aivm")
JOB_TTL_SECONDS = 72 * 3600

# ---------------------------------------------------------------------------
# Modal app / image / secrets
# ---------------------------------------------------------------------------
app = modal.App("ai-video-worker")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.7.1",
        # LTX-2 family pipelines need diffusers >= 0.41
        "diffusers>=0.41,<1",
        "transformers>=5,<6",
        "accelerate>=1.2,<2",
        "sentencepiece==0.2.0",
        "protobuf==5.29.2",
        "boto3==1.35.90",
        "Pillow==11.0.0",
        "redis==5.2.1",
        "numpy<2",
    )
)

model_volume = modal.Volume.from_name("ltx-models", create_if_missing=True)
r2_secret = modal.Secret.from_name("r2-credentials")
redis_secret = modal.Secret.from_name("upstash-redis")


# ---------------------------------------------------------------------------
# Helpers (inside the GPU container)
# ---------------------------------------------------------------------------
def report(job_id: str, **fields) -> None:
    """Best-effort progress write to Redis. Never raises."""
    try:
        import redis

        client = redis.Redis.from_url(
            os.environ["REDIS_URL"], socket_timeout=5, socket_connect_timeout=5
        )
        key = f"job:{job_id}"
        client.hset(key, mapping={k: str(v) for k, v in fields.items()})
        client.expire(key, JOB_TTL_SECONDS)
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


def r2_upload(local_path: Path, key: str) -> None:
    r2_client().upload_file(
        str(local_path), os.environ["R2_BUCKET"], key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def r2_download(key: str, local_path: Path) -> None:
    r2_client().download_file(os.environ["R2_BUCKET"], key, str(local_path))


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {proc.returncode}): {proc.stderr[-2000:]}"
        )


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-500:]}")
    return float(proc.stdout.strip())


def extract_last_frame(video: Path, out_png: Path) -> None:
    _run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(video),
         "-update", "1", "-frames:v", "1", str(out_png)]
    )


def crossfade_stitch(a: Path, b: Path, out: Path,
                     fade: float = CROSSFADE_SECONDS) -> None:
    offset = max(probe_duration(a) - fade, 0.0)
    # re-encode so both segments share codec settings; drop source audio
    # (stitched segments have independent generated soundtracks)
    _run(
        ["ffmpeg", "-y", "-i", str(a), "-i", str(b),
         "-filter_complex",
         f"[0:v][1:v]xfade=transition=fade:duration={fade:.2f}:offset={offset:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    )


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=60,
    timeout=1200,               # LTX-2.3 is heavier: raise the cap to 20 min
    cpu=4,
    memory=20480,
    max_containers=2,
    volumes={MODEL_DIR: model_volume},
    secrets=[r2_secret, redis_secret],
)
class VideoWorker:
    """One warm container serves many jobs; pipelines load once."""

    @modal.enter()
    def load_models(self) -> None:
        import torch

        os.environ["HF_HOME"] = MODEL_DIR
        self.family = MODEL_FAMILY

        if self.family == "ltx2":
            from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline

            # t2v loads every component once; i2v shares them via from_pipe
            # (no duplicate VRAM). Output includes synchronized AUDIO.
            self.t2v = LTX2Pipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16
            )
            self.i2v = LTX2ImageToVideoPipeline.from_pipe(self.t2v)
            self.t2v.to("cuda")
            self.i2v.to("cuda")
            try:
                self.audio_sample_rate = int(
                    self.t2v.vocoder.config.output_sampling_rate  # 24000
                )
            except Exception:
                self.audio_sample_rate = 24000
        else:
            from diffusers import LTXImageToVideoPipeline, LTXPipeline
            from transformers import T5EncoderModel

            # LTX-Video 1.x family: shared T5-XXL keeps both pipelines ~18GB
            text_encoder = T5EncoderModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16
            )
            load_kwargs: dict = {
                "torch_dtype": torch.bfloat16,
                "text_encoder": text_encoder,
            }
            if VARIANT:
                load_kwargs["variant"] = VARIANT

            self.t2v = LTXPipeline.from_pretrained(MODEL_ID, **load_kwargs)
            self.i2v = LTXImageToVideoPipeline.from_pretrained(
                MODEL_ID, **load_kwargs
            )
            self.t2v.to("cuda")
            self.i2v.to("cuda")

        for pipe in (self.t2v, self.i2v):
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _generate_clip(self, prompt: str, num_frames: int, image=None):
        """Run the right pipeline; return (frames, audio|None)."""
        import torch

        seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        common = dict(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        )
        if self.family == "ltx2":
            out = (self.i2v if image is not None else self.t2v)(
                num_frames=num_frames,
                frame_rate=FPS,
                **common,
                **({"image": image} if image is not None else {}),
            )
            frames = out.frames
            if frames and isinstance(frames[0], (list, tuple)):
                frames = frames[0]
            if not frames:
                raise RuntimeError("Pipeline returned no frames")
            audio = getattr(out, "audio", None)
            if audio is not None:
                try:
                    audio = audio[0]
                except Exception:
                    audio = None
            return frames, audio

        out = (self.i2v if image is not None else self.t2v)(
            num_frames=num_frames, **common,
            **({"image": image} if image is not None else {}),
        )
        frames = out.frames
        if frames and isinstance(frames[0], (list, tuple)):
            frames = frames[0]
        if not frames:
            raise RuntimeError("Pipeline returned no frames")
        return frames, None

    def _export_clip(self, frames, audio, path: Path) -> None:
        if self.family == "ltx2" and audio is not None:
            try:
                from diffusers.utils.export_utils import encode_video

                encode_video(
                    frames, fps=FPS, output_path=str(path),
                    audio=audio, audio_sample_rate=self.audio_sample_rate,
                )
                return
            except Exception:
                pass  # fall back to a silent export
        from diffusers.utils import export_to_video

        export_to_video(frames, str(path), fps=FPS)

    # ------------------------------------------------------------------
    @modal.method()
    def generate_base(self, payload: dict) -> dict:
        """Text -> ~5s clip."""
        job_id, prompt = payload["job_id"], payload["prompt"]
        workdir = TMP_ROOT / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        clip = workdir / "clip.mp4"
        try:
            report(job_id, status="running", stage="generating", progress=10)
            frames, audio = self._generate_clip(prompt, BASE_FRAMES)
            self._export_clip(frames, audio, clip)

            report(job_id, stage="uploading", progress=85)
            key = f"videos/{job_id}/final.mp4"
            r2_upload(clip, key)

            duration = probe_duration(clip)
            report(job_id, status="done", stage="done", progress=100,
                   r2_key=key, duration=f"{duration:.2f}")
            return {"r2_key": key, "duration": duration}
        except Exception as exc:
            report(job_id, status="failed", stage="failed",
                   error=str(exc)[:500])
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    @modal.method()
    def extend_clip(self, payload: dict) -> dict:
        """Last frame -> i2v continuation -> crossfade stitch -> upload."""
        job_id = payload["job_id"]
        prompt = payload["prompt"]
        prev_key = payload["prev_r2_key"]

        workdir = TMP_ROOT / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        prev, nxt, merged, last = (
            workdir / "prev.mp4",
            workdir / "next.mp4",
            workdir / "merged.mp4",
            workdir / "last_frame.png",
        )
        try:
            report(job_id, status="running", stage="downloading", progress=5)
            r2_download(prev_key, prev)

            report(job_id, stage="generating", progress=20)
            from PIL import Image

            extract_last_frame(prev, last)
            image = Image.open(last).convert("RGB").resize((WIDTH, HEIGHT))
            frames, audio = self._generate_clip(
                prompt, EXTEND_FRAMES, image=image
            )
            self._export_clip(frames, audio, nxt)

            report(job_id, stage="stitching", progress=70)
            crossfade_stitch(prev, nxt, merged)

            report(job_id, stage="uploading", progress=92)
            key = f"videos/{job_id}/final.mp4"
            r2_upload(merged, key)

            duration = probe_duration(merged)
            report(job_id, status="done", stage="done", progress=100,
                   r2_key=key, duration=f"{duration:.2f}")
            return {"r2_key": key, "duration": duration}
        except Exception as exc:
            report(job_id, status="failed", stage="failed",
                   error=str(exc)[:500])
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    @modal.method()
    def generate_story(self, payload: dict) -> dict:
        """2-6 scenes -> chained clips -> one stitched video (~30s)."""
        job_id = payload["job_id"]
        scenes = [s.strip() for s in payload["scenes"] if s.strip()][:6]
        if len(scenes) < 2:
            raise ValueError("A story needs at least 2 scenes")

        workdir = TMP_ROOT / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        acc = workdir / "scene_000.mp4"
        last = workdir / "last_frame.png"
        try:
            report(job_id, status="running", stage="scene_1", progress=5)
            frames, audio = self._generate_clip(scenes[0], BASE_FRAMES)
            self._export_clip(frames, audio, acc)

            for i, scene in enumerate(scenes[1:], start=2):
                report(job_id, stage=f"scene_{i}",
                       progress=5 + int(85 * (i - 1) / len(scenes)))
                from PIL import Image

                extract_last_frame(acc, last)
                image = Image.open(last).convert("RGB").resize((WIDTH, HEIGHT))
                nxt = workdir / f"scene_{i:03d}.mp4"
                frames, audio = self._generate_clip(
                    scene, BASE_FRAMES, image=image
                )
                self._export_clip(frames, audio, nxt)

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
        except Exception as exc:
            report(job_id, status="failed", stage="failed",
                   error=str(exc)[:500])
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Local smoke test:  modal run worker.py --prompt "a red balloon floating"
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def smoke_test(prompt: str = "A cheerful puppy running through a sunny park, "
                             "bright colors, dynamic camera"):
    result = VideoWorker.generate_base.remote(
        {"job_id": f"smoke-{uuid.uuid4().hex[:8]}", "prompt": prompt}
    )
    print("Base clip uploaded:", result)
