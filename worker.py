"""
AI Video Maker — Vast.ai PyWorker agent
=======================================
Vast's PyWorker proxies endpoint requests to our model server
(model_server.py on 127.0.0.1:8000) and manages readiness/health.

Docs: https://docs.vast.ai/guides/serverless/creating-new-pyworkers.md
"""

from vastai import (
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)


def generate_benchmark_payload() -> dict:
    """Engine benchmark के दौरान ये payload भेजेगा (1 test video)।"""
    import uuid

    return {
        "job_id": f"bench-{uuid.uuid4().hex[:8]}",
        "prompt": "A colorful balloon floating over a city, smooth motion",
    }


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="/var/log/portal/model.log",

    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,   # one GPU job at a time
            max_queue_time=900.0,
            workload_calculator=lambda payload: 1.0,
            benchmark_config=BenchmarkConfig(   # engine को एक benchmark ज़रूरी है
                generator=generate_benchmark_payload,
                runs=1,
                concurrency=1,
            ),
        ),
        HandlerConfig(
            route="/extend",
            allow_parallel_requests=False,
            max_queue_time=900.0,
            workload_calculator=lambda payload: 1.0,
        ),
        HandlerConfig(
            route="/story",
            allow_parallel_requests=False,
            max_queue_time=1200.0,
            workload_calculator=lambda payload: 5.0,  # stories cost more quota
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=["Model ready — accepting jobs"],
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError",
            "CUDA out of memory",
        ],
        on_info=["stage:"],
    ),
)

Worker(worker_config).run()
