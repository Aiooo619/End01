from __future__ import annotations

import argparse
import json

from .config import load_settings
from .storage import IngestError
from .trainer import TrainingWorker


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queued SDXL LoRA training jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", type=str, default="")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    worker = TrainingWorker(load_settings())
    job_path = worker.settings.data_root / "queues" / args.job if args.job else worker.next_job()
    if job_path is None or not job_path.exists():
        print("training_job=none")
        return 0
    if args.register_only:
        print(f"registered_models={worker.register_job_outputs(job_path)}")
        return 0
    try:
        prepared = worker.run(job_path, dry_run=args.dry_run)
    except IngestError as exc:
        print(f"training_blocked={exc}")
        return 2
    print(f"training_job={job_path.name}")
    print(f"training_images={prepared.image_count}")
    print(f"training_output={prepared.output_dir / (prepared.output_name + '.safetensors')}")
    if args.dry_run:
        print("training_command=" + json.dumps(prepared.command, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
