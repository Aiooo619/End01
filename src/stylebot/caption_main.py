from __future__ import annotations

import argparse
import logging
import time

from .captioner import process_caption_jobs
from .config import load_settings
from .storage import DatasetStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Process queued image caption jobs")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    while True:
        backfilled = DatasetStore(settings).queue_missing_captions()
        completed, failed = process_caption_jobs(settings, max(1, args.limit))
        if backfilled or completed or failed or not args.watch:
            print(f"caption_backfilled={backfilled}", flush=True)
            print(f"caption_completed={completed} caption_failed={failed}", flush=True)
        if not args.watch:
            return 1 if failed else 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
