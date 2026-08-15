# Discord Style Trainer

Local pipeline for receiving categorized reference images from Discord and
training a separate SDXL LoRA for each visual style.

## Repository layout

- `config/` — tracked bot, style, and training configuration
- `datasets/` — local style datasets; image contents are ignored by Git
- `records/` — manifests and append-only ingestion/training records
- `queues/` — pending approval and training jobs
- `models/` — local LoRA versions; model weights are ignored by Git
- `outputs/` — generated previews and test grids
- `logs/` — runtime logs
- `state/` — local database and process state
- `docs/` — operating and data-format documentation

## Data flow

1. Upload images to a mapped Discord style channel or slash command.
2. The bot downloads attachments immediately and records their hashes.
3. Images enter `incoming`, then move to `approved` or `rejected`.
4. Approved images are captioned and queued for versioned LoRA training.
5. Test previews are generated before a model version is published.

## Secrets

Copy `.env.example` to `.env`. Keep the Discord bot token only in `.env`.
The `.env` file, datasets, model weights, logs, and local databases are excluded
from Git.

