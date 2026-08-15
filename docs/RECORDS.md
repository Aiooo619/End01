# Record formats

Runtime records use JSON Lines (`.jsonl`): one JSON object per line. They are
append-only so interrupted bot or training processes do not corrupt older data.

## Ingestion event

Stored under `records/events/YYYY-MM-DD.jsonl`.

```json
{"event":"image_received","timestamp":"2026-08-15T12:00:00Z","style_id":"japanese_film","discord_message_id":"","discord_user_id":"","original_filename":"image.png","sha256":"","local_path":"datasets/japanese_film/incoming/image.png","status":"incoming"}
```

## Training run

Stored under `records/training-runs.jsonl`.

```json
{"run_id":"japanese_film-v001","style_id":"japanese_film","started_at":"","finished_at":"","dataset_revision":"","image_count":0,"config_path":"config/training.yaml","output_model":"models/japanese_film/v001/model.safetensors","status":"queued"}
```

Do not record the Discord bot token, authorization headers, or private message
contents in any event or log file.

