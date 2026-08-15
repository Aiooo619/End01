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

## Discord bot setup

Requirements: Python 3.12 and a Discord application with a bot user.

1. Run `powershell -ExecutionPolicy Bypass -File scripts/setup.ps1`.
2. Fill `.env` with the bot token, guild ID, and allowed Discord user IDs.
3. Copy channel IDs into `config/styles.yaml` and rename/add styles as needed.
4. In Discord Developer Portal, enable the **Message Content Intent** if images
   should be accepted by dropping them directly into mapped channels.
5. Invite the bot with `bot` and `applications.commands` scopes. Give it View
   Channel, Read Message History, Send Messages, and Attach Files permissions.
6. Run `powershell -ExecutionPolicy Bypass -File scripts/start.ps1`.

Available slash commands:

- `/upload_style` — upload one image and choose its style
- `/register_style` — register the current forum post/thread as a style
- `/material_status` — view incoming/approved/rejected counts
- `/pending_images` — list pending filenames, dimensions, and short hashes
- `/approve_style` — approve incoming images
- `/reject_style` — reject incoming images in a batch
- `/train_style` — create a versioned training queue job
- `/models` — list draft, production, and archived model checkpoints
- `/review_model` — generate one reproducible review image in a model thread
- `/review_suite` — run four fixed costume-design review prompts
- `/iterate_test` — compare four checkpoints with one prompt, seed, and strength
- `/iterate_auto` — continuously generate bounded multi-round checkpoint comparisons
- `/stop_iteration` — safely stop continuous comparisons after the current image
- `/iteration_report` — show checkpoint win rates and tagged defect counts
- `/prepare_iteration` — prepare a confirmed next-version plan without starting training
- `/feedback` — attach categorized feedback to a model or generation
- `/start_iteration` — record the change plan for the next model version
- `/promote_model` — publish the selected checkpoint as production
- `/generate` — generate with a production model, optional second LoRA and pose image

To reject only the images attached to one Discord message, open the message
context menu and choose **Apps → Reject training images**. Approving an image
also creates a caption job under `queues/captions/`.

Run `scripts/run-captioner.ps1` to process approved images with the local
Apache-2.0 `SmilingWolf/wd-vit-tagger-v3` ONNX model. The first run downloads
about 379MB; later runs use the local cache. Captions are written to each
style's `datasets/<style>/captions/` directory and start with its trigger token.
Use `scripts/start-caption-worker.ps1` to keep a worker running; it scans for
newly approved images every 15 seconds and captions them automatically.

## SDXL LoRA training

The training worker uses the official `kohya-ss/sd-scripts` checkout under
the ignored `work/` directory. `/train_style` creates a job only after the
style reaches its minimum approved-image count. Run `scripts/run-trainer-dry.ps1`
to validate a queued job without downloading SDXL or using the GPU, then run
`scripts/run-trainer.ps1` to start training. Outputs are versioned under
`models/<style>/v001/`, `v002/`, and so on.

The default 16GB profile trains SDXL UNet-only LoRA with rank 32, BF16,
gradient checkpointing, latent/text-encoder disk caches, SDPA, and an 8-bit
optimizer. Raw images, caches, base models, logs, and model weights stay local.

Captions are intentionally ordered for costume-design learning: trigger token,
`character costume design`, clothing/accessories, character traits, composition,
then rendering or art-method tags. Character-name predictions are excluded and
caption shuffling is disabled so this structure remains stable during training.

Model types are kept modular: `design`, `art`, `character`, and `control`.
Only production checkpoints appear in `/generate`; review images, seeds, LoRA
strengths, selections, feedback, and iteration plans are stored in the local
registry for reproducibility. A pose attachment activates the optional SDXL
OpenPose ControlNet path instead of training exact poses into a style LoRA.

After training, run `scripts/setup-inference.ps1` once to install the LoRA and
OpenPose inference dependencies. The bot prefers a materialized local SDXL
Diffusers model at `models/base/sdxl-base-1.0/`; this avoids Windows symlink
permission failures and keeps later generations offline-capable.

Direct uploads support multiple attachments when the Discord channel ID is
mapped to a style. Accepted formats are JPEG, PNG, and WebP; the shortest side
must be at least 512 pixels.

For forum mode, set `DISCORD_FORUM_CHANNEL_ID` or use
`config/runtime.local.yaml`. Each forum post is a thread and maps to one style.
New posts can be registered from inside the post with `/register_style`.
