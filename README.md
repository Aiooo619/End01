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

To reject only the images attached to one Discord message, open the message
context menu and choose **Apps → Reject training images**. Approving an image
also creates a caption job under `queues/captions/`.

Direct uploads support multiple attachments when the Discord channel ID is
mapped to a style. Accepted formats are JPEG, PNG, and WebP; the shortest side
must be at least 512 pixels.

For forum mode, set `DISCORD_FORUM_CHANNEL_ID` or use
`config/runtime.local.yaml`. Each forum post is a thread and maps to one style.
New posts can be registered from inside the post with `/register_style`.
