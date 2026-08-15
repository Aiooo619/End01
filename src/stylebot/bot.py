from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path

import discord
import yaml
from discord import app_commands
from discord.ext import commands

from .config import Settings, StyleConfig
from .arknights_curator import accept_record, update_record
from .inference import InferenceRunner
from .registry import ModelRegistry
from .storage import DatasetStore, IngestError


LOGGER = logging.getLogger(__name__)
REVIEW_PROMPTS = (
    "character costume design, full body female character, layered practical outfit, detailed accessories, neutral standing pose, simple background",
    "character costume design, full body male character, structured long coat, functional equipment, neutral standing pose, simple background",
    "character costume design, androgynous character, asymmetrical outfit, complex silhouette, utility accessories, three-quarter view",
    "character costume design, full body character, alternate color palette, light clothing layers, readable silhouette, plain background",
)
ITERATION_NEGATIVE = (
    "photorealistic, close-up, portrait crop, cropped feet, extra limbs, extra arms, "
    "extra legs, malformed hands, fused face, fused accessories, duplicate character, "
    "busy background, text, logo, watermark"
)
ISSUE_LABELS = {
    "good_design": "服裝設計好",
    "good_color": "配色好",
    "good_anatomy": "人體結構穩定",
    "good_silhouette": "人物輪廓清晰",
    "good_clothing_structure": "服裝結構清楚",
    "good_materials": "材質表現好",
    "good_accessories": "配件位置合理",
    "good_pose": "動作自然",
    "good_style": "畫風符合",
    "good_prompt_match": "符合提示詞",
    "extra_limbs": "多肢體",
    "concept_bleeding": "五官／飾品融合",
    "clothing_fusion": "衣服部件融合",
    "mechanical_sleeves": "袖子機械化",
    "bad_accessory_placement": "配件位置錯誤",
    "bad_pose": "動作錯誤",
    "copied_material": "太像原始素材",
    "bad_anatomy": "人體結構錯誤",
    "bad_hands": "手部錯誤",
    "bad_face": "五官錯誤",
    "bad_composition": "構圖錯誤",
    "cropped_body": "身體被裁切",
    "wrong_style": "畫風不符",
    "prompt_mismatch": "不符合提示詞",
    "too_busy": "畫面過度混亂",
}


def estimated_clip_tokens(prompt: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", prompt))
    latin_words = len(re.findall(r"[A-Za-z0-9_'-]+", prompt))
    punctuation = len(re.findall(r"[,.;:，。；：]", prompt))
    return cjk * 2 + latin_words + punctuation


class CandidateIssueSelect(discord.ui.Select):
    def __init__(self, bot: "StyleBot", session_id: str, generation_id: str):
        self.style_bot = bot
        self.session_id = session_id
        self.generation_id = generation_id
        super().__init__(
            placeholder="標記這張圖的優點或問題",
            min_values=1,
            max_values=5,
            options=[discord.SelectOption(label=label, value=value) for value, label in ISSUE_LABELS.items()],
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if not self.style_bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        for tag in self.values:
            self.style_bot.registry.tag_candidate(
                self.session_id, self.generation_id, str(interaction.user.id), tag
            )
        labels = "、".join(ISSUE_LABELS[tag] for tag in self.values)
        await interaction.response.send_message(f"✅ 已記錄：{labels}", ephemeral=True)


class CandidateView(discord.ui.View):
    def __init__(self, bot: "StyleBot", session_id: str, generation_id: str):
        super().__init__(timeout=86400)
        self.style_bot = bot
        self.session_id = session_id
        self.generation_id = generation_id
        self.add_item(CandidateIssueSelect(bot, session_id, generation_id))

    @discord.ui.button(label="選這張", style=discord.ButtonStyle.success)
    async def choose(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.style_bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        self.style_bot.registry.choose_candidate(self.session_id, self.generation_id)
        await interaction.response.send_message(
            f"✅ 已選擇 `{self.generation_id}`；同組其他選擇已取消。", ephemeral=True
        )


class PrepareIterationView(discord.ui.View):
    def __init__(self, bot: "StyleBot", model_id: str, summary: str):
        super().__init__(timeout=3600)
        self.style_bot = bot
        self.model_id = model_id
        self.summary = summary

    @discord.ui.button(label="確認建立迭代計畫", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.style_bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        iteration_id = self.style_bot.registry.create_iteration(self.model_id, self.summary)
        button.disabled = True
        await interaction.response.edit_message(
            content=(
                f"✅ 已建立 `{iteration_id}`。這只建立 v002 計畫，尚未啟動 GPU 訓練；"
                "完成 caption 清洗後再執行 `/train_style`。"
            ),
            view=self,
        )


class CurationModal(discord.ui.Modal, title="確認角色與服裝標註"):
    form = discord.ui.TextInput(label="角色／形態是否正確", max_length=100)
    structure = discord.ui.TextInput(label="主要服裝結構", style=discord.TextStyle.paragraph, max_length=500)
    materials = discord.ui.TextInput(label="材質與配件", style=discord.TextStyle.paragraph, max_length=500)
    notes = discord.ui.TextInput(label="排除內容或修正備註", style=discord.TextStyle.paragraph, required=False, max_length=500)

    def __init__(self, bot: "StyleBot", record: dict, focus: str):
        super().__init__()
        self.style_bot = bot
        self.record = record
        self.focus = focus
        self.form.default = record["preferred"]
        analysis = record.get("analysis", {})
        self.structure.default = analysis.get("structure", "")[:500]
        self.materials.default = analysis.get("materials", "")[:500]
        self.notes.default = f"自動設計點：{analysis.get('design_points', '待校對')}"[:500]

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if not self.style_bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        style = self.style_bot.settings.styles["arknights_portrait"]
        source = (
            self.style_bot.settings.project_root / "datasets" / "arknights_top50" /
            "curation" / self.record["filename"]
        )
        result = self.style_bot.store.ingest(
            style, source.read_bytes(), self.record["filename"],
            str(interaction.message.id), str(interaction.user.id),
        )
        self.style_bot.store.approve_one(style.style_id, result.sha256)
        accept_record(
            self.style_bot.settings.project_root,
            self.record["rank"],
            {
                "form": str(self.form), "structure": str(self.structure),
                "materials": str(self.materials), "notes": str(self.notes), "focus": self.focus,
            }, result.filename,
        )
        await interaction.response.edit_message(
            content=f"✅ **#{self.record['rank']} {self.record['group']}** 已確認並寫入 caption／審核文檔。",
            attachments=[], view=None,
        )


class CurationFocusSelect(discord.ui.Select):
    def __init__(self, view: "CurationView"):
        self.curation_view = view
        super().__init__(
            placeholder="先選這張圖最值得學習的部分",
            options=[
                discord.SelectOption(label="服裝結構", value="clothing structure"),
                discord.SelectOption(label="人物輪廓", value="character silhouette"),
                discord.SelectOption(label="材質與配件", value="materials and accessories"),
                discord.SelectOption(label="配色系統", value="color system"),
                discord.SelectOption(label="畫風（次要）", value="art style, secondary priority"),
            ],
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        self.curation_view.focus = self.values[0]
        await interaction.response.send_message(f"已選學習重點：`{self.values[0]}`", ephemeral=True)


class CurationView(discord.ui.View):
    def __init__(self, bot: "StyleBot", record: dict):
        super().__init__(timeout=604800)
        self.style_bot = bot
        self.record = record
        self.focus = "clothing structure"
        self.add_item(CurationFocusSelect(self))

    @discord.ui.button(label="確認並填寫標註", style=discord.ButtonStyle.success)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(CurationModal(self.style_bot, self.record, self.focus))

    @discord.ui.button(label="拒絕此素材", style=discord.ButtonStyle.danger)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.style_bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        update_record(self.style_bot.settings.project_root, self.record["rank"], status="rejected")
        await interaction.response.edit_message(content=f"❌ **#{self.record['rank']} {self.record['group']}** 已拒絕。", attachments=[], view=None)


class StyleBot(commands.Bot):
    def __init__(self, settings: Settings, store: DatasetStore):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!stylebot ", intents=intents)
        self.settings = settings
        self.store = store
        self.registry = ModelRegistry(settings)
        self.inference = InferenceRunner(settings)
        self.generation_lock = asyncio.Lock()
        self.background_tasks: set[asyncio.Task] = set()
        self.training_jobs: set[str] = set()
        self._training_scan_started = False
        self._synced = False
        self._curation_started = False

    async def setup_hook(self) -> None:
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            LOGGER.info("Slash commands synced to guild %s", self.settings.guild_id)
        else:
            await self.tree.sync()
            LOGGER.info("Slash commands synced globally")

    async def on_ready(self) -> None:
        LOGGER.info("Logged in as %s (%s)", self.user, getattr(self.user, "id", "?"))
        if not self._training_scan_started:
            self._training_scan_started = True
            await self.start_pending_training_monitors()
        if not self._curation_started:
            self._curation_started = True
            task = asyncio.create_task(self.post_pending_curation())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def post_pending_curation(self) -> None:
        manifest = self.settings.project_root / "state" / "arknights_top50_curation.json"
        style = self.settings.styles.get("arknights_portrait")
        if not manifest.exists() or not style or not style.discord_channel_id:
            return
        channel = self.get_channel(style.discord_channel_id) or await self.fetch_channel(style.discord_channel_id)
        records = json.loads(manifest.read_text(encoding="utf-8"))
        pending = [item for item in records if item.get("status") == "downloaded"]
        if not pending:
            return
        await channel.send(
            f"<@{next(iter(self.settings.allowed_user_ids), '')}>\n"
            f"📚 前 50 人氣角色素材審核開始：已解析 `{len(pending)}` 張。"
            "請先選學習重點，再按『確認並填寫標註』；未確認素材不會進入訓練。"
        )
        root = self.settings.project_root / "datasets" / "arknights_top50" / "curation"
        for record in pending:
            flags = []
            if record.get("collaboration"):
                flags.append("⚠️ 聯動角色")
            if record.get("unusual_form"):
                flags.append("⚠️ 非人形／特殊形態")
            warning = " · ".join(flags) or "標準角色素材"
            message = await channel.send(
                f"**#{record['rank']} {record['group']}** · `{record['preferred']}`\n"
                f"{warning}\n"
                f"**自動分層：** {record.get('analysis', {}).get('structure', '待分析')}\n"
                f"**設計點：** {record.get('analysis', {}).get('design_points', '待分析')}\n"
                f"**配件：** {record.get('analysis', {}).get('materials', '待分析')}\n"
                "來源：公開遊戲資源鏡像；你只需校對並選最佳學習部分。",
                file=discord.File(root / record["filename"]),
                view=CurationView(self, record),
            )
            update_record(self.settings.project_root, record["rank"], status="posted", discord_message_id=str(message.id))
            await asyncio.sleep(1.2)

    @staticmethod
    def training_progress(log_path: Path) -> tuple[int, int, float, str, str] | None:
        if not log_path.exists():
            return None
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 200_000))
            text = handle.read().decode("utf-8", errors="replace")
        matches = list(re.finditer(r"(\d+)/(\d+)\s+\[[^\]<]*<([^,\]]+)", text))
        if not matches:
            return None
        match = matches[-1]
        step, total = int(match.group(1)), int(match.group(2))
        losses = re.findall(r"avr_loss=([0-9.]+)", text[match.start():])
        return step, total, step / total * 100, match.group(3).strip(), losses[-1] if losses else "—"

    @staticmethod
    def progress_bar(percent: float, width: int = 20) -> str:
        filled = min(width, max(0, round(percent / 100 * width)))
        return "█" * filled + "░" * (width - filled)

    async def start_pending_training_monitors(self) -> None:
        for job_path in sorted((self.settings.data_root / "queues").glob("*.json")):
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("status") not in {"queued", "running"}:
                continue
            style = self.settings.styles.get(job.get("style_id", ""))
            if not style or not style.discord_channel_id:
                continue
            channel = self.get_channel(style.discord_channel_id)
            if channel is None:
                channel = await self.fetch_channel(style.discord_channel_id)
            task = asyncio.create_task(self.monitor_training_job(job_path, channel))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def monitor_training_job(self, job_path: Path, channel) -> None:
        job_id = job_path.stem
        if job_id in self.training_jobs:
            return
        self.training_jobs.add(job_id)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        mention = " ".join(f"<@{user_id}>" for user_id in self.settings.allowed_user_ids)
        message = await channel.send(
            f"{mention}\n🚀 訓練 `{job_id}` 準備中…\n`░░░░░░░░░░░░░░░░░░░░` 0%"
        )
        process = None
        try:
            if job.get("status") == "queued":
                python = self.settings.project_root / "work" / "sd-scripts" / "venv" / "Scripts" / "python.exe"
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(self.settings.project_root / "src")
                process = await asyncio.create_subprocess_exec(
                    str(python), "-m", "stylebot.trainer_main", "--job", job_path.name,
                    cwd=self.settings.project_root, env=environment,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
            log_path = self.settings.project_root / "work" / "training_runs" / job_id / "training.log"
            last_content = ""
            while True:
                await asyncio.sleep(15)
                job = json.loads(job_path.read_text(encoding="utf-8"))
                progress = self.training_progress(log_path)
                if progress:
                    step, total, percent, eta, loss = progress
                    content = (
                        f"{mention}\n🏋️ 訓練 `{job_id}` · `{job.get('version', '準備中')}`\n"
                        f"`{self.progress_bar(percent)}` **{percent:.1f}%**\n"
                        f"step `{step}/{total}` · loss `{loss}` · ETA `{eta}`"
                    )
                    if content != last_content:
                        await message.edit(content=content)
                        last_content = content
                if job.get("status") in {"completed", "failed"}:
                    break
                if process and process.returncode is not None:
                    break
            if process:
                await process.communicate()
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("status") == "completed":
                await message.edit(content=(
                    f"{mention}\n✅ 訓練 `{job_id}` 已完成 · `{job.get('version', '')}`\n"
                    f"`{self.progress_bar(100)}` **100%** · 已登錄 {job.get('registered_models', 0)} 個 checkpoint"
                ))
            else:
                await message.edit(content=f"{mention}\n❌ 訓練 `{job_id}` 失敗：{job.get('error', '請查看日誌')}")
        except Exception as exc:
            LOGGER.exception("Training monitor failed")
            await message.edit(content=f"{mention}\n❌ 訓練監控失敗：{str(exc)[-1000:]}")
        finally:
            self.training_jobs.discard(job_id)

    def is_allowed(self, user_id: int) -> bool:
        return not self.settings.allowed_user_ids or user_id in self.settings.allowed_user_ids

    def style_for_channel(self, channel_id: int) -> StyleConfig | None:
        return next(
            (style for style in self.settings.styles.values() if style.discord_channel_id == channel_id),
            None,
        )

    def is_style_forum_thread(self, channel: object) -> bool:
        return (
            isinstance(channel, discord.Thread)
            and self.settings.forum_channel_id is not None
            and channel.parent_id == self.settings.forum_channel_id
        )

    def register_thread_style(
        self,
        thread_id: int,
        style_id: str,
        display_name: str,
        minimum_approved_images: int,
        model_type: str,
    ) -> StyleConfig:
        normalized = re.sub(r"[^a-z0-9_-]+", "_", style_id.strip().lower()).strip("_")
        if not normalized:
            raise IngestError("style_id 只能使用小寫英文字母、數字、底線或連字號。")
        if any(
            item.discord_channel_id == thread_id and item.style_id != normalized
            for item in self.settings.styles.values()
        ):
            raise IngestError("這個子區已註冊為另一個風格。")
        style = StyleConfig(
            style_id=normalized,
            display_name=display_name.strip() or normalized,
            discord_channel_id=thread_id,
            trigger_token=f"{normalized}_style",
            minimum_approved_images=minimum_approved_images,
            model_type=model_type,
            enabled=True,
        )
        self.settings.styles[normalized] = style
        path = self.settings.project_root / "config" / "styles.local.yaml"
        payload = {
            "styles": {
                item.style_id: {
                    "display_name": item.display_name,
                    "discord_channel_id": str(item.discord_channel_id or ""),
                    "trigger_token": item.trigger_token,
                    "minimum_approved_images": item.minimum_approved_images,
                    "model_type": item.model_type,
                    "enabled": item.enabled,
                }
                for item in self.settings.styles.values()
            }
        }
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        for bucket in ("incoming", "approved", "rejected", "captions"):
            (self.settings.data_root / "datasets" / normalized / bucket).mkdir(
                parents=True, exist_ok=True
            )
        return style

    def style_by_name(self, value: str) -> StyleConfig | None:
        normalized = value.strip().casefold()
        return next(
            (
                style
                for style in self.settings.styles.values()
                if normalized in {style.style_id.casefold(), style.display_name.casefold()}
            ),
            None,
        )

    async def ingest_attachment(
        self,
        style: StyleConfig,
        attachment: discord.Attachment,
        message_id: str,
        user_id: str,
    ) -> str:
        if attachment.size > self.settings.max_attachment_mb * 1024 * 1024:
            return f"❌ `{attachment.filename}` 超過大小限制"
        try:
            payload = await attachment.read(use_cached=True)
            result = self.store.ingest(style, payload, attachment.filename, message_id, user_id)
        except IngestError as exc:
            return f"❌ `{attachment.filename}`：{exc}"
        except discord.HTTPException:
            LOGGER.exception("Failed to download attachment %s", attachment.id)
            return f"❌ `{attachment.filename}`：下載失敗"
        if result.status == "duplicate":
            return f"♻️ `{attachment.filename}` 已存在於 `{result.style_id}`"
        return f"✅ `{attachment.filename}` → **{style.display_name}**（{result.width}×{result.height}）"

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.attachments:
            return
        if not self.is_allowed(message.author.id):
            return
        style = self.style_for_channel(message.channel.id)
        if not style:
            if self.is_style_forum_thread(message.channel):
                await message.reply(
                    "這個子區尚未註冊。請先執行 `/register_style`。", mention_author=False
                )
            return
        responses = [
            await self.ingest_attachment(style, item, str(message.id), str(message.author.id))
            for item in message.attachments
        ]
        await message.reply("\n".join(responses), mention_author=False)


def register_commands(bot: StyleBot) -> None:
    async def style_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        del interaction
        current = current.casefold()
        return [
            app_commands.Choice(name=style.display_name, value=style.style_id)
            for style in bot.settings.styles.values()
            if current in style.style_id.casefold() or current in style.display_name.casefold()
        ][:25]

    async def model_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        del interaction
        current = current.casefold()
        return [
            app_commands.Choice(
                name=f"{model.style_id} {model.version} {model.checkpoint} [{model.status}]",
                value=model.model_id,
            )
            for model in bot.registry.list_models()
            if current in model.model_id.casefold()
        ][:25]

    async def production_model_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        del interaction
        current = current.casefold()
        return [
            app_commands.Choice(
                name=f"{model.style_id} {model.version} {model.checkpoint}",
                value=model.model_id,
            )
            for model in bot.registry.list_models(production_only=True)
            if current in model.model_id.casefold()
        ][:25]

    def checkpoint_candidates(parent) -> list:
        items = [
            item for item in bot.registry.list_models(parent.style_id)
            if item.version == parent.version
        ]
        final = next((item for item in items if item.checkpoint == "final"), None)
        epochs = sorted(
            (item for item in items if item.checkpoint.startswith("epoch-")),
            key=lambda item: int(item.checkpoint.rsplit("-", 1)[-1]),
        )
        if not epochs:
            return [final] if final else []
        indexes = {
            min(len(epochs) - 1, max(0, math.ceil(len(epochs) * fraction) - 1))
            for fraction in (0.25, 0.50, 0.75)
        }
        selected = [epochs[index] for index in sorted(indexes)]
        if final:
            selected.append(final)
        elif epochs[-1] not in selected:
            selected.append(epochs[-1])
        return selected[:4]

    async def run_continuous_generation(
        run_id: str, channel: discord.abc.Messageable, parent,
        prompt: str, strength: float, seed: int, rounds: int,
    ) -> None:
        candidates = checkpoint_candidates(parent)
        try:
            for round_index in range(rounds):
                if not bot.registry.continuous_run_active(run_id):
                    break
                round_seed = seed + round_index
                session_id = bot.registry.create_comparison(
                    parent.style_id, parent.version, prompt, ITERATION_NEGATIVE, round_seed
                )
                await channel.send(
                    f"🔄 `{run_id}` · 第 {round_index + 1}/{rounds} 輪 · "
                    f"seed `{round_seed}` · strength `{strength}`"
                )
                for index, candidate in enumerate(candidates, start=1):
                    if not bot.registry.continuous_run_active(run_id):
                        break
                    async with bot.generation_lock:
                        result = await asyncio.to_thread(
                            bot.inference.generate,
                            [(candidate, strength)], prompt, ITERATION_NEGATIVE,
                            round_seed, 768, 1024, "comparison",
                        )
                    generation_id = bot.registry.record_generation(
                        candidate.model_id, prompt, ITERATION_NEGATIVE,
                        result.seed, strength, result.image_path, "comparison",
                    )
                    bot.registry.add_comparison_candidate(
                        session_id, generation_id, candidate.model_id, strength
                    )
                    message = await channel.send(
                        content=(
                            f"候選 {index}/{len(candidates)} · `{candidate.model_id}`\n"
                            f"比較組 `{session_id}` · generation `{generation_id}`"
                        ),
                        file=discord.File(result.image_path),
                        view=CandidateView(bot, session_id, generation_id),
                    )
                    bot.registry.attach_message(generation_id, str(message.id))
                if bot.registry.continuous_run_active(run_id):
                    bot.registry.complete_continuous_round(run_id)
            status = "已完成" if bot.registry.continuous_run_active(run_id) is False else "已結束"
            await channel.send(f"⏹️ 連續評測 `{run_id}` {status}。可使用 `/iteration_report` 查看統計。")
        except Exception as exc:
            LOGGER.exception("Continuous iteration failed")
            bot.registry.stop_continuous_runs(parent.style_id)
            await channel.send(f"連續評測 `{run_id}` 失敗：{str(exc)[-1200:]}")

    @bot.tree.command(name="upload_style", description="上傳一張圖片到指定風格資料集")
    @app_commands.describe(style="風格名稱", image="JPEG、PNG 或 WebP 圖片")
    @app_commands.autocomplete(style=style_autocomplete)
    async def upload_style(
        interaction: discord.Interaction, style: str, image: discord.Attachment
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.ingest_attachment(
            selected, image, str(interaction.id), str(interaction.user.id)
        )
        await interaction.followup.send(result, ephemeral=True)

    @bot.tree.command(name="register_style", description="將目前論壇子區註冊為一個新風格")
    @app_commands.describe(
        style_id="資料夾識別名稱，例如 arknights_portrait",
        display_name="顯示名稱；留空時使用目前子區名稱",
        minimum_images="允許建立訓練工作的最低圖片數",
        model_type="design、art、character 或 control",
    )
    async def register_style(
        interaction: discord.Interaction,
        style_id: str,
        display_name: str = "",
        minimum_images: app_commands.Range[int, 1, 1000] = 50,
        model_type: str = "design",
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        channel = interaction.channel
        if channel is None or not bot.is_style_forum_thread(channel):
            await interaction.response.send_message(
                "這個指令只能在設定的訓練論壇子區內使用。", ephemeral=True
            )
            return
        try:
            model_type = model_type.strip().lower()
            if model_type not in {"design", "art", "character", "control"}:
                raise IngestError("model_type 必須是 design、art、character 或 control。")
            style = bot.register_thread_style(
                channel.id,
                style_id,
                display_name or getattr(channel, "name", style_id),
                minimum_images,
                model_type,
            )
        except IngestError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"✅ 此子區已註冊為 **{style.display_name}** (`{style.style_id}`)。",
            ephemeral=True,
        )

    @bot.tree.command(name="material_status", description="查看風格素材數量")
    @app_commands.describe(style="風格名稱")
    @app_commands.autocomplete(style=style_autocomplete)
    async def material_status(interaction: discord.Interaction, style: str) -> None:
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        counts = bot.store.status(selected.style_id)
        await interaction.response.send_message(
            f"**{selected.display_name}**\n待審核：{counts['incoming']}\n已批准：{counts['approved']}\n已拒絕：{counts['rejected']}",
            ephemeral=True,
        )

    @bot.tree.command(name="pending_images", description="列出等待審核的圖片")
    @app_commands.describe(style="風格名稱", limit="顯示數量")
    @app_commands.autocomplete(style=style_autocomplete)
    async def pending_images(
        interaction: discord.Interaction,
        style: str,
        limit: app_commands.Range[int, 1, 25] = 10,
    ) -> None:
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        rows = bot.store.pending(selected.style_id, limit)
        if not rows:
            await interaction.response.send_message("目前沒有待審核圖片。", ephemeral=True)
            return
        lines = [
            f"`{row['sha256'][:12]}` {row['filename']} ({row['width']}×{row['height']})"
            for row in rows
        ]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @bot.tree.command(name="approve_style", description="批准風格的待審核素材")
    @app_commands.describe(style="風格名稱", limit="本次最多批准數量")
    @app_commands.autocomplete(style=style_autocomplete)
    async def approve_style(
        interaction: discord.Interaction, style: str, limit: app_commands.Range[int, 1, 500] = 100
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        count = bot.store.approve(selected.style_id, limit)
        await interaction.response.send_message(
            f"已批准 **{selected.display_name}** 的 {count} 張素材。", ephemeral=True
        )

    @bot.tree.command(name="reject_style", description="拒絕風格的待審核素材")
    @app_commands.describe(style="風格名稱", limit="本次最多拒絕數量")
    @app_commands.autocomplete(style=style_autocomplete)
    async def reject_style(
        interaction: discord.Interaction, style: str, limit: app_commands.Range[int, 1, 500] = 100
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        count = bot.store.reject(selected.style_id, limit)
        await interaction.response.send_message(
            f"已拒絕 **{selected.display_name}** 的 {count} 張素材。", ephemeral=True
        )

    @bot.tree.context_menu(name="Reject training images")
    async def reject_training_images(
        interaction: discord.Interaction, message: discord.Message
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        selected = bot.style_for_channel(message.channel.id)
        if not selected:
            await interaction.response.send_message("這不是已註冊的風格子區。", ephemeral=True)
            return
        count = bot.store.reject(selected.style_id, 100, str(message.id))
        await interaction.response.send_message(
            f"已拒絕這則訊息中的 {count} 張待審核圖片。", ephemeral=True
        )

    @bot.tree.command(name="train_style", description="建立指定風格的訓練工作")
    @app_commands.describe(style="風格名稱")
    @app_commands.autocomplete(style=style_autocomplete)
    async def train_style(interaction: discord.Interaction, style: str) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        selected = bot.style_by_name(style)
        if not selected:
            await interaction.response.send_message("找不到這個風格。", ephemeral=True)
            return
        try:
            job = bot.store.queue_training(selected)
        except IngestError as exc:
            await interaction.response.send_message(f"尚未建立訓練：{exc}", ephemeral=True)
            return
        await interaction.response.send_message(
            f"✅ 已建立訓練工作 `{job.stem}`，正在啟動；進度會顯示在此子區。", ephemeral=True
        )
        task = asyncio.create_task(bot.monitor_training_job(job, interaction.channel))
        bot.background_tasks.add(task)
        task.add_done_callback(bot.background_tasks.discard)

    @bot.tree.command(name="models", description="查看已註冊的模型版本")
    async def models(interaction: discord.Interaction) -> None:
        records = bot.registry.list_models()
        if not records:
            await interaction.response.send_message("目前沒有已完成的模型。", ephemeral=True)
            return
        lines = [
            f"`{item.model_id}` · {item.model_type} · **{item.status}**"
            for item in records[:25]
        ]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @bot.tree.command(name="promote_model", description="將驗收通過的模型發布為 production")
    @app_commands.describe(model="要發布的模型 checkpoint")
    @app_commands.autocomplete(model=model_autocomplete)
    async def promote_model(interaction: discord.Interaction, model: str) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        try:
            record = bot.registry.promote(model)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"✅ `{record.model_id}` 已發布為 production。", ephemeral=False
        )

    @bot.tree.command(name="review_model", description="在子區生成模型驗收圖")
    @app_commands.describe(
        model="要驗收的 checkpoint",
        prompt="測試提示詞；留空使用服飾設計標準測試",
        strength="LoRA 強度",
        seed="固定種子；-1 表示隨機",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def review_model(
        interaction: discord.Interaction,
        model: str,
        prompt: str = "",
        strength: app_commands.Range[float, 0.1, 1.5] = 0.8,
        seed: int = 42,
    ) -> None:
        record = bot.registry.get_model(model)
        if not record:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        style = bot.settings.styles.get(record.style_id)
        if not style or interaction.channel_id != style.discord_channel_id:
            await interaction.response.send_message(
                "請在此模型所屬的訓練子區執行驗收。", ephemeral=True
            )
            return
        prompt = prompt.strip() or (
            f"{style.trigger_token}, character costume design, full body character, "
            "detailed layered outfit, functional accessories, neutral standing pose, simple background"
        )
        if style.trigger_token.lower() not in prompt.lower():
            prompt = f"{style.trigger_token}, {prompt}"
        await interaction.response.defer(thinking=True)
        try:
            async with bot.generation_lock:
                result = await asyncio.to_thread(
                    bot.inference.generate,
                    [(record, strength)],
                    prompt,
                    ITERATION_NEGATIVE,
                    seed,
                    768,
                    1024,
                    "review",
                )
            generation_id = bot.registry.record_generation(
                record.model_id, prompt, ITERATION_NEGATIVE, result.seed, strength,
                result.image_path, "review"
            )
            message = await interaction.followup.send(
                content=(
                    f"驗收 `{record.model_id}` · seed `{result.seed}` · strength `{strength}`\n"
                    f"generation `{generation_id}`"
                ),
                file=discord.File(result.image_path),
                wait=True,
            )
            bot.registry.attach_message(generation_id, str(message.id))
        except Exception as exc:
            LOGGER.exception("Review generation failed")
            await interaction.followup.send(f"生圖失敗：{str(exc)[-1500:]}", ephemeral=True)

    @bot.tree.command(name="feedback", description="記錄模型或驗收圖的結構化意見")
    @app_commands.describe(
        model="模型 checkpoint",
        category="clothing、silhouette、color、character、art、pose 或 other",
        comment="希望保留或修改的內容",
        generation_id="可選：驗收圖的 generation ID",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def feedback(
        interaction: discord.Interaction,
        model: str,
        category: str,
        comment: str,
        generation_id: str = "",
    ) -> None:
        if not bot.registry.get_model(model):
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        category = category.strip().lower()
        if category not in {"clothing", "silhouette", "color", "character", "art", "pose", "other"}:
            await interaction.response.send_message("不支援的 feedback category。", ephemeral=True)
            return
        feedback_id = bot.registry.add_feedback(
            model, str(interaction.user.id), category, comment, generation_id or None
        )
        await interaction.response.send_message(
            f"✅ 意見已記錄：`{feedback_id}`。", ephemeral=True
        )

    @bot.tree.command(name="start_iteration", description="根據回饋建立下一版本的迭代計畫")
    @app_commands.describe(model="作為基準的模型", change_summary="下一版要保留與修改的內容")
    @app_commands.autocomplete(model=model_autocomplete)
    async def start_iteration(
        interaction: discord.Interaction, model: str, change_summary: str
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        try:
            iteration_id = bot.registry.create_iteration(model, change_summary)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"✅ 已建立迭代 `{iteration_id}`。整理素材後再次使用 `/train_style` 建立下一版本。",
            ephemeral=False,
        )

    @bot.tree.command(name="review_suite", description="用固定提示詞生成四張標準驗收圖")
    @app_commands.describe(model="要驗收的 checkpoint", strength="LoRA 強度", seed="起始種子")
    @app_commands.autocomplete(model=model_autocomplete)
    async def review_suite(
        interaction: discord.Interaction,
        model: str,
        strength: app_commands.Range[float, 0.1, 1.5] = 0.8,
        seed: int = 42,
    ) -> None:
        record = bot.registry.get_model(model)
        if not record:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        style = bot.settings.styles.get(record.style_id)
        if not style or interaction.channel_id != style.discord_channel_id:
            await interaction.response.send_message("請在模型所屬子區執行。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            async with bot.generation_lock:
                for index, base_prompt in enumerate(REVIEW_PROMPTS, start=1):
                    prompt = f"{style.trigger_token}, {base_prompt}"
                    result = await asyncio.to_thread(
                        bot.inference.generate,
                        [(record, strength)], prompt, "", seed + index - 1,
                        1024, 1024, "review",
                    )
                    generation_id = bot.registry.record_generation(
                        record.model_id, prompt, "", result.seed, strength,
                        result.image_path, "review"
                    )
                    message = await interaction.followup.send(
                        content=(
                            f"驗收 {index}/4 · `{record.model_id}` · seed `{result.seed}`\n"
                            f"generation `{generation_id}`"
                        ),
                        file=discord.File(result.image_path),
                        wait=True,
                    )
                    bot.registry.attach_message(generation_id, str(message.id))
        except Exception as exc:
            LOGGER.exception("Review suite failed")
            await interaction.followup.send(f"驗收套件失敗：{str(exc)[-1500:]}", ephemeral=True)

    @bot.tree.context_menu(name="Select review image")
    async def select_review_image(
        interaction: discord.Interaction, message: discord.Message
    ) -> None:
        generation_id = bot.registry.select_by_message(str(message.id))
        if not generation_id:
            await interaction.response.send_message("這不是已記錄的驗收圖片。", ephemeral=True)
            return
        await interaction.response.send_message(
            f"✅ 已選擇驗收圖片 `{generation_id}`。", ephemeral=True
        )

    @bot.tree.command(name="iterate_test", description="以相同條件生成四張 checkpoint 候選圖")
    @app_commands.describe(
        model="作為基準的模型版本",
        prompt="建議使用精簡英文提示詞；系統會自動補觸發詞",
        strength="統一 LoRA 強度",
        seed="四張圖共用的固定 seed",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def iterate_test(
        interaction: discord.Interaction,
        model: str,
        prompt: str,
        strength: app_commands.Range[float, 0.1, 1.2] = 0.55,
        seed: int = 42,
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        parent = bot.registry.get_model(model)
        if not parent:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        style = bot.settings.styles.get(parent.style_id)
        if not style or interaction.channel_id != style.discord_channel_id:
            await interaction.response.send_message("請在模型所屬子區執行。", ephemeral=True)
            return
        clean_prompt = prompt.strip()
        if not clean_prompt:
            await interaction.response.send_message("提示詞不能為空。", ephemeral=True)
            return
        if style.trigger_token.lower() not in clean_prompt.lower():
            clean_prompt = f"{style.trigger_token}, {clean_prompt}"
        budget = estimated_clip_tokens(clean_prompt)
        if budget > 70:
            await interaction.response.send_message(
                f"提示詞估算為 {budget} tokens，可能超過 SDXL 的 77-token 上限。"
                "請縮短到 70 以下再生成。",
                ephemeral=True,
            )
            return
        candidates = checkpoint_candidates(parent)
        if len(candidates) < 2:
            await interaction.response.send_message("同版本至少需要兩個 checkpoint。", ephemeral=True)
            return
        session_id = bot.registry.create_comparison(
            parent.style_id, parent.version, clean_prompt, ITERATION_NEGATIVE, seed
        )
        await interaction.response.defer(thinking=True)
        try:
            async with bot.generation_lock:
                for index, candidate in enumerate(candidates, start=1):
                    result = await asyncio.to_thread(
                        bot.inference.generate,
                        [(candidate, strength)], clean_prompt, ITERATION_NEGATIVE,
                        seed, 768, 1024, "comparison",
                    )
                    generation_id = bot.registry.record_generation(
                        candidate.model_id, clean_prompt, ITERATION_NEGATIVE,
                        result.seed, strength, result.image_path, "comparison",
                    )
                    bot.registry.add_comparison_candidate(
                        session_id, generation_id, candidate.model_id, strength
                    )
                    message = await interaction.followup.send(
                        content=(
                            f"候選 {index}/{len(candidates)} · `{candidate.model_id}` · "
                            f"strength `{strength}` · seed `{seed}`\n"
                            f"比較組 `{session_id}` · generation `{generation_id}`"
                        ),
                        file=discord.File(result.image_path),
                        view=CandidateView(bot, session_id, generation_id),
                        wait=True,
                    )
                    bot.registry.attach_message(generation_id, str(message.id))
        except Exception as exc:
            LOGGER.exception("Iteration comparison failed")
            await interaction.followup.send(f"候選生成失敗：{str(exc)[-1500:]}", ephemeral=True)

    @bot.tree.command(name="iterate_auto", description="連續生成多輪四圖比較，直到完成或手動停止")
    @app_commands.describe(
        model="作為基準的模型版本",
        prompt="精簡英文提示詞；系統自動補觸發詞",
        rounds="生成輪數；每輪四張，最多十輪",
        strength="統一 LoRA 強度",
        seed="第一輪 seed；後續每輪自動加一",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def iterate_auto(
        interaction: discord.Interaction,
        model: str,
        prompt: str,
        rounds: app_commands.Range[int, 1, 10] = 3,
        strength: app_commands.Range[float, 0.1, 1.2] = 0.50,
        seed: int = 42,
    ) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        parent = bot.registry.get_model(model)
        if not parent:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        style = bot.settings.styles.get(parent.style_id)
        if not style or interaction.channel_id != style.discord_channel_id:
            await interaction.response.send_message("請在模型所屬子區執行。", ephemeral=True)
            return
        clean_prompt = prompt.strip()
        if not clean_prompt:
            await interaction.response.send_message("提示詞不能為空。", ephemeral=True)
            return
        if style.trigger_token.lower() not in clean_prompt.lower():
            clean_prompt = f"{style.trigger_token}, {clean_prompt}"
        budget = estimated_clip_tokens(clean_prompt)
        if budget > 70:
            await interaction.response.send_message(
                f"提示詞估算為 {budget} tokens；請縮短到 70 以下，避免再次亂生成。",
                ephemeral=True,
            )
            return
        if len(checkpoint_candidates(parent)) < 2:
            await interaction.response.send_message("同版本至少需要兩個 checkpoint。", ephemeral=True)
            return
        run_id = bot.registry.create_continuous_run(
            parent.style_id, parent.version, clean_prompt, ITERATION_NEGATIVE,
            strength, seed, rounds, str(interaction.channel_id),
        )
        await interaction.response.send_message(
            f"✅ 已啟動 `{run_id}`：{rounds} 輪、每輪四個 checkpoint。"
            "你可以邊生成邊選圖；使用 `/stop_iteration` 可安全停止。",
            ephemeral=False,
        )
        task = asyncio.create_task(
            run_continuous_generation(
                run_id, interaction.channel, parent, clean_prompt, strength, seed, rounds
            )
        )
        bot.background_tasks.add(task)
        task.add_done_callback(bot.background_tasks.discard)

    @bot.tree.command(name="stop_iteration", description="停止目前風格的連續評測")
    @app_commands.describe(model="此風格的任一模型")
    @app_commands.autocomplete(model=model_autocomplete)
    async def stop_iteration(interaction: discord.Interaction, model: str) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        record = bot.registry.get_model(model)
        if not record:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        count = bot.registry.stop_continuous_runs(record.style_id)
        await interaction.response.send_message(
            f"⏹️ 已停止 {count} 個連續評測；目前正在完成的單張圖不會被強制中斷。",
            ephemeral=True,
        )

    @bot.tree.command(name="iteration_report", description="查看 checkpoint 勝率與缺陷統計")
    @app_commands.describe(model="此風格的任一模型")
    @app_commands.autocomplete(model=model_autocomplete)
    async def iteration_report(interaction: discord.Interaction, model: str) -> None:
        record = bot.registry.get_model(model)
        if not record:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        report = bot.registry.comparison_report(record.style_id)
        candidate_lines = []
        for row in report["candidates"][:8]:
            rate = (row["wins"] / row["appearances"] * 100) if row["appearances"] else 0
            candidate_lines.append(
                f"`{row['model_id']}` · strength {row['strength']:.2f} · "
                f"{row['wins']}/{row['appearances']} 勝（{rate:.0f}%）"
            )
        tag_lines = [f"{ISSUE_LABELS.get(row['tag'], row['tag'])}：{row['count']}" for row in report["tags"]]
        content = (
            f"比較組：{report['sessions']}\n\n**候選表現**\n"
            + ("\n".join(candidate_lines) or "尚無選擇資料")
            + "\n\n**標記統計**\n"
            + ("\n".join(tag_lines) or "尚無缺陷標記")
        )
        await interaction.response.send_message(content, ephemeral=True)

    @bot.tree.command(name="prepare_iteration", description="根據比較結果準備下一版迭代計畫")
    @app_commands.describe(model="下一版的父模型")
    @app_commands.autocomplete(model=model_autocomplete)
    async def prepare_iteration(interaction: discord.Interaction, model: str) -> None:
        if not bot.is_allowed(interaction.user.id):
            await interaction.response.send_message("你沒有使用權限。", ephemeral=True)
            return
        record = bot.registry.get_model(model)
        if not record:
            await interaction.response.send_message("找不到模型。", ephemeral=True)
            return
        report = bot.registry.comparison_report(record.style_id)
        if report["sessions"] < 1:
            await interaction.response.send_message("請先完成至少一組 `/iterate_test`。", ephemeral=True)
            return
        defects = [
            ISSUE_LABELS.get(row["tag"], row["tag"])
            for row in report["tags"]
            if not row["tag"].startswith("good_")
        ]
        summary = (
            "v002 human-guided iteration: preserve selected clothing/design results; "
            f"prioritize fixing {', '.join(defects[:5]) or 'caption consistency and anatomy'}; "
            "clean contradictory captions; use repeats=3, epochs=8, learning_rate=5e-5; "
            "do not add generated comparison images to the training dataset."
        )
        await interaction.response.send_message(
            f"準備建立以下迭代計畫：\n```{summary}```\n確認後仍不會直接啟動 GPU 訓練。",
            view=PrepareIterationView(bot, record.model_id, summary),
            ephemeral=True,
        )

    @bot.tree.command(name="generate", description="使用已發布模型直接生圖")
    @app_commands.describe(
        model="主要 production 模型",
        prompt="提示詞",
        strength="主要 LoRA 強度",
        art_model="可選的第二個 production 畫風模型",
        art_strength="第二個 LoRA 強度",
        pose="可選的姿勢參考圖片",
        seed="固定種子；-1 表示隨機",
    )
    @app_commands.autocomplete(model=production_model_autocomplete, art_model=production_model_autocomplete)
    async def generate(
        interaction: discord.Interaction,
        model: str,
        prompt: str,
        strength: app_commands.Range[float, 0.1, 1.5] = 0.8,
        art_model: str = "",
        art_strength: app_commands.Range[float, 0.1, 1.5] = 0.5,
        pose: discord.Attachment | None = None,
        seed: int = -1,
    ) -> None:
        primary = bot.registry.get_model(model)
        if not primary or primary.status != "production":
            await interaction.response.send_message("主要模型尚未發布為 production。", ephemeral=True)
            return
        adapters = [(primary, strength)]
        if art_model:
            secondary = bot.registry.get_model(art_model)
            if not secondary or secondary.status != "production":
                await interaction.response.send_message("第二個模型尚未發布。", ephemeral=True)
                return
            adapters.append((secondary, art_strength))
        pose_path: Path | None = None
        if pose:
            if pose.size > bot.settings.max_attachment_mb * 1024 * 1024:
                await interaction.response.send_message("姿勢圖片過大。", ephemeral=True)
                return
            pose_root = bot.settings.project_root / "work" / "pose_inputs"
            pose_root.mkdir(parents=True, exist_ok=True)
            suffix = Path(pose.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                await interaction.response.send_message("姿勢附件必須是圖片。", ephemeral=True)
                return
            pose_path = pose_root / f"{uuid.uuid4().hex[:16]}{suffix}"
            pose_path.write_bytes(await pose.read(use_cached=True))
        await interaction.response.defer(thinking=True)
        full_prompt = f"{bot.settings.styles[primary.style_id].trigger_token}, {prompt}"
        try:
            async with bot.generation_lock:
                result = await asyncio.to_thread(
                    bot.inference.generate,
                    adapters,
                    full_prompt,
                    "",
                    seed,
                    1024,
                    1024,
                    "generate",
                    pose_path,
                    0.8,
                )
            generation_id = bot.registry.record_generation(
                primary.model_id, full_prompt, "", result.seed, strength,
                result.image_path, "generate", pose_path.as_posix() if pose_path else None
            )
            message = await interaction.followup.send(
                content=f"seed `{result.seed}` · generation `{generation_id}`",
                file=discord.File(result.image_path),
                wait=True,
            )
            bot.registry.attach_message(generation_id, str(message.id))
        except Exception as exc:
            LOGGER.exception("Generation failed")
            await interaction.followup.send(f"生圖失敗：{str(exc)[-1500:]}", ephemeral=True)


def create_bot(settings: Settings) -> StyleBot:
    store = DatasetStore(settings)
    bot = StyleBot(settings, store)
    register_commands(bot)
    return bot
