from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

import discord
import yaml
from discord import app_commands
from discord.ext import commands

from .config import Settings, StyleConfig
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
        self._synced = False

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
            f"✅ 已建立訓練工作 `{job.stem}`。", ephemeral=True
        )

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
        await interaction.response.defer(thinking=True)
        try:
            async with bot.generation_lock:
                result = await asyncio.to_thread(
                    bot.inference.generate,
                    [(record, strength)],
                    prompt,
                    "",
                    seed,
                    1024,
                    1024,
                    "review",
                )
            generation_id = bot.registry.record_generation(
                record.model_id, prompt, "", result.seed, strength,
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
