import os
from aiogram import Bot
import logging
from models import to_tensor, Identifier


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


async def download_photo(bot: Bot, file_id: str, destination: str):
    try:
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, destination)
        logger.info(f"Фото загружено: {destination}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке фото: {e}")

def save_additional_info(path: str, info: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(info)
        logger.info(f"Дополнительная информация сохранена: {path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении информации: {e}")