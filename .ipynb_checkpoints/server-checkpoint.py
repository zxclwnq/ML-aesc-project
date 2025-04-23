import logging
import asyncio
from aiogram import Bot, Dispatcher
from ApiKeys import TgApiKey
from handlers import router

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    # Инициализация бота и диспетчера
    bot = Bot(token=TgApiKey.token)
    dp = Dispatcher()

    # Подключение роутеров с обработчиками
    dp.include_router(router)

    # Запуск polling
    logger.info("Бот запущен и готов к работе!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен пользователем")
