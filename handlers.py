import os
import json
import logging
import torch
from aiogram import types, Router
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from models import to_tensor, Recognizer, CatsVsDogs, Identifier, BreedExtractor
from utils import download_photo

# Пути для хранения
LOST_PATH = "lost_pets_images"
FOUND_PATH = "found_pets_images"
os.makedirs(LOST_PATH, exist_ok=True)
os.makedirs(FOUND_PATH, exist_ok=True)

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Клавиатуры
main_menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Я потерял питомца")],
        [KeyboardButton(text="Я нашёл питомца")],
        [KeyboardButton(text="Каталог найденных питомцев")],
    ], resize_keyboard=True
)

type_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Кошку")],
        [KeyboardButton(text="Собаку")],
        [KeyboardButton(text="Назад")],
    ], resize_keyboard=True
)

back_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Назад")] ], resize_keyboard=True
)

# Кнопка «Без описания» для info
info_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Без описания")],
        [KeyboardButton(text="Назад")],
    ], resize_keyboard=True
)

browse_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Следующее")],
        [KeyboardButton(text="Назад")],
    ], resize_keyboard=True
)

# FSM-состояния
class LostPetStates(StatesGroup):
    waiting_for_type = State()
    waiting_for_photos = State()
    waiting_for_info = State()

class FoundPetStates(StatesGroup):
    waiting_for_type = State()
    waiting_for_photos = State()
    waiting_for_info = State()

class CatalogStates(StatesGroup):
    choosing_type = State()
    browsing = State()

# Создаём роутер
router = Router()

# --- Общие хэндлеры ---
@router.message(lambda m: m.text == "/start")
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Добро пожаловать! Чем я могу помочь?", reply_markup=main_menu_keyboard)

@router.message(lambda m: m.text == "Назад")
async def cmd_back(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Вы вернулись в главное меню.", reply_markup=main_menu_keyboard)

# --- Flow: Я потерял питомца ---
@router.message(lambda m: m.text == "Я потерял питомца")
async def lost_pet_start(message: types.Message, state: FSMContext):
    await state.set_state(LostPetStates.waiting_for_type)
    await message.answer("Кого вы потеряли?", reply_markup=type_keyboard)

@router.message(LostPetStates.waiting_for_type)
async def lost_pet_type(message: types.Message, state: FSMContext):
    if message.text not in ["Кошку", "Собаку"]:
        return await message.answer("Пожалуйста, выберите категорию 'Кошка' или 'Собака'.")
    await state.update_data(pet_type=message.text)
    await state.set_state(LostPetStates.waiting_for_photos)
    await message.answer("Пришлите фото потерянного питомца.", reply_markup=back_keyboard)

@router.message(LostPetStates.waiting_for_photos, lambda m: m.content_type == types.ContentType.PHOTO)
async def lost_pet_photos(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    # Проверка размера
    if photo.file_size and photo.file_size > 5 * 1024 * 1024:
        return await message.answer("Файл >5МБ. Отправьте фото меньшего размера.")

    data = await state.get_data()
    pet_type = data['pet_type']
    file_id = photo.file_id
    folder = os.path.join(LOST_PATH, f"{message.chat.id}_{file_id}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_id}.jpg")
    await download_photo(message.bot, file_id, path)

    # Валидация изображения
    try:
        from PIL import Image
        Image.open(path).verify()
    except Exception:
        os.remove(path)
        return await message.answer("Это некорректное изображение. Попробуйте ещё раз.")

    # Классификация питомца
    if torch.argmax(Recognizer(to_tensor(path)), dim=1).item() != 0:
        os.remove(path)
        return await message.answer("Это не похоже на питомца. Попробуйте другое фото.")
    is_cat = torch.argmax(CatsVsDogs(to_tensor(path)), dim=1).item() == 0
    if (pet_type == "Кошку" and not is_cat) or (pet_type == "Собаку" and is_cat):
        os.remove(path)
        return await message.answer("Тип животного не совпадает.")

    photos = data.get('photos', []) + [path]
    await state.update_data(photos=photos)
    # Предлагаем описать или пропустить
    await message.answer("Фото сохранено.\n Напишите дополнительную информацию, например, где нашли или нажмите на кнопку ниже.", reply_markup=info_keyboard)

@router.message(LostPetStates.waiting_for_photos)
async def prompt_lost_info(message: types.Message, state: FSMContext):
    await state.set_state(LostPetStates.waiting_for_info)
    await lost_pet_info(message, state)

@router.message(LostPetStates.waiting_for_info)
async def lost_pet_info(message: types.Message, state: FSMContext):
    data = await state.get_data()
    photos = data['photos']
    additional = message.text if message.text != "Без описания" else ""
    # Добавляем тег владельца для связи
    owner_tag = f"@{message.from_user.username}" if message.from_user.username else None
    info = {
        'type': data['pet_type'],
        'info': additional,
        'owner_id': message.chat.id,
        'owner_tag': owner_tag
    }
    info_path = os.path.join(os.path.dirname(photos[0]), 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Отправляем подтверждение с тегом владельца
    confirm_text = f"Информация сохранена!\nС вами могут связаться в ближайшее время!"
    await message.answer(confirm_text, reply_markup=main_menu_keyboard)

    # Поиск найденных питомцев
    threshold = 0.2
    best_dist = float('inf')
    best_match = None
    for root, _, files in os.walk(FOUND_PATH):
        if 'info.json' in files:
            found_record = json.load(open(os.path.join(root, 'info.json'), encoding='utf-8'))
            found_photo = next((f for f in files if f.lower().endswith('.jpg')), None)
            if not found_photo:
                continue
            found_path = os.path.join(root, found_photo)
            # вычисляем расстояние между embedding-ами
            dist = Identifier.get_distance(to_tensor(photos[0]), to_tensor(found_path), mode='cosine',
                                           catdog_checker=CatsVsDogs, breed_extractor=BreedExtractor)
            logger.info(f"Distance found->lost: {dist}")
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_match = {'photo': found_path, 'info': found_record}
    if best_match:
        # уведомляем потерявшего
        await message.answer("Найден похожий найденный питомец:")
        await message.answer_photo(FSInputFile(best_match['photo']))
        text1 = f"Тип: {best_match['info']['type']}\n{best_match['info']['info']}\nТег нашедшего: {best_match['info'].get('finder_tag')}"
        text2 = f"Тип: {data['pet_type']}\n{additional}\nТег владельца: {owner_tag}"
        await message.answer(text1)
        # уведомляем того, кто нашёл
        finder_id = best_match['info'].get('finder_id')
        if finder_id:
            try:
                await message.bot.send_message(finder_id, "Возможно, ваш найденный питомец совпал с только что добавленным потерянным:")
                await message.bot.send_photo(finder_id, FSInputFile(photos[0]), caption=text2)
            except Exception as e:
                logger.error(f"Не удалось уведомить нашедшего {finder_id}: {e}")

    await state.clear()

# --- Flow: Я нашёл питомца ---
@router.message(lambda m: m.text == "Я нашёл питомца")
async def found_pet_start(message: types.Message, state: FSMContext):
    await state.set_state(FoundPetStates.waiting_for_type)
    await message.answer("Кого вы нашли?", reply_markup=type_keyboard)

@router.message(FoundPetStates.waiting_for_type)
async def found_pet_type(message: types.Message, state: FSMContext):
    if message.text not in ["Кошку", "Собаку"]:
        return await message.answer("Выберите 'Кошку' или 'Собаку'.")
    await state.update_data(pet_type=message.text)
    await state.set_state(FoundPetStates.waiting_for_photos)
    await message.answer("Пришлите фото найденного питомца.", reply_markup=back_keyboard)

@router.message(FoundPetStates.waiting_for_photos, lambda m: m.content_type == types.ContentType.PHOTO)
async def found_pet_photos(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > 5 * 1024 * 1024:
        return await message.answer("Файл >5МБ. Отправьте фото меньшего размера.")
    data = await state.get_data()
    pet_type = data['pet_type']
    file_id = photo.file_id
    folder = os.path.join(FOUND_PATH, f"{message.chat.id}_{file_id}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_id}.jpg")
    await download_photo(message.bot, file_id, path)
    try:
        from PIL import Image
        Image.open(path).verify()
    except Exception:
        os.remove(path)
        return await message.answer("Это некорректное изображение. Попробуйте ещё раз.")
    if torch.argmax(Recognizer(to_tensor(path)), dim=1).item() != 0:
        os.remove(path)
        return await message.answer("Это не похоже на питомца. Попробуйте другое фото.")
    is_cat = torch.argmax(CatsVsDogs(to_tensor(path)), dim=1).item() == 0
    if (pet_type == "Кошку" and not is_cat) or (pet_type == "Собаку" and is_cat):
        os.remove(path)
        return await message.answer("Тип животного не совпадает.")
    photos = data.get('photos', []) + [path]
    await state.update_data(photos=photos)
    await message.answer("Фото сохранено.\n Напишите дополнительную информацию, например, где нашли или нажмите на кнопку ниже", reply_markup=info_keyboard)

@router.message(FoundPetStates.waiting_for_photos)
async def prompt_found_info(message: types.Message, state: FSMContext):
    await state.set_state(FoundPetStates.waiting_for_info)
    await found_pet_info(message, state)

@router.message(FoundPetStates.waiting_for_info)
async def found_pet_info(message: types.Message, state: FSMContext):
    data = await state.get_data()
    photos = data['photos']
    text = message.text if message.text != "Без описания" else ""
    # Добавляем тег нашедшего для связи
    finder_tag = f"@{message.from_user.username}" if message.from_user.username else None
    info = {
        'type': data['pet_type'],
        'info': text,
        'finder_id': message.chat.id,
        'finder_tag': finder_tag
    }
    info_path = os.path.join(os.path.dirname(photos[0]), 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    confirm_text = f"Информация сохранена!\nС вами могут связаться в ближайшее время!"
    await message.answer(confirm_text, reply_markup=main_menu_keyboard)

    # Поиск совпадения среди потерянных
    threshold = 0.2
    best_dist = float('inf')
    best_match = None
    for root, _, files in os.walk(LOST_PATH):
        if 'info.json' in files:
            lost_record = json.load(open(os.path.join(root, 'info.json'), encoding='utf-8'))
            lost_photo = next((f for f in files if f.lower().endswith('.jpg')), None)
            if not lost_photo: continue
            lost_path = os.path.join(root, lost_photo)
            dist = Identifier.get_distance(to_tensor(photos[0]), to_tensor(lost_path), mode='cosine', catdog_checker=CatsVsDogs, breed_extractor=BreedExtractor)
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_match = {'photo': lost_path, 'info': lost_record}
    if best_match:
        await message.answer("Найден возможный совпавший потерянный питомец:")
        await message.answer_photo(FSInputFile(best_match['photo']))
        await message.answer(f"Тип: {best_match['info']['type']}\n{best_match['info']['info']}\nТег владельца: {best_match['info'].get('owner_tag')}")
        owner_id = best_match['info'].get('owner_id')
        if owner_id:
            await message.bot.send_message(owner_id, "Возможно, ваш питомец найден. Просмотрите фото ниже:")
            await message.bot.send_photo(owner_id, FSInputFile(photos[0]), caption=confirm_text)

    await state.clear()

# --- Flow: Каталог найденных питомцев ---
@router.message(lambda m: m.text == "Каталог найденных питомцев")
async def catalog_start(message: types.Message, state: FSMContext):
    await state.set_state(CatalogStates.choosing_type)
    await message.answer("Выберите тип питомца для просмотра:", reply_markup=type_keyboard)

@router.message(CatalogStates.choosing_type)
async def catalog_choose_type(message: types.Message, state: FSMContext):
    if message.text not in ["Кошку", "Собаку"]:
        return await message.answer("Пожалуйста, выберите категорию 'Кошка' или 'Собака'.")
    pets = []
    for root, _, files in os.walk(FOUND_PATH):
        if 'info.json' in files:
            record = json.load(open(os.path.join(root, 'info.json'), encoding='utf-8'))
            if record.get('type') == message.text:
                photo_f = next((f for f in files if f.lower().endswith('.jpg')), None)
                if photo_f:
                    pets.append({'photo': os.path.join(root, photo_f), 'info': record})
    if not pets:
        await message.answer("На данный момент нет найденных питомцев этого типа.", reply_markup=main_menu_keyboard)
        return await state.clear()
    await state.update_data(pets=pets, index=0)
    await state.set_state(CatalogStates.browsing)
    await show_catalog_item(message, state)

async def show_catalog_item(message: types.Message, state: FSMContext):
    data = await state.get_data()
    pet = data['pets'][data['index']]
    await message.answer_photo(FSInputFile(pet['photo']))
    # Отображаем информацию вместе с тегом для связи
    info_text = f"Тип: {pet['info']['type']}\n{pet['info']['info']}\nТег для связи: {pet['info'].get('finder_tag') or pet['info'].get('owner_tag')}"
    await message.answer(info_text, reply_markup=browse_keyboard)

@router.message(CatalogStates.browsing)
async def catalog_browse(message: types.Message, state: FSMContext):
    if message.text == "Назад":
        await message.answer("Возврат в главное меню.", reply_markup=main_menu_keyboard)
        return await state.clear()
    data = await state.get_data()
    idx = data['index']
    if message.text == "Следующее":
        idx = (idx + 1) % len(data['pets'])
        await state.update_data(index=idx)
        return await show_catalog_item(message, state)

# Роутер готов к подключению в server.py
