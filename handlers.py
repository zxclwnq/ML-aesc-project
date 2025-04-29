import os
import json
import logging
import shutil
import uuid
import torch
from aiogram import types, Router
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, FSInputFile, InputMediaPhoto, \
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
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

# Для проверки подтверждения и удаления фотографий
match_confirmations: dict[str, dict] = {}

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

# --- Я потерял питомца ---
@router.message(lambda m: m.text == "Я потерял питомца")
async def lost_pet_start(message: types.Message, state: FSMContext):
    await state.set_state(LostPetStates.waiting_for_type)
    await message.answer("Кого вы потеряли?", reply_markup=type_keyboard)

@router.message(LostPetStates.waiting_for_type)
async def lost_pet_type(message: types.Message, state: FSMContext):
    if message.text not in ["Кошку", "Собаку"]:
        return await message.answer("Пожалуйста, выберите категорию 'Кошка' или 'Собака'.")
    await state.update_data(pet_type=message.text[:-1] + 'а')
    await state.set_state(LostPetStates.waiting_for_photos)
    await message.answer("Пришлите фото потерянного питомца.", reply_markup=back_keyboard)

@router.message(LostPetStates.waiting_for_photos, lambda m: m.content_type == types.ContentType.PHOTO)
async def lost_pet_photos(message: types.Message, state: FSMContext):
     data = await state.get_data()
     pet_type = data['pet_type']
     saved_paths = data.get('photos', [])

     # берём только самое большое представление из списка размеров
     photo = message.photo[-1]
     # проверка размера
     if photo.file_size and photo.file_size > 5 * 1024 * 1024:
         return await message.answer("Файл >5МБ. Отправьте фото меньшего размера.")

     folder = (await state.get_data()).get('folder')
     if not folder:
        folder = os.path.join(LOST_PATH, f"{message.chat.id}_{uuid.uuid4().hex}")
        os.makedirs(folder, exist_ok=True)
        await state.update_data(folder=folder, photos=[])

     file_id = photo.file_id
     path = os.path.join(folder, f"{file_id}.jpg")
     await download_photo(message.bot, file_id, path)

     # верификация изображения
     try:
         from PIL import Image
         Image.open(path).verify()
     except Exception:
         os.remove(path)
         return await message.answer("Некорректное изображение. Пробуйте другое фото.")

     # проверка, что это питомец нужного типа
     if torch.argmax(Recognizer(to_tensor(path)), dim=1).item() != 0:
         os.remove(path)
         return await message.answer("Это не похоже на питомца. Попробуйте другое фото.")
     is_cat = torch.argmax(CatsVsDogs(to_tensor(path)), dim=1).item() == 0
     if (pet_type == "Кошка" and not is_cat) or (pet_type == "Собака" and is_cat):
         os.remove(path)
         return await message.answer("Тип животного не совпадает.")

     # сохраняем путь и просим ещё или инфо
     saved_paths.append(path)
     await state.update_data(photos=saved_paths)
     count = len(saved_paths)
     await message.answer(f"Фото сохранено ({count}). Пришлите ещё или напиши доп информацию (если её нет, нажмите «Без описания»).", reply_markup=info_keyboard)

@router.message(LostPetStates.waiting_for_photos)
async def prompt_lost_info(message: types.Message, state: FSMContext):
    await state.set_state(LostPetStates.waiting_for_info)
    await lost_pet_info(message, state)

@router.message(LostPetStates.waiting_for_info)
async def lost_pet_info(message: types.Message, state: FSMContext):
    data = await state.get_data()
    photos = data['photos']
    additional = message.text if message.text != "Без описания" else ""
    owner_tag = f"@{message.from_user.username}" if message.from_user.username else None

    # сохраняем info.json
    info = {
        'type': data['pet_type'],
        'info': additional,
        'owner_id': message.chat.id,
        'owner_tag': owner_tag
    }
    info_path = os.path.join(os.path.dirname(photos[0]), 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    await message.answer("Информация сохранена!", reply_markup=main_menu_keyboard)

    # ищем лучший match в FOUND_PATH
    threshold = 0.2
    best_dist = float('inf')
    best_match = None

    for root, _, files in os.walk(FOUND_PATH):
        if 'info.json' not in files:
            continue

        # найдём первое фото в папке found
        found_photo = next((f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))), None)
        if not found_photo:
            continue

        found_path = os.path.join(root, found_photo)

        # считаем дистанции от каждого lost фото до найденного
        dists = [
            Identifier.get_distance(
                to_tensor(lp), to_tensor(found_path),
                mode='cosine',
                catdog_checker=CatsVsDogs,
                breed_extractor=BreedExtractor
            )
            for lp in photos
        ]
        dist = min(dists)

        if dist < threshold and dist < best_dist:
            best_dist = dist
            # прочитаем info найденного
            found_info = json.load(open(os.path.join(root, 'info.json'), encoding='utf-8'))
            best_match = {
                'folder': root,
                'info': found_info
            }

    if best_match:
        await message.answer("Найден похожий найденный питомец:")

        match_id = uuid.uuid4().hex
        match_confirmations[match_id] = {
            # все папки, где лежат фото потерянного
            'lost_folders': [os.path.basename(os.path.dirname(photos[0]))],
            # все папки найденного
            'found_folders': [os.path.basename(best_match['folder'])],
            # сохраним ID владельца (lost) и нашедшего (found)
            'lost_user_id': message.chat.id,
            'found_user_id': best_match['info'].get('finder_id'),
            'lost': None,
            'found': None
        }
        # собрать и отправить все фото из папки совпадения
        media = []
        for fn in os.listdir(best_match['folder']):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(best_match['folder'], fn)
                media.append(InputMediaPhoto(media=FSInputFile(path)))

        # отправляем инфо нашедшего
        fi = best_match['info']
        found_desc = fi.get('info', '').strip() or "Без описания"
        found_tag = fi.get('finder_tag') or "—"
        found_capt = f"Тип: {fi['type']}\nОписание: {found_desc}\nТег нашедшего: {found_tag}"

        lost_desc = additional.strip() or "Без описания"
        lost_type = data['pet_type']
        lost_tag = owner_tag or "—"
        lost_capt = f"Возможно, ваш найденный питомец совпал с только что добавленным потерянным:\nТип: {lost_type}\nОписание: {lost_desc}\nТег владельца: {lost_tag}"

        if media:
            media[0].caption = found_capt
            await message.answer_media_group(media)

        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Да", callback_data=f"match:{match_id}:lost:yes"),
            InlineKeyboardButton(text="Нет", callback_data=f"match:{match_id}:lost:no"),
        ]])
        await message.answer("Это ваш питомец?", reply_markup=kb)

        # уведомляем нашедшего
        finder_id = best_match['info'].get('finder_id')
        if finder_id:
            try:
                await message.bot.send_message(finder_id, "Возможно ваш питомец совпал с только что добавленным потерянным:")
                # отправка фото потерянного
                media2 = [InputMediaPhoto(media=FSInputFile(p)) for p in photos]
                media2[0].caption = lost_capt
                kb2 = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="Да", callback_data=f"match:{match_id}:found:yes"),
                    InlineKeyboardButton(text="Нет", callback_data=f"match:{match_id}:found:no"),
                ]])
                await message.bot.send_media_group(finder_id, media2)
                await message.bot.send_message(
                    finder_id,
                    "Пожалуйста, подтвердите — это ваш найденный питомец?",
                    reply_markup=kb2
                )
            except Exception as e:
                logger.error(f"Не удалось уведомить нашедшего {finder_id}: {e}")

    await state.clear()


# --- Я нашёл питомца ---
@router.message(lambda m: m.text == "Я нашёл питомца")
async def found_pet_start(message: types.Message, state: FSMContext):
    await state.set_state(FoundPetStates.waiting_for_type)
    await message.answer("Кого вы нашли?", reply_markup=type_keyboard)

@router.message(FoundPetStates.waiting_for_type)
async def found_pet_type(message: types.Message, state: FSMContext):
    if message.text not in ["Кошку", "Собаку"]:
        return await message.answer("Выберите 'Кошка' или 'Собака'.")
    await state.update_data(pet_type=message.text[:-1] + 'а')
    await state.set_state(FoundPetStates.waiting_for_photos)
    await message.answer("Пришлите фото найденного питомца.", reply_markup=back_keyboard)

@router.message(FoundPetStates.waiting_for_photos, lambda m: m.content_type == types.ContentType.PHOTO)
async def found_pet_photos(message: types.Message, state: FSMContext):
    data = await state.get_data()
    pet_type = data['pet_type']
    saved_paths = data.get('photos', [])

    # берём только самое большое из доступных размеров
    photo = message.photo[-1]
    # проверка размера
    if photo.file_size and photo.file_size > 5 * 1024 * 1024:
        return await message.answer("Файл >5МБ. Отправьте фото меньшего размера.")

    # если ещё нет сессионной папки — создаём её и сохраняем в state
    folder = (await state.get_data()).get('folder')
    if not folder:
        folder = os.path.join(FOUND_PATH, f"{message.chat.id}_{uuid.uuid4().hex}")
        os.makedirs(folder, exist_ok=True)
        await state.update_data(folder=folder, photos=[])

    file_id = photo.file_id
    path = os.path.join(folder, f"{file_id}.jpg")
    await download_photo(message.bot, file_id, path)

    # верификация изображения
    try:
        from PIL import Image
        Image.open(path).verify()
    except Exception:
        os.remove(path)
        return await message.answer("Некорректное изображение. Попробуйте другое фото.")

    # проверка, что это питомец
    if torch.argmax(Recognizer(to_tensor(path)), dim=1).item() != 0:
        os.remove(path)
        return await message.answer("Это не похоже на питомца. Попробуйте другое фото.")
    is_cat = torch.argmax(CatsVsDogs(to_tensor(path)), dim=1).item() == 0
    if (pet_type == "Кошка" and not is_cat) or (pet_type == "Собака" and is_cat):
        os.remove(path)
        return await message.answer("Тип животного не совпадает.")

    # сохраняем путь
    saved_paths.append(path)
    await state.update_data(photos=saved_paths)
    count = len(saved_paths)
    await message.answer(f"Фото сохранено ({count}). Пришлите ещё или напиши доп информацию (если её нет, нажмите «Без описания»)", reply_markup=info_keyboard)

@router.message(FoundPetStates.waiting_for_photos)
async def prompt_found_info(message: types.Message, state: FSMContext):
    await state.set_state(FoundPetStates.waiting_for_info)
    await found_pet_info(message, state)

@router.message(FoundPetStates.waiting_for_info)
async def found_pet_info(message: types.Message, state: FSMContext):
    data = await state.get_data()
    photos = data['photos']
    text = message.text if message.text != "Без описания" else ""
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

    await message.answer("Информация сохранена!", reply_markup=main_menu_keyboard)

    # ищем лучший match в LOST_PATH
    threshold = 0.2
    best_dist = float('inf')
    best_match = None

    for root, _, files in os.walk(LOST_PATH):
        if 'info.json' not in files:
            continue

        lost_photo = next((f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))), None)
        if not lost_photo:
            continue

        lost_path = os.path.join(root, lost_photo)

        dists = [
            Identifier.get_distance(
                to_tensor(fp), to_tensor(lost_path),
                mode='cosine',
                catdog_checker=CatsVsDogs,
                breed_extractor=BreedExtractor
            )
            for fp in photos
        ]
        dist = min(dists)

        if dist < threshold and dist < best_dist:
            best_dist = dist
            lost_info = json.load(open(os.path.join(root, 'info.json'), encoding='utf-8'))
            best_match = {
                'folder': root,
                'info': lost_info
            }

    if best_match:
        await message.answer("Найден возможный совпавший потерянный питомец:")

        match_id = uuid.uuid4().hex
        match_confirmations[match_id] = {
            # все папки потерянного
            'lost_folders': [os.path.basename(best_match['folder'])],
            # все папки, где лежат фото найденного
            'found_folders': [os.path.basename(os.path.dirname(photos[0]))],
            # сохраним ID владельца (lost) и нашедшего (found)
            'lost_user_id': best_match['info'].get('owner_id'),
            'found_user_id': message.chat.id,
            'lost': None,
            'found': None
        }

        # отправляем все фото из папки совпадения
        media = []
        for fn in os.listdir(best_match['folder']):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(best_match['folder'], fn)
                media.append(InputMediaPhoto(media=FSInputFile(path)))

        li = best_match['info']
        lost_desc = li.get('info', '').strip() or "Без описания"
        lost_tag = li.get('owner_tag') or "—"
        lost_capt = f"Тип: {li['type']}\nОписание: {lost_desc}\nТег владельца: {lost_tag}"

        found_desc = text.strip() or "Без описания"
        found_type = data['pet_type']
        found_tag = finder_tag or "—"
        found_capt = f"Тип: {found_type}\nОписание: {found_desc}\nТег нашедшего: {found_tag}"

        if media:
            media[0].caption = lost_capt
            await message.answer_media_group(media)

        # 2) отдельно шлём текст с inline-кнопками
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Да", callback_data=f"match:{match_id}:found:yes"),
            InlineKeyboardButton(text="Нет", callback_data=f"match:{match_id}:found:no"),
        ]])
        await message.answer("Это ваш найденный питомец?", reply_markup=kb)

        # уведомляем владельца
        owner_id = li.get('owner_id')
        if owner_id:
            try:
                await message.bot.send_message(owner_id, "Возможно, ваш питомец найден. Смотрите фото:")
                media2 = [InputMediaPhoto(media=FSInputFile(p)) for p in photos]
                media2[0].caption = found_capt
                kb2 = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="Да", callback_data=f"match:{match_id}:lost:yes"),
                    InlineKeyboardButton(text="Нет", callback_data=f"match:{match_id}:lost:no"),
                ]])
                await message.bot.send_media_group(owner_id, media2)
                await message.bot.send_message(
                    owner_id,
                    "Пожалуйста, подтвердите — это ваш потерянный питомец?",
                    reply_markup=kb2
                )
            except Exception as e:
                logger.error(f"Не удалось уведомить владельца {owner_id}: {e}")

    await state.clear()

# --- Каталог найденных питомцев ---
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
    desc = pet['info'].get('info', '').strip() or "Без описания"
    contact = pet['info'].get('finder_tag') or pet['info'].get('owner_tag') or "—"
    info_text = (
        f"Тип: {pet['info']['type']}\n"
        f"Описание: {desc}\n"
        f"Тег для связи: {contact}"
    )

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

# --- Обработка inline-кнопок ---
@router.callback_query(lambda c: c.data and c.data.startswith("match:"))
async def on_match_confirm(cb: CallbackQuery):
    # формат: match:<match_id>:<role>:<yes|no>
    _, match_id, role, resp = cb.data.split(":")
    entry = match_confirmations.get(match_id)
    if not entry:
        return await cb.answer("Сессия устарела.", show_alert=True)

    entry[role] = (resp == "yes")
    await cb.answer("Ваш выбор сохранён.")

    # если кто-то отказался
    if entry['lost'] is False or entry['found'] is False:
        match_confirmations.pop(match_id, None)
        return

    # если оба подтвердили
    if entry['lost'] and entry['found']:
        # удаляем папки
        print(entry['lost_folders'])
        print(entry['found_folders'])
        for lf in entry['lost_folders']:
            shutil.rmtree(os.path.join(LOST_PATH, lf), ignore_errors=True)

        for ff in entry['found_folders']:
            shutil.rmtree(os.path.join(FOUND_PATH, ff), ignore_errors=True)

        lost_id = entry.get('lost_user_id')
        found_id = entry.get('found_user_id')
        text = "Совпадение подтверждено — все данные удалены."
        if lost_id:
            await cb.bot.send_message(lost_id, text)

        if found_id:
            await cb.bot.send_message(found_id, text)

        match_confirmations.pop(match_id, None)

# Роутер готов к подключению в server.py
