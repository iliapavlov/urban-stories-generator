import argparse
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

# Завантаження змінних оточення з .env
load_dotenv()

# Налаштування та константи
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_INPUT_DIR = Path("./input_photos")
DEFAULT_OUTPUT_DIR = Path("./output_ideas")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RATE_LIMIT_DELAY = 2

SYSTEM_PROMPT = """
Роль: Ти — чуйний міський спостерігач, вуличний філософ та копірайтер для зборів на потреби військових. Твій підхід базується на прямій горизонтальній взаємодії «людина для людини» без моралізаторства, пафосу чи почуття провини.

Вхідні дані: Зображення повсякденного міського простору (вулиця, транспорт, побутові деталі, гра світла, текстури, артефакти міста).

Твоє завдання:
1. Проаналізувати зображення: виділити ключовий візуальний фокус (текст, деталь, світловий акцент, міський контекст).
2. Створити 3 варіанти структури сторіз (серія з 2–3 слайдів на одне фото).

Ключова філософія тексту:
- Фокус на особистому виборі та сприйнятті реальності (фреймінг повсякденності).
- Пряма горизонтальна солідарність: допомога тому, хто поруч і хто просить про закриття конкретної базової потреби, поза великою політикою чи глобальними суперечками.
- Жодної токсичності, звинувачень чи штучного нагнітання. Чиста спостережливість, яка природно виходить на дію.

Мова: Українська. Тон: Спокійний, автентичний, влучний, без канцеляризмів.
"""


# Схеми типізованого виводу
class StoryConcept(BaseModel):
    title: str = Field(description="Коротка назва концепту")
    slide_1_hook: str = Field(description="Текст-гачок, прив'язаний до конкретної візуальної деталі фото (до 15-20 слів)")
    slide_2_reflection: str = Field(description="Зміна оптики: від побутового роздратування/байдужості до усвідомленого вибору (20-30 слів)")
    slide_3_cta: str = Field(description="Органічний перехід до банки на цільову потребу")


class StoriesResponse(BaseModel):
    visual_focus: str = Field(description="Короткий аналіз ключового візуального фокусу зображення")
    variants: list[StoryConcept] = Field(description="Рівно 3 варіанти структури сторіз")


# Перевикористовувана конфігурація запиту
GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7,
    response_mime_type="application/json",
    response_schema=StoriesResponse,
)


def get_gemini_client() -> genai.Client:
    """Ініціалізує та повертає клієнт Google GenAI."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Помилка: GEMINI_API_KEY не знайдено у змінних середовища або файлі .env")
    return genai.Client(api_key=api_key)


def process_image(client: genai.Client, image_path: Path, out_path: Path, save_json: bool = False) -> bool:
    """Обробляє фотографію через Gemini Vision API та зберігає результат у Markdown (і опційно JSON)."""
    try:
        with Image.open(image_path) as pil_image:
            chat = client.chats.create(
                model=MODEL_NAME,
                config=GENERATION_CONFIG,
            )
            response = chat.send_message(
                message=[pil_image, "Проаналізуй фото та створи концепції сторіз."]
            )

        data: StoriesResponse = response.parsed

        # 1. Збереження Markdown
        markdown_lines = [
            f"# Ідеї для сторіз: {image_path.name}\n",
            f"**Ключовий візуальний фокус:** {data.visual_focus}\n",
            "---\n",
        ]

        for idx, variant in enumerate(data.variants, start=1):
            markdown_lines.append(f"### Варіант {idx}: {variant.title}")
            markdown_lines.append(f"* **Слайд 1 (Спостереження)**: {variant.slide_1_hook}")
            markdown_lines.append(f"* **Слайд 2 (Рефлексія/Вибір)**: {variant.slide_2_reflection}")
            markdown_lines.append(f"* **Слайд 3 (Call-to-Action)**: {variant.slide_3_cta}\n")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(markdown_lines), encoding="utf-8")

        # 2. Опційне збереження сирого JSON поруч
        if save_json:
            json_path = out_path.with_suffix(".json")
            json_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")

        return True

    except Exception as e:
        print(f"\n[!] Помилка обробки {image_path.name}: {e}")
        return False


def run_batch(client: genai.Client, input_dir: Path, output_dir: Path, save_json: bool):
    """Пакетна обробка всіх файлів у теці."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    photos = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    total = len(photos)
    if total == 0:
        print(f"[*] У папці {input_dir.resolve()} не знайдено фотографій для обробки.")
        return

    print(f"[*] Режим BATCH: знайдено фотографій — {total}. Модель: {MODEL_NAME}.")
    if save_json:
        print("[*] Збереження JSON активовано.")
    print("Початок роботи...\n")

    processed = 0
    skipped = 0

    for idx, photo_path in enumerate(photos, start=1):
        out_file = output_dir / f"{photo_path.stem}_stories.md"

        if out_file.exists():
            print(f"[{idx}/{total}] Пропущено (вже існує): {photo_path.name}")
            skipped += 1
            continue

        print(f"[{idx}/{total}] Обробка: {photo_path.name} ...", end=" ", flush=True)
        if process_image(client, photo_path, out_file, save_json=save_json):
            print("Готово [✓]")
            processed += 1
        else:
            print("Помилка [✗]")

        if idx < total:
            time.sleep(RATE_LIMIT_DELAY)

    print("\n" + "=" * 40)
    print(f"Підсумок: оброблено — {processed}, пропущено — {skipped}")
    print(f"Збережено в: {output_dir.resolve()}")
    print("=" * 40)


def run_watcher(client: genai.Client, input_dir: Path, output_dir: Path, save_json: bool):
    """Фоновий моніторинг папки на появу нових файлів."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    class PhotoHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                out_file = output_dir / f"{file_path.stem}_stories.md"
                if out_file.exists():
                    return
                print(f"\n[+] Нове фото виявлено: {file_path.name}")
                # Пауза для завершення запису файлу на диск ОС Windows
                time.sleep(1.5)
                print(f"[*] Генерація ідей через {MODEL_NAME}...")
                if process_image(client, file_path, out_file, save_json=save_json):
                    saved_files = f"{out_file.name}" + (f" та {out_file.stem}.json" if save_json else "")
                    print(f"[✓] Успішно створено: {saved_files}")

    event_handler = PhotoHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(input_dir), recursive=False)
    observer.start()

    print(f"[*] Режим WATCHER активний (Модель: {MODEL_NAME}).")
    if save_json:
        print("[*] Збереження JSON активовано.")
    print(f"[*] Моніторинг теки: {input_dir.resolve()}")
    print("Натисніть Ctrl+C для зупинки.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nЗупинка моніторингу...")
        observer.stop()
    observer.join()


def main():
    parser = argparse.ArgumentParser(
        description="CLI-генератор концептів сторіз із міських фото для зборів."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", action="store_true", help="Обробити всі наявні фото в папці")
    group.add_argument("--watch", action="store_true", help="Запустити моніторинг папки в реальному часі")

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Шлях до вхідної папки з фото")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Шлях до папки з результатами")
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=os.getenv("SAVE_JSON", "false").lower() in ("true", "1", "yes"),
        help="Зберігати сирий валідований JSON поруч із файлом Markdown",
    )

    args = parser.parse_args()
    client = get_gemini_client()

    if args.batch:
        run_batch(client, args.input, args.output, save_json=args.save_json)
    elif args.watch:
        run_watcher(client, args.input, args.output, save_json=args.save_json)


if __name__ == "__main__":
    main()