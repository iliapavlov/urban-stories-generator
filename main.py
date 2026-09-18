import argparse
import functools
import os
import random
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
from pydantic import BaseModel, Field

# Завантаження змінних оточення з .env
load_dotenv()

# Налаштування шляхів та моделей
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
PROMPTS_DIR = Path("./prompts")
DEFAULT_INPUT_DIR = Path("./input_photos")
DEFAULT_OUTPUT_DIR = Path("./output_ideas")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RATE_LIMIT_DELAY = 2

# CLI UI текстові константи для інтерактивного режиму
UI_BANNER_SEPARATOR = "=" * 50
UI_SECTION_SEPARATOR = "-" * 50
UI_INTERACTIVE_HEADER = "\n[*] ІНТЕРАКТИВНИЙ РЕЖИМ: {photo_name}"
UI_DEEP_DIVE_START = "\n[*] Сканування кадру та генерація Deep Dive аналізу..."
UI_DEEP_DIVE_TITLE = "\n🔍 ЗНАХІДКИ ТА ВІЗУАЛЬНИЙ АНАЛІЗ МОДЕЛІ:"
UI_PROMPT_FOCUS_HEADER = (
    "\nЩо беремо за основу?\n"
    "Підкажи моделі, за яку деталь зачепитися (або натисни Enter для автоматичного вибору):"
)
UI_PROMPT_FOCUS_INPUT = "\nТвій фокус/коментар > "
UI_STORIES_GENERATION_START = "\n[*] Генерація концептів сторіз..."
UI_REFINEMENT_START = "\n[*] Внесення правок..."
UI_COMMANDS_HELP = (
    "\nКоманди:\n"
    "  [Enter]  — Затвердити та зберегти результат\n"
    "  [текст]  — Написати зауваження для правки (наприклад: 'зроби 2 варіант іронічнішим')\n"
    "  q        — Вийти без збереження"
)
UI_COMMAND_INPUT = "\nДія > "
UI_CANCELLED_MSG = "[*] Скасовано. Файли не збережено."
UI_SUCCESS_SAVE_MSG = "\n[✓] Успішно збережено в: {file_path}"
UI_RETRY_PROMPT = "\n[?] Спробувати надіслати цей запит ще раз? (y/n) > "


def deprecated(reason: str):
    """Декоратор для позначення застарілих функцій (PEP 702 сумісний)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} є застарілою: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def load_prompt(filename: str) -> str:
    """Завантажує текст промпта із директорії prompts/."""
    file_path = PROMPTS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Файл промпта не знайдено: {file_path.resolve()}")
    return file_path.read_text(encoding="utf-8").strip()


# Pydantic-схеми для структурованого виводу
class StoryConcept(BaseModel):
    title: str = Field(description="Коротка назва концепту")
    slide_1_hook: str = Field(
        description="Текст-гачок, прив'язаний до конкретної візуальної деталі фото (до 15-20 слів)"
    )
    slide_2_reflection: str = Field(
        description="Зміна оптики: від побутового роздратування/байдужості до усвідомленого вибору (20-30 слів)"
    )
    slide_3_cta: str = Field(description="Органічний перехід до банки на цільову потребу")


class StoriesResponse(BaseModel):
    visual_focus: str = Field(description="Аналіз ключового візуального фокусу зображення")
    variants: list[StoryConcept] = Field(description="Рівно 3 варіанти структури сторіз")


# Конфігурація генерації для фінальних сторіз
GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=load_prompt("system_role.txt"),
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


def send_message_with_retry(
    chat,
    message,
    config=None,
    max_retries: int = 4,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
):
    """
    Надсилає повідомлення у сесію чату з автоматичними повторами
    при тимчасових збоях сервера (503, 429, 500).
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return chat.send_message(message=message, config=config)
        except APIError as e:
            last_exception = e
            # Обробка перевантаження сервера або лімітів запитів
            if e.code in (503, 429, 500):
                jitter = random.uniform(0.8, 1.3)
                sleep_time = delay * jitter
                print(
                    f"\n[!] Модель тимчасово недоступна ({e.code}). "
                    f"Автоматична спроба {attempt}/{max_retries}. Очікування {sleep_time:.1f}с..."
                )
                time.sleep(sleep_time)
                delay *= backoff_factor
            else:
                # Фатальні помилки (400 Bad Request, 401 Unauthorized тощо) викидаємо одразу
                raise e
        except Exception as e:
            last_exception = e
            print(f"\n[!] Мережевий збій: {e}. Спроба {attempt}/{max_retries}...")
            time.sleep(delay)
            delay *= backoff_factor

    print(f"\n[✗] Не вдалося отримати відповідь після {max_retries} спроб: {last_exception}")
    return None


def execute_turn_with_recovery(chat, message, config=None):
    """
    Виконує крок діалогу. У разі вичерпання авто-ретраїв дає користувачеві
    змогу повторити спробу без перезапуску скрипту та втрати сесії.
    """
    while True:
        response = send_message_with_retry(chat, message=message, config=config)
        if response is not None:
            return response

        choice = input(UI_RETRY_PROMPT).strip().lower()
        if choice not in ("y", "yes", "так"):
            return None


def format_stories_to_markdown(data: StoriesResponse, photo_name: str, deep_dive_text: str = "") -> str:
    """Форматує структуровану відповідь у Markdown."""
    lines = [f"# Ідеї для сторіз: {photo_name}\n"]
    if deep_dive_text:
        lines.extend([
            "## 🔍 Первинний Deep Dive аналіз\n",
            deep_dive_text,
            "\n---\n",
        ])

    lines.extend([
        f"**Ключовий візуальний фокус:** {data.visual_focus}\n",
        "---\n",
    ])

    for idx, variant in enumerate(data.variants, start=1):
        lines.append(f"### Варіант {idx}: {variant.title}")
        lines.append(f"* **Слайд 1 (Спостереження)**: {variant.slide_1_hook}")
        lines.append(f"* **Слайд 2 (Рефлексія/Вибір)**: {variant.slide_2_reflection}")
        lines.append(f"* **Слайд 3 (Call-to-Action)**: {variant.slide_3_cta}\n")

    return "\n".join(lines)


def run_interactive(client: genai.Client, photo_path: Path, output_dir: Path, save_json: bool):
    """Основний режим: двоетапний діалог із Deep Dive та циклом правок."""
    if not photo_path.exists():
        print(f"[!] Файл не знайдено: {photo_path.resolve()}")
        return

    print(f"\n{UI_BANNER_SEPARATOR}")
    print(UI_INTERACTIVE_HEADER.format(photo_name=photo_path.name))
    print(UI_BANNER_SEPARATOR)

    # 1. Відкриття сесії чату
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=load_prompt("system_role.txt"),
            temperature=0.7,
        ),
    )

    # 2. КРОК 1: Deep Dive аналіз кадру з відновленням
    print(UI_DEEP_DIVE_START)
    with Image.open(photo_path) as pil_img:
        deep_dive_prompt = load_prompt("deep_dive.txt")
        response_deep_dive = execute_turn_with_recovery(
            chat,
            message=[pil_img, deep_dive_prompt]
        )

    if response_deep_dive is None:
        print(UI_CANCELLED_MSG)
        return

    deep_dive_text = response_deep_dive.text
    print(UI_DEEP_DIVE_TITLE)
    print(UI_SECTION_SEPARATOR)
    print(deep_dive_text)
    print(UI_SECTION_SEPARATOR)

    # 3. КРОК 2: Вибір фокусу автором та генерація зі структурованою схемою
    print(UI_PROMPT_FOCUS_HEADER)
    user_focus = input(UI_PROMPT_FOCUS_INPUT).strip()

    base_generation_prompt = load_prompt("story_generation.txt")
    if user_focus:
        full_generation_request = f"Фокусуйся на цій деталі: '{user_focus}'.\n{base_generation_prompt}"
    else:
        full_generation_request = f"Використай найсильніші знахідки з цього аналізу.\n{base_generation_prompt}"

    print(UI_STORIES_GENERATION_START)

    response_stories = execute_turn_with_recovery(
        chat,
        message=full_generation_request,
        config=GENERATION_CONFIG,
    )

    if response_stories is None:
        print(UI_CANCELLED_MSG)
        return

    current_data: StoriesResponse = response_stories.parsed

    # 4. КРОК 3: Цикл доопрацювання (Refinement Loop)
    while True:
        rendered_md = format_stories_to_markdown(current_data, photo_path.name)
        print(f"\n{UI_BANNER_SEPARATOR}")
        print(rendered_md)
        print(UI_BANNER_SEPARATOR)

        print(UI_COMMANDS_HELP)
        feedback = input(UI_COMMAND_INPUT).strip()

        if feedback.lower() == "q":
            print(UI_CANCELLED_MSG)
            return

        if not feedback:
            # Збереження затвердженого результату
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{photo_path.stem}_stories.md"
            final_markdown = format_stories_to_markdown(
                current_data, photo_path.name, deep_dive_text=deep_dive_text
            )
            out_file.write_text(final_markdown, encoding="utf-8")

            if save_json:
                json_path = out_file.with_suffix(".json")
                json_path.write_text(current_data.model_dump_json(indent=2), encoding="utf-8")

            print(UI_SUCCESS_SAVE_MSG.format(file_path=out_file.resolve()))
            break

        print(UI_REFINEMENT_START)
        refinement_request = (
            f"Внеси правки відповідно до зауваження: '{feedback}'. "
            f"Поверни оновлені 3 варіанти відповідно до заданої JSON-схеми."
        )
        response_refinement = execute_turn_with_recovery(
            chat,
            message=refinement_request,
            config=GENERATION_CONFIG,
        )

        if response_refinement is not None:
            current_data = response_refinement.parsed
        else:
            print("\n[!] Збережено поточну версію варіантів без останньої невдалої правки.")


def process_image(client: genai.Client, image_path: Path, out_path: Path, save_json: bool = False) -> bool:
    """Обробка одного фото для фонового моніторингу (Watcher) із ретраями."""
    try:
        with Image.open(image_path) as pil_image:
            chat = client.chats.create(
                model=MODEL_NAME,
                config=GENERATION_CONFIG,
            )
            story_prompt = load_prompt("story_generation.txt")
            response = send_message_with_retry(
                chat,
                message=[pil_image, story_prompt],
                config=GENERATION_CONFIG,
            )

        if response is None:
            return False

        data: StoriesResponse = response.parsed
        markdown_text = format_stories_to_markdown(data, image_path.name)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown_text, encoding="utf-8")

        if save_json:
            json_path = out_path.with_suffix(".json")
            json_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")

        return True

    except Exception as e:
        print(f"\n[!] Помилка обробки {image_path.name}: {e}")
        return False


@deprecated("Використовуйте інтерактивний режим (--interactive) або режим моніторингу (--watch).")
def run_batch(client: genai.Client, input_dir: Path, output_dir: Path, save_json: bool):
    """Пакетна сліпа обробка всіх файлів у теці без діалогу."""
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

    print(f"[*] Режим BATCH (DEPRECATED): знайдено фотографій — {total}. Модель: {MODEL_NAME}.")
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
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

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
                time.sleep(1.5)
                print(f"[*] Автоматична генерація ідей через {MODEL_NAME}...")
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
        description="CLI-інструмент для інтерактивного та автоматичного створення концептів сторіз для зборів."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--interactive", "-i",
        type=Path,
        metavar="IMAGE_PATH",
        help="Запустити інтерактивний режим співавторства з Deep Dive для конкретного фото",
    )
    group.add_argument(
        "--watch",
        action="store_true",
        help="Запустити фоновий моніторинг папки в реальному часі",
    )
    group.add_argument(
        "--batch",
        action="store_true",
        help="[DEPRECATED] Обробити всі наявні фото в папці без діалогу",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Шлях до вхідної папки з фото (для --watch та --batch)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Шлях до папки з результатами",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=os.getenv("SAVE_JSON", "false").lower() in ("true", "1", "yes"),
        help="Зберігати сирий валідований JSON поруч із файлом Markdown",
    )

    args = parser.parse_args()
    client = get_gemini_client()

    if args.interactive:
        run_interactive(client, args.interactive, args.output, save_json=args.save_json)
    elif args.watch:
        run_watcher(client, args.input, args.output, save_json=args.save_json)
    elif args.batch:
        run_batch(client, args.input, args.output, save_json=args.save_json)


if __name__ == "__main__":
    main()