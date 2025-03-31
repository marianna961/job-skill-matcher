import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.save_to_db import save_data_to_db
from format_checker import check_format
from resume_parser import parse_resume

def main():
    file_path = input("Введите путь к файлу: ").strip()

    format_type = check_format(file_path)
    if format_type == 'unknown':
        print("Формат файла не поддерживается.")
        return

    print(f"Определен формат файла: {format_type}")

    data = parse_resume(file_path, format_type)
    if data:
        print("Извлеченные данные:")
        print(data)

        try:
            save_data_to_db(data)
            print("Данные успешно сохранены в базу данных!")
        except Exception as e:
            print(f"Ошибка при сохранении данных в базу: {e}")
    else:
        print("Не удалось получить данные из файла.")

if __name__ == "__main__":
    main()