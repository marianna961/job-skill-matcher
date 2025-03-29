from format_checker import check_format
from resume_parser import parse_resume

def main():
    file_path = input("path: ").strip()

    format_type = check_format(file_path)
    print(f"Входной формат файла: {format_type}")

    text = parse_resume(file_path, format_type)
    if text:
        print(text)
    else:
        print("Не удалось получить данные из файла.")

if __name__ == "__main__":
    main()