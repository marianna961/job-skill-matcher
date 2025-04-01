import sqlite3

def init_database():
    connection = sqlite3.connect("resumes.db")
    cursor = connection.cursor()

    # Создаем таблицу users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        location TEXT
    )
    """)

    # Создаем таблицу resumes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        desired_position TEXT,
        specialization TEXT,
        education TEXT,
        skills TEXT,
        experience TEXT,
        about_me TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    connection.commit()
    connection.close()

    print("База данных успешно инициализирована!")

if __name__ == "__main__":
    init_database()