import pandas as pd

# Загружаем CSV
df = pd.read_csv("output.csv", encoding="utf-8-sig")

df = df.applymap(
    lambda x: x.replace(" ##", "")
              .replace("##", "")
              .replace("  ", " ")
              .replace(" @ ", "@") 
              .replace(" . ", ".") 
              .strip()
    if isinstance(x, str) else x
)

# Сохраняем обратно
df.to_csv("output_clean.csv", encoding="utf-8-sig", index=False)

print("Готово! Чистый файл сохранён как output_clean.csv")
