import requests
import re
from bs4 import BeautifulSoup as bs
import time

# выполняю запрос и записываю содержимое страницы в файл, далее работаю только с файлом, не обращаясь к странице (7-22 стр.)
# url = 'https://www.avito.ru/moskva/rezume?cd=1&q=data+scientist'

# headers = {
#     'Accept': '*/*',
#     'UserAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 YaBrowser/25.2.0.0 Safari/537.36'
# }

# try:
#     req = requests.get(url, headers=headers)
# except Exception as e:
#     print('Не удалось получить страницу')
# print(req.text)
# soup = bs(req.text, 'lxml')

# with open('index.html', 'w', encoding='utf-8') as file:
#     file.write(req.text)


with open('index.html', encoding='utf-8') as file:
    src = file.read()

soup = bs(src, 'lxml')

resumes = []
links = []

resume_items = soup.find_all('div', {'class': 'iva-item-root-Se7z4', 'data-marker': 'item'})
for item in resume_items:
    link_elem = item.find('a', {'data-marker': 'item-title'})

    if link_elem and 'href' in link_elem.attrs:
        link = 'https://www.avito.ru' + link_elem['href']

    links.append(link)

print(links)

headers = {
        'Accept': '*/*',
        'UserAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
}
count = 0
for link in links:
    count += 1
    req = requests.get(links[0], headers=headers)
    soup = bs(req.text, 'html.parser')

    all_text = soup.get_text(strip=True)
    with open(f'{count}_resume.txt', 'w', encoding='utf-8') as file:
        file.write(all_text)

    print(f'{count} успешно обработан')
    time.sleep(100)
