import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def get_page_data(start):
    url = f"https://movie.douban.com/top250?start={start}"  # 豆瓣每一页网址

    headers = {
        "User-Agent":
            "Mozilla/5.0"
    }  # 模拟浏览器访问，避免反爬虫

    response = requests.get(
        url,
        headers=headers
    )  # 获取HTML

    soup = BeautifulSoup(response.text,
                         "lxml"
                         )  # 将HTML转化为可读内容

    movies = soup.find_all("div", class_="item")  # 找到所有电影

    movies_data = []  # 数据集

    for movie in movies:  # 针对每一部电影提取数据
        title = movie.find(
            "span",
            class_="title"
        ).text

        score = float(
            movie.find(
                "span",
                class_="rating_num"
            ).text
        )

        info = movie.find(
            "div",
            class_="bd"
        ).p.text.strip()

        lines = info.split("\n")  # 对数据分行

        detail = lines[-1].strip()  # 提取年份国家类型一栏

        detail = detail.replace("\xa0", "")  # 去掉HTML的不换行空格代码

        parts = detail.split("/")  # 将年份国家类型分成3部分

        year = parts[0].strip()[:4]
        country = parts[1].strip()
        genre = parts[2].strip()

        movies_info = {
            "title": title,
            "score": score,
            "year": int(year),
            "country": country,
            "genre": genre
        }

        movies_data.append(movies_info)

    return movies_data

all_movies = []

for start in range(0, 250, 25):

    print(f"正在爬取{start}")

    page_data = get_page_data(start)

    all_movies.extend(page_data)

    time.sleep(1)

df = pd.DataFrame(all_movies)

print(df.head())

df.to_csv("../data/movies.csv",
          index=False,
          encoding="utf-8-sig"
          )

print("csv保存成功")