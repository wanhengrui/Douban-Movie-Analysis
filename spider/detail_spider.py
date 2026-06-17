"""
豆瓣Top250电影详情页爬虫（增强版）

在列表页爬取（douban_spider.py）的基础上，
新增导演、演员、片长、语言、剧情简介字段。

策略：
  - 列表页（桌面版）：已含导演、演员信息，补充解析即可
  - 详情页（手机版）：获取片长、语言、剧情简介等

输出: data/movies_detail.csv（11个字段的完整数据集）
"""
import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
import re


# ============================================================
#  第一部分：列表页爬取（桌面版）
# ============================================================

def get_list_page_data(start):
    """
    爬取单个列表页（25部电影），提取包括导演、演员在内的全部信息。

    列表页每部电影的 <p> 结构：
        Line 1: "导演: 弗兰克·德拉邦特 Frank Darabont   主演: 蒂姆·罗宾斯 Tim Robbins /..."
        Line 2: "1994 / 美国 / 犯罪 剧情"

    返回: list[dict]，每部电影一条记录
    """
    url = f"https://movie.douban.com/top250?start={start}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        print(f"    ✗ 列表页请求失败 (start={start}): {e}")
        return []
    soup = BeautifulSoup(response.text, "lxml")

    items = soup.find_all("div", class_="item")
    movies_data = []

    for item in items:
        # --- 电影名称 ---
        title_tag = item.find("span", class_="title")
        title = title_tag.text if title_tag else ""

        # --- 评分 ---
        score_tag = item.find("span", class_="rating_num")
        try:
            score = float(score_tag.text) if score_tag else 0.0
        except (ValueError, AttributeError):
            score = 0.0

        # --- 信息行解析 ---
        bd_div = item.find("div", class_="bd")
        if not bd_div:
            continue
        info_p = bd_div.find("p")
        if not info_p:
            continue

        info_text = info_p.get_text("\n", strip=True)
        # 将 \xa0（HTML不换行空格）替换为普通空格
        info_text = info_text.replace("\xa0", " ")
        lines = info_text.split("\n")

        # Line 1: "导演: XXX   主演: XXX / YYY /..."
        first_line = lines[0].strip() if len(lines) > 0 else ""

        director = ""
        actors = ""
        if "导演:" in first_line and "主演:" in first_line:
            # 按 "主演:" 分割，左边是导演，右边是演员
            parts = first_line.split("主演:", 1)
            director = parts[0].replace("导演:", "").strip()
            # 去掉末尾的截断符号 "..."/"…" 和残留的 " /"
            actors = parts[1].strip().rstrip(".…/ ").strip()

        # Line 2: "1994 / 美国 / 犯罪 剧情"
        second_line = lines[1].strip() if len(lines) > 1 else ""

        detail_parts = [p.strip() for p in second_line.split("/")]
        year = 0
        country = ""
        genre = ""
        if len(detail_parts) >= 3:
            year_str = detail_parts[0]
            try:
                year = int(year_str)
            except ValueError:
                year = int(year_str[:4]) if year_str[:4].isdigit() else 0
            country = detail_parts[1]
            genre = detail_parts[2]
        elif len(detail_parts) == 2:
            country = detail_parts[0]
            genre = detail_parts[1]

        # --- 详情页URL ---
        link_tag = item.find("a")
        detail_url = link_tag["href"] if link_tag and link_tag.get("href") else ""

        movies_data.append({
            "title": title,
            "score": score,
            "year": year,
            "country": country,
            "genre": genre,
            "director": director,
            "actors": actors,
            "detail_url": detail_url
        })

    return movies_data


# ============================================================
#  第二部分：手机详情页爬取（补充片长、语言、简介）
# ============================================================

def get_mobile_detail(url):
    """
    从手机版详情页获取片长、语言、剧情简介。

    手机版页面结构：
      - sub-meta: "国家 / 类型 / 上映日期上映 / 片长XXX分钟"
      - subject-intro: 剧情简介全文

    参数:
        url: 桌面版详情页URL（如 https://movie.douban.com/subject/1292052/）

    返回: dict，包含 runtime, language, summary
    """
    # 从桌面版URL提取subject_id
    match = re.search(r"/subject/(\d+)/", url)
    if not match:
        return {"runtime": "", "language": "", "summary": ""}

    subject_id = match.group(1)
    mobile_url = f"https://m.douban.com/movie/subject/{subject_id}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
    }

    try:
        response = requests.get(mobile_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")
    except Exception as e:
        print(f"    ✗ 手机页请求失败: {e}")
        return {"runtime": "", "language": "", "summary": ""}

    # --- 片长 ---
    runtime = ""
    meta = soup.find("div", class_="sub-meta")
    if meta:
        meta_text = meta.get_text(" ", strip=True)
        # 从 "片长142分钟" 中提取
        rt_match = re.search(r"片长(\d+\s*分?钟?)", meta_text, re.UNICODE)
        if rt_match:
            runtime = rt_match.group(0)

    # --- 语言 ---
    # 手机版详情页通常不直接显示语言，使用 meta 中的信息
    # 部分电影在 meta 中有语言信息
    language = ""
    if meta:
        # 尝试匹配 "语言: XXX" 或从sub-meta中提取
        lang_match = re.search(r"语言[：:]\s*([^\s/]+)", meta.get_text(" ", strip=True))
        if lang_match:
            language = lang_match.group(1)

    # --- 剧情简介 ---
    summary = ""
    intro = soup.find("section", class_="subject-intro")
    if intro:
        # 提取 "剧情简介" 之后的文字
        full_text = intro.get_text(" ", strip=True)
        # 去掉 "剧情简介" 标签和 "本片获得..." 之类的获奖信息
        summary_text = full_text.replace("剧情简介", "").strip()
        # 截取主要简介部分（去掉获奖信息等附加内容）
        extra_match = re.search(
            r"(?:本片|该片|影片|此片|电影)(?:获得|根据|改编|入围|荣获|提名)",
            summary_text
        )
        if extra_match:
            summary_text = summary_text[:extra_match.start()].strip()
        summary = re.sub(r"\s+", " ", summary_text)

    return {
        "runtime": runtime,
        "language": language,
        "summary": summary
    }


# ============================================================
#  第三部分：主流程
# ============================================================

def main():
    print("=" * 60)
    print("  豆瓣Top250电影 —— 详情页爬虫（增强版）")
    print("=" * 60)
    print()

    # 步骤1：从列表页获取基础信息（含导演、演员）
    print("[阶段 1/2] 爬取列表页，提取基础信息 + 导演 + 演员...")
    all_movies = []

    for page in range(10):
        start = page * 25
        print(f"  列表页 第{page + 1}页 (start={start})...")
        page_data = get_list_page_data(start)
        all_movies.extend(page_data)
        time.sleep(1)

    print(f"  ✓ 列表页爬取完成，共 {len(all_movies)} 部电影\n")

    # 步骤2：逐部爬取手机详情页，补充片长、语言、简介
    print("[阶段 2/2] 爬取手机详情页，补充片长 + 语言 + 简介...")
    total = len(all_movies)

    for i, movie in enumerate(all_movies):
        url = movie["detail_url"]
        print(f"  ({i + 1}/{total}) {movie['title']}")

        extra = get_mobile_detail(url)
        movie["runtime"] = extra["runtime"]
        movie["language"] = extra["language"]
        movie["summary"] = extra["summary"]

        if extra["runtime"] or extra["summary"]:
            print(f"    ✓ 片长={extra['runtime'][:20]}, "
                  f"简介={len(extra['summary'])}字")
        else:
            print(f"    ⚠ 未获取到补充信息")

        time.sleep(1.5)

    # 步骤3：保存数据
    print(f"\n{'=' * 60}")
    print("  保存数据...")

    df = pd.DataFrame(all_movies)
    # 调整列顺序
    column_order = [
        "title", "score", "year", "country", "genre",
        "director", "actors", "runtime", "language", "summary",
        "detail_url"
    ]
    df = df[column_order]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(base_dir, "data", "movies_detail.csv"),
              index=False, encoding="utf-8-sig")
    print(f"  ✓ 成功保存 {len(df)} 条记录到 data/movies_detail.csv")
    print(f"{'=' * 60}")

    # 打印数据预览
    print(f"\n数据预览（前5行）:\n")
    print(df.head().to_string(max_colwidth=30))


if __name__ == "__main__":
    main()
