"""
豆瓣Top250电影详情页爬虫（增强版）

在列表页爬取（douban_spider.py）的基础上，
新增导演、演员、片长、语言、剧情简介字段。

策略：
  - 列表页（桌面版）：已含导演、演员信息，补充解析即可
  - 详情页（手机版）：获取片长、剧情简介
  - 语言字段：通过"国家→语言"映射表推断（手机页无语言信息）

输出: data/movies_detail.csv（11个字段的完整数据集）
"""
import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import re


# ============================================================
#  国家 → 语言 映射表
#  手机版详情页不显示语言，根据制片国家推断
# ============================================================

COUNTRY_LANG_MAP = {
    "美国": "英语",
    "英国": "英语",
    "加拿大": "英语",
    "澳大利亚": "英语",
    "爱尔兰": "英语",
    "新西兰": "英语",
    "中国大陆": "汉语普通话",
    "中国香港": "粤语",
    "中国台湾": "汉语普通话",
    "日本": "日语",
    "韩国": "韩语",
    "法国": "法语",
    "德国": "德语",
    "意大利": "意大利语",
    "西班牙": "西班牙语",
    "印度": "印地语",
    "巴西": "葡萄牙语",
    "葡萄牙": "葡萄牙语",
    "俄罗斯": "俄语",
    "苏联": "俄语",
    "阿根廷": "西班牙语",
    "墨西哥": "西班牙语",
    "瑞典": "瑞典语",
    "丹麦": "丹麦语",
    "波兰": "波兰语",
    "泰国": "泰语",
    "伊朗": "波斯语",
    "土耳其": "土耳其语",
    "荷兰": "荷兰语",
    "比利时": "法语",
    "瑞士": "德语",
    "挪威": "挪威语",
    "芬兰": "芬兰语",
    "希腊": "希腊语",
    "捷克": "捷克语",
    "斯洛伐克": "捷克语",
    "匈牙利": "匈牙利语",
    "罗马尼亚": "罗马尼亚语",
    "保加利亚": "保加利亚语",
    "南斯拉夫": "塞尔维亚语",
    "捷克共和国": "捷克语",
    "奥地利": "德语",
    "以色列": "希伯来语",
    "南非": "英语",
    "塞内加尔": "法语",
    "阿尔及利亚": "阿拉伯语",
    "摩洛哥": "阿拉伯语",
    "阿富汗": "达里语",
    "越南": "越南语",
    "菲律宾": "菲律宾语",
    "印度尼西亚": "印尼语",
    "马来西亚": "马来语",
    "新加坡": "英语",
    "冰岛": "冰岛语",
    "哥伦比亚": "西班牙语",
    "智利": "西班牙语",
    "秘鲁": "西班牙语",
    "古巴": "西班牙语",
    "乌克兰": "乌克兰语",
    "阿联酋": "阿拉伯语",
    "黎巴嫩": "阿拉伯语",
}


def infer_language(country_str):
    """
    根据制片国家/地区推断语言。

    参数:
        country_str: 如 "美国"、"中国大陆 中国香港"、"美国 英国"

    返回:
        推断的语言字符串，如 "英语"、"汉语普通话 / 粤语"
    """
    if not country_str:
        return ""

    countries = country_str.split()
    langs = []
    for c in countries:
        lang = COUNTRY_LANG_MAP.get(c, "")
        if lang and lang not in langs:
            langs.append(lang)

    return " / ".join(langs)


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
#  第二部分：手机详情页爬取（补充片长、简介）
#  语言不在此处获取，后续通过国家映射推断
# ============================================================

# 移动端 User-Agent 池，减少被限流概率
MOBILE_UA_POOL = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36",
]


def get_mobile_detail(url, max_retries=3):
    """
    从手机版详情页获取片长、剧情简介。
    带重试机制：请求失败或返回防爬页面时自动重试。

    手机版页面结构：
      - sub-meta: "国家 / 类型 / 上映日期上映 / 片长XXX分钟"
      - subject-intro: 剧情简介全文

    参数:
        url: 桌面版详情页URL
        max_retries: 最大重试次数

    返回: dict，包含 runtime, summary
    """
    match = re.search(r"/subject/(\d+)/", url)
    if not match:
        return {"runtime": "", "summary": ""}

    subject_id = match.group(1)
    mobile_url = f"https://m.douban.com/movie/subject/{subject_id}/"

    for attempt in range(1, max_retries + 1):
        headers = {
            "User-Agent": random.choice(MOBILE_UA_POOL),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        try:
            response = requests.get(mobile_url, headers=headers, timeout=15)

            # 检查是否被反爬（返回的是JS挑战页面而非真实内容）
            if len(response.content) < 5000:
                if attempt < max_retries:
                    wait = attempt * 3 + random.uniform(1, 3)
                    print(f"    ⚠ 疑似反爬页面 (len={len(response.content)}), "
                          f"第{attempt}次重试, 等待{wait:.1f}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"    ✗ 重试{max_retries}次后仍失败")
                    return {"runtime": "", "summary": ""}

            soup = BeautifulSoup(response.content, "lxml")

            # --- 片长 ---
            runtime = ""
            meta = soup.find("div", class_="sub-meta")
            if meta:
                meta_text = meta.get_text(" ", strip=True)
                rt_match = re.search(r"片长(\d+\s*分?钟?)", meta_text, re.UNICODE)
                if rt_match:
                    runtime = rt_match.group(0)

            # --- 剧情简介 ---
            summary = ""
            intro = soup.find("section", class_="subject-intro")
            if intro:
                full_text = intro.get_text(" ", strip=True)
                summary_text = full_text.replace("剧情简介", "").strip()
                # 截掉获奖信息等附加内容
                extra_match = re.search(
                    r"(?:本片|该片|影片|此片|电影)(?:获得|根据|改编|入围|荣获|提名)",
                    summary_text
                )
                if extra_match:
                    summary_text = summary_text[:extra_match.start()].strip()
                summary = re.sub(r"\s+", " ", summary_text)

            # 至少获取到片长或简介才算成功
            if runtime or summary:
                return {"runtime": runtime, "summary": summary}

            # 页面正常但没提取到内容（极少见）
            if attempt < max_retries:
                time.sleep(2)
            else:
                return {"runtime": "", "summary": ""}

        except Exception as e:
            if attempt < max_retries:
                wait = attempt * 2 + random.uniform(0.5, 1.5)
                print(f"    ⚠ 请求异常: {e}, 第{attempt}次重试, 等待{wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"    ✗ 请求异常，重试{max_retries}次后放弃: {e}")
                return {"runtime": "", "summary": ""}

    return {"runtime": "", "summary": ""}


# ============================================================
#  第三部分：主流程
# ============================================================

def main():
    print("=" * 60)
    print("  豆瓣Top250电影 —— 详情页爬虫（增强版）")
    print("=" * 60)
    print()

    # ---- 阶段1：列表页获取基础信息（含导演、演员） ----
    print("[阶段 1/2] 爬取列表页，提取基础信息 + 导演 + 演员...")
    all_movies = []

    for page in range(10):
        start = page * 25
        print(f"  列表页 第{page + 1}页 (start={start})...")
        page_data = get_list_page_data(start)
        all_movies.extend(page_data)
        time.sleep(1 + random.uniform(0, 0.5))

    print(f"  ✓ 列表页爬取完成，共 {len(all_movies)} 部电影\n")

    # ---- 阶段2：手机详情页补充片长 + 简介 ----
    print("[阶段 2/2] 爬取手机详情页，补充片长 + 简介...")
    total = len(all_movies)
    failed_indices = []  # 记录失败的索引，最后统一重试

    for i, movie in enumerate(all_movies):
        url = movie["detail_url"]
        print(f"  ({i + 1}/{total}) {movie['title']}")

        extra = get_mobile_detail(url)
        movie["runtime"] = extra["runtime"]
        movie["summary"] = extra["summary"]

        if extra["runtime"] or extra["summary"]:
            print(f"    ✓ 片长={extra['runtime'][:20]}, "
                  f"简介={len(extra['summary'])}字")
        else:
            print(f"    ⚠ 未获取到补充信息")
            failed_indices.append(i)

        # 随机延迟 1.5~2.5 秒，降低被限流概率
        time.sleep(1.5 + random.uniform(0, 1.0))

        # 每 50 部电影多休息 10 秒，进一步降低触发反爬的概率
        if (i + 1) % 50 == 0 and (i + 1) < total:
            print(f"    --- 已完成 {i + 1}/{total}，休息 10 秒 ---")
            time.sleep(10)

    # ---- 阶段2补充：重试失败的 ----
    if failed_indices:
        print(f"\n  ⚠ {len(failed_indices)} 部电影未获取到详情，进行重试...")
        for idx in failed_indices:
            movie = all_movies[idx]
            print(f"  重试: {movie['title']}")
            time.sleep(3 + random.uniform(0, 2))  # 重试间隔更长
            extra = get_mobile_detail(movie["detail_url"], max_retries=5)
            movie["runtime"] = extra["runtime"]
            movie["summary"] = extra["summary"]
            if extra["runtime"] or extra["summary"]:
                print(f"    ✓ 重试成功")
            else:
                print(f"    ✗ 重试仍失败")

    # ---- 阶段3：推断语言 + 保存 ----
    print(f"\n{'=' * 60}")
    print("  推断语言 & 保存数据...")

    # 根据国家推断语言
    for movie in all_movies:
        movie["language"] = infer_language(movie.get("country", ""))

    df = pd.DataFrame(all_movies)
    column_order = [
        "title", "score", "year", "country", "genre",
        "director", "actors", "runtime", "language", "summary",
        "detail_url"
    ]
    df = df[column_order]

    df.to_csv("../data/movies_detail.csv", index=False, encoding="utf-8-sig")

    # 统计
    has_runtime = (df["runtime"] != "").sum()
    has_summary = (df["summary"] != "").sum()
    has_language = (df["language"] != "").sum()

    print(f"  ✓ 成功保存 {len(df)} 条记录到 data/movies_detail.csv")
    print(f"    片长覆盖: {has_runtime}/250")
    print(f"    简介覆盖: {has_summary}/250")
    print(f"    语言覆盖: {has_language}/250")
    print(f"{'=' * 60}")

    print(f"\n数据预览（前5行）:\n")
    print(df.head().to_string(max_colwidth=30))


if __name__ == "__main__":
    main()
