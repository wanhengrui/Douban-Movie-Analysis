"""
补爬脚本：修复 movies_detail.csv 中缺失的字段

读取已生成的 movies_detail.csv，
对片长/简介为空的电影重新爬取手机详情页。
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import time
import random


MOBILE_UA = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36",
]


def fix_one(url):
    """爬取单个手机详情页，返回 runtime, summary"""
    match = re.search(r"/subject/(\d+)/", url)
    if not match:
        return "", ""

    mobile_url = f"https://m.douban.com/movie/subject/{match.group(1)}/"

    for attempt in range(3):
        try:
            headers = {
                "User-Agent": random.choice(MOBILE_UA),
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
            resp = requests.get(mobile_url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.content, "lxml")

            # 检查反爬
            if len(resp.content) < 5000:
                time.sleep(3)
                continue

            runtime = ""
            meta = soup.find("div", class_="sub-meta")
            if meta:
                m = re.search(r"片长(\d+\s*分?钟?)", meta.get_text(" ", strip=True))
                if m:
                    runtime = m.group(0)

            summary = ""
            intro = soup.find("section", class_="subject-intro")
            if intro:
                text = intro.get_text(" ", strip=True).replace("剧情简介", "").strip()
                # 截掉获奖信息
                cut = re.search(r"(?:本片|该片|影片|此片|电影)(?:获得|根据|改编|入围|荣获|提名)", text)
                if cut:
                    text = text[:cut.start()].strip()
                summary = re.sub(r"\s+", " ", text)

            if runtime or summary:
                return runtime, summary

        except Exception as e:
            time.sleep(2)

    return "", ""


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "movies_detail.csv")
    df = pd.read_csv(csv_path)

    # 补全缺失字段
    for col in ["runtime", "summary", "director", "actors", "language"]:
        if col not in df.columns:
            df[col] = ""

    need_fix = df[
        df["runtime"].isna() | (df["runtime"] == "") |
        df["summary"].isna() | (df["summary"] == "")
    ]

    print(f"需要修复: {len(need_fix)} 部\n")

    fixed = 0
    for i, (idx, row) in enumerate(need_fix.iterrows()):
        print(f"({i+1}/{len(need_fix)}) {row['title']}")

        runtime, summary = fix_one(row["detail_url"])
        if runtime:
            df.at[idx, "runtime"] = runtime
        if summary:
            df.at[idx, "summary"] = summary

        if runtime or summary:
            fixed += 1
            print(f"  [OK] runtime={runtime[:20]}, summary={len(summary)}字")
        else:
            print(f"  [FAIL]")

        time.sleep(1.5 + random.uniform(0, 1))

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n修复完成: {fixed}/{len(need_fix)}")
    print(f"数据已保存到 data/movies_detail.csv")


if __name__ == "__main__":
    main()
