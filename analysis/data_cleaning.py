"""
数据清洗模块

对爬取数据进行标准化处理：
  - 修复列表页解析错位（年份含多余日期信息）
  - 修复列表页解析失败（导演/演员标签缺失）
  - 国家字段标准化
  - 类型字段标准化
  - 导演字段标准化
  - 演员字段标准化
  - 缺失值处理
  - 删除无用列（language）

输入: data/movies_detail.csv（如不存在则用 data/movies.csv）
输出: data/movies_cleaned.csv
"""
import pandas as pd
import re
import os


# ============================================================
#  1. 数据加载
# ============================================================

def load_data():
    """自动选择最完整的数据文件加载"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detail_path = os.path.join(base_dir, "data", "movies_detail.csv")
    base_path = os.path.join(base_dir, "data", "movies.csv")

    if os.path.exists(detail_path):
        print(f"[加载] {detail_path}")
        return pd.read_csv(detail_path)
    elif os.path.exists(base_path):
        print(f"[加载] {base_path}（基础版）")
        return pd.read_csv(base_path)
    else:
        raise FileNotFoundError("未找到数据文件，请先运行爬虫")


# ============================================================
#  2. 修复列表页解析错位与缺失
# ============================================================

def fix_parse_errors(df):
    """
    修复两种类型的列表页解析问题：

    问题A（3部）: 年份含重映日期 → country/genre 串位
    问题B（15部）: <p> 标签中缺少 "导演:"/"主演:" 标签 → 导演/演员为空

    这15部电影在列表页中的信息行结构与其他电影不同，
    导致正则解析失败。由于数据量小（15/250），手动补全。
    """
    fixes = {
        # ---- 问题A: 日期串位 ----
        "大闹天宫":   {"year": 1961, "country": "中国大陆", "genre": "动画 奇幻"},
        "天书奇谭":   {"year": 1983, "country": "中国大陆", "genre": "动画 奇幻"},
        "高山下的花环": {"year": 1984, "country": "中国大陆", "genre": "剧情 战争"},

        # ---- 问题B: 导演/演员缺失 ----
        "触不可及":    {"director": "奥利维耶·纳卡什 Olivier Nakache / 埃里克·托莱达诺 Eric Toledano",
                       "actors": "弗朗索瓦·克鲁塞 Francois Cluzet / 奥玛·希 Omar Sy"},
        "黑客帝国":    {"director": "莉莉·沃卓斯基 Lilly Wachowski / 拉娜·沃卓斯基 Lana Wachowski",
                       "actors": "基努·里维斯 Keanu Reeves / 劳伦斯·菲什伯恩 Laurence Fishburne"},
        "窃听风暴":    {"director": "弗洛里安·亨克尔·冯·多纳斯马尔克 Florian Henckel von Donnersmarck",
                       "actors": "乌尔里希·穆埃 Ulrich Muhe / 马蒂娜·格德克 Martina Gedeck"},
        "蝴蝶效应":    {"director": "埃里克·布雷斯 Eric Bress / J·麦基·格鲁伯 J. Mackye Gruber",
                       "actors": "阿什顿·库彻 Ashton Kutcher / 艾米·斯马特 Amy Smart"},
        "头脑特工队":  {"director": "彼特·道格特 Pete Docter / 罗纳尔多·德尔·卡门 Ronaldo Del Carmen",
                       "actors": "艾米·波勒 Amy Poehler / 菲利丝·史密斯 Phyllis Smith"},
        "黑客帝国3：矩阵革命": {"director": "莉莉·沃卓斯基 Lilly Wachowski / 拉娜·沃卓斯基 Lana Wachowski",
                         "actors": "基努·里维斯 Keanu Reeves / 劳伦斯·菲什伯恩 Laurence Fishburne"},
        "疯狂原始人":  {"director": "柯克·德·米科 Kirk De Micco / 克里斯·桑德斯 Chris Sanders",
                       "actors": "尼古拉斯·凯奇 Nicolas Cage / 艾玛·斯通 Emma Stone"},
        "上帝之城":    {"director": "费尔南多·梅里尔斯 Fernando Meirelles / 卡迪亚·兰德 Katia Lund",
                       "actors": "亚历桑德雷·罗德里格斯 Alexandre Rodrigues / 莱安德鲁·菲尔米诺 Leandro Firmino"},
        "冰川时代":    {"director": "克里斯·韦奇 Chris Wedge / 卡洛斯·沙尔丹哈 Carlos Saldanha",
                       "actors": "雷·罗马诺 Ray Romano / 约翰·雷吉扎莫 John Leguizamo"},
        "初恋这件小事": {"director": "普特鹏·普罗萨卡·那·萨克那卡林 Puttipong Promsaka Na Sakolnakorn / 华森·波克彭 Wasin Pokpong",
                       "actors": "马里奥·毛瑞尔 Mario Maurer / 平采娜·乐维瑟派布恩 Pimchanok Luevisadpaibul"},
        "黑客帝国2：重装上阵": {"director": "莉莉·沃卓斯基 Lilly Wachowski / 拉娜·沃卓斯基 Lana Wachowski",
                         "actors": "基努·里维斯 Keanu Reeves / 劳伦斯·菲什伯恩 Laurence Fishburne"},
        "蜘蛛侠：平行宇宙": {"director": "鲍勃·佩尔西凯蒂 Bob Persichetti / 彼得·拉姆齐 Peter Ramsey / 罗德尼·罗斯曼 Rodney Rothman",
                         "actors": "沙梅克·摩尔 Shameik Moore / 杰克·约翰逊 Jake Johnson"},
        "寻梦环游记":  {"actors": "安东尼·冈萨雷斯 Anthony Gonzalez / 盖尔·加西亚·贝纳尔 Gael Garcia Bernal"},
        "驯龙高手":    {"actors": "杰伊·巴鲁切尔 Jay Baruchel / 杰拉德·巴特勒 Gerard Butler"},
        "神偷奶爸":    {"actors": "史蒂夫·卡瑞尔 Steve Carell / 杰森·席格尔 Jason Segel"},
    }

    for title, fix in fixes.items():
        mask = df["title"] == title
        if mask.any():
            for key, val in fix.items():
                current = df.loc[mask, key].values[0]
                if pd.isna(current) or current == "":
                    df.loc[mask, key] = val
            print(f"  [OK] 修复: {title}")

    return df


# ============================================================
#  3. 国家字段标准化
# ============================================================

def clean_country(df):
    """
    国家字段标准化：
      - 多国合拍片统一用 " / " 分隔
    """
    if "country" not in df.columns:
        return df

    df["country"] = df["country"].apply(
        lambda c: " / ".join(c.split()) if isinstance(c, str) else c
    )
    return df


# ============================================================
#  4. 类型字段标准化
# ============================================================

def clean_genre(df):
    """类型字段：合并多余空白"""
    if "genre" not in df.columns:
        return df

    df["genre"] = df["genre"].apply(
        lambda g: re.sub(r"\s+", " ", g.strip()) if isinstance(g, str) else g
    )
    return df


# ============================================================
#  5. 导演字段标准化
# ============================================================

def clean_director(df):
    """导演字段：去首尾空白"""
    if "director" not in df.columns:
        return df

    df["director"] = df["director"].apply(
        lambda d: d.strip() if isinstance(d, str) else d
    )
    return df


# ============================================================
#  6. 演员字段标准化
# ============================================================

def clean_actors(df):
    """
    演员字段标准化：
      - 去除列表页截断符号 "..."
      - 合并多余空白
    """
    if "actors" not in df.columns:
        return df

    def standardize(s):
        if not isinstance(s, str):
            return s
        s = s.rstrip(".…").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    df["actors"] = df["actors"].apply(standardize)
    return df


# ============================================================
#  7. 缺失值处理
# ============================================================

def handle_missing(df):
    """
    缺失值处理：
      - 数值列填中位数
      - 文本列填 "未知"
    """
    print("\n=== 缺失值统计 ===")

    for col in df.columns:
        null_count = df[col].isna().sum()
        empty_count = (df[col] == "").sum() if df[col].dtype == object else 0
        total_missing = null_count + empty_count
        if total_missing > 0:
            print(f"  {col}: {total_missing} 条缺失")

    if "score" in df.columns:
        df["score"] = df["score"].fillna(df["score"].median())

    text_cols = ["country", "genre", "director", "actors",
                 "runtime", "summary"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("未知")
            df[col] = df[col].replace("", "未知")

    return df


# ============================================================
#  8. 删除无用列
# ============================================================

def drop_useless_columns(df):
    """
    删除无用的列：
      - language: 手机详情页无此字段，全为空
    """
    if "language" in df.columns:
        df.drop(columns=["language"], inplace=True)
        print("  [OK] 已删除无用列: language")

    return df


# ============================================================
#  9. 主流程
# ============================================================

def main():
    print("=" * 55)
    print("  数据清洗")
    print("=" * 55)

    df = load_data()
    print(f"  原始数据: {len(df)} 条记录\n")

    print("[1/6] 修复列表页解析错位与缺失...")
    df = fix_parse_errors(df)

    print("\n[2/6] 标准化国家字段...")
    df = clean_country(df)

    print("[3/6] 标准化类型/导演/演员字段...")
    df = clean_genre(df)
    df = clean_director(df)
    df = clean_actors(df)

    print("[4/6] 删除无用列...")
    df = drop_useless_columns(df)

    print("\n[5/6] 处理缺失值...")
    df = handle_missing(df)

    print(f"\n[6/6] 保存...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "data", "movies_cleaned.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 已保存 {len(df)} 条记录到 data/movies_cleaned.csv")

    print(f"\n数据预览（前5行）:")
    pd.set_option("display.max_colwidth", 25)
    print(df.head())
    print(f"\n列信息:")
    print(df.dtypes.tolist())


if __name__ == "__main__":
    main()
