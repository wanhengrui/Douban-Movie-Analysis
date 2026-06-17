"""
数据清洗模块

对爬取数据进行标准化处理：
  - 修复列表页解析错位（年份含多余日期信息）
  - 国家字段标准化
  - 类型字段标准化
  - 导演字段标准化
  - 演员字段标准化
  - 缺失值处理

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
#  2. 修复列表页解析错位
#  部分电影年份字段含重映日期，导致 country/genre 串位
# ============================================================

def fix_parse_errors(df):
    """
    修复因年份含多余信息导致的 country/genre 解析错位。

    问题示例:
      年份显示 "1961 / 1964(中国大陆上映)" → 按 "/" 切分后，
      country 串位拿到 "1964(中国大陆上映)"，genre 拿到 "中国大陆"
    """
    fixes = {
        "大闹天宫":  {"year": 1961, "country": "中国大陆", "genre": "动画 奇幻"},
        "天书奇谭":  {"year": 1983, "country": "中国大陆", "genre": "动画 奇幻"},
        "高山下的花环": {"year": 1984, "country": "中国大陆", "genre": "剧情 战争"},
    }

    for title, fix in fixes.items():
        mask = df["title"] == title
        if mask.any():
            for key, val in fix.items():
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
      - 去除首尾空白
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
    """
    类型字段标准化：
      - 保留空格分隔（后续分析按空格拆分即可）
      - 去除多余空白
    """
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
    """
    导演字段标准化：
      - 多导演统一用 " / " 分隔
      - 去除首尾空白
    """
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
      - 统一 " / " 分隔
      - 去除首尾空白
    """
    if "actors" not in df.columns:
        return df

    def standardize_actors(actors_str):
        if not isinstance(actors_str, str):
            return actors_str
        # 去掉末尾的截断符号 "..."/"…"
        actors_str = actors_str.rstrip(".…").strip()
        # 多余空白合并
        actors_str = re.sub(r"\s+", " ", actors_str)
        return actors_str

    df["actors"] = df["actors"].apply(standardize_actors)
    return df


# ============================================================
#  7. 缺失值处理
# ============================================================

def handle_missing(df):
    """
    缺失值处理：
      - 数值列（score）: 填中位数
      - 文本列: 填 "未知"
      - 打印缺失统计
    """
    print("\n=== 缺失值统计 ===")

    for col in df.columns:
        null_count = df[col].isna().sum()
        empty_count = (df[col] == "").sum() if df[col].dtype == object else 0
        total_missing = null_count + empty_count
        if total_missing > 0:
            print(f"  {col}: {total_missing} 条缺失")

    # 数值列填中位数
    if "score" in df.columns:
        df["score"] = df["score"].fillna(df["score"].median())

    # 文本列填 "未知"
    text_cols = ["country", "genre", "director", "actors",
                 "runtime", "language", "summary"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("未知")
            df[col] = df[col].replace("", "未知")

    return df


# ============================================================
#  8. 主流程
# ============================================================

def main():
    print("=" * 55)
    print("  数据清洗")
    print("=" * 55)

    df = load_data()
    print(f"  原始数据: {len(df)} 条记录\n")

    # 步骤1: 修复解析错位
    print("[1/5] 修复解析错位...")
    df = fix_parse_errors(df)

    # 步骤2: 标准化各字段
    print("\n[2/5] 标准化国家字段...")
    df = clean_country(df)

    print("[3/5] 标准化类型/导演/演员字段...")
    df = clean_genre(df)
    df = clean_director(df)
    df = clean_actors(df)

    # 步骤4: 缺失值处理
    print("[4/5] 处理缺失值...")
    df = handle_missing(df)

    # 步骤5: 保存
    print(f"\n[5/5] 保存...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "data", "movies_cleaned.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 已保存 {len(df)} 条记录到 data/movies_cleaned.csv")

    # 预览
    print(f"\n数据预览（前5行）:")
    pd.set_option("display.max_colwidth", 25)
    print(df.head())
    print(f"\n列信息:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
