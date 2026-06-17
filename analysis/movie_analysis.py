"""
电影数据分析模块

包含：国家分析、类型分析、年代分析、导演分析、演员分析、
     评分分布、可视化图表

输入: data/movies_cleaned.csv（如不存在则用 movies_detail.csv）
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "images")


# ============================================================
#  数据加载
# ============================================================

def load_data():
    """自动选择最完整的数据文件"""
    cleaned = os.path.join(BASE_DIR, "data", "movies_cleaned.csv")
    detail = os.path.join(BASE_DIR, "data", "movies_detail.csv")
    base = os.path.join(BASE_DIR, "data", "movies.csv")

    for path in [cleaned, detail, base]:
        if os.path.exists(path):
            print(f"[加载] {os.path.basename(path)}")
            return pd.read_csv(path)

    raise FileNotFoundError("未找到数据文件")


# ============================================================
#  辅助函数：拆分多值字段
# ============================================================

def explode_column(df, col, sep=" / "):
    """
    将多值字段拆分为多行，便于统计。

    例如: "中国大陆 / 中国香港" → 两行，分别对应两个国家
    返回一个新的 DataFrame，每行只包含一个值。
    """
    df_copy = df.copy()
    df_copy[col] = df_copy[col].str.split(sep)
    return df_copy.explode(col)


# ============================================================
#  模块3: 国家分析
# ============================================================

def analyze_country(df):
    """
    国家分析：
      - 出现次数统计
      - 平均评分
      - 评分排名

    合拍片（如"美国 / 英国"）会被拆分，
    每个国家各自计入统计。
    """
    print("\n" + "=" * 55)
    print("  国家分析")
    print("=" * 55)

    # 拆分合拍片
    exploded = explode_column(df, "country")

    # 1. 出现次数
    country_count = exploded["country"].value_counts()
    print(f"\n[国家出现次数 Top10]")
    for country, count in country_count.head(10).items():
        print(f"  {country:8s}  {count:3d} 部")

    # 2. 平均评分（按国家分组）
    country_score = exploded.groupby("country")["score"].agg(["mean", "count"])
    country_score = country_score[country_score["count"] >= 3]  # 至少3部才有统计意义
    country_score = country_score.sort_values("mean", ascending=False)

    print(f"\n[国家平均评分 Top10（>=3部）]")
    for country, row in country_score.head(10).iterrows():
        print(f"  {country:8s}  {row['mean']:.2f} 分  ({int(row['count'])}部)")

    # 3. 合拍统计
    multi = df[df["country"].str.contains(" / ", na=False)]
    print(f"\n[合拍片统计]")
    print(f"  合拍片: {len(multi)} 部 ({len(multi)/len(df)*100:.1f}%)")
    print(f"  单一国家/地区: {len(df)-len(multi)} 部")

    return country_count, country_score


# ============================================================
#  可视化
# ============================================================

def draw_score_distribution(df):
    """评分分布直方图"""
    plt.figure(figsize=(8, 5))
    plt.hist(df["score"], bins=12, edgecolor="white", color="#4CAF50")
    plt.title("豆瓣Top250 评分分布")
    plt.xlabel("评分")
    plt.ylabel("电影数量")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "score_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] {path}")


def draw_country_distribution(country_count):
    """国家分布横向条形图"""
    top10 = country_count.head(10)
    # 倒序让数值大的在上面
    countries = list(top10.index)[::-1]
    counts = list(top10.values)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(countries, counts, color="#2196F3")
    plt.title("豆瓣Top250 国家/地区分布 Top10")
    plt.xlabel("电影数量")
    for i, v in enumerate(counts):
        plt.text(v + 0.5, i, str(v), va="center")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "country_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] {path}")


def draw_country_score_chart(country_score):
    """国家平均评分图"""
    top10 = country_score.head(10)

    plt.figure(figsize=(10, 6))
    countries = list(top10.index)[::-1]
    scores = list(top10["mean"].values)[::-1]

    bars = plt.barh(countries, scores, color="#FF9800")
    plt.title("豆瓣Top250 国家/地区平均评分 Top10（>=3部）")
    plt.xlabel("平均评分")

    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                 f"{score:.2f}", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "country_score.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] {path}")


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 55)
    print("  豆瓣Top250 电影数据分析")
    print("=" * 55)

    df = load_data()
    print(f"  共 {len(df)} 部电影\n")

    # 模块3: 国家分析
    country_count, country_score = analyze_country(df)

    # 可视化
    print(f"\n{'=' * 55}")
    print("  生成图表")
    print("=" * 55)
    draw_score_distribution(df)
    draw_country_distribution(country_count)
    draw_country_score_chart(country_score)

    print(f"\n分析完成。")


if __name__ == "__main__":
    main()
