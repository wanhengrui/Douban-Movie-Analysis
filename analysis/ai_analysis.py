"""
AI 智能分析模块

将数据统计结果发送给 DeepSeek 大模型，
由 AI 解读数据、发现规律、生成分析报告。

输入: data/movies_cleaned.csv
输出: report/ai_analysis_report.md
"""
import pandas as pd
import requests
import os
import json
from datetime import datetime

# ============================================================
#  配置
# ============================================================

DEEPSEEK_API_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"  # deepseek-v4-flash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_cleaned.csv")
REPORT_DIR = os.path.join(BASE_DIR, "report")
REPORT_PATH = os.path.join(REPORT_DIR, "ai_analysis_report.md")


# ============================================================
#  数据加载与统计
# ============================================================

def load_and_summarize():
    """加载数据，生成结构化统计摘要"""
    df = pd.read_csv(DATA_PATH)
    summary = {}

    # ---- 基本概况 ----
    summary["total"] = len(df)
    summary["score_range"] = f"{df['score'].min():.1f} ~ {df['score'].max():.1f}"
    summary["score_mean"] = round(df["score"].mean(), 2)
    summary["score_median"] = round(df["score"].median(), 2)
    summary["year_range"] = f"{int(df['year'].min())} ~ {int(df['year'].max())}"

    # ---- 国家统计 ----
    def explode_col(series, sep=" / "):
        return series.dropna().str.split(sep).explode()

    country_exploded = explode_col(df["country"])
    country_count = country_exploded.value_counts().head(15)
    summary["country_top15"] = [
        f"{c}: {n}部" for c, n in country_count.items()
    ]

    # 国家平均评分 (>=3部)
    cs = pd.DataFrame({"country": country_exploded, "score": df.loc[country_exploded.index, "score"]})
    cs_group = cs.groupby("country")["score"].agg(["mean", "count"])
    cs_group = cs_group[cs_group["count"] >= 3].sort_values("mean", ascending=False).head(10)
    summary["country_score_top10"] = [
        f"{c}: 均分{row['mean']:.2f} ({int(row['count'])}部)"
        for c, row in cs_group.iterrows()
    ]

    # ---- 类型统计 ----
    genre_exploded = explode_col(df["genre"], sep=" ")
    genre_count = genre_exploded.value_counts().head(15)
    summary["genre_top15"] = [
        f"{g}: {n}部" for g, n in genre_count.items()
    ]

    gs = pd.DataFrame({"genre": genre_exploded, "score": df.loc[genre_exploded.index, "score"]})
    gs_group = gs.groupby("genre")["score"].agg(["mean", "count"])
    gs_group = gs_group[gs_group["count"] >= 5].sort_values("mean", ascending=False).head(10)
    summary["genre_score_top10"] = [
        f"{g}: 均分{row['mean']:.2f} ({int(row['count'])}部)"
        for g, row in gs_group.iterrows()
    ]

    # ---- 年代统计 ----
    df_copy = df.copy()
    df_copy["decade"] = (df_copy["year"] // 10) * 10
    decade_count = df_copy["decade"].value_counts().sort_index()
    summary["decade_distribution"] = [
        f"{int(d)}年代: {int(n)}部" for d, n in decade_count.items()
    ]

    decade_score = df_copy.groupby("decade")["score"].agg(["mean", "count"])
    decade_score = decade_score[decade_score["count"] >= 3].sort_index()
    summary["decade_score"] = [
        f"{int(d)}年代: 均分{row['mean']:.2f} ({int(row['count'])}部)"
        for d, row in decade_score.iterrows()
    ]

    # ---- 导演统计 ----
    director_exploded = explode_col(df["director"])
    director_exploded = director_exploded[director_exploded != "未知"]
    director_count = director_exploded.value_counts().head(10)
    summary["director_top10"] = [
        f"{d}: {n}部" for d, n in director_count.items()
    ]

    ds = pd.DataFrame({"director": director_exploded, "score": df.loc[director_exploded.index, "score"]})
    ds_group = ds.groupby("director")["score"].agg(["mean", "count"])
    ds_group = ds_group[ds_group["count"] >= 2].sort_values("mean", ascending=False).head(10)
    summary["director_score_top10"] = [
        f"{d}: 均分{row['mean']:.2f} ({int(row['count'])}部)"
        for d, row in ds_group.iterrows()
    ]

    # ---- 演员统计 ----
    actor_exploded = explode_col(df["actors"])
    actor_exploded = actor_exploded[actor_exploded != "未知"]
    actor_count = actor_exploded.value_counts().head(10)
    summary["actor_top10"] = [
        f"{a}: {n}部" for a, n in actor_count.items()
    ]

    as_ = pd.DataFrame({"actor": actor_exploded, "score": df.loc[actor_exploded.index, "score"]})
    as_group = as_.groupby("actor")["score"].agg(["mean", "count"])
    as_group = as_group[as_group["count"] >= 2].sort_values("mean", ascending=False).head(10)
    summary["actor_score_top10"] = [
        f"{a}: 均分{row['mean']:.2f} ({int(row['count'])}部)"
        for a, row in as_group.iterrows()
    ]

    return df, summary


# ============================================================
#  Prompt 构建
# ============================================================

def build_prompt(summary):
    """根据统计摘要构造发送给AI的prompt"""
    prompt = f"""你是一位资深电影数据分析师。请根据以下"豆瓣Top250电影"数据集统计结果，
撰写一份专业的数据分析报告。

## 数据集概况
- 电影总数：{summary['total']} 部
- 评分范围：{summary['score_range']}
- 平均评分：{summary['score_mean']}，中位数：{summary['score_median']}
- 年代跨度：{summary['year_range']}

## 各维度统计数据

### 国家/地区分布 Top15
{chr(10).join(f"- {c}" for c in summary['country_top15'])}

### 国家/地区平均评分 Top10（>=3部）
{chr(10).join(f"- {c}" for c in summary['country_score_top10'])}

### 电影类型分布 Top15
{chr(10).join(f"- {g}" for g in summary['genre_top15'])}

### 电影类型平均评分 Top10（>=5部）
{chr(10).join(f"- {g}" for g in summary['genre_score_top10'])}

### 年代分布
{chr(10).join(f"- {d}" for d in summary['decade_distribution'])}

### 年代平均评分
{chr(10).join(f"- {d}" for d in summary['decade_score'])}

### 导演出现次数 Top10
{chr(10).join(f"- {d}" for d in summary['director_top10'])}

### 导演平均评分 Top10（>=2部）
{chr(10).join(f"- {d}" for d in summary['director_score_top10'])}

### 演员出现次数 Top10
{chr(10).join(f"- {a}" for a in summary['actor_top10'])}

### 演员平均评分 Top10（>=2部）
{chr(10).join(f"- {a}" for a in summary['actor_score_top10'])}

## 分析要求
请从以下角度进行分析，使用Markdown格式：

1. **数据概览**：评分分布特点、年代分布特点
2. **国家/地区洞察**：哪些国家在Top250中占主导？高分国家有什么特点？
3. **类型趋势**：什么类型最受欢迎？高分类型集中在哪些领域？
4. **年代变迁**：电影质量随年代如何变化？哪些年代是"黄金期"？
5. **人才分析**：上榜最多的导演/演员有什么共同特点？
6. **总体结论**：从这250部电影中能得出什么关于"好电影"的规律？

请用中文撰写，语言专业但不晦涩，适合本科生阅读。
每个分析点都要引用具体数据支撑。"""
    return prompt


# ============================================================
#  API 调用
# ============================================================

def call_deepseek(prompt):
    """调用 DeepSeek API 获取分析报告"""
    print("  正在调用 DeepSeek API...")

    resp = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "你是一位专业的数据分析师，擅长从数据中发现规律和洞察。请用中文回复，使用Markdown格式。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.7
        },
        timeout=120
    )

    if resp.status_code != 200:
        print(f"  API调用失败: {resp.status_code}")
        print(f"  {resp.text[:500]}")
        return None

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {})
    print(f"  [OK] 模型: {data.get('model', 'N/A')}")
    print(f"  [OK] Token消耗: {tokens.get('total_tokens', 'N/A')}")
    return content


# ============================================================
#  保存报告
# ============================================================

def save_report(content, df, summary):
    """保存AI生成的分析报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)

    header = f"""# 豆瓣Top250电影 - AI智能分析报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 分析模型：DeepSeek V4 Flash
> 数据来源：豆瓣Top250 (movies_cleaned.csv)
> 数据集：{summary['total']}部电影，评分 {summary['score_range']}

---

"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(header + content)

    print(f"  [OK] 报告已保存到: {REPORT_PATH}")


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 60)
    print("  豆瓣Top250 - AI 智能分析")
    print("=" * 60)

    # 1. 加载数据并生成统计
    print("\n[1/3] 加载数据 & 计算统计摘要...")
    df, summary = load_and_summarize()
    print(f"  加载 {summary['total']} 部电影")
    print(f"  评分: {summary['score_range']}, 均分 {summary['score_mean']}")

    # 2. 构建Prompt并调用AI
    print("\n[2/3] 构建分析 Prompt & 调用 AI...")
    prompt = build_prompt(summary)
    print(f"  Prompt长度: {len(prompt)} 字符")

    content = call_deepseek(prompt)
    if content is None:
        print("\n  分析失败，请检查API Key和网络连接。")
        return

    # 3. 保存报告
    print("\n[3/3] 保存分析报告...")
    save_report(content, df, summary)

    print(f"\n{'=' * 60}")
    print("  分析完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
