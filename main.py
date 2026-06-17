"""
豆瓣Top250电影分析系统 —— 主入口

命令行交互菜单，集成数据查询、AI问答、电影推荐等功能。
"""
import pandas as pd
import requests
import os
import sys

# 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_cleaned.csv")

# DeepSeek API 配置
DEEPSEEK_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# 全局数据
df = None


# ============================================================
#  数据加载
# ============================================================

def load_data():
    global df
    for p in [DATA_PATH,
              os.path.join(BASE_DIR, "data", "movies_detail.csv"),
              os.path.join(BASE_DIR, "data", "movies.csv")]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return True
    return False


# ============================================================
#  辅助函数
# ============================================================

def explode_col(series, sep):
    return series.dropna().str.split(sep).explode()


def print_divider(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


# ============================================================
#  菜单选项 1~4: 数据查询
# ============================================================

def show_country_analysis():
    print_divider("国家/地区分析")

    exploded = explode_col(df["country"], " / ")
    print("\n[出现次数 Top10]")
    for c, n in exploded.value_counts().head(10).items():
        print(f"  {c:10s} {n:3d} 部")

    cs = pd.DataFrame({"country": exploded, "score": df.loc[exploded.index, "score"]})
    g = cs.groupby("country")["score"].agg(["mean", "count"])
    g = g[g["count"] >= 3].sort_values("mean", ascending=False).head(10)
    print("\n[平均评分 Top10 (>=3部)]")
    for c, row in g.iterrows():
        print(f"  {c:10s} {row['mean']:.2f} 分 ({int(row['count'])}部)")


def show_genre_analysis():
    print_divider("类型分析")

    exploded = explode_col(df["genre"], " ")
    print("\n[出现次数 Top10]")
    for g, n in exploded.value_counts().head(10).items():
        print(f"  {g:8s} {n:3d} 部")

    gs = pd.DataFrame({"genre": exploded, "score": df.loc[exploded.index, "score"]})
    g = gs.groupby("genre")["score"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean", ascending=False).head(10)
    print("\n[平均评分 Top10 (>=5部)]")
    for t, row in g.iterrows():
        print(f"  {t:8s} {row['mean']:.2f} 分 ({int(row['count'])}部)")


def show_director_analysis():
    print_divider("导演分析")

    exploded = explode_col(df["director"], " / ")
    print("\n[出现次数 Top10]")
    for d, n in exploded.value_counts().head(10).items():
        print(f"  {d} : {n} 部")

    ds = pd.DataFrame({"director": exploded, "score": df.loc[exploded.index, "score"]})
    g = ds.groupby("director")["score"].agg(["mean", "count"])
    g = g[g["count"] >= 2].sort_values("mean", ascending=False).head(10)
    print("\n[平均评分 Top10 (>=2部)]")
    for d, row in g.iterrows():
        print(f"  {row['mean']:.2f} 分 - {d} ({int(row['count'])}部)")


def show_actor_analysis():
    print_divider("演员分析")

    exploded = explode_col(df["actors"], " / ")
    print("\n[出现次数 Top10]")
    for a, n in exploded.value_counts().head(10).items():
        print(f"  {a} : {n} 部")

    as_ = pd.DataFrame({"actor": exploded, "score": df.loc[exploded.index, "score"]})
    g = as_.groupby("actor")["score"].agg(["mean", "count"])
    g = g[g["count"] >= 2].sort_values("mean", ascending=False).head(10)
    print("\n[平均评分 Top10 (>=2部)]")
    for a, row in g.iterrows():
        print(f"  {row['mean']:.2f} 分 - {a} ({int(row['count'])}部)")


# ============================================================
#  菜单选项 5: AI 自由问答
# ============================================================

def build_data_context():
    """构造数据上下文，让AI了解数据集"""
    exploded_c = explode_col(df["country"], " / ")
    exploded_g = explode_col(df["genre"], " ")
    exploded_d = explode_col(df["director"], " / ")
    exploded_a = explode_col(df["actors"], " / ")

    ctx = f"""## 数据集概况
豆瓣Top250电影数据集，共{len(df)}部电影。
评分范围 {df['score'].min():.1f}~{df['score'].max():.1f}，均分 {df['score'].mean():.2f}。
年代跨度 {int(df['year'].min())}~{int(df['year'].max())}。

## 国家Top5
{', '.join(f'{c}({n})' for c, n in exploded_c.value_counts().head(5).items())}

## 类型Top5
{', '.join(f'{g}({n})' for g, n in exploded_g.value_counts().head(5).items())}

## 导演Top5
{', '.join(f'{d}({n})' for d, n in exploded_d.value_counts().head(5).items())}

## 演员Top5
{', '.join(f'{a}({n})' for a, n in exploded_a.value_counts().head(5).items())}

## 年代分布
{', '.join(f'{int(d)}年代({int(n)}部)' for d, n in df['year'].apply(lambda y: y//10*10).value_counts().sort_index().items())}

## 最高分电影Top5
{chr(10).join(f'- {r["title"]} ({r["score"]}分, {r["year"]}年)' for _, r in df.nlargest(5, 'score').iterrows())}

## 完整数据列
{', '.join(df.columns.tolist())}"""
    return ctx


def ai_chat():
    print_divider("AI 智能问答 (输入 'q' 退出)")
    print("  你可以自由提问关于这250部电影的任何问题")
    print("  例如: '为什么日本动画片评分这么高？'")
    print("        '帮我推荐5部悬疑片'")
    print("        '1990年代最好的10部电影是什么？'")

    data_ctx = build_data_context()
    messages = [
        {"role": "system", "content": f"""你是一个豆瓣电影数据分析助手。你可以访问以下数据集进行回答：

{data_ctx}

请根据数据回答问题，给出具体的数据和电影名称。如果用户要推荐电影，从数据集中筛选。
回答简洁专业，用中文。如果用户问数据中不存在的信息，诚实说明。"""}
    ]

    while True:
        try:
            q = input("\n💬 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("q", "quit", "exit", ""):
            print("  已退出AI问答")
            break

        print("  AI思考中...")
        messages.append({"role": "user", "content": q})

        try:
            resp = requests.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": messages,
                      "max_tokens": 1500, "temperature": 0.7},
                timeout=60
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                print(f"\n🤖 {answer}")
                messages.append({"role": "assistant", "content": answer})
            else:
                print(f"  API错误: {resp.status_code}")
        except Exception as e:
            print(f"  请求失败: {e}")


# ============================================================
#  菜单选项 6: 生成AI报告
# ============================================================

def generate_report():
    print_divider("生成AI分析报告")
    print("  正在调用 DeepSeek 生成完整分析报告...")
    print("  请稍候，约需30秒...")

    # 导入并运行 ai_analysis 模块
    sys.path.insert(0, os.path.join(BASE_DIR, "analysis"))
    from ai_analysis import load_and_summarize, build_prompt, call_deepseek, save_report

    df_local, summary = load_and_summarize()
    prompt = build_prompt(summary)
    content = call_deepseek(prompt)

    if content:
        save_report(content, df_local, summary)
        print(f"\n  报告已生成: report/ai_analysis_report.md")
    else:
        print("  生成失败，请检查网络连接")


# ============================================================
#  菜单选项 7: 电影推荐 (占位)
# ============================================================

def movie_recommend():
    print_divider("电影推荐")
    print("  该功能将在模块8中实现（TF-IDF + Cosine Similarity）")
    input("\n按回车返回菜单...")


# ============================================================
#  主菜单
# ============================================================

def show_menu():
    print(f"\n{'=' * 50}")
    print(f"  豆瓣Top250电影分析系统")
    print(f"{'=' * 50}")
    print(f"  1. 查看国家分析")
    print(f"  2. 查看类型分析")
    print(f"  3. 查看导演分析")
    print(f"  4. 查看演员分析")
    print(f"  5. AI 智能问答（自由提问）")
    print(f"  6. 生成AI分析报告")
    print(f"  7. 获取电影推荐")
    print(f"  0. 退出")
    print(f"{'=' * 50}")


def main():
    print("=" * 50)
    print("  豆瓣Top250电影分析系统")
    print("=" * 50)

    if not load_data():
        print("错误: 未找到数据文件，请先运行爬虫")
        return

    print(f"  已加载 {len(df)} 部电影\n")

    while True:
        show_menu()
        try:
            choice = input("请选择 (0-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break

        if choice == "1":
            show_country_analysis()
        elif choice == "2":
            show_genre_analysis()
        elif choice == "3":
            show_director_analysis()
        elif choice == "4":
            show_actor_analysis()
        elif choice == "5":
            ai_chat()
        elif choice == "6":
            generate_report()
        elif choice == "7":
            movie_recommend()
        elif choice == "0":
            print("  再见!")
            break
        else:
            print("  无效选择，请重新输入")


if __name__ == "__main__":
    main()
