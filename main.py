"""
豆瓣Top250电影分析系统 —— GUI版

基于 tkinter 的图形界面，提供数据分析、AI问答、电影推荐等功能。
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import requests
import os
import threading
import re
import sys

# ============================================================
#  配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# ============================================================
#  数据与工具
# ============================================================

def load_data():
    for name in ["movies_cleaned.csv", "movies_detail.csv", "movies.csv"]:
        path = os.path.join(BASE_DIR, "data", name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def explode_col(series, sep):
    return series.dropna().str.split(sep).explode()


def top_table(series, n=10):
    """返回 top N 的格式化文本"""
    lines = []
    for i, (k, v) in enumerate(series.head(n).items(), 1):
        lines.append(f"{i:2}. {k:30s} {v:4d} 部")
    return "\n".join(lines)


def score_table(grouped, n=10):
    """返回均分排名格式化文本"""
    lines = []
    for i, (k, row) in enumerate(grouped.head(n).iterrows(), 1):
        lines.append(f"{i:2}. {row['mean']:.2f} 分  {k}  ({int(row['count'])}部)")
    return "\n".join(lines)


# ============================================================
#  AI 调用（后台线程）
# ============================================================

def call_ai(messages, callback):
    """异步调用 DeepSeek API"""
    def _run():
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
                callback(answer)
            else:
                callback(f"[API错误 {resp.status_code}]")
        except Exception as e:
            callback(f"[网络错误: {e}]")
    threading.Thread(target=_run, daemon=True).start()


def build_data_context(df):
    """构造AI数据上下文"""
    ec = explode_col(df["country"], " / ")
    eg = explode_col(df["genre"], " ")
    ed = explode_col(df["director"], " / ")
    ea = explode_col(df["actors"], " / ")
    decades = df["year"].apply(lambda y: f"{y // 10 * 10}年代").value_counts()

    return f"""豆瓣Top250数据集：{len(df)}部电影。
评分{df['score'].min():.1f}~{df['score'].max():.1f}，均分{df['score'].mean():.2f}。
年代{int(df['year'].min())}~{int(df['year'].max())}。

国家Top5: {', '.join(f'{c}({n})' for c,n in ec.value_counts().head(5).items())}
类型Top5: {', '.join(f'{g}({n})' for g,n in eg.value_counts().head(5).items())}
导演Top5: {', '.join(f'{d}({n})' for d,n in ed.value_counts().head(5).items())}
演员Top5: {', '.join(f'{a}({n})' for a,n in ea.value_counts().head(5).items())}
年代分布: {', '.join(f'{d}({int(n)})' for d,n in decades.items())}

高分Top5: {', '.join(f'{r["title"]}({r["score"]})' for _,r in df.nlargest(5,'score').iterrows())}

数据列: {', '.join(df.columns)}"""


# ============================================================
#  GUI
# ============================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("豆瓣Top250 电影分析系统")
        self.root.geometry("900x680")
        self.root.configure(bg="#f0f0f0")

        # 加载数据
        self.df = load_data()
        if self.df is None:
            messagebox.showerror("错误", "未找到数据文件")
            root.destroy()
            return

        # AI对话历史
        self.ai_messages = [
            {"role": "system", "content": f"你是电影数据分析助手。回答简洁专业。数据:\n{build_data_context(self.df)}"}
        ]

        self.setup_ui()
        self.status(f"已加载 {len(self.df)} 部电影  |  评分 {self.df['score'].min():.1f}~{self.df['score'].max():.1f}")

    # ---- 布局 ----

    def setup_ui(self):
        # 左侧面板 - 数据查询
        left = ttk.Frame(self.root, width=400)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left.pack_propagate(False)

        ttk.Label(left, text="数据分析", font=("Microsoft YaHei", 14, "bold")).pack(pady=(0, 10))

        btns = [
            ("国家/地区分析", self.show_country),
            ("类型分析", self.show_genre),
            ("导演分析", self.show_director),
            ("演员分析", self.show_actor),
            ("生成AI分析报告", self.generate_report),
        ]
        for text, cmd in btns:
            ttk.Button(left, text=text, command=cmd, width=25).pack(pady=3)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 左侧结果显示
        self.result_text = scrolledtext.ScrolledText(
            left, width=48, height=28, font=("Consolas", 10), wrap=tk.WORD
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 右侧面板 - AI问答
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(right, text="AI 智能问答", font=("Microsoft YaHei", 14, "bold")).pack(pady=(0, 5))
        ttk.Label(right, text="自由提问，AI根据数据回答", foreground="gray").pack(pady=(0, 10))

        # 快捷提问
        quick_frame = ttk.Frame(right)
        quick_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quick_frame, text="快捷提问:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        quick_qs = [
            "推荐5部悬疑片",
            "评分最高的10部电影",
            "日本动画为什么分高",
            "1990年代最好电影",
        ]
        for q in quick_qs:
            ttk.Button(quick_frame, text=q, width=18,
                       command=lambda q=q: self.quick_ask(q)).pack(side=tk.LEFT, padx=2)

        # 聊天显示区
        self.chat_text = scrolledtext.ScrolledText(
            right, width=50, height=24, font=("Microsoft YaHei", 10), wrap=tk.WORD
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.chat_text.tag_config("user", foreground="#1a73e8", font=("Microsoft YaHei", 10, "bold"))
        self.chat_text.tag_config("ai", foreground="#0d904f")
        self.chat_text.tag_config("system", foreground="gray", font=("Microsoft YaHei", 9))
        self.chat_text.insert(tk.END, "AI: 你好！我是电影数据分析助手。\n", "system")
        self.chat_text.insert(tk.END, "    可以问我任何关于这250部电影的问题。\n\n", "system")
        self.chat_text.see(tk.END)

        # 输入区
        input_frame = ttk.Frame(right)
        input_frame.pack(fill=tk.X)
        self.input_entry = ttk.Entry(input_frame, font=("Microsoft YaHei", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", lambda e: self.ask_ai())
        ttk.Button(input_frame, text="发送", command=self.ask_ai, width=8).pack(side=tk.RIGHT)

    # ---- 数据查询方法 ----

    def show_result(self, title, content):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"{'=' * 40}\n")
        self.result_text.insert(tk.END, f"  {title}\n")
        self.result_text.insert(tk.END, f"{'=' * 40}\n\n")
        self.result_text.insert(tk.END, content)

    def show_country(self):
        e = explode_col(self.df["country"], " / ")
        cs = pd.DataFrame({"country": e, "score": self.df.loc[e.index, "score"]})
        g = cs.groupby("country")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 3].sort_values("mean", ascending=False)

        text = "[出现次数 Top10]\n"
        text += top_table(e.value_counts())
        text += f"\n\n[平均评分 Top10 (>=3部)]\n"
        text += score_table(g)
        self.show_result("国家/地区分析", text)
        self.status("已显示国家分析")

    def show_genre(self):
        e = explode_col(self.df["genre"], " ")
        gs = pd.DataFrame({"genre": e, "score": self.df.loc[e.index, "score"]})
        g = gs.groupby("genre")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 5].sort_values("mean", ascending=False)

        text = "[出现次数 Top10]\n"
        text += top_table(e.value_counts())
        text += f"\n\n[平均评分 Top10 (>=5部)]\n"
        text += score_table(g)
        self.show_result("类型分析", text)
        self.status("已显示类型分析")

    def show_director(self):
        e = explode_col(self.df["director"], " / ")
        ds = pd.DataFrame({"director": e, "score": self.df.loc[e.index, "score"]})
        g = ds.groupby("director")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 2].sort_values("mean", ascending=False)

        text = "[出现次数 Top10]\n"
        text += top_table(e.value_counts())
        text += f"\n\n[平均评分 Top10 (>=2部)]\n"
        text += score_table(g)
        self.show_result("导演分析", text)
        self.status("已显示导演分析")

    def show_actor(self):
        e = explode_col(self.df["actors"], " / ")
        as_ = pd.DataFrame({"actor": e, "score": self.df.loc[e.index, "score"]})
        g = as_.groupby("actor")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 2].sort_values("mean", ascending=False)

        text = "[出现次数 Top10]\n"
        text += top_table(e.value_counts())
        text += f"\n\n[平均评分 Top10 (>=2部)]\n"
        text += score_table(g)
        self.show_result("演员分析", text)
        self.status("已显示演员分析")

    # ---- AI 报告 ----

    def generate_report(self):
        self.show_result("生成中...", "正在调用 DeepSeek 生成分析报告，请稍候...")
        self.status("正在生成AI报告...")

        # 构建完整统计
        sys.path.insert(0, os.path.join(BASE_DIR, "analysis"))
        from ai_analysis import load_and_summarize, build_prompt, call_deepseek, save_report
        df, summary = load_and_summarize()
        prompt = build_prompt(summary)

        def on_done(content):
            if content and not content.startswith("["):
                save_report(content, df, summary)
                self.show_result("AI分析报告", f"报告已生成！\n\n文件: report/ai_analysis_report.md\n\n{content[:2000]}...")
                self.status("报告生成完成")
            else:
                self.show_result("错误", f"生成失败: {content}")
                self.status("报告生成失败")

        call_ai([
            {"role": "system", "content": "你是专业电影数据分析师，请生成详细报告。"},
            {"role": "user", "content": prompt}
        ], on_done)

    # ---- AI 问答 ----

    def quick_ask(self, question):
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, question)
        self.ask_ai()

    def ask_ai(self):
        q = self.input_entry.get().strip()
        if not q:
            return
        self.input_entry.delete(0, tk.END)

        # 显示用户消息
        self.chat_text.insert(tk.END, f"你: {q}\n", "user")
        self.chat_text.insert(tk.END, "AI思考中...\n", "system")
        self.chat_text.see(tk.END)
        self.status("AI思考中...")

        self.ai_messages.append({"role": "user", "content": q})

        def on_done(answer):
            # 删除 "AI思考中..."
            last_line_start = self.chat_text.get("end-2c linestart", "end-1c")
            self.chat_text.delete("end-2c linestart", "end-1c")
            self.chat_text.insert(tk.END, f"AI: {answer}\n\n", "ai")
            self.chat_text.see(tk.END)
            self.ai_messages.append({"role": "assistant", "content": answer})
            self.status("就绪")

        call_ai(self.ai_messages, on_done)

    def status(self, msg):
        self.status_bar.config(text=f"  {msg}")


# ============================================================
#  启动
# ============================================================

def main():
    root = tk.Tk()
    # 设置中文字体兼容
    style = ttk.Style()
    style.theme_use("clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
