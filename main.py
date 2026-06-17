"""
豆瓣Top250电影分析系统 —— GUI版
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import requests
import os
import threading
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# ---- 配色 ----
C_BG      = "#f5f0eb"   # 背景
C_PANEL   = "#ffffff"   # 面板
C_ACCENT  = "#2d8c4a"   # 豆瓣绿
C_ACCENT2 = "#e09040"   # 暖橙
C_TEXT    = "#333333"   # 正文
C_SUB     = "#888888"   # 次要文字
C_USER    = "#1a6fb5"   # 用户消息
C_AI      = "#2d8c4a"   # AI消息
C_BORDER  = "#e0d8cf"   # 边框


def load_data():
    for name in ["movies_cleaned.csv", "movies_detail.csv", "movies.csv"]:
        path = os.path.join(BASE_DIR, "data", name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def explode_col(series, sep):
    return series.dropna().str.split(sep).explode()


def top_table(series, n=10):
    lines = []
    for i, (k, v) in enumerate(series.head(n).items(), 1):
        lines.append(f"  {i:2}.  {k:28s} {v:4d} 部")
    return "\n".join(lines)


def score_table(grouped, n=10):
    lines = []
    for i, (k, row) in enumerate(grouped.head(n).iterrows(), 1):
        lines.append(f"  {i:2}.  {row['mean']:.2f} 分  {k}  ({int(row['count'])}部)")
    return "\n".join(lines)


def call_ai(messages, callback):
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
                callback(resp.json()["choices"][0]["message"]["content"])
            else:
                callback(f"[API错误 {resp.status_code}]")
        except Exception as e:
            callback(f"[网络错误: {e}]")
    threading.Thread(target=_run, daemon=True).start()


def build_data_context(df):
    ec = explode_col(df["country"], " / ")
    eg = explode_col(df["genre"], " ")
    ed = explode_col(df["director"], " / ")
    ed = ed[ed != "未知"]
    ea = explode_col(df["actors"], " / ")
    ea = ea[ea != "未知"]
    decades = df["year"].apply(lambda y: f"{y // 10 * 10}年代").value_counts()
    return f"""豆瓣Top250：{len(df)}部电影，评分{df['score'].min():.1f}~{df['score'].max():.1f}，均分{df['score'].mean():.2f}。
年代{int(df['year'].min())}~{int(df['year'].max())}。
国家Top5: {', '.join(f'{c}({n})' for c,n in ec.value_counts().head(5).items())}
类型Top5: {', '.join(f'{g}({n})' for g,n in eg.value_counts().head(5).items())}
导演Top5: {', '.join(f'{d}({n})' for d,n in ed.value_counts().head(5).items())}
演员Top5: {', '.join(f'{a}({n})' for a,n in ea.value_counts().head(5).items())}
年代: {', '.join(f'{d}({int(n)})' for d,n in decades.items())}
高分Top5: {', '.join(f'{r["title"]}({r["score"]})' for _,r in df.nlargest(5,'score').iterrows())}"""


# ============================================================
#  GUI
# ============================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("豆瓣Top250 电影分析系统")
        self.root.geometry("960x700")
        self.root.configure(bg=C_BG)
        self.root.minsize(800, 550)

        self.df = load_data()
        if self.df is None:
            messagebox.showerror("错误", "未找到数据文件")
            root.destroy()
            return

        self.ai_messages = [
            {"role": "system", "content": f"你是电影数据分析助手，回答简洁专业，引用数据。\n{build_data_context(self.df)}"}
        ]

        self.setup_ui()
        self.set_status(f"已加载 {len(self.df)} 部电影    |    评分 {self.df['score'].min():.1f} ~ {self.df['score'].max():.1f}    |    均分 {self.df['score'].mean():.2f}")

    # ---- UI 构建 ----

    def setup_ui(self):
        # 顶部标题栏
        header = tk.Frame(self.root, bg=C_ACCENT, height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="豆瓣Top250 电影分析系统", font=("Microsoft YaHei", 16, "bold"),
                 bg=C_ACCENT, fg="white").pack(pady=10)

        # 主体
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 0))

        # -- 左侧面板 --
        left = tk.Frame(body, bg=C_PANEL, bd=0, highlightbackground=C_BORDER,
                        highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))
        left.pack_propagate(False)
        left.configure(width=380)

        # 标题
        tk.Label(left, text="数据查询", font=("Microsoft YaHei", 13, "bold"),
                 bg=C_PANEL, fg=C_TEXT).pack(pady=(15, 12))

        # 按钮
        btns = [
            ("国家/地区分析", self.show_country),
            ("类型分析", self.show_genre),
            ("导演分析", self.show_director),
            ("演员分析", self.show_actor),
        ]
        for text, cmd in btns:
            self._make_btn(left, text, cmd).pack(pady=3, padx=20, fill=tk.X)

        # 分隔
        tk.Frame(left, bg=C_BORDER, height=1).pack(fill=tk.X, padx=20, pady=12)

        self._make_btn(left, "生成 AI 完整分析报告", self.generate_report,
                       accent=True).pack(pady=3, padx=20, fill=tk.X)

        # 结果显示（只读）
        result_container = tk.Frame(left, bg=C_PANEL)
        result_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(12, 15))

        self.result_text = tk.Text(
            result_container, font=("Consolas", 10), wrap=tk.WORD,
            bg="#fafaf7", fg=C_TEXT, relief=tk.FLAT, bd=0,
            padx=12, pady=8, state=tk.DISABLED
        )
        result_scroll = ttk.Scrollbar(result_container, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._result_insert("欢迎\n\n点击左侧按钮查看数据统计\n\n右侧窗口可与 AI 自由对话")

        # -- 右侧面板 --
        right = tk.Frame(body, bg=C_PANEL, bd=0, highlightbackground=C_BORDER,
                         highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(right, text="AI 智能问答", font=("Microsoft YaHei", 13, "bold"),
                 bg=C_PANEL, fg=C_TEXT).pack(pady=(15, 5))
        tk.Label(right, text="自由提问，AI 基于数据实时回答",
                 font=("Microsoft YaHei", 9), bg=C_PANEL, fg=C_SUB).pack(pady=(0, 10))

        # 快捷提问
        quick_frame = tk.Frame(right, bg=C_PANEL)
        quick_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        tk.Label(quick_frame, text="快捷提问", font=("Microsoft YaHei", 9),
                 bg=C_PANEL, fg=C_SUB).pack(anchor=tk.W, pady=(0, 4))

        qrow1 = tk.Frame(quick_frame, bg=C_PANEL)
        qrow1.pack(fill=tk.X, pady=(0, 2))
        qrow2 = tk.Frame(quick_frame, bg=C_PANEL)
        qrow2.pack(fill=tk.X)

        quick_qs = [
            "推荐5部高分悬疑片", "1990年代最好的10部电影",
            "日本动画为什么评分高", "评分最高的导演是谁",
        ]
        for i, q in enumerate(quick_qs):
            parent = qrow1 if i < 2 else qrow2
            self._make_quick_btn(parent, q).pack(side=tk.LEFT, padx=(0 if i % 2 else 0, 4 if i % 2 == 0 else 0))

        # 聊天区
        chat_container = tk.Frame(right, bg=C_PANEL)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 0))

        self.chat_text = tk.Text(
            chat_container, font=("Microsoft YaHei", 10), wrap=tk.WORD,
            bg="#fafaf7", fg=C_TEXT, relief=tk.FLAT, bd=0,
            padx=12, pady=8, state=tk.DISABLED
        )
        chat_scroll = ttk.Scrollbar(chat_container, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=chat_scroll.set)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text.tag_config("user", foreground=C_USER, font=("Microsoft YaHei", 10, "bold"),
                                  lmargin1=20, lmargin2=20, spacing1=4, spacing3=4)
        self.chat_text.tag_config("ai", foreground=C_AI, lmargin1=10, lmargin2=10,
                                  spacing1=4, spacing3=4)
        self.chat_text.tag_config("system", foreground=C_SUB, font=("Microsoft YaHei", 9),
                                  lmargin1=10, lmargin2=10)

        self._chat_insert("AI:  你好！我是电影数据分析助手。\n", "ai")
        self._chat_insert("     可以问我任何关于这250部电影的问题。\n\n", "system")
        self._chat_insert("提示:  点击上方快捷按钮，或在下方输入框自由提问\n\n", "system")

        # 输入区
        input_frame = tk.Frame(right, bg=C_PANEL)
        input_frame.pack(fill=tk.X, padx=15, pady=(8, 15))

        self.input_entry = tk.Entry(input_frame, font=("Microsoft YaHei", 11),
                                    relief=tk.SOLID, bd=1,
                                    highlightbackground=C_BORDER, highlightthickness=1)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=4)
        self.input_entry.bind("<Return>", lambda e: self.ask_ai())
        self.input_entry.focus_set()

        send_btn = tk.Button(input_frame, text="发送", command=self.ask_ai,
                             bg=C_ACCENT, fg="white", font=("Microsoft YaHei", 10, "bold"),
                             relief=tk.FLAT, padx=18, pady=4, cursor="hand2",
                             activebackground="#23703a", activeforeground="white")
        send_btn.pack(side=tk.RIGHT)

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="", font=("Microsoft YaHei", 9),
                                   bg="#e8e3dc", fg=C_SUB, anchor=tk.W, padx=12, pady=3)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---- 自定义按钮 ----

    def _make_btn(self, parent, text, cmd, accent=False):
        bg = C_ACCENT if accent else C_PANEL
        fg = "white" if accent else C_TEXT
        hover_bg = "#23703a" if accent else "#f0ebe2"
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Microsoft YaHei", 10),
                        bg=bg, fg=fg, relief=tk.FLAT, padx=12, pady=8,
                        cursor="hand2", activebackground=hover_bg,
                        activeforeground="white" if accent else C_TEXT,
                        highlightbackground=C_BORDER if not accent else bg,
                        highlightthickness=1 if not accent else 0)
        return btn

    def _make_quick_btn(self, parent, text):
        btn = tk.Button(parent, text=text,
                        command=lambda: self.quick_ask(text),
                        font=("Microsoft YaHei", 9), bg="#f5f2ed", fg=C_TEXT,
                        relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
                        activebackground="#e8e0d4", activeforeground=C_TEXT,
                        highlightbackground=C_BORDER, highlightthickness=1)
        return btn

    # ---- 只读聊天区操作 ----

    def _chat_insert(self, text, tag=None):
        self.chat_text.configure(state=tk.NORMAL)
        if tag:
            self.chat_text.insert(tk.END, text, tag)
        else:
            self.chat_text.insert(tk.END, text)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _chat_delete_last_line(self):
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete("end-2c linestart", "end-1c")
        self.chat_text.configure(state=tk.DISABLED)

    # ---- 结果显示（只读） ----

    def _result_insert(self, text):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def show_result(self, title, content):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"  {title}\n")
        self.result_text.insert(tk.END, f"  {'─' * 38}\n\n")
        self.result_text.insert(tk.END, content)
        self.result_text.configure(state=tk.DISABLED)

    def show_country(self):
        e = explode_col(self.df["country"], " / ")
        cs = pd.DataFrame({"country": e, "score": self.df.loc[e.index, "score"]})
        g = cs.groupby("country")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 3].sort_values("mean", ascending=False)
        text = "▎出现次数 Top10\n" + top_table(e.value_counts())
        text += "\n\n▎平均评分 Top10 (≥3部)\n" + score_table(g)
        self.show_result("国家/地区分析", text)
        self.set_status("国家分析已完成")

    def show_genre(self):
        e = explode_col(self.df["genre"], " ")
        gs = pd.DataFrame({"genre": e, "score": self.df.loc[e.index, "score"]})
        g = gs.groupby("genre")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 5].sort_values("mean", ascending=False)
        text = "▎出现次数 Top10\n" + top_table(e.value_counts())
        text += "\n\n▎平均评分 Top10 (≥5部)\n" + score_table(g)
        self.show_result("类型分析", text)
        self.set_status("类型分析已完成")

    def show_director(self):
        e = explode_col(self.df["director"], " / ")
        e = e[e != "未知"]  # 剔除缺失值
        ds = pd.DataFrame({"director": e, "score": self.df.loc[e.index, "score"]})
        g = ds.groupby("director")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 2].sort_values("mean", ascending=False)
        text = "▎出现次数 Top10\n" + top_table(e.value_counts())
        text += "\n\n▎平均评分 Top10 (≥2部)\n" + score_table(g)
        self.show_result("导演分析", text)
        self.set_status("导演分析已完成")

    def show_actor(self):
        e = explode_col(self.df["actors"], " / ")
        e = e[e != "未知"]  # 剔除缺失值
        ag = pd.DataFrame({"actor": e, "score": self.df.loc[e.index, "score"]})
        g = ag.groupby("actor")["score"].agg(["mean", "count"])
        g = g[g["count"] >= 2].sort_values("mean", ascending=False)
        text = "▎出现次数 Top10\n" + top_table(e.value_counts())
        text += "\n\n▎平均评分 Top10 (≥2部)\n" + score_table(g)
        self.show_result("演员分析", text)
        self.set_status("演员分析已完成")

    # ---- AI 报告 ----

    def generate_report(self):
        self.show_result("生成中...", "正在调用 DeepSeek 生成完整分析报告，请稍候...\n\n约需 15~30 秒")
        self.set_status("正在生成AI报告...")
        sys.path.insert(0, os.path.join(BASE_DIR, "analysis"))
        from ai_analysis import load_and_summarize, build_prompt, call_deepseek, save_report
        df, summary = load_and_summarize()
        prompt = build_prompt(summary)

        def on_done(content):
            if content and not content.startswith("["):
                save_report(content, df, summary)
                self.show_result("AI 分析报告", f"报告已生成!\n\n文件: report/ai_analysis_report.md\n\n{'─'*38}\n\n{content[:1500]}...")
                self.set_status("报告生成完成")
            else:
                self.show_result("错误", f"生成失败: {content}")
                self.set_status("报告生成失败")

        call_ai([{"role": "system", "content": "你是专业电影数据分析师，生成详细Markdown报告。"},
                 {"role": "user", "content": prompt}], on_done)

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
        self.input_entry.configure(state=tk.DISABLED)

        self._chat_insert(f"你: {q}\n", "user")
        self._chat_insert("AI 思考中...\n", "system")
        self.set_status("AI 思考中...")
        self.ai_messages.append({"role": "user", "content": q})

        def on_done(answer):
            self._chat_delete_last_line()
            self._chat_insert(f"AI: {answer}\n\n", "ai")
            self.ai_messages.append({"role": "assistant", "content": answer})
            self.input_entry.configure(state=tk.NORMAL)
            self.input_entry.focus_set()
            self.set_status("就绪")

        call_ai(self.ai_messages, on_done)

    def set_status(self, msg):
        self.status_bar.config(text=f"    {msg}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
