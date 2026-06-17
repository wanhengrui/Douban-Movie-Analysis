"""
豆瓣Top250电影分析系统 —— Web版
Flask + TF-IDF推荐 + AI解释
"""
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import requests
import os
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# ============================================================
#  数据加载
# ============================================================

def load_data():
    for name in ["movies_cleaned.csv", "movies_detail.csv", "movies.csv"]:
        path = os.path.join(BASE_DIR, "data", name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df = load_data()

# 补齐必要字段
for col in ["summary", "director", "actors", "runtime"]:
    if col not in df.columns:
        df[col] = ""

# ============================================================
#  TF-IDF 推荐引擎
# ============================================================

# 把 summary 中为 NaN 或 "未知" 的填成空字符串
summaries = df["summary"].fillna("").replace("未知", "").tolist()
vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix = vectorizer.fit_transform(summaries)

# 电影标题列表用于模糊匹配
titles = df["title"].tolist()


def find_movie(query):
    """模糊匹配电影名，返回 (index, title)"""
    q = query.strip().lower()
    # 精确匹配
    for i, t in enumerate(titles):
        if q == t.lower():
            return i, t
    # 包含匹配
    for i, t in enumerate(titles):
        if q in t.lower() or t.lower() in q:
            return i, t
    return None, None


def recommend_by_tfidf(query, top_n=5):
    """
    根据电影名利用 TF-IDF + Cosine Similarity 推荐相似电影。

    原理：
    1. 将所有电影剧情简介转为 TF-IDF 向量（词频-逆文档频率）
    2. 找到输入电影的向量
    3. 计算与所有其他电影的余弦相似度
    4. 返回相似度最高的 top_n 部
    """
    idx, title = find_movie(query)
    if idx is None:
        return None, f"未找到电影: {query}"

    query_vec = tfidf_matrix[idx]
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # 排除自己
    sims[idx] = -1
    top_indices = np.argsort(sims)[::-1][:top_n]

    results = []
    for i in top_indices:
        s = float(sims[i])
        if s < 0.05:
            break
        row = df.iloc[i]
        results.append({
            "title": row["title"],
            "score": float(row["score"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else 0,
            "director": str(row["director"]) if pd.notna(row["director"]) else "",
            "genre": str(row["genre"]) if pd.notna(row["genre"]) else "",
            "similarity": round(s, 4),
            "summary": str(row["summary"])[:200] if pd.notna(row["summary"]) and row["summary"] != "未知" else "",
        })

    return {"query_title": title, "query_summary": str(df.iloc[idx]["summary"])[:300],
            "results": results}, None


# ============================================================
#  统计
# ============================================================

def explode_col(series, sep):
    return series.dropna().str.split(sep).explode()

def compute_stats():
    ec = explode_col(df["country"], " / ")
    eg = explode_col(df["genre"], " ")
    ed = explode_col(df["director"], " / "); ed = ed[ed != "未知"]
    ea = explode_col(df["actors"], " / ");   ea = ea[ea != "未知"]

    cs = pd.DataFrame({"val": ec, "score": df.loc[ec.index, "score"]})
    gs = pd.DataFrame({"val": eg, "score": df.loc[eg.index, "score"]})
    ds = pd.DataFrame({"val": ed, "score": df.loc[ed.index, "score"]})
    ag = pd.DataFrame({"val": ea, "score": df.loc[ea.index, "score"]})

    def agg(gdf, mc):
        g = gdf.groupby("val")["score"].agg(["mean", "count"])
        return g[g["count"] >= mc].sort_values("mean", ascending=False)

    return {
        "total": len(df), "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()), "score_mean": round(float(df["score"].mean()), 2),
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
        "country_top": _ft(ec.value_counts().head(10)),
        "country_score": _fs(agg(cs, 3)),
        "genre_top": _ft(eg.value_counts().head(10)),
        "genre_score": _fs(agg(gs, 5)),
        "director_top": _ft(ed.value_counts().head(10)),
        "director_score": _fs(agg(ds, 2)),
        "actor_top": _ft(ea.value_counts().head(10)),
        "actor_score": _fs(agg(ag, 2)),
    }

def _ft(s): return [{"name": k, "count": int(v)} for k, v in s.items()]
def _fs(g): return [{"name": k, "mean": round(float(r["mean"]), 2), "count": int(r["count"])}
                     for k, r in g.head(10).iterrows()]

def build_data_ctx():
    s = compute_stats()
    ctx = f"豆瓣Top250：{s['total']}部，评分{s['score_min']}~{s['score_max']}，均分{s['score_mean']}。\n"
    ctx += f"年代{s['year_min']}~{s['year_max']}。\n"
    ctx += "国家Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["country_top"][:5]) + "\n"
    ctx += "类型Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["genre_top"][:5]) + "\n"
    ctx += "导演Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["director_top"][:5]) + "\n"
    ctx += f"高分Top5: " + ", ".join(f"{r['title']}({r['score']})" for _, r in df.nlargest(5, 'score').iterrows()) + "\n"
    ctx += "\n你有能力调用 recommend 功能：当用户说'推荐和XXX类似的电影'时，先说明可以通过TF-IDF分析剧情简介找到相似电影，然后列出可能的匹配结果。"
    return ctx

# ============================================================
#  API
# ============================================================

@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats())

@app.route("/api/recommend")
def api_recommend():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "请输入电影名称"})

    data, error = recommend_by_tfidf(q)
    if error:
        return jsonify({"ok": False, "error": error})

    return jsonify({"ok": True, **data})


@app.route("/api/recommend_ai", methods=["POST"])
def api_recommend_ai():
    """TF-IDF推荐 + AI解释"""
    data = request.get_json()
    q = data.get("q", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "请输入电影名称"})

    rec_data, error = recommend_by_tfidf(q)
    if error:
        return jsonify({"ok": False, "error": error})

    # 构建推荐结果文本发给AI
    rec_text = f"输入: {rec_data['query_title']}\n剧情: {rec_data['query_summary'][:200]}\n\n相似推荐:\n"
    for i, r in enumerate(rec_data["results"], 1):
        rec_text += f"{i}. {r['title']} ({r['score']}分, {r['year']}年, {r['genre']})\n"
        rec_text += f"   相似度:{r['similarity']:.2f} 导演:{r['director']}\n"
        rec_text += f"   简介:{r['summary'][:150]}\n"

    prompt = f"""以下是基于TF-IDF算法分析剧情简介后得到的电影推荐结果：

{rec_text}

请用自然语言向用户解释这些推荐：
1. 简要说明TF-IDF算法的原理（1-2句话，通俗易懂）
2. 分析为什么这些电影相似（从剧情主题、风格等角度，引用简介中的关键词）
3. 给每条推荐一句话点评
4. 整体风格：热情、专业、不过于学术化，200字以内"""

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 800, "temperature": 0.7},
            timeout=60
        )
        if resp.status_code == 200:
            explanation = resp.json()["choices"][0]["message"]["content"]
            return jsonify({"ok": True, "explanation": explanation, **rec_data})

        return jsonify({"ok": False, "error": f"API错误 {resp.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    history = data.get("history", [])

    # 检测是否在问推荐相关的问题
    rec_keywords = ["推荐", "相似", "类似", "差不多", "同类型", "差不多"]
    is_rec_q = any(k in question for k in rec_keywords)

    # 尝试从中提取电影名
    if is_rec_q:
        # 尝试用每个电影名去匹配问题中提到的电影
        mentioned = None
        for t in titles:
            if t in question:
                mentioned = t
                break

        if mentioned:
            rec_data, error = recommend_by_tfidf(mentioned)
            if rec_data and len(rec_data["results"]) > 0:
                rec_text = "\n".join(
                    f"{i+1}. {r['title']} ({r['score']}分, 相似度{r['similarity']:.2f}) - {r['genre']}"
                    for i, r in enumerate(rec_data["results"])
                )
                question = f"""{question}

[系统自动补充：已通过TF-IDF分析剧情简介，找到与"{mentioned}"最相似的电影：
{rec_text}

请基于以上TF-IDF推荐结果回答用户问题。简要解释TF-IDF原理，并点评这些推荐。]"""

    messages = [
        {"role": "system", "content": f"你是电影数据分析助手。\n{build_data_ctx()}"}
    ]
    for h in history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": question})

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": messages,
                  "max_tokens": 1200, "temperature": 0.7},
            timeout=60
        )
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"]
            return jsonify({"ok": True, "answer": answer})
        return jsonify({"ok": False, "error": f"API {resp.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/report", methods=["POST"])
def api_report():
    sys.path.insert(0, os.path.join(BASE_DIR, "analysis"))
    from ai_analysis import load_and_summarize, build_prompt, call_deepseek, save_report
    df_r, summary = load_and_summarize()
    prompt = build_prompt(summary)
    content = call_deepseek(prompt)
    if content:
        save_report(content, df_r, summary)
        return jsonify({"ok": True, "preview": content[:3000]})
    return jsonify({"ok": False, "error": "生成失败"})


# ============================================================
#  页面
# ============================================================

INDEX = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>豆瓣Top250 电影分析系统</title>
<style>
:root {
  --green: #2d8c4a; --green-dark: #1f6b35; --green-light: #e8f5e9;
  --orange: #e09040; --orange-dark: #c87830;
  --bg: #f5f0eb; --card: #fff; --card-alt: #fafaf7;
  --text: #333; --text-sub: #888; --border: #e0d8cf;
  --blue: #1a6fb5; --blue-dark: #145a91;
  --radius: 12px; --shadow: 0 2px 12px rgba(0,0,0,.06);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
/* ---- header ---- */
header{background:linear-gradient(135deg,var(--green),var(--green-dark));color:#fff;padding:16px 28px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 8px rgba(0,0,0,.12)}
header h1{font-size:20px;font-weight:700;letter-spacing:.5px}
header .status{font-size:13px;opacity:.9;background:rgba(255,255,255,.15);padding:6px 14px;border-radius:20px}
/* ---- layout ---- */
main{flex:1;display:flex;overflow:hidden;gap:0}
/* ---- sidebar ---- */
aside{width:420px;background:var(--card);display:flex;flex-direction:column;flex-shrink:0;box-shadow:var(--shadow);z-index:1}
aside .section{padding:16px 20px 8px}
aside .section h2{font-size:14px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:6px}
aside .section h2 .dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block}
aside .btn-row{padding:4px 20px 12px;display:grid;grid-template-columns:1fr 1fr;gap:8px}
aside .btn-row button{padding:10px 8px;border:1px solid var(--border);border-radius:8px;background:var(--card-alt);cursor:pointer;font-size:13px;color:var(--text);transition:all .2s;font-weight:500}
aside .btn-row button:hover{background:var(--green);color:#fff;border-color:var(--green);transform:translateY(-1px);box-shadow:0 3px 8px rgba(45,140,74,.25)}
aside .btn-row button.report{grid-column:1/-1;background:var(--orange);color:#fff;border-color:var(--orange)}
aside .btn-row button.report:hover{background:var(--orange-dark);border-color:var(--orange-dark);box-shadow:0 3px 8px rgba(224,144,64,.3)}
/* ---- rec panel ---- */
.rec-panel{margin:0 20px 12px;padding:14px 16px;background:var(--green-light);border-radius:var(--radius);border:1px solid #d0e8d4}
.rec-panel label{font-size:13px;font-weight:700;color:var(--green);display:block;margin-bottom:8px}
.rec-panel input{width:100%;padding:10px 14px;border:1px solid #d0e8d4;border-radius:8px;font-size:13px;outline:none;background:#fff;transition:border .2s}
.rec-panel input:focus{border-color:var(--green);box-shadow:0 0 0 3px rgba(45,140,74,.1)}
.rec-panel .hint{font-size:11px;color:var(--text-sub);margin-top:5px}
/* ---- result area ---- */
.result-area{flex:1;margin:0 20px 16px;background:var(--card-alt);border-radius:var(--radius);padding:16px;overflow:auto;font-size:13px;line-height:1.8;border:1px solid var(--border)}
.result-area .empty{color:var(--text-sub);text-align:center;padding:40px 0;line-height:2}
.result-area .empty .icon{font-size:36px;display:block;margin-bottom:8px}
/* rec cards */
.rec-card{background:var(--card);border-radius:10px;padding:14px 16px;margin-bottom:10px;border:1px solid var(--border);transition:all .2s;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.rec-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.08);border-color:#ccc}
.rec-card .rec-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.rec-card .rec-title{font-size:15px;font-weight:700;color:var(--text)}
.rec-card .rec-score{font-size:13px;font-weight:700;color:var(--orange);background:#fff7ef;padding:3px 10px;border-radius:12px}
.rec-card .rec-sim{font-size:11px;color:var(--green);font-weight:600}
.rec-card .rec-meta{font-size:12px;color:var(--text-sub);margin-bottom:4px}
.rec-card .rec-summary{font-size:12px;color:#666;line-height:1.6;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.rec-card .rec-rank{display:inline-block;width:24px;height:24px;line-height:24px;text-align:center;border-radius:50%;background:var(--green);color:#fff;font-size:12px;font-weight:700;margin-right:10px}
.ai-note{background:#f8f4ff;border:1px solid #e0d4f0;border-radius:10px;padding:14px 16px;margin-top:14px;font-size:13px;line-height:1.8;color:#555}
.ai-note::before{content:'AI 分析';display:block;font-size:11px;font-weight:700;color:#7c5cbf;margin-bottom:6px;letter-spacing:1px;text-transform:uppercase}
.stats-table{font-family:Consolas,"Courier New",monospace;font-size:13px;line-height:1.8}
.stats-table .section-title{font-weight:700;color:var(--green);margin:14px 0 6px;font-size:14px}
/* ---- chat ---- */
section.chat{flex:1;background:var(--card);display:flex;flex-direction:column;position:relative}
section.chat .chat-header{padding:18px 24px 12px;border-bottom:1px solid var(--border);background:var(--card-alt)}
section.chat .chat-header h2{font-size:14px;font-weight:700;color:var(--text)}
section.chat .quick-row{padding:10px 24px;display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--card-alt)}
section.chat .quick-row button{padding:7px 16px;border:1px solid var(--border);border-radius:18px;background:var(--card);cursor:pointer;font-size:12px;color:#666;transition:all .2s;white-space:nowrap}
section.chat .quick-row button:hover{background:var(--green);color:#fff;border-color:var(--green)}
.messages{flex:1;padding:20px 24px;overflow:auto;display:flex;flex-direction:column;gap:14px;background:linear-gradient(180deg,#fafaf7 0%,#f5f2ed 100%)}
.messages .msg{max-width:80%;padding:12px 16px;border-radius:14px;font-size:13px;line-height:1.7;animation:slideUp .25s ease-out;word-break:break-word}
.messages .msg.user{align-self:flex-end;background:var(--blue);color:#fff;border-bottom-right-radius:4px;box-shadow:0 2px 8px rgba(26,111,181,.25)}
.messages .msg.ai{align-self:flex-start;background:var(--card);color:var(--text);border-bottom-left-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #eee}
.messages .msg.thinking{align-self:flex-start;color:var(--text-sub);font-size:12px;padding:4px 0;animation:pulse 1.2s infinite}
.messages .msg.thinking::after{content:'...';animation:dots 1.5s steps(3,end) infinite}
@keyframes slideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
@keyframes dots{0%{content:'.'}33%{content:'..'}66%{content:'...'}}
.chat-input{padding:14px 24px 18px;display:flex;gap:10px;border-top:1px solid var(--border);background:var(--card-alt)}
.chat-input input{flex:1;padding:12px 18px;border:1px solid var(--border);border-radius:24px;font-size:14px;outline:none;transition:all .2s;background:var(--card)}
.chat-input input:focus{border-color:var(--green);box-shadow:0 0 0 3px rgba(45,140,74,.08)}
.chat-input button{padding:12px 28px;background:var(--green);color:#fff;border:none;border-radius:24px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;letter-spacing:.5px}
.chat-input button:hover{background:var(--green-dark);transform:translateY(-1px);box-shadow:0 4px 12px rgba(45,140,74,.3)}
.chat-input button:disabled{opacity:.5;cursor:not-allowed;transform:none}
/* ---- responsive ---- */
@media(max-width:768px){
  main{flex-direction:column}
  aside{width:100%;max-height:50vh}
  section.chat{min-height:50vh}
}
</style>
</head>
<body>
<header>
  <h1>豆瓣Top250 电影分析系统</h1>
  <span class="status">250部 | 均分 {{s.score_mean}} | {{s.year_min}}-{{s.year_max}}</span>
</header>
<main>
<aside>
  <div class="section"><h2><span class="dot"></span>数据统计查询</h2></div>
  <div class="btn-row">
    <button onclick="showStats('country')">国家/地区</button>
    <button onclick="showStats('genre')">类型分析</button>
    <button onclick="showStats('director')">导演排行</button>
    <button onclick="showStats('actor')">演员排行</button>
    <button class="report" onclick="genReport()">生成 AI 分析报告</button>
  </div>
  <div class="rec-panel">
    <label>TF-IDF 电影推荐</label>
    <input id="recInput" placeholder="输入电影名，如: 肖申克的救赎" onkeydown="if(event.key==='Enter')doRecommend()" autocomplete="off">
    <div class="hint">基于剧情简介的文本相似度 — 找到内容最接近的电影</div>
  </div>
  <div class="result-area" id="result">
    <div class="empty">
      <span class="icon"></span>
      点击上方按钮查看数据统计<br>或在推荐框中输入电影名获取推荐
    </div>
  </div>
</aside>
<section class="chat">
  <div class="chat-header"><h2>AI 智能问答</h2></div>
  <div class="quick-row">
    <button onclick="quickAsk('推荐和肖申克的救赎类似的电影')">推荐类似电影</button>
    <button onclick="quickAsk('1990年代最好的10部电影')">1990年代TOP10</button>
    <button onclick="quickAsk('日本动画为什么评分高')">日本动画分析</button>
    <button onclick="quickAsk('评分最高的导演是谁')">最佳导演</button>
  </div>
  <div class="messages" id="messages">
    <div class="msg ai">你好！我是电影数据分析助手。<br><br>你可以自由提问，我会基于250部电影的详细数据为你解答。<br><br>左侧的 <b>TF-IDF 推荐</b> 可以帮你发现相似的好电影——试试输入"肖申克的救赎"看看。</div>
  </div>
  <div class="chat-input">
    <input id="question" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter')ask()" autocomplete="off">
    <button id="sendBtn" onclick="ask()">发送</button>
  </div>
</section>
</main>
<script>
let history=[];
let statsData=null;
fetch('/api/stats').then(r=>r.json()).then(d=>{statsData=d});

function showStats(type){
  if(!statsData){setTimeout(()=>showStats(type),200);return}
  let d=statsData,a=d[type+'_top'],b=d[type+'_score'];
  let title={'country':'国家/地区','genre':'类型','director':'导演','actor':'演员'}[type];
  let h='<div class="stats-table">';
  h+='<div class="section-title">'+title+'出现次数 Top10</div>';
  a.forEach((x,i)=>{h+=`  <b>${i+1}.</b>  ${x.name.padEnd(22)} <b>${x.count}</b> 部<br>`});
  if(b){h+='<div class="section-title">'+title+'平均评分 Top10</div>';
  b.forEach((x,i)=>{h+=`  <b>${i+1}.</b>  <span style="color:#e09040;font-weight:700">${x.mean.toFixed(2)}</span> 分  ${x.name}  (${x.count}部)<br>`});}
  h+='</div>';
  document.getElementById('result').innerHTML=h;
}

function doRecommend(){
  let q=document.getElementById('recInput').value.trim();
  if(!q){return}
  document.getElementById('result').innerHTML='<div class="empty">正在分析剧情简介，计算文本相似度...</div>';
  fetch('/api/recommend_ai',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:q})})
  .then(r=>r.json()).then(d=>{
    if(!d.ok){document.getElementById('result').innerHTML='<div class="empty">错误: '+d.error+'</div>';return}
    let h=`<div style="font-size:13px;color:#888;margin-bottom:10px">输入: <b style="color:#333">${d.query_title}</b> &nbsp;|&nbsp; ${d.query_summary.slice(0,80)}...</div>`;
    d.results.forEach((r,i)=>{
      let pct=(r.similarity*100).toFixed(0);
      let cls=pct>15?'#2d8c4a':pct>8?'#e09040':'#999';
      h+=`<div class="rec-card">
        <div class="rec-header">
          <div><span class="rec-rank">${i+1}</span><span class="rec-title">${r.title}</span></div>
          <span class="rec-score">${r.score}分</span>
        </div>
        <div class="rec-meta">${r.year}年 &nbsp;|&nbsp; ${r.genre} &nbsp;|&nbsp; 导演: ${r.director} &nbsp;|&nbsp; <span style="color:${cls};font-weight:600">相似度 ${pct}%</span></div>
        <div class="rec-summary">${r.summary}</div>
      </div>`;
    });
    if(d.explanation)h+='<div class="ai-note">'+d.explanation.replace(/\n/g,'<br>')+'</div>';
    document.getElementById('result').innerHTML=h;
  });
}

function addMsg(role,text){
  let d=document.createElement('div');
  d.className='msg '+role;d.innerHTML=text;
  document.getElementById('messages').appendChild(d);
  document.getElementById('messages').scrollTop=document.getElementById('messages').scrollHeight;
  return d;
}

function ask(){
  let inp=document.getElementById('question'),btn=document.getElementById('sendBtn');
  let q=inp.value.trim();if(!q)return;
  inp.value='';inp.disabled=btn.disabled=true;
  addMsg('user',q);
  let think=addMsg('thinking','AI 思考中');
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,history:history})})
  .then(r=>r.json()).then(d=>{
    think.remove();
    if(d.ok){addMsg('ai',d.answer.replace(/\n/g,'<br>'));history.push({role:'user',content:q});history.push({role:'assistant',content:d.answer})}
    else addMsg('ai','[错误] '+d.error);
    inp.disabled=btn.disabled=false;inp.focus();
  });
}

function quickAsk(q){document.getElementById('question').value=q;ask()}

function genReport(){
  document.getElementById('result').innerHTML='<div class="empty">正在调用 DeepSeek 生成完整分析报告...<br><small>预计需要 15-30 秒</small></div>';
  fetch('/api/report',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok)document.getElementById('result').innerHTML='<div style="font-size:13px;line-height:1.8"><b style="color:#2d8c4a">报告已生成!</b><br><br>文件: <code>report/ai_analysis_report.md</code><br><br>'+d.preview.replace(/\n/g,'<br>')+'</div>';
    else document.getElementById('result').innerHTML='<div class="empty">生成失败: '+d.error+'</div>';
  });
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    s = compute_stats()
    return render_template_string(INDEX, s=s)


if __name__ == "__main__":
    print(f"启动服务: http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
