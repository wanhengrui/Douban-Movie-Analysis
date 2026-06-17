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

INDEX = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>豆瓣Top250 电影分析系统</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f0eb;color:#333;height:100vh;display:flex;flex-direction:column}
header{background:#2d8c4a;color:#fff;padding:14px 24px;font-size:18px;font-weight:bold;display:flex;align-items:center;justify-content:space-between}
header span.status{font-size:13px;font-weight:normal;opacity:.85}
main{flex:1;display:flex;overflow:hidden}
/* ---- 左侧 ---- */
aside{width:400px;background:#fff;border-right:1px solid #e0d8cf;display:flex;flex-direction:column;flex-shrink:0}
aside .section-title{padding:18px 20px 10px;font-size:15px;font-weight:bold;color:#333}
aside .btns{padding:0 16px;display:flex;flex-wrap:wrap;gap:8px}
aside .btns button{flex:1;min-width:calc(50% - 4px);padding:10px 6px;border:1px solid #e0d8cf;border-radius:6px;background:#fafaf7;cursor:pointer;font-size:13px;color:#444;transition:all .15s}
aside .btns button:hover{background:#2d8c4a;color:#fff;border-color:#2d8c4a}
aside .btns button.accent{min-width:100%;background:#e09040;color:#fff;border-color:#e09040}
aside .btns button.accent:hover{background:#c87830}
aside .rec-box{margin:12px 16px;padding:12px;background:#f0f7f2;border-radius:8px;border:1px solid #d0e8d4}
aside .rec-box .title{font-size:13px;font-weight:bold;color:#2d8c4a;margin-bottom:6px}
aside .rec-box input{width:100%;padding:8px 12px;border:1px solid #d0e8d4;border-radius:6px;font-size:13px;outline:none}
aside .rec-box input:focus{border-color:#2d8c4a}
aside .rec-box .hint{font-size:11px;color:#999;margin-top:4px}
aside .result{flex:1;margin:0 16px 16px;background:#fafaf7;border-radius:8px;padding:16px;overflow:auto;font-family:Consolas,monospace;font-size:13px;line-height:1.7;white-space:pre-wrap}
/* ---- 右侧 ---- */
section.chat{flex:1;background:#fff;display:flex;flex-direction:column;border-left:1px solid #e0d8cf}
section.chat .header{padding:18px 24px 10px;font-size:15px;font-weight:bold;color:#333;border-bottom:1px solid #f0e8dc}
section.chat .quick{padding:10px 20px;display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid #f5f0eb;background:#fafaf7}
section.chat .quick button{padding:6px 14px;border:1px solid #e0d8cf;border-radius:14px;background:#fff;cursor:pointer;font-size:12px;color:#666;transition:all .15s}
section.chat .quick button:hover{background:#2d8c4a;color:#fff;border-color:#2d8c4a}
section.chat .messages{flex:1;padding:16px 24px;overflow:auto;display:flex;flex-direction:column;gap:12px}
section.chat .messages .msg{max-width:85%;padding:10px 14px;border-radius:10px;font-size:13px;line-height:1.6;animation:fadeIn .3s}
section.chat .messages .msg.user{align-self:flex-end;background:#1a6fb5;color:#fff;border-bottom-right-radius:3px}
section.chat .messages .msg.ai{align-self:flex-start;background:#f0f0f0;color:#333;border-bottom-left-radius:3px}
section.chat .messages .msg.thinking{align-self:flex-start;color:#aaa;font-size:12px;padding:0}
section.chat .input-box{padding:12px 20px 16px;display:flex;gap:10px;border-top:1px solid #f0e8dc;background:#fafaf7}
section.chat .input-box input{flex:1;padding:10px 16px;border:1px solid #e0d8cf;border-radius:20px;font-size:14px;outline:none;transition:border .2s}
section.chat .input-box input:focus{border-color:#2d8c4a}
section.chat .input-box button{padding:10px 24px;background:#2d8c4a;color:#fff;border:none;border-radius:20px;font-size:14px;cursor:pointer;transition:background .15s}
section.chat .input-box button:hover{background:#23703a}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<header>
  豆瓣Top250 电影分析系统
  <span class="status">已加载 {{s.total}} 部电影 | 评分 {{s.score_min}}~{{s.score_max}} | 均分 {{s.score_mean}}</span>
</header>
<main>
  <aside>
    <div class="section-title">数据查询</div>
    <div class="btns">
      <button onclick="showStats('country')">国家/地区</button>
      <button onclick="showStats('genre')">类型分析</button>
      <button onclick="showStats('director')">导演分析</button>
      <button onclick="showStats('actor')">演员分析</button>
      <button class="accent" onclick="genReport()">生成 AI 分析报告</button>
    </div>
    <div class="rec-box">
      <div class="title">🎬 TF-IDF 电影推荐</div>
      <input id="recInput" placeholder="输入电影名，如: 肖申克的救赎" onkeydown="if(event.key==='Enter')doRecommend()">
      <div class="hint">基于剧情简介的文本相似度算法，找到内容最接近的电影</div>
    </div>
    <div class="result" id="result">点击按钮查看数据统计
或在推荐框中输入电影名</div>
  </aside>
  <section class="chat">
    <div class="header">AI 智能问答</div>
    <div class="quick">
      <button onclick="quickAsk('推荐和肖申克的救赎类似的电影')">推荐类似电影</button>
      <button onclick="quickAsk('1990年代最好的10部电影')">1990年代TOP10</button>
      <button onclick="quickAsk('日本动画为什么评分高')">日本动画分析</button>
      <button onclick="quickAsk('评分最高的导演是谁')">最佳导演</button>
    </div>
    <div class="messages" id="messages">
      <div class="msg ai">你好！我是电影数据分析助手。<br><br>左侧可以查看各维度数据统计，也可以输入电影名获得 <b>TF-IDF 智能推荐</b>。<br><br>有问题随时问我！</div>
    </div>
    <div class="input-box">
      <input id="question" placeholder="输入问题..." onkeydown="if(event.key==='Enter')ask()">
      <button onclick="ask()">发送</button>
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
  let h=`▎${title}出现次数 Top10\\n`+a.map((x,i)=>`  ${(i+1+'').padStart(2)}.  ${x.name.padEnd(22)} ${(x.count+'').padStart(3)} 部`).join('\\n');
  if(b)h+=`\\n\\n▎${title}平均评分 Top10\\n`+b.map((x,i)=>`  ${(i+1+'').padStart(2)}.  ${x.mean.toFixed(2)} 分  ${x.name}  (${x.count}部)`).join('\\n');
  document.getElementById('result').textContent=h;
}

function doRecommend(){
  let q=document.getElementById('recInput').value.trim();
  if(!q)return;
  document.getElementById('result').textContent='正在分析剧情简介...';
  fetch('/api/recommend_ai',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:q})})
  .then(r=>r.json()).then(d=>{
    if(!d.ok){document.getElementById('result').textContent='错误: '+d.error;return}
    let h=`■ 输入: ${d.query_title}\\n   剧情: ${d.query_summary.slice(0,150)}...\\n\\n`;
    h+=`■ TF-IDF 相似推荐 (基于剧情简介的余弦相似度)\\n\\n`;
    d.results.forEach((r,i)=>{
      h+=`${i+1}. ${r.title} (${r.score}分, ${r.year}年)\\n`;
      h+=`   相似度: ${(r.similarity*100).toFixed(1)}% | ${r.genre}\\n`;
      h+=`   导演: ${r.director}\\n`;
      h+=`   简介: ${r.summary.slice(0,120)}...\\n\\n`;
    });
    if(d.explanation)h+=`── AI 分析 ──\\n${d.explanation}`;
    document.getElementById('result').textContent=h;
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
  let inp=document.getElementById('question');
  let q=inp.value.trim();if(!q)return;
  inp.value='';inp.disabled=true;
  addMsg('user',q);
  let think=addMsg('thinking','AI 思考中...');
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,history:history})})
  .then(r=>r.json()).then(d=>{
    think.remove();
    if(d.ok){addMsg('ai',d.answer.replace(/\\n/g,'<br>'));history.push({role:'user',content:q});history.push({role:'assistant',content:d.answer})}
    else addMsg('ai','[错误] '+d.error);
    inp.disabled=false;inp.focus();
  });
}

function quickAsk(q){document.getElementById('question').value=q;ask()}

function genReport(){
  document.getElementById('result').textContent='正在生成AI分析报告，请稍候...';
  fetch('/api/report',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok)document.getElementById('result').textContent='报告已生成！\\n\\n文件: report/ai_analysis_report.md\\n\\n'+d.preview;
    else document.getElementById('result').textContent='生成失败: '+d.error;
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
