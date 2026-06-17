"""
豆瓣Top250电影分析系统 —— Web版
Flask + 内嵌HTML，无需额外前端文件
"""
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import requests
import os
import re
import threading
import sys

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_KEY = "sk-ee6a047554b24ba7923b7e8b35d76e3d"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# ---- 数据 ----

def load_data():
    for name in ["movies_cleaned.csv", "movies_detail.csv", "movies.csv"]:
        path = os.path.join(BASE_DIR, "data", name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df = load_data()

def explode_col(series, sep):
    return series.dropna().str.split(sep).explode()

def compute_stats():
    ec = explode_col(df["country"], " / ")
    eg = explode_col(df["genre"], " ")
    ed = explode_col(df["director"], " / ")
    ed = ed[ed != "未知"]
    ea = explode_col(df["actors"], " / ")
    ea = ea[ea != "未知"]

    cs = pd.DataFrame({"val": ec, "score": df.loc[ec.index, "score"]})
    gs = pd.DataFrame({"val": eg, "score": df.loc[eg.index, "score"]})
    ds = pd.DataFrame({"val": ed, "score": df.loc[ed.index, "score"]})
    ag = pd.DataFrame({"val": ea, "score": df.loc[ea.index, "score"]})

    def agg(gdf, min_count):
        g = gdf.groupby("val")["score"].agg(["mean", "count"])
        g = g[g["count"] >= min_count].sort_values("mean", ascending=False)
        return g

    return {
        "total": len(df),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "score_mean": round(float(df["score"].mean()), 2),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "country_top": _format_top(ec.value_counts().head(10), "部"),
        "country_score": _format_score(agg(cs, 3)),
        "genre_top": _format_top(eg.value_counts().head(10), "部"),
        "genre_score": _format_score(agg(gs, 5)),
        "director_top": _format_top(ed.value_counts().head(10), "部"),
        "director_score": _format_score(agg(ds, 2)),
        "actor_top": _format_top(ea.value_counts().head(10), "部"),
        "actor_score": _format_score(agg(ag, 2)),
    }

def _format_top(series, unit):
    return [{"name": k, "count": int(v)} for k, v in series.items()]

def _format_score(grouped):
    return [{"name": k, "mean": round(float(row["mean"]), 2), "count": int(row["count"])}
            for k, row in grouped.head(10).iterrows()]

def build_data_ctx():
    s = compute_stats()
    ctx = f"豆瓣Top250：{s['total']}部，评分{s['score_min']}~{s['score_max']}，均分{s['score_mean']}。"
    ctx += f"年代{s['year_min']}~{s['year_max']}。"
    ctx += "国家Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["country_top"][:5]) + "。"
    ctx += "类型Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["genre_top"][:5]) + "。"
    ctx += "导演Top5: " + ", ".join(f"{c['name']}({c['count']})" for c in s["director_top"][:5]) + "。"
    ctx += f"高分: " + ", ".join(f"{r['title']}({r['score']})" for _, r in df.nlargest(5, 'score').iterrows()) + "。"
    return ctx

# ---- API ----

@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats())

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    history = data.get("history", [])

    messages = [
        {"role": "system", "content": f"你是电影数据分析助手，回答简洁专业。\n{build_data_ctx()}"}
    ]
    # 只保留最近6条历史
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

    def _call():
        return call_deepseek(prompt)

    content = _call()
    if content:
        save_report(content, df_r, summary)
        return jsonify({"ok": True, "preview": content[:3000]})
    return jsonify({"ok": False, "error": "生成失败"})

# ---- 页面 ----

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
aside{width:380px;background:#fff;border-right:1px solid #e0d8cf;display:flex;flex-direction:column;flex-shrink:0}
aside .header{padding:18px 20px 10px;font-size:15px;font-weight:bold;color:#333}
aside .btns{padding:0 16px;display:flex;flex-wrap:wrap;gap:8px}
aside .btns button{flex:1;min-width:calc(50% - 4px);padding:10px 6px;border:1px solid #e0d8cf;border-radius:6px;background:#fafaf7;cursor:pointer;font-size:13px;color:#444;transition:all .15s}
aside .btns button:hover{background:#2d8c4a;color:#fff;border-color:#2d8c4a}
aside .btns button.report{min-width:100%;background:#e09040;color:#fff;border-color:#e09040}
aside .btns button.report:hover{background:#c87830}
aside .result{flex:1;margin:12px 16px 16px;background:#fafaf7;border-radius:8px;padding:16px;overflow:auto;font-family:Consolas,monospace;font-size:13px;line-height:1.7;white-space:pre-wrap}
/* ---- 右侧 ---- */
section.chat{flex:1;background:#fff;margin:0;display:flex;flex-direction:column;border-left:1px solid #e0d8cf}
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
    <div class="header">数据查询</div>
    <div class="btns">
      <button onclick="showStats('country')">国家/地区分析</button>
      <button onclick="showStats('genre')">类型分析</button>
      <button onclick="showStats('director')">导演分析</button>
      <button onclick="showStats('actor')">演员分析</button>
      <button class="report" onclick="genReport()">生成 AI 分析报告</button>
    </div>
    <div class="result" id="result">点击上方按钮查看数据统计</div>
  </aside>
  <section class="chat">
    <div class="header">AI 智能问答</div>
    <div class="quick">
      <button onclick="quickAsk('推荐5部高分悬疑片')">推荐5部悬疑片</button>
      <button onclick="quickAsk('1990年代最好的10部电影')">1990年代TOP10</button>
      <button onclick="quickAsk('日本动画为什么评分高')">日本动画分析</button>
      <button onclick="quickAsk('评分最高的导演是谁')">最佳导演</button>
    </div>
    <div class="messages" id="messages">
      <div class="msg ai">你好！我是电影数据分析助手，可以问我任何关于这250部电影的问题。</div>
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

function addMsg(role,text){
  let d=document.createElement('div');
  d.className='msg '+role;d.textContent=text;
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
    if(d.ok){addMsg('ai',d.answer);history.push({role:'user',content:q});history.push({role:'assistant',content:d.answer})}
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
