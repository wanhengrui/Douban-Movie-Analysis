# 豆瓣Top250电影数据采集、分析与推荐系统

> 本科课程项目 | 2026

基于豆瓣Top250电影数据，实现**数据爬取 → 清洗 → 分析 → 推荐 → AI解读**的完整数据流水线。

---

## 功能特性

| 模块 | 说明 | 技术栈 |
|------|------|--------|
| 数据爬取 | Top250列表页 + 手机详情页爬虫 | Requests + BeautifulSoup |
| 数据清洗 | 解析错误修复、字段标准化、缺失值处理 | Pandas + Regex |
| 数据分析 | 国家/类型/导演/演员/年代多维度统计 | Pandas + NumPy |
| 电影推荐 | 基于剧情简介的TF-IDF + 余弦相似度 | Scikit-learn |
| AI 解读 | DeepSeek大模型自动分析数据并解释推荐 | DeepSeek API |
| Web 界面 | 数据查询 + AI问答 + 电影推荐的统一入口 | Flask + HTML/CSS/JS |

## 项目结构

```
Douban-Movie-Analysis/
├── web_app.py               # Web主程序 (Flask)
├── spider/                  # 爬虫模块
│   ├── douban_spider.py     #   Top250列表页爬虫
│   ├── detail_spider.py     #   详情页爬虫 (导演/演员/片长/简介)
│   └── fix_data.py          #   缺失数据补爬脚本
├── analysis/                # 分析模块
│   ├── data_cleaning.py     #   数据清洗 (修复错位/标准化/缺失值)
│   ├── ai_analysis.py       #   AI自动生成分析报告
│   └── movie_analysis.py    #   独立图表生成
├── data/                    # 数据文件
│   ├── movies.csv           #   基础数据 (5字段)
│   ├── movies_detail.csv    #   详情数据 (11字段)
│   └── movies_cleaned.csv   #   清洗后数据
├── images/                  # 可视化图表
│   ├── score_distribution.png
│   ├── country_distribution.png
│   └── country_score.png
├── report/
│   └── ai_analysis_report.md  # AI生成的分析报告
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行爬虫（如需重新采集数据）

```bash
cd spider

# 第一步: 爬取Top250列表页
python douban_spider.py        # 生成 data/movies.csv

# 第二步: 爬取详情页
python detail_spider.py        # 生成 data/movies_detail.csv

# 第三步: 修复缺失数据
python fix_data.py              # 修复 runtime/summary 缺失项
cd ..
```

### 3. 数据清洗

```bash
cd analysis
python data_cleaning.py        # 生成 data/movies_cleaned.csv
cd ..
```

### 4. 启动 Web 系统

```bash
python web_app.py
```

浏览器打开 http://127.0.0.1:5000

## 数据集

- **数据源**: 豆瓣电影 Top250
- **记录数**: 250部电影
- **字段**: 电影名称、评分、年份、国家/地区、类型、导演、演员、片长、剧情简介
- **评分范围**: 8.4 ~ 9.7
- **年代跨度**: 1931 ~ 2023

## 推荐系统原理

```
用户输入电影名
    │
    ▼
查找电影 → 获取剧情简介
    │
    ▼
TF-IDF 向量化 (250篇简介 → 250个向量)
    │
    ▼
余弦相似度计算 (输入电影 vs 所有电影)
    │
    ▼
Top5 最相似电影 ← AI 解释推荐理由
```

**TF-IDF (词频-逆文档频率)**: 评估一个词对一部电影简介的重要程度。词频(TF)越高且在其他电影中出现越少(IDF)，则该词对该电影的区分力越强。

**余弦相似度**: 将两部电影的TF-IDF向量之间的夹角余弦值作为相似度。值越接近1表示两部电影的剧情内容越相似。

## AI 集成

系统通过 DeepSeek API (deepseek-v4-flash) 实现两项AI功能：

1. **AI 问答**: 用户可自由提问关于数据集的问题，AI根据数据统计和电影详情回答
2. **推荐解释**: TF-IDF找到相似电影后，AI基于剧情简介解释为什么这些电影相似
3. **报告生成**: 自动计算6个维度的统计摘要，交由AI生成专业分析报告

## 答辩要点

1. **数据采集**: 双策略爬取(列表页+手机页)，解决反爬问题
2. **数据处理**: 完整的ETL流水线，18条手动补全(7.2%边缘情况)
3. **推荐算法**: TF-IDF + 余弦相似度，基于真实爬取的剧情文本
4. **AI 集成**: 大模型作为分析层，提升系统智能化程度
5. **工程化**: 模块化设计，Web界面，CSV数据持久化

## 已知限制

- 演员列表仅包含前1~3位主演（列表页截断）
- 部分电影剧情简介缺失（3/250）
- 豆瓣偶有反爬措施，需添加Cookie访问
- 推荐仅基于剧情文本，未纳入导演/演员/类型等结构化信息

## License

MIT
