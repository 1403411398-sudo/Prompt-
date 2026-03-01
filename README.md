# 智能Prompt工程平台 (Intelligent Prompt Engineering Platform)

PromptForge 是一套面向大模型应用的多任务 Prompt 自动优化系统，围绕“生成—评估—迭代—优化”构建闭环框架。系统支持自动生成 Prompt（关键词组合、模板扩展、语义增强），并调用主流大模型完成文本分类、摘要生成与中英翻译任务。系统级联 Qwen3.5、DeepSeek-V3 及 Kimi-K2 等前沿大模型，结合 Accuracy、ROUGE、BLEU 等指标进行量化评估。内置随机搜索、遗传算法与贝叶斯优化三种策略，自动搜索最优提示词，同时输出最优 Prompt、得分曲线与搜索过程可视化结果，满足任务对自动优化、多指标评估与过程展示的要求。

## 项目概述

随着大语言模型（LLM）的飞速发展，Prompt工程已成为发挥模型能力的关键。然而，针对不同NLP任务手动设计高质量Prompt并客观评估，是一个复杂、耗时且高度依赖经验的过程。本项目通过构建一个智能Prompt工程平台，实现了Prompt自动生成、迭代优化和效果评估的功能。

## 项目功能

- **Prompt 智能生成**: 用户输入任务描述后，系统自动生成多种候选Prompt方案，快速启动任务。
- **Prompt 迭代优化**: 基于模型输出和评估反馈，自动对现有Prompt进行迭代优化，持续提升效果。
- **多维度效果评估**: 针对不同任务类型，自动计算并展示准确率、ROUGE分数等关键指标，量化模型性能。
- **多任务场景支持**: 支持文本分类、文本摘要、机器翻译等多种NLP任务，满足多样化的业务需求。
- **直观友好的界面**: 提供直观、易用的Web界面，用户可以轻松配置任务、查看结果并进行对比分析。

## 技术栈

- **前端**: React.js, JavaScript, HTML5, CSS3
- **后端**: Python, Flask, Pandas
- **AI模型与NLP**: OpenAI API, Transformers
- **部署与协作**: Docker, Git

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/intelligent-prompt-platform.git
cd intelligent-prompt-platform

2. 安装依赖

确保你已经安装了 Python 和 Node.js，接下来安装项目的依赖：

前端依赖:

cd frontend
npm install

后端依赖:

cd backend
pip install -r requirements.txt
3. 启动项目

启动前端:

cd frontend
npm start

启动后端:

cd backend
python app.py

访问前端页面：http://localhost:3000

项目结构
intelligent-prompt-platform/
│
├── frontend/                # 前端代码
│   ├── index.html           # 主页面
│   ├── app.js               # 主JS文件
│   └── style.css            # 样式文件
│
├── backend/                 # 后端代码
│   ├── app.py               # 后端入口文件
│   ├── evaluator.py         # 评估引擎
│   └── requirements.txt     # Python依赖包
│
└── README.md                # 项目说明文档
