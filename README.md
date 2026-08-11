# PaperQ — AI Research Agent for Condensed Matter Physics

A Jupyter notebook agent that uses the DeepSeek API to analyze academic papers, answer physics questions, and generate MATLAB analysis code. Designed for low-temperature electron transport research in low-dimensional semiconductor devices.

## Features

- **Paper Analysis** — Upload a PDF and get a structured literature review covering material system, device structure, transport phenomena, numerical results, and fitting models. Bibliographic metadata (title, authors, year, DOI) is extracted automatically from the first page.

- **Multi-Project Chat** — Organize conversations into named projects. Switch between research topics without losing context. All history is persisted to disk and can be reloaded.

- **Markdown Display** — Full conversation history and paper analyses render as formatted Markdown with proper LaTeX equation rendering in Jupyter.

- **Paper Database** — All analyzed papers are stored in a searchable database. Search by material, phenomenon, year range, or author. Recall papers into the chat context for follow-up questions.

- **MATLAB Code Generation** — The agent generates standalone `.m` files rather than inline code blocks. Files are auto-saved to `matlab_output/` with automatic naming.

- **BibTeX Export** — Export any paper in the database as a BibTeX citation string.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
   ```bash
   set DEEPSEEK_API_KEY=your-key-here
   ```

3. **Open the notebook** in VS Code or Jupyter and run the first cell.

4. **Analyze a paper:**
   ```python
   analyze_paper()  # prompts you to paste the file path
   ```

5. **Ask a question:**
   ```python
   ask("What limits mobility in modulation-doped GaAs quantum wells?")
   ```

> **Project structure:** All agent logic lives in [`paperq.py`](paperq.py); `paper_q.ipynb` is a thin driver that imports it (`from paperq import *`, then `paperq.startup()`). The command names are unchanged.

## Commands

| Command | Description |
|---------|-------------|
| `analyze_paper()` | Analyze a PDF (paste path interactively) |
| `new_project('name')` | Create a new project |
| `commands('keyword')` | List commands with descriptions (keyword optional) |
| `switch_project('name')` | Switch to a project (partial names auto-complete) |
| `ask('question')` | Ask a physics question |
| `ask_matlab('question')` | Request MATLAB code |
| `ask_python('question')` | Request Python code |
| `list_papers()` | List all papers in database |
| `search_papers(...)` | Search by material, phenomenon, year, author |
| `recall_paper(n)` | Load a paper into chat context |
| `display_paper(n)` | Display paper analysis as Markdown |
| `display_history()` | Show conversation as Markdown |
| `show_history()` | Show conversation as plain text |
| `export_bibtex(n)` | Export paper as BibTeX |
| `forget_paper()` | Remove paper context from chat |
| `delete_paper(n)` | Delete a paper from database |
| `update_paper(n, field, val)` | Update paper metadata |

## System Prompt

The agent is configured as a condensed matter physicist specializing in:
- Weak localization / antilocalization
- Universal conductance fluctuations
- Shubnikov–de Haas oscillations & quantum Hall effect
- Coulomb blockade
- Landauer–Büttiker formalism
- Non-equilibrium Green's functions

By default it generates MATLAB code. Request Python explicitly for Kwant simulations or ML tasks.

## License

MIT — see [LICENSE](LICENSE).

---

# PaperQ — 凝聚态物理 AI 研究助手

一个基于 Jupyter Notebook 的 AI 助手，使用 DeepSeek API 分析学术论文、回答物理问题并生成 MATLAB 分析代码。专为低维半导体器件中的低温电子输运研究而设计。

## 功能特性

- **论文分析** — 上传 PDF 即可获得结构化文献综述，涵盖材料体系、器件结构、输运现象、数值结果和拟合模型。书目元数据（标题、作者、年份、DOI）自动从首页提取。

- **多项目对话** — 将对话组织为命名项目。在研究主题之间自由切换，上下文不丢失。所有历史记录持久化保存并可重新加载。

- **Markdown 显示** — 完整的对话历史和论文分析以格式化 Markdown 渲染，支持 LaTeX 公式在 Jupyter 中正确显示。

- **论文数据库** — 所有分析的论文存储在可搜索数据库中。支持按材料、现象、年份范围或作者进行检索。可将论文召回至对话上下文中进行追问。

- **MATLAB 代码生成** — 智能体生成独立的 `.m` 文件而非内联代码块。文件自动保存至 `matlab_output/` 并自动命名。

- **BibTeX 导出** — 将数据库中的任意论文导出为 BibTeX 引用字符串。

## 快速开始

1. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

2. **设置 API 密钥：**
   ```bash
   set DEEPSEEK_API_KEY=your-key-here
   ```

3. **打开 Notebook**，在 VS Code 或 Jupyter 中运行第一个单元格。

4. **分析论文：**
   ```python
   analyze_paper()  # 提示你粘贴文件路径
   ```

5. **提出问题：**
   ```python
   ask("调制掺杂 GaAs 量子阱中迁移率的限制因素是什么？")
   ```

> **项目结构：** 所有智能体逻辑位于 [`paperq.py`](paperq.py)；`paper_q.ipynb` 是精简的驱动程序，导入该模块（`from paperq import *`，然后 `paperq.startup()`）。命令用法不变。

## 命令参考

| 命令 | 描述 |
|---------|-------------|
| `analyze_paper()` | 分析 PDF（交互式粘贴路径） |
| `new_project('name')` | 创建新项目 |
| `commands('keyword')` | 列出命令及说明（可按关键词筛选） |
| `switch_project('name')` | 切换到已有项目（支持部分名称自动补全） |
| `ask('question')` | 提问物理问题 |
| `ask_matlab('question')` | 请求 MATLAB 代码 |
| `ask_python('question')` | 请求 Python 代码 |
| `list_papers()` | 列出数据库中所有论文 |
| `search_papers(...)` | 按材料、现象、年份、作者检索 |
| `recall_paper(n)` | 将论文加载到对话上下文 |
| `display_paper(n)` | 以 Markdown 显示论文分析 |
| `display_history()` | 以 Markdown 显示对话历史 |
| `show_history()` | 以纯文本显示对话历史 |
| `export_bibtex(n)` | 导出论文为 BibTeX |
| `forget_paper()` | 从对话中移除论文上下文 |
| `delete_paper(n)` | 从数据库中删除论文 |
| `update_paper(n, field, val)` | 更新论文元数据 |

## 系统提示词

该助手被配置为凝聚态物理学家，专长领域包括：
- 弱局域化 / 反弱局域化
- 普适电导涨落
- Shubnikov–de Haas 振荡与量子霍尔效应
- 库仑阻塞
- Landauer–Büttiker 理论
- 非平衡格林函数

默认生成 MATLAB 代码。如需进行 Kwant 模拟或机器学习任务，请明确请求 Python 代码。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
