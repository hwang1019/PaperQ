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

## Commands

| Command | Description |
|---------|-------------|
| `analyze_paper()` | Analyze a PDF (paste path interactively) |
| `new_project('name')` | Create a new project |
| `switch_project('name')` | Switch to an existing project |
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
