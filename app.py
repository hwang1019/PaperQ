"""PaperQ — Gradio GUI for the condensed-matter physics research agent.

A thin graphical front end over paperq.py. Run it with:

    python app.py

This starts a *local* web server on your own machine and opens a browser tab
at http://127.0.0.1:7860. Nothing is deployed or shared; closing the terminal
stops the app. All agent logic (PDF analysis, manuscript review, the paper
databases, MATLAB file generation) still lives in paperq.py — this file only
wraps those functions with a graphical interface.
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr

# Anchor the working directory to this file's location. paperq.py reads its
# databases and histories with relative paths (paper_database.json,
# notebook_histories/, matlab_output/), so launching `python app.py` from any
# directory other than the repo root would silently load an empty database.
# Switching here makes the app work no matter where it is launched from.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR and os.path.isdir(_BASE_DIR):
    os.chdir(_BASE_DIR)


# ============================================================
# Startup / API-key handling
# ============================================================

def _has_api_key() -> bool:
    """Return True if a DeepSeek key (or an OpenAI key) is available."""
    return bool(
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


# Importing paperq constructs the OpenAI client immediately, which raises
# without an API key. Guard the import so the app still launches and can show
# a friendly "set your key" message instead of a traceback.
PAPERQ_OK = False
IMPORT_ERROR = "DEEPSEEK_API_KEY is not set."

if _has_api_key():
    try:
        import paperq  # noqa: E402
        PAPERQ_OK = True
        IMPORT_ERROR = ""
    except Exception as exc:  # noqa: BLE001
        PAPERQ_OK = False
        IMPORT_ERROR = str(exc)


# ============================================================
# Internationalisation (bilingual UI)
# ============================================================

LANG = "en"  # current display language: "en" or "zh"

# Turns accumulated while "Don't save" is checked. They are included in the
# LLM context for a coherent conversation but are never written to any
# project file, and are dropped when switching/creating projects or clearing.
_session_unsaved = []

# KaTeX: Gradio only renders $$...$$ by default, which makes inline $...$
# show up as raw source. Add the inline delimiter explicitly.
LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

STRINGS = {
    "en": {
        "app_title": "PaperQ — Condensed-Matter Physics Research Agent",
        "app_subtitle": "Analyze papers, review manuscripts, and chat with an expert "
                        "tuned for low-temperature electron transport. MATLAB code is "
                        "generated and saved to `matlab_output/` automatically.",
        "chatbot_label": "Conversation",
        "question_label": "Question",
        "question_placeholder": "e.g. What limits mobility in modulation-doped GaAs quantum wells?",
        "model_label": "Model",
        "model_info": "pro = best quality · flash = faster / cheaper",
        "send": "Send",
        "clear_history": "Clear history",
        "paper_tab_desc": "Upload a published PDF for a structured literature review.",
        "paper_file_label": "Paper PDF",
        "analyze_paper": "Analyze paper",
        "ms_tab_desc": "Upload an unpublished draft for an honest, constructive review.",
        "ms_file_label": "Manuscript PDF",
        "review_manuscript": "Review manuscript",
        "db_tab_desc": "Browse analyzed papers and reviewed manuscripts: view, load into "
                       "chat, export BibTeX, delete, or re-analyze / re-review.",
        "refresh": "🔄 Refresh lists",
        "papers_heading": "### Papers",
        "manuscripts_heading": "### Manuscripts",
        "paper_dd_label": "Papers",
        "ms_dd_label": "Manuscripts",
        "view_paper": "View paper",
        "view_manuscript": "View manuscript",
        "load_chat": "Load into chat",
        "export_bibtex": "Export BibTeX",
        "delete": "Delete",
        "reanalyze": "Re-analyze",
        "rereview": "Re-review",
        "reanalyze_file_label": "PDF for re-analysis",
        "rereview_file_label": "PDF for re-review",
        "projects_tab_desc": "Organize conversations into named projects. Selecting a project "
                             "from the list switches to it instantly — chat history is saved "
                             "automatically and reloaded on switch.",
        "project_dd_label": "Project",
        "switch_project": "Switch to project",
        "switch_on_select_info": "The project switches as soon as you select it.",
        "create_project": "Create project",
        "new_project_label": "New project name",
        "new_project_placeholder": "e.g. quantum_hall",
        "current_project": "Current project",
        "refresh_projects": "🔄 Refresh",
        "settings_tab_desc": "Choose the display language. This changes the interface "
                             "only — your data and analyses are unaffected.",
        "lang_label": "Language",
        "lang_info": "Select English or Chinese.",
        "footer": "All analysis results are stored in `paper_database.json` and "
                  "`manuscript_database.json`; generated MATLAB scripts land in "
                  "`matlab_output/`. These persist across sessions.",
        # Dynamic strings
        "no_papers": "(no papers yet)",
        "no_manuscripts": "(no manuscripts yet)",
        "upload_first": "**⚠️ Please upload a PDF first.**",
        "already_analyzed": "**Already analyzed** as record #{idx}. Use the *Database* tab "
                            "to load it into the chat.",
        "already_reviewed": "**Already reviewed** as record #{idx}. Use the *Database* tab "
                            "to load it into the chat.",
        "analysis_failed": "**❌ Analysis failed.** Check that the PDF is valid and your "
                           "API key is correct.",
        "review_failed": "**❌ Review failed.** Check that the PDF is valid and your "
                         "API key is correct.",
        "saved_paper": "**✅ Saved to the database as record #{idx}.** Discuss it in the "
                       "*Chat* tab (or load it there first).",
        "saved_manuscript": "**✅ Saved to the database as record #{idx}.** Discuss it in the "
                            "*Chat* tab (or load it there first).",
        "no_such_record": "**⚠️ No such record.**",
        "select_paper_view": "**⚠️ Select a paper to view.**",
        "select_ms_view": "**⚠️ Select a manuscript to view.**",
        "select_paper_load": "**⚠️ Select a paper to load.**",
        "select_ms_load": "**⚠️ Select a manuscript to load.**",
        "loaded_paper_status": "**✅ Loaded record #{idx} into the Chat context.**",
        "loaded_ms_status": "**✅ Loaded record #{idx} into the Chat context.**",
        "chat_loaded_paper": "📄 Loaded paper *{title}* into the chat context.",
        "chat_loaded_ms": "📝 Loaded manuscript *{title}* into the chat context.",
        "toast_paper_title": "Paper loaded",
        "toast_ms_title": "Manuscript loaded",
        "toast_paper_msg": "Record #{idx} · {title} is now in the chat context.",
        "toast_ms_msg": "Record #{idx} · {title} is now in the chat context.",
        "deleted_paper": "Deleted paper '{title}'.",
        "deleted_ms": "Deleted manuscript '{title}'.",
        "toast_deleted_title": "Deleted",
        "reanalyzed_paper": "Re-analyzed paper '{title}'.",
        "rereviewed_ms": "Re-reviewed manuscript '{title}'.",
        "toast_reanalyzed_title": "Done",
        "select_project": "Select a project.",
        "enter_project_name": "Enter a project name.",
        "switched_project": "Switched to project '{name}'.",
        "created_project": "Created project '{name}'.",
        "toast_project_title": "Project",
        "error_prefix": "**⚠️ Error:**",
        "record": "Record",
        "full_analysis": "### Full analysis",
        "review_heading": "### Review",
        "no_analysis": "No analysis available.",
        "no_review": "No review available.",
        "field_col": "Field",
        "value_col": "Value",
        "loaded_context_paper": "📄 *Paper context loaded from the database.*",
        "loaded_context_ms": "📝 *Manuscript context loaded from the database.*",
        "dont_save": "Don't save to project",
        "dont_save_info": "When checked, this conversation is not written to the project.",
        "analyzing_elapsed": "⏱ **Analyzing…** time elapsed: `{elapsed}`",
        "reviewing_elapsed": "⏱ **Reviewing…** time elapsed: `{elapsed}`",
        "completed_in": "⏱ Completed in {elapsed}",
    },
    "zh": {
        "app_title": "PaperQ — 凝聚态物理研究助手",
        "app_subtitle": "分析论文、审阅手稿，并与专注于低温电子输运的专家对话。"
                        "MATLAB 代码会自动生成并保存到 `matlab_output/`。",
        "chatbot_label": "对话",
        "question_label": "问题",
        "question_placeholder": "例如：调制掺杂 GaAs 量子阱中迁移率的限制因素是什么？",
        "model_label": "模型",
        "model_info": "pro = 最佳质量 · flash = 更快 / 更便宜",
        "send": "发送",
        "clear_history": "清空历史",
        "paper_tab_desc": "上传已发表论文 PDF，进行结构化文献综述。",
        "paper_file_label": "论文 PDF",
        "analyze_paper": "分析论文",
        "ms_tab_desc": "上传未发表草稿，获得诚实、建设性的审阅。",
        "ms_file_label": "手稿 PDF",
        "review_manuscript": "审阅手稿",
        "db_tab_desc": "浏览已分析的论文和已审阅的手稿：查看、加载到对话、导出 BibTeX、"
                       "删除或重新分析 / 重新审阅。",
        "refresh": "🔄 刷新列表",
        "papers_heading": "### 论文",
        "manuscripts_heading": "### 手稿",
        "paper_dd_label": "论文",
        "ms_dd_label": "手稿",
        "view_paper": "查看论文",
        "view_manuscript": "查看手稿",
        "load_chat": "加载到对话",
        "export_bibtex": "导出 BibTeX",
        "delete": "删除",
        "reanalyze": "重新分析",
        "rereview": "重新审阅",
        "reanalyze_file_label": "重新分析的 PDF",
        "rereview_file_label": "重新审阅的 PDF",
        "projects_tab_desc": "将对话组织为命名项目。从列表中选择项目即可立即切换——"
                             "对话历史会自动保存，切换时自动加载。",
        "project_dd_label": "项目",
        "switch_project": "切换到项目",
        "switch_on_select_info": "选择后立即切换项目。",
        "create_project": "创建项目",
        "new_project_label": "新项目名称",
        "new_project_placeholder": "例如：quantum_hall",
        "current_project": "当前项目",
        "refresh_projects": "🔄 刷新",
        "settings_tab_desc": "选择显示语言。该设置只影响界面显示，"
                             "你的数据和分析结果不受影响。",
        "lang_label": "语言",
        "lang_info": "选择英文或中文。",
        "footer": "所有分析结果存储在 `paper_database.json` 和 `manuscript_database.json` 中；"
                  "生成的 MATLAB 脚本保存在 `matlab_output/` 中。这些内容会在会话之间保留。",
        # Dynamic strings
        "no_papers": "（暂无论文）",
        "no_manuscripts": "（暂无手稿）",
        "upload_first": "**⚠️ 请先上传 PDF 文件。**",
        "already_analyzed": "**已分析过**（记录 #{idx}）。请在*数据库*标签页中将其加载到对话。",
        "already_reviewed": "**已审阅过**（记录 #{idx}）。请在*数据库*标签页中将其加载到对话。",
        "analysis_failed": "**❌ 分析失败。** 请检查 PDF 是否有效以及 API 密钥是否正确。",
        "review_failed": "**❌ 审阅失败。** 请检查 PDF 是否有效以及 API 密钥是否正确。",
        "saved_paper": "**✅ 已保存到数据库（记录 #{idx}）。** 可在*对话*标签页中讨论"
                       "（或先将其加载到对话）。",
        "saved_manuscript": "**✅ 已保存到数据库（记录 #{idx}）。** 可在*对话*标签页中讨论"
                            "（或先将其加载到对话）。",
        "no_such_record": "**⚠️ 没有此记录。**",
        "select_paper_view": "**⚠️ 请选择要查看的论文。**",
        "select_ms_view": "**⚠️ 请选择要查看的手稿。**",
        "select_paper_load": "**⚠️ 请选择要加载的论文。**",
        "select_ms_load": "**⚠️ 请选择要加载的手稿。**",
        "loaded_paper_status": "**✅ 已将记录 #{idx} 加载到对话。**",
        "loaded_ms_status": "**✅ 已将记录 #{idx} 加载到对话。**",
        "chat_loaded_paper": "📄 已将论文《{title}》加载到对话上下文。",
        "chat_loaded_ms": "📝 已将手稿《{title}》加载到对话上下文。",
        "toast_paper_title": "论文已加载",
        "toast_ms_title": "手稿已加载",
        "toast_paper_msg": "记录 #{idx} ·《{title}》已加载到对话上下文。",
        "toast_ms_msg": "记录 #{idx} ·《{title}》已加载到对话上下文。",
        "deleted_paper": "已删除论文《{title}》。",
        "deleted_ms": "已删除手稿《{title}》。",
        "toast_deleted_title": "已删除",
        "reanalyzed_paper": "已重新分析论文《{title}》。",
        "rereviewed_ms": "已重新审阅手稿《{title}》。",
        "toast_reanalyzed_title": "完成",
        "select_project": "请选择项目。",
        "enter_project_name": "请输入项目名称。",
        "switched_project": "已切换到项目「{name}」。",
        "created_project": "已创建项目「{name}」。",
        "toast_project_title": "项目",
        "error_prefix": "**⚠️ 错误：**",
        "record": "记录",
        "full_analysis": "### 完整分析",
        "review_heading": "### 审阅意见",
        "no_analysis": "暂无分析。",
        "no_review": "暂无审阅。",
        "field_col": "字段",
        "value_col": "值",
        "loaded_context_paper": "📄 *已从数据库加载论文上下文。*",
        "loaded_context_ms": "📝 *已从数据库加载手稿上下文。*",
        "dont_save": "不保存到项目",
        "dont_save_info": "勾选后，本次对话不会写入项目。",
        "analyzing_elapsed": "⏱ **正在分析…** 已耗时：`{elapsed}`",
        "reviewing_elapsed": "⏱ **正在审阅…** 已耗时：`{elapsed}`",
        "completed_in": "⏱ 用时 {elapsed}",
    },
}

# Metadata field labels, keyed by database field name.
FIELD_LABELS = {
    "authors": {"en": "Authors", "zh": "作者"},
    "year": {"en": "Year", "zh": "年份"},
    "journal": {"en": "Journal", "zh": "期刊"},
    "doi": {"en": "DOI", "zh": "DOI"},
    "arxiv_id": {"en": "arXiv", "zh": "arXiv"},
    "material_system": {"en": "Material system", "zh": "材料体系"},
    "device_structure": {"en": "Device structure", "zh": "器件结构"},
    "temperature_range": {"en": "Temperature range", "zh": "温度范围"},
    "magnetic_field_range": {"en": "Magnetic field range", "zh": "磁场范围"},
    "mobility": {"en": "Mobility", "zh": "迁移率"},
    "carrier_density": {"en": "Carrier density", "zh": "载流子密度"},
    "mean_free_path": {"en": "Mean free path", "zh": "平均自由程"},
    "phase_coherence_length": {"en": "Phase coherence length", "zh": "相位相干长度"},
    "uploaded_at": {"en": "Uploaded", "zh": "上传时间"},
    "id": {"en": "ID", "zh": "ID"},
    "status": {"en": "Status", "zh": "状态"},
}

PAPER_FIELD_KEYS = [
    "authors", "year", "journal", "doi", "arxiv_id", "material_system",
    "device_structure", "temperature_range", "magnetic_field_range", "mobility",
    "carrier_density", "mean_free_path", "phase_coherence_length",
]
MANUSCRIPT_FIELD_KEYS = [
    "authors", "uploaded_at", "id", "status", "material_system",
    "device_structure", "temperature_range", "magnetic_field_range", "mobility",
    "carrier_density", "mean_free_path", "phase_coherence_length",
]


def t(key: str) -> str:
    """Return the current-language string for `key`."""
    return STRINGS[LANG].get(key, key)


def _flabel(field_key: str) -> str:
    """Return the current-language label for a metadata field."""
    return FIELD_LABELS.get(field_key, {}).get(LANG, field_key)


# ============================================================
# Metadata helpers (for the Database browser)
# ============================================================

def _md_table(record: dict, field_keys: list) -> str:
    """Render a record's metadata as a compact Markdown table."""
    rows = []
    for key in field_keys:
        val = record.get(key)
        if val in (None, "", "Not reported", "Not Reported"):
            continue
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val)
        rows.append(f"| {_flabel(key)} | {val} |")
    if not rows:
        return ""
    return f"| {t('field_col')} | {t('value_col')} |\n|---|---|\n" + "\n".join(rows)


def _paper_markdown(idx: int) -> str:
    if idx < 1 or idx > len(paperq.paper_database):
        return t("no_such_record")
    p = paperq.paper_database[idx - 1]
    md = [f"## 📄 {t('record')} #{idx}: {p.get('title', 'Unknown')}", ""]
    table = _md_table(p, PAPER_FIELD_KEYS)
    if table:
        md.append(table)
        md.append("")
    analysis = p.get("analysis") or p.get("summary") or t("no_analysis")
    md.append(t("full_analysis"))
    md.append("")
    md.append(paperq._render(analysis))
    return "\n".join(md)


def _manuscript_markdown(idx: int) -> str:
    if idx < 1 or idx > len(paperq.manuscript_database):
        return t("no_such_record")
    m = paperq.manuscript_database[idx - 1]
    md = [f"## 📝 {t('record')} #{idx}: {m.get('title', 'Unknown')}", ""]
    table = _md_table(m, MANUSCRIPT_FIELD_KEYS)
    if table:
        md.append(table)
        md.append("")
    critique = m.get("critique") or m.get("summary") or t("no_review")
    md.append(t("review_heading"))
    md.append("")
    md.append(paperq._render(critique))
    return "\n".join(md)


# ============================================================
# Chat (ask)
# ============================================================

def _recalled_title(content: str, prefix: str) -> str:
    """Extract the title from a "[... context recalled: TITLE]" message."""
    if content.startswith(prefix):
        return content[len(prefix):].splitlines()[0].strip().rstrip("]").strip() or "?"
    return "?"


def _seed_visible_history() -> list:
    """Convert paperq.messages into chatbot-visible history.

    The internal message list also contains the system prompt, injected
    "[Paper context ...]"/"[Manuscript context ...]" blobs plus their acks,
    and "[Paper analyzed ...]"/"[Manuscript analyzed ...]" markers. Context
    injections are shown as a one-line note; analysis markers are hidden.
    """
    out = []
    skip_next = False
    for m in paperq.messages:
        role = m.get("role")
        content = m.get("content") or ""
        if skip_next:
            skip_next = False
            continue
        if role == "system":
            continue
        if role == "user" and content.startswith("[Paper context"):
            title = _recalled_title(content, "[Paper context recalled: ")
            out.append({"role": "assistant",
                        "content": t("chat_loaded_paper").format(title=title)})
            skip_next = True
            continue
        if role == "user" and content.startswith("[Manuscript context"):
            title = _recalled_title(content, "[Manuscript context recalled: ")
            out.append({"role": "assistant",
                        "content": t("chat_loaded_ms").format(title=title)})
            skip_next = True
            continue
        if role == "user" and (content.startswith("[Paper analyzed")
                               or content.startswith("[Manuscript analyzed")):
            skip_next = True  # also skip the following assistant ack
            continue
        out.append({"role": role, "content": paperq._render(content)})
    return out


def _respond(message: str, history: list, model_choice: str, dont_save: bool):
    """Chat turn generator: streams the reply token-by-token into the UI.

    When dont_save is True the turn is kept in memory only (so the agent
    keeps context) but is never written into the current project file.
    """
    history = list(history or [])
    if not message or not message.strip():
        yield history, ""
        return

    history = history + [{"role": "user", "content": message}]

    fast = model_choice == "deepseek-v4-flash"
    model = paperq.MODEL_FAST if fast else paperq.MODEL_PRO
    effort = "low" if fast else "high"

    context = paperq.messages + _session_unsaved + [{"role": "user", "content": message}]

    reply = ""
    try:
        response = paperq.client.chat.completions.create(
            model=model,
            messages=context,
            stream=True,
            reasoning_effort=effort,
        )
        for chunk in response:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                reply += delta
                yield history + [{"role": "assistant",
                                  "content": paperq._render(reply)}], ""
    except Exception as exc:  # noqa: BLE001
        yield history + [{"role": "assistant",
                          "content": f"{t('error_prefix')} {exc}"}], ""
        return

    # Save any ```matlab filename="..."``` blocks to .m files, then render.
    clean = paperq._save_matlab_blocks(reply)
    if dont_save:
        _session_unsaved.append({"role": "user", "content": message})
        _session_unsaved.append({"role": "assistant", "content": reply})
    else:
        paperq.messages.append({"role": "user", "content": message})
        paperq.messages.append({"role": "assistant", "content": reply})
        paperq.save_project()
    yield history + [{"role": "assistant",
                      "content": paperq._render(clean)}], ""


def _clear_chat():
    _session_unsaved.clear()
    paperq.clear_history()
    return [], ""


# ============================================================
# Analyze paper / Review manuscript
# ============================================================

def _fmt_elapsed(secs: float) -> str:
    """Format a duration in seconds as 'm:ss'."""
    secs = int(secs)
    mm, ss = divmod(secs, 60)
    return f"{mm}:{ss:02d}"


def _run_with_elapsed(work, label: str, pack):
    """Run `work` in a background thread, streaming elapsed time to the UI.

    A generator: every 0.25 s while `work` is still running it yields
    `pack(label.format(elapsed=...))`. `pack` must return the *full* output
    tuple for the Gradio event (e.g. a 2-tuple `(timer_line, output)`), so the
    values yielded here always match the event's number of outputs.

    When the worker finishes, the generator's return value is `work()`'s result
    (raise the worker's exception if one occurred). Use it with `yield from`
    inside the event handler.

    The analysis calls (paperq.analyze_paper / analyze_manuscript) block for
    30–60+ seconds while talking to the API, so they cannot stream progress
    themselves; running them in a daemon thread lets this generator keep
    yielding updates and keeps the rest of the app responsive.
    """
    started = time.monotonic()
    state = {"done": False, "value": None, "error": None}

    def _worker():
        try:
            state["value"] = work()
        except Exception as exc:  # noqa: BLE001
            state["error"] = exc
        finally:
            state["done"] = True

    threading.Thread(target=_worker, daemon=True).start()

    while not state["done"]:
        time.sleep(0.25)
        yield pack(label.format(elapsed=_fmt_elapsed(time.monotonic() - started)))

    if state["error"] is not None:
        raise state["error"]
    return state["value"]


def _strip_analysis_record(n_before: int):
    """Drop any chat-history messages appended by analyze_paper/analyze_manuscript.

    paperq's analyze/review functions append a "[... analyzed]" marker plus an
    ack into the conversation. Analysis actions should not be recorded in the
    chat history, so remove whatever they appended and re-save.
    """
    if len(paperq.messages) > n_before:
        del paperq.messages[n_before:]
        paperq.save_project()


def _analyze_paper(pdf):
    """Analyze an uploaded paper PDF, streaming a live elapsed-time counter.

    Yields (timer_line, output_markdown) so Gradio updates both the elapsed
    timer and the analysis output while the API works in the background.
    """
    if not pdf:
        yield "", t("upload_first")
        return
    path = str(Path(pdf))
    fname = Path(path).name

    for i, p in enumerate(paperq.paper_database, 1):
        if p.get("filename") == fname:
            yield "", (_paper_markdown(i)
                       + f"\n\n> ⚠️ {t('already_analyzed').format(idx=i)}")
            return

    n_before = len(paperq.messages)
    started = time.monotonic()
    try:
        analysis = yield from _run_with_elapsed(
            lambda: paperq.analyze_paper(pdf_path=path),
            t("analyzing_elapsed"),
            lambda status: (status, ""),
        )
    except Exception as exc:  # noqa: BLE001
        yield "", f"{t('error_prefix')} {exc}"
        return
    if not analysis:
        yield "", t("analysis_failed")
        return
    _strip_analysis_record(n_before)
    idx = len(paperq.paper_database)
    done = t("completed_in").format(elapsed=_fmt_elapsed(time.monotonic() - started))
    yield done, (paperq._render(analysis)
                 + f"\n\n---\n{t('saved_paper').format(idx=idx)}")


def _review_manuscript(pdf):
    """Review an uploaded manuscript PDF, streaming a live elapsed-time counter.

    Yields (timer_line, output_markdown) so Gradio updates both the elapsed
    timer and the review output while the API works in the background.
    """
    if not pdf:
        yield "", t("upload_first")
        return
    path = str(Path(pdf))
    fname = Path(path).name

    for i, m in enumerate(paperq.manuscript_database, 1):
        if m.get("filename") == fname:
            yield "", (_manuscript_markdown(i)
                       + f"\n\n> ⚠️ {t('already_reviewed').format(idx=i)}")
            return

    n_before = len(paperq.messages)
    started = time.monotonic()
    try:
        critique = yield from _run_with_elapsed(
            lambda: paperq.analyze_manuscript(pdf_path=path),
            t("reviewing_elapsed"),
            lambda status: (status, ""),
        )
    except Exception as exc:  # noqa: BLE001
        yield "", f"{t('error_prefix')} {exc}"
        return
    if not critique:
        yield "", t("review_failed")
        return
    _strip_analysis_record(n_before)
    idx = len(paperq.manuscript_database)
    done = t("completed_in").format(elapsed=_fmt_elapsed(time.monotonic() - started))
    yield done, (paperq._render(critique)
                 + f"\n\n---\n{t('saved_manuscript').format(idx=idx)}")


# ============================================================
# Database browser
# ============================================================

def _paper_choices() -> list:
    if not paperq.paper_database:
        return [t("no_papers")]
    return [f"{i}. {paperq.paper_database[i - 1].get('title', '?')[:60]}"
            for i in range(1, len(paperq.paper_database) + 1)]


def _manuscript_choices() -> list:
    if not paperq.manuscript_database:
        return [t("no_manuscripts")]
    return [f"{i}. {paperq.manuscript_database[i - 1].get('title', '?')[:60]}"
            for i in range(1, len(paperq.manuscript_database) + 1)]


def _parse_index(choice: str) -> Optional[int]:
    try:
        return int(str(choice).split(".")[0])
    except (ValueError, AttributeError):
        return None


def _refresh_lists():
    return gr.update(choices=_paper_choices(), value=None), \
        gr.update(choices=_manuscript_choices(), value=None)


def _view_paper(choice: str) -> str:
    idx = _parse_index(choice)
    return _paper_markdown(idx) if idx else t("select_paper_view")


def _view_manuscript(choice: str) -> str:
    idx = _parse_index(choice)
    return _manuscript_markdown(idx) if idx else t("select_ms_view")


def _load_paper(choice: str, history: list):
    """Load a paper into the chat context: toast confirmation + chat note."""
    idx = _parse_index(choice)
    history = list(history or [])
    if not idx:
        gr.Warning(t("select_paper_load"), duration=None)
        return history
    paperq.recall_paper(idx)
    title = paperq.paper_database[idx - 1].get("title", "?")
    note = {"role": "assistant", "content": t("chat_loaded_paper").format(title=title)}
    history = history + [note]
    gr.Success(t("toast_paper_msg").format(idx=idx, title=title),
               title=t("toast_paper_title"), duration=None)
    return history


def _load_manuscript(choice: str, history: list):
    """Load a manuscript into the chat context: toast confirmation + chat note."""
    idx = _parse_index(choice)
    history = list(history or [])
    if not idx:
        gr.Warning(t("select_ms_load"), duration=None)
        return history
    paperq.recall_manuscript(idx)
    title = paperq.manuscript_database[idx - 1].get("title", "?")
    note = {"role": "assistant", "content": t("chat_loaded_ms").format(title=title)}
    history = history + [note]
    gr.Success(t("toast_ms_msg").format(idx=idx, title=title),
               title=t("toast_ms_title"), duration=None)
    return history


def _export_bibtex(choice: str) -> str:
    idx = _parse_index(choice)
    if not idx:
        gr.Warning(t("select_paper_view"), duration=None)
        return ""
    return paperq.export_bibtex(idx)


def _delete_paper(choice: str):
    idx = _parse_index(choice)
    if not idx:
        gr.Warning(t("select_paper_view"), duration=None)
        return gr.update(), "", ""
    title = paperq.paper_database[idx - 1].get("title", "?")
    paperq.paper_database.pop(idx - 1)
    paperq.save_paper_database()
    gr.Success(t("deleted_paper").format(title=title),
               title=t("toast_deleted_title"), duration=None)
    return gr.update(choices=_paper_choices(), value=None), "", ""


def _delete_manuscript(choice: str):
    idx = _parse_index(choice)
    if not idx:
        gr.Warning(t("select_ms_view"), duration=None)
        return gr.update(), ""
    title = paperq.manuscript_database[idx - 1].get("title", "?")
    paperq.manuscript_database.pop(idx - 1)
    paperq.save_manuscript_database()
    gr.Success(t("deleted_ms").format(title=title),
               title=t("toast_deleted_title"), duration=None)
    return gr.update(choices=_manuscript_choices(), value=None), ""


def _reanalyze_paper(choice: str, pdf):
    idx = _parse_index(choice)
    if not idx:
        gr.Warning(t("select_paper_view"), duration=None)
        yield gr.update(choices=_paper_choices(), value=None), "", t("select_paper_view")
        return
    if not pdf:
        gr.Warning(t("upload_first"), duration=None)
        yield gr.update(choices=_paper_choices(), value=choice), "", t("upload_first")
        return
    n_before = len(paperq.messages)
    started = time.monotonic()
    keep_dd = gr.update(choices=_paper_choices(), value=choice)
    try:
        analysis = yield from _run_with_elapsed(
            lambda: paperq.analyze_paper(pdf_path=str(Path(pdf)), replace_index=idx),
            t("analyzing_elapsed"),
            lambda status: (keep_dd, status, ""),
        )
    except Exception as exc:  # noqa: BLE001
        yield keep_dd, "", f"{t('error_prefix')} {exc}"
        return
    if not analysis:
        yield keep_dd, "", t("analysis_failed")
        return
    _strip_analysis_record(n_before)
    title = paperq.paper_database[idx - 1].get("title", "?")
    choices = _paper_choices()
    value = choices[idx - 1] if 0 <= idx - 1 < len(choices) else None
    gr.Success(t("reanalyzed_paper").format(title=title),
               title=t("toast_reanalyzed_title"), duration=None)
    done = t("completed_in").format(elapsed=_fmt_elapsed(time.monotonic() - started))
    yield gr.update(choices=choices, value=value), done, paperq._render(analysis)


def _rereview_manuscript(choice: str, pdf):
    idx = _parse_index(choice)
    if not idx:
        gr.Warning(t("select_ms_view"), duration=None)
        yield gr.update(choices=_manuscript_choices(), value=None), "", t("select_ms_view")
        return
    if not pdf:
        gr.Warning(t("upload_first"), duration=None)
        yield gr.update(choices=_manuscript_choices(), value=choice), "", t("upload_first")
        return
    n_before = len(paperq.messages)
    started = time.monotonic()
    keep_dd = gr.update(choices=_manuscript_choices(), value=choice)
    try:
        critique = yield from _run_with_elapsed(
            lambda: paperq.analyze_manuscript(pdf_path=str(Path(pdf)), replace_index=idx),
            t("reviewing_elapsed"),
            lambda status: (keep_dd, status, ""),
        )
    except Exception as exc:  # noqa: BLE001
        yield keep_dd, "", f"{t('error_prefix')} {exc}"
        return
    if not critique:
        yield keep_dd, "", t("review_failed")
        return
    _strip_analysis_record(n_before)
    title = paperq.manuscript_database[idx - 1].get("title", "?")
    choices = _manuscript_choices()
    value = choices[idx - 1] if 0 <= idx - 1 < len(choices) else None
    gr.Success(t("rereviewed_ms").format(title=title),
               title=t("toast_reanalyzed_title"), duration=None)
    done = t("completed_in").format(elapsed=_fmt_elapsed(time.monotonic() - started))
    yield gr.update(choices=choices, value=value), done, paperq._render(critique)


# ============================================================
# Projects
# ============================================================

def _project_choices() -> list:
    return paperq.get_project_files() or []


def _current_project_md() -> str:
    return f"**{t('current_project')}:** `{paperq.current_project or '—'}`"


def _switch_project(name: str):
    if not name:
        gr.Warning(t("select_project"), duration=None)
        return _seed_visible_history(), _current_project_md(), _current_project_md()
    if name == paperq.current_project:
        # Already on this project (re-selected, or the create-project flow just
        # switched to it); nothing to do beyond refreshing the visible state.
        return _seed_visible_history(), _current_project_md(), _current_project_md()
    _session_unsaved.clear()
    paperq.switch_project(name)
    gr.Success(t("switched_project").format(name=name),
               title=t("toast_project_title"), duration=None)
    return _seed_visible_history(), _current_project_md(), _current_project_md()


def _create_project(name: str):
    name = (name or "").strip()
    if not name:
        gr.Warning(t("enter_project_name"), duration=None)
        return (_seed_visible_history(),
                gr.update(choices=_project_choices(), value=paperq.current_project),
                _current_project_md(), "", _current_project_md())
    _session_unsaved.clear()
    paperq.new_project(name)
    gr.Success(t("created_project").format(name=name),
               title=t("toast_project_title"), duration=None)
    return (_seed_visible_history(),
            gr.update(choices=_project_choices(), value=name),
            _current_project_md(), "", _current_project_md())


# ============================================================
# App assembly
# ============================================================

def _header_md_text() -> str:
    return f"# {t('app_title')}\n{t('app_subtitle')}"


def _footer_md_text() -> str:
    return "---\n*" + t("footer") + "*"


def _set_language(choice: str):
    """Return gr.update() calls for every localised component, in wiring order."""
    global LANG
    LANG = "zh" if choice == "中文" else "en"
    return (
        gr.update(value=_header_md_text()),                                    # header_md
        gr.update(value=_current_project_md()),                                 # chat_proj_md
        gr.update(label=t("chatbot_label")),                                    # chatbot
        gr.update(label=t("question_label"), placeholder=t("question_placeholder")),  # msg
        gr.update(label=t("model_label"), info=t("model_info")),                # model_dd
        gr.update(label=t("dont_save"), info=t("dont_save_info")),              # dont_save_chk
        gr.update(value=t("send")),                                             # send_btn
        gr.update(value=t("clear_history")),                                    # clear_btn
        gr.update(value=t("paper_tab_desc")),                                   # paper_desc
        gr.update(label=t("paper_file_label")),                                 # paper_file
        gr.update(value=t("analyze_paper")),                                    # paper_btn
        gr.update(value=t("ms_tab_desc")),                                      # ms_desc
        gr.update(label=t("ms_file_label")),                                    # ms_file
        gr.update(value=t("review_manuscript")),                                # ms_btn
        gr.update(value=t("db_tab_desc")),                                      # db_desc
        gr.update(value=t("refresh")),                                          # refresh_btn
        gr.update(value=t("papers_heading")),                                   # papers_heading
        gr.update(value=t("manuscripts_heading")),                              # ms_heading
        gr.update(label=t("paper_dd_label")),                                   # paper_dd
        gr.update(value=t("view_paper")),                                       # paper_view_btn
        gr.update(value=t("load_chat")),                                        # paper_load_btn
        gr.update(value=t("export_bibtex")),                                    # paper_export_btn
        gr.update(value=t("delete")),                                           # paper_delete_btn
        gr.update(label=t("reanalyze_file_label")),                             # paper_re_file
        gr.update(value=t("reanalyze")),                                        # paper_re_btn
        gr.update(label=t("ms_dd_label")),                                      # ms_dd
        gr.update(value=t("view_manuscript")),                                  # ms_view_btn
        gr.update(value=t("load_chat")),                                        # ms_load_btn
        gr.update(value=t("delete")),                                           # ms_delete_btn
        gr.update(label=t("rereview_file_label")),                              # ms_re_file
        gr.update(value=t("rereview")),                                         # ms_re_btn
        gr.update(value=t("projects_tab_desc")),                                # proj_desc
        gr.update(label=t("project_dd_label"), info=t("switch_on_select_info")),  # proj_dd
        gr.update(value=t("refresh_projects")),                                 # refresh_proj_btn
        gr.update(label=t("new_project_label"), placeholder=t("new_project_placeholder")),  # new_proj_tb
        gr.update(value=t("create_project")),                                   # create_proj_btn
        gr.update(value=_current_project_md()),                                 # cur_proj_md
        gr.update(value=t("settings_tab_desc")),                                # settings_desc
        gr.update(label=t("lang_label"), info=t("lang_info")),                  # lang_radio
        gr.update(value=_footer_md_text()),                                     # footer_md
    )


def build_app() -> gr.Blocks:
    if not PAPERQ_OK:
        return _build_missing_key_app()

    # Initialise paperq state (databases + a dedicated GUI project so chat
    # history persists between sessions without touching your notebook projects).
    paperq.load_paper_database()
    paperq.load_manuscript_database()
    paperq.current_project = "gui"
    paperq.messages = paperq.load_project("gui")
    print(f"[PaperQ] Working directory: {os.getcwd()}")
    print(f"[PaperQ] Loaded {len(paperq.paper_database)} paper(s) and "
          f"{len(paperq.manuscript_database)} manuscript(s).")

    with gr.Blocks(title="PaperQ") as demo:
        header_md = gr.Markdown(_header_md_text(), latex_delimiters=LATEX_DELIMS)

        # ---- Chat ----
        with gr.Tab("Chat · 对话"):
            chat_proj_md = gr.Markdown(_current_project_md())
            chatbot = gr.Chatbot(
                label=t("chatbot_label"),
                height=560,
                value=_seed_visible_history(),
                latex_delimiters=LATEX_DELIMS,
            )
            with gr.Row():
                msg = gr.Textbox(
                    label=t("question_label"),
                    placeholder=t("question_placeholder"),
                    scale=5,
                )
                model_dd = gr.Dropdown(
                    choices=["deepseek-v4-pro", "deepseek-v4-flash"],
                    value="deepseek-v4-pro",
                    label=t("model_label"),
                    info=t("model_info"),
                    scale=1,
                )
            with gr.Row():
                dont_save_chk = gr.Checkbox(
                    label=t("dont_save"),
                    value=False,
                    info=t("dont_save_info"),
                )
                send_btn = gr.Button(t("send"), variant="primary")
                clear_btn = gr.Button(t("clear_history"))

            send_btn.click(_respond, [msg, chatbot, model_dd, dont_save_chk], [chatbot, msg])
            msg.submit(_respond, [msg, chatbot, model_dd, dont_save_chk], [chatbot, msg])
            clear_btn.click(_clear_chat, None, [chatbot, msg])

        # ---- Analyze Paper ----
        with gr.Tab("Analyze Paper · 分析论文"):
            paper_desc = gr.Markdown(t("paper_tab_desc"))
            paper_file = gr.File(label=t("paper_file_label"), file_types=[".pdf"], type="filepath")
            paper_btn = gr.Button(t("analyze_paper"), variant="primary")
            paper_timer = gr.Markdown("")
            paper_out = gr.Markdown(latex_delimiters=LATEX_DELIMS)

            paper_btn.click(_analyze_paper, paper_file, [paper_timer, paper_out])

        # ---- Review Manuscript ----
        with gr.Tab("Review Manuscript · 审阅手稿"):
            ms_desc = gr.Markdown(t("ms_tab_desc"))
            ms_file = gr.File(label=t("ms_file_label"), file_types=[".pdf"], type="filepath")
            ms_btn = gr.Button(t("review_manuscript"), variant="primary")
            ms_timer = gr.Markdown("")
            ms_out = gr.Markdown(latex_delimiters=LATEX_DELIMS)

            ms_btn.click(_review_manuscript, ms_file, [ms_timer, ms_out])

        # ---- Database ----
        with gr.Tab("Database · 数据库"):
            db_desc = gr.Markdown(t("db_tab_desc"))
            refresh_btn = gr.Button(t("refresh"))

            with gr.Row():
                with gr.Column():
                    papers_heading = gr.Markdown(t("papers_heading"))
                    paper_dd = gr.Dropdown(choices=_paper_choices(), label=t("paper_dd_label"))
                    with gr.Row():
                        paper_view_btn = gr.Button(t("view_paper"))
                        paper_load_btn = gr.Button(t("load_chat"))
                    with gr.Row():
                        paper_export_btn = gr.Button(t("export_bibtex"))
                        paper_delete_btn = gr.Button(t("delete"))
                    paper_bibtex = gr.Textbox(label="BibTeX", lines=6)
                    with gr.Row():
                        paper_re_file = gr.File(label=t("reanalyze_file_label"),
                                                file_types=[".pdf"], type="filepath")
                        paper_re_btn = gr.Button(t("reanalyze"))
                    paper_re_timer = gr.Markdown("")
                    paper_view = gr.Markdown(latex_delimiters=LATEX_DELIMS)
                with gr.Column():
                    ms_heading = gr.Markdown(t("manuscripts_heading"))
                    ms_dd = gr.Dropdown(choices=_manuscript_choices(), label=t("ms_dd_label"))
                    with gr.Row():
                        ms_view_btn = gr.Button(t("view_manuscript"))
                        ms_load_btn = gr.Button(t("load_chat"))
                    ms_delete_btn = gr.Button(t("delete"))
                    with gr.Row():
                        ms_re_file = gr.File(label=t("rereview_file_label"),
                                             file_types=[".pdf"], type="filepath")
                        ms_re_btn = gr.Button(t("rereview"))
                    ms_re_timer = gr.Markdown("")
                    ms_view = gr.Markdown(latex_delimiters=LATEX_DELIMS)

            refresh_btn.click(_refresh_lists, None, [paper_dd, ms_dd])
            paper_view_btn.click(_view_paper, paper_dd, paper_view)
            paper_load_btn.click(_load_paper, [paper_dd, chatbot], chatbot)
            paper_export_btn.click(_export_bibtex, paper_dd, paper_bibtex)
            paper_delete_btn.click(_delete_paper, paper_dd, [paper_dd, paper_view, paper_bibtex])
            paper_re_btn.click(_reanalyze_paper, [paper_dd, paper_re_file],
                               [paper_dd, paper_re_timer, paper_view])
            ms_view_btn.click(_view_manuscript, ms_dd, ms_view)
            ms_load_btn.click(_load_manuscript, [ms_dd, chatbot], chatbot)
            ms_delete_btn.click(_delete_manuscript, ms_dd, [ms_dd, ms_view])
            ms_re_btn.click(_rereview_manuscript, [ms_dd, ms_re_file],
                            [ms_dd, ms_re_timer, ms_view])

        # ---- Projects ----
        with gr.Tab("Projects · 项目"):
            proj_desc = gr.Markdown(t("projects_tab_desc"))
            proj_dd = gr.Dropdown(choices=_project_choices(), value=paperq.current_project,
                                  label=t("project_dd_label"),
                                  info=t("switch_on_select_info"))
            refresh_proj_btn = gr.Button(t("refresh_projects"))
            cur_proj_md = gr.Markdown(_current_project_md())
            with gr.Row():
                new_proj_tb = gr.Textbox(label=t("new_project_label"),
                                         placeholder=t("new_project_placeholder"), scale=3)
                create_proj_btn = gr.Button(t("create_project"), scale=1)

            # Selecting a project from the list switches to it immediately.
            proj_dd.change(_switch_project, proj_dd,
                           [chatbot, cur_proj_md, chat_proj_md])
            refresh_proj_btn.click(
                lambda: gr.update(choices=_project_choices(), value=paperq.current_project),
                None, proj_dd)
            create_proj_btn.click(_create_project, new_proj_tb,
                                  [chatbot, proj_dd, cur_proj_md, new_proj_tb, chat_proj_md])

        # ---- Settings ----
        with gr.Tab("Settings · 设置"):
            settings_desc = gr.Markdown(t("settings_tab_desc"))
            lang_radio = gr.Radio(
                choices=["English", "中文"],
                value="English" if LANG == "en" else "中文",
                label=t("lang_label"),
                info=t("lang_info"),
            )

        footer_md = gr.Markdown(_footer_md_text())

        lang_radio.change(
            _set_language,
            inputs=lang_radio,
            outputs=[
                header_md, chat_proj_md, chatbot, msg, model_dd, dont_save_chk,
                send_btn, clear_btn,
                paper_desc, paper_file, paper_btn, ms_desc, ms_file, ms_btn,
                db_desc, refresh_btn, papers_heading, ms_heading,
                paper_dd, paper_view_btn, paper_load_btn, paper_export_btn,
                paper_delete_btn, paper_re_file, paper_re_btn,
                ms_dd, ms_view_btn, ms_load_btn, ms_delete_btn, ms_re_file, ms_re_btn,
                proj_desc, proj_dd, refresh_proj_btn,
                new_proj_tb, create_proj_btn, cur_proj_md,
                settings_desc, lang_radio, footer_md,
            ],
        )

    return demo


def _build_missing_key_app() -> gr.Blocks:
    """Minimal app shown when no API key is set, so the user still gets a
    working GUI that tells them what to do."""
    with gr.Blocks(title="PaperQ — Setup required") as demo:
        gr.Markdown(
            "# PaperQ — API key required / 需要 API 密钥\n\n"
            f"**⚠️ Could not load the agent / 无法加载智能体:** {IMPORT_ERROR}\n\n"
            "Set your DeepSeek API key and restart the app / "
            "设置 DeepSeek API 密钥后重新启动：\n\n"
            "**Windows (Command Prompt):**\n"
            "```\nset DEEPSEEK_API_KEY=your-key-here\npython app.py\n```\n\n"
            "**Windows (PowerShell) / macOS / Linux:**\n"
            "```\n$env:DEEPSEEK_API_KEY=\"your-key-here\"  # PowerShell\n"
            "export DEEPSEEK_API_KEY=your-key-here        # bash\n"
            "python app.py\n```\n\n"
            "Get a key from https://platform.deepseek.com."
        )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch(inbrowser=True, show_error=True)
