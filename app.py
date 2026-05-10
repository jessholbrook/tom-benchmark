"""Streamlit dashboard for running and comparing benchmark runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from tom_benchmark.adapters import MODEL_REGISTRY
from tom_benchmark.judge import JudgeConfig
from tom_benchmark.loader import CATEGORY_FILES, category_counts, load_scenarios
from tom_benchmark.models import BenchmarkRun
from tom_benchmark.runner import BenchmarkRunner

load_dotenv()

RESULTS_DIR = Path(".benchmark_runs")
RESULTS_DIR.mkdir(exist_ok=True)


def _save_run(run: BenchmarkRun) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = run.model_name.replace("/", "_")
    path = RESULTS_DIR / f"{ts}_{safe_model}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_runs() -> list[tuple[Path, BenchmarkRun]]:
    runs: list[tuple[Path, BenchmarkRun]] = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            run = BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))
            runs.append((path, run))
        except Exception:
            continue
    return runs


def _apply_session_keys() -> None:
    """Push the session-state API keys into os.environ for adapters to pick up."""
    for env_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        val = st.session_state.get(f"_{env_key}")
        if val:
            os.environ[env_key] = val


def _render_sidebar() -> None:
    st.sidebar.title("🧠 ToM Benchmark")
    st.sidebar.markdown(
        "Evaluate LLMs on **Theory of Mind** tasks across 6 cognitive categories."
    )

    st.sidebar.divider()
    st.sidebar.subheader("API keys (session-only)")
    st.sidebar.caption(
        "Keys are kept in your browser session and used only to call the model "
        "providers. They are not stored or logged."
    )

    # Anthropic
    current_anthropic = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_input = st.sidebar.text_input(
        "ANTHROPIC_API_KEY",
        value=st.session_state.get("_ANTHROPIC_API_KEY", ""),
        type="password",
        placeholder="sk-ant-..." if not current_anthropic else "(loaded from env)",
        help="Optional. Required to run Claude models.",
        key="_ANTHROPIC_API_KEY",
    )
    if anthropic_input:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_input

    # OpenAI
    current_openai = os.getenv("OPENAI_API_KEY", "")
    openai_input = st.sidebar.text_input(
        "OPENAI_API_KEY",
        value=st.session_state.get("_OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-..." if not current_openai else "(loaded from env)",
        help="Optional. Required to run GPT models.",
        key="_OPENAI_API_KEY",
    )
    if openai_input:
        os.environ["OPENAI_API_KEY"] = openai_input

    st.sidebar.subheader("Status")
    st.sidebar.markdown(
        f"- Anthropic: {'✅ ready' if os.getenv('ANTHROPIC_API_KEY') else '⚪ not set'}\n"
        f"- OpenAI: {'✅ ready' if os.getenv('OPENAI_API_KEY') else '⚪ not set'}"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Dataset")
    counts = category_counts()
    total = sum(counts.values())
    st.sidebar.markdown(f"**{total} scenarios** across {len(counts)} categories")
    for cat, n in counts.items():
        st.sidebar.markdown(f"- `{cat}`: {n}")


def render_browse_tab() -> None:
    st.header("Browse Scenarios")
    st.caption(
        "The dataset is hand-crafted. Filter, read, and inspect any scenario "
        "without running a model."
    )

    col1, col2 = st.columns(2)
    with col1:
        cat_filter = st.selectbox("Category", options=["(all)"] + list(CATEGORY_FILES.keys()))
    with col2:
        tier_filter = st.selectbox("Tier", options=["(all)", "easy", "medium", "hard", "expert"])

    cat = None if cat_filter == "(all)" else cat_filter
    tier = None if tier_filter == "(all)" else tier_filter
    scenarios = load_scenarios(category=cat, tier=tier)

    st.markdown(f"**{len(scenarios)} scenarios** match.")

    for s in scenarios:
        with st.expander(f"`{s.id}` · {s.name} · {s.category}/{s.tier}"):
            st.markdown(f"**Scenario.** {s.scenario}")
            st.markdown(f"**Question.** {s.question}")
            st.markdown(f"**Expected answer.** `{s.expected_answer}`")
            if s.answer_aliases:
                aliases = ", ".join(f"`{a}`" for a in s.answer_aliases)
                st.markdown(f"**Aliases.** {aliases}")
            st.markdown(f"**Rubric.** {s.rubric}")
            st.markdown(f"**Why this is the answer.** {s.explanation}")
            st.json(json.loads(s.model_dump_json()), expanded=False)


def render_run_tab() -> None:
    st.header("Run Benchmark")
    available_models = list(MODEL_REGISTRY.keys())

    no_keys = not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if no_keys:
        st.warning(
            "No API keys detected. Paste an Anthropic or OpenAI key in the sidebar "
            "to enable benchmark runs. Keys live only in your session."
        )

    models = st.multiselect(
        "Models to evaluate",
        options=available_models,
        default=[available_models[0]] if available_models else [],
    )
    categories = st.multiselect(
        "Categories (empty = all)",
        options=list(CATEGORY_FILES.keys()),
    )
    tiers = st.multiselect(
        "Tiers (empty = all)",
        options=["easy", "medium", "hard", "expert"],
    )

    judge_model = st.selectbox(
        "Judge model (Layer 2 — optional)",
        options=["(none)"] + available_models,
    )
    judge_always = st.checkbox("Run judge on every response", value=False)
    structured = st.checkbox("Enable Layer 3 (structured JSON output)", value=False)

    can_run = bool(models) and not no_keys
    if st.button("Run", type="primary", disabled=not can_run):
        scenarios = []
        if categories:
            for cat in categories:
                scenarios.extend(load_scenarios(category=cat))
        else:
            scenarios = load_scenarios()
        if tiers:
            scenarios = [s for s in scenarios if s.tier in tiers]
        if not scenarios:
            st.error("No scenarios match those filters.")
            return

        judge_cfg = (
            JudgeConfig(judge_model=judge_model, always=judge_always)
            if judge_model != "(none)"
            else None
        )

        for model in models:
            st.subheader(f"Running {model}")
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            try:
                runner = BenchmarkRunner(model_name=model, judge=judge_cfg, structured=structured)
            except Exception as e:
                st.error(f"Could not start runner for {model}: {e}")
                continue

            def _progress(idx: int, total: int, scen) -> None:
                progress_bar.progress(idx / total)
                status_box.write(f"[{idx}/{total}] {scen.id} — {scen.category}/{scen.tier}")

            try:
                run = runner.run(scenarios, progress=_progress)
            except Exception as e:
                st.error(f"Run failed for {model}: {e}")
                continue

            saved_path = _save_run(run)
            st.success(
                f"{model}: accuracy {run.accuracy:.2%} "
                f"({run.correct_count}/{run.total}) — saved to {saved_path.name}"
            )
            with st.expander("Per-scenario results"):
                for r in run.results:
                    label = "✅" if r.correct else ("❓" if r.correct is None else "❌")
                    st.write(
                        f"{label} `{r.scenario_id}` ({r.category}/{r.tier}) — {r.response[:160]}"
                    )


def render_results_tab() -> None:
    st.header("Results")
    runs = _load_runs()
    if not runs:
        st.info("No saved runs yet. Use the Run Benchmark tab to create one.")
        return

    options = {f"{p.name} — {r.model_name}": (p, r) for p, r in runs}
    selected = st.selectbox("Select a run", options=list(options.keys()))
    if not selected:
        return
    _, run = options[selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{run.accuracy:.2%}")
    c2.metric("Correct", f"{run.correct_count} / {run.total}")
    c3.metric("Model", run.model_name)

    by_cat = run.accuracy_by_category()
    if by_cat:
        try:
            import pandas as pd
            import plotly.express as px
        except ImportError:
            st.write(by_cat)
        else:
            df = pd.DataFrame({"category": list(by_cat), "accuracy": list(by_cat.values())})
            fig = px.bar(df, x="category", y="accuracy", title="Accuracy by category", range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw results"):
        st.json(json.loads(run.model_dump_json()))


def render_compare_tab() -> None:
    st.header("Compare Models")
    runs = _load_runs()
    if len(runs) < 2:
        st.info("Need at least two saved runs to compare.")
        return

    options = {f"{p.name} — {r.model_name}": (p, r) for p, r in runs}
    selected = st.multiselect("Pick runs to compare", options=list(options.keys()))
    if len(selected) < 2:
        return

    chosen = [options[s] for s in selected]
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        for _, run in chosen:
            st.write(run.model_name, run.accuracy_by_category())
        return

    rows = []
    for _, run in chosen:
        for cat, acc in run.accuracy_by_category().items():
            rows.append({"model": run.model_name, "category": cat, "accuracy": acc})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="category",
        y="accuracy",
        color="model",
        barmode="group",
        title="Accuracy by category, grouped by model",
        range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="🧠 ToM Benchmark", layout="wide", page_icon="🧠")
    _render_sidebar()

    st.title("🧠 Theory of Mind Benchmark")
    st.caption(
        "A benchmark for evaluating LLMs on social cognition tasks. "
        "Browse scenarios, run models against them, and compare results."
    )

    tab_browse, tab_run, tab_results, tab_compare = st.tabs(
        ["Browse Scenarios", "Run Benchmark", "Results", "Compare Models"]
    )
    with tab_browse:
        render_browse_tab()
    with tab_run:
        render_run_tab()
    with tab_results:
        render_results_tab()
    with tab_compare:
        render_compare_tab()


if __name__ == "__main__":
    main()
