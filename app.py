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
from tom_benchmark.loader import CATEGORY_FILES, load_scenarios
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


def _render_sidebar() -> None:
    st.sidebar.title("ToM Benchmark")
    st.sidebar.subheader("API key status")
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.sidebar.markdown(
        f"- Anthropic: {'OK' if anthropic_ok else 'missing'}\n"
        f"- OpenAI: {'OK' if openai_ok else 'missing'}"
    )


def render_run_tab() -> None:
    st.header("Run Benchmark")
    available_models = list(MODEL_REGISTRY.keys())
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

    if st.button("Run", type="primary", disabled=not models):
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
            runner = BenchmarkRunner(model_name=model, judge=judge_cfg, structured=structured)

            def _progress(idx: int, total: int, scen) -> None:
                progress_bar.progress(idx / total)
                status_box.write(f"[{idx}/{total}] {scen.id} — {scen.category}/{scen.tier}")

            run = runner.run(scenarios, progress=_progress)
            saved_path = _save_run(run)
            st.success(
                f"{model}: accuracy {run.accuracy:.2%} ({run.correct_count}/{run.total}) — saved to {saved_path}"
            )
            with st.expander("Per-scenario results"):
                for r in run.results:
                    label = "OK" if r.correct else ("?" if r.correct is None else "X")
                    st.write(
                        f"**{label}** `{r.scenario_id}` ({r.category}/{r.tier}) — {r.response[:120]}"
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
    st.metric("Accuracy", f"{run.accuracy:.2%}", help=f"{run.correct_count}/{run.total}")

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
    st.set_page_config(page_title="ToM Benchmark", layout="wide")
    _render_sidebar()
    tab1, tab2, tab3 = st.tabs(["Run Benchmark", "Results", "Compare Models"])
    with tab1:
        render_run_tab()
    with tab2:
        render_results_tab()
    with tab3:
        render_compare_tab()


if __name__ == "__main__":
    main()
