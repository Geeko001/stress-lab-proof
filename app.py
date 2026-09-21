"""Linear Attention Stress-Test Lab — Streamlit frontend.

Local experimental validation environment. All results are reported as
observed measurements / candidate regions, never proof.
"""

import io
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from experiments.associative_recall import run_associative_recall, run_dimensionality_sweep
from experiments.context_length import run_length_sweep
from experiments.numerical_saturation import (
    shrinking_update_experiment,
    decay_experiment,
)
from models.linear_attention import FEATURE_MAPS
from utils.generators import estimate_memory_bytes
from utils.metrics import candidate_breakpoint_region
from utils.plotting import plot_accuracy_vs_bindings, plot_multi_curve, plot_simple, save_high_res

# ---------------------------------------------------------------- constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DISCLAIMER = (
    "These experiments test hypotheses proposed in the research paper. "
    "Observed results are not assumed in advance and do not constitute "
    "proof of the paper's theoretical claims."
)

METHODOLOGY = (
    "This laboratory is a local experimental implementation designed to test "
    "specific hypotheses derived from the accompanying research paper. It is "
    "not a reproduction of production-scale linear-attention models and should "
    "not be interpreted as evidence about every implementation of Linear "
    "Attention, RetNet, RWKV, Mamba, or SSM architectures."
)

SOFTMAX_NOTE = (
    "The softmax baseline is a tiny exact-attention reference that retains "
    "individual key-value bindings. It is a conceptual reference, not a "
    "production Transformer."
)

# Default safety caps (4 GB RAM target machine)
MAX_DIM = 128
MAX_TRIALS = 50
MAX_SEQ_DEFAULT = 32768
DTYPE_MAP = {"float64": np.float64, "float32": np.float32, "float16": np.float16}

st.set_page_config(page_title="Linear Attention Stress-Test Lab", layout="wide")

# ---------------------------------------------------------------- helpers
def env_info():
    import numpy
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "streamlit": st.__version__,
    }


def save_run(subdir, config, df):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_DIR / subdir / ts
    outdir.mkdir(parents=True, exist_ok=True)
    full_config = {**config, "timestamp": ts, "env": env_info()}
    (outdir / "config.json").write_text(json.dumps(full_config, indent=2, default=str))
    df.to_csv(outdir / "results.csv", index=False)
    for i, fig in enumerate(st.session_state.get("_figs_to_save", [])):
        fig.savefig(outdir / f"plot_{i}.png", dpi=120)
    # stash for the "Export Video Assets" button (rendered per page)
    st.session_state["last_figs"] = list(st.session_state.get("_figs_to_save", []))
    st.session_state["last_exp"] = subdir
    st.session_state["_figs_to_save"] = []
    return outdir


def export_buttons(df, config, key_prefix):
    csv = df.to_csv(index=False).encode()
    payload = {"config": config, "env": env_info(), "rows": df.to_dict(orient="records")}
    st.download_button("Export CSV", csv, f"{key_prefix}_results.csv", "text/csv",
                       key=f"{key_prefix}_csv")
    st.download_button("Export JSON", json.dumps(payload, indent=2, default=str),
                       f"{key_prefix}_results.json", "application/json",
                       key=f"{key_prefix}_json")


def maybe_export_video_assets():
    """Render 'Export Video Assets' button if a run produced figures."""
    figs = st.session_state.get("last_figs", [])
    exp = st.session_state.get("last_exp", "exp")
    if not figs:
        return
    if st.button("Export Video Assets (300 DPI dark-mode PNGs)", key=f"video_{exp}"):
        outdir = RESULTS_DIR / "figures"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = [save_high_res(fig, outdir / f"{exp}_{ts}_{i}.png",
                               dpi=300, dark_mode=True)
                 for i, fig in enumerate(figs)]
        st.success(f"Saved {len(paths)} high-res figure(s) to {outdir}")
        for p in paths:
            st.caption(p)


def repro_panel(config):
    st.subheader("Reproducible experiment configuration")
    st.code(json.dumps(config, indent=2, default=str), language="json")
    st.caption("Copy this configuration to reproduce the run.")


def interpretation_panel(lines, x_vals=None, y_vals=None):
    st.subheader("Research interpretation (descriptive only)")
    for line in lines:
        st.write(f"Observation: {line}")
    st.caption("Interpretation should be performed by the researcher.")
    if x_vals is not None and y_vals is not None:
        idx = candidate_breakpoint_region(x_vals, y_vals)
        if idx is not None and idx > 0:
            st.info(
                f"Candidate breakpoint region near x = {x_vals[idx]} "
                f"(heuristic: first point ≥0.15 below max observed; "
                f"descriptive only, not proof)."
            )
        else:
            st.info("No candidate breakpoint region flagged by the heuristic.")


def safety_check(dim, trials, seq_len=None, allow_override=False):
    problems = []
    if dim > MAX_DIM:
        problems.append(f"State dimension d={dim} exceeds default cap {MAX_DIM}.")
    if trials > MAX_TRIALS:
        problems.append(f"Trials={trials} exceeds default cap {MAX_TRIALS}.")
    if seq_len is not None and seq_len > MAX_SEQ_DEFAULT:
        problems.append(
            f"Sequence length N={seq_len} exceeds default cap {MAX_SEQ_DEFAULT}. "
            "Large runs may exhaust ~4 GB RAM. Confirmation required."
        )
    if problems and not allow_override:
        for p in problems:
            st.warning("Configuration may be too expensive for this machine. " + p)
        return False
    return True


def results_table(df, experiment_name):
    st.subheader("Results table")
    st.dataframe(df)
    return df


def sidebar_nav():
    st.sidebar.title("LINEAR ATTENTION STRESS-TEST LAB")
    return st.sidebar.radio("Page", [
        "Dashboard",
        "Paper",
        "Associative Recall",
        "Sequence Length",
        "Numerical Saturation",
        "Compare Runs",
        "Results",
        "Paper Hypotheses",
        "About / Methodology",
    ])


def global_settings():
    st.sidebar.header("Global settings")
    dim = st.sidebar.selectbox("State dimension", [8, 16, 32, 64, 128], index=2)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    trials = st.sidebar.selectbox("Trials", [1, 5, 10, 25, 50], index=2)
    precision = st.sidebar.selectbox("Precision", ["float64", "float32", "float16"], index=0)
    feature_map = st.sidebar.selectbox("Feature map", list(FEATURE_MAPS.keys()), index=0)
    override = st.sidebar.checkbox("Override safety limits (advanced)", value=False)
    return dim, int(seed), int(trials), precision, feature_map, override

# ---------------------------------------------------------------- pages
def page_dashboard():
    st.title("Linear Attention Stress-Test Lab")
    st.subheader("Experimental validation environment for predicted architectural breakpoints.")
    st.warning(DISCLAIMER)
    st.info(METHODOLOGY)
    col1, col2, col3 = st.columns(3)
    # count completed runs
    n_runs = sum(1 for _ in RESULTS_DIR.rglob("config.json")) if RESULTS_DIR.exists() else 0
    col1.metric("Completed runs", n_runs)
    col2.metric("Python", env_info()["python"])
    col3.metric("NumPy", env_info()["numpy"])
    st.markdown("### Experiments")
    st.markdown("- **Associative Recall** — density-driven retrieval interference (H1)")
    st.markdown("- **Sequence Length** — out-of-distribution length scaling (H2)")
    st.markdown("- **Numerical Saturation** — finite-precision state updates (H3)")
    st.markdown("### First-run plan")
    st.markdown("1. Experiment 1: d=16, m ≤ 128, trials=5\n2. Experiment 2: d=16, N ≤ 4096\n3. Experiment 3: small numerical sweep")


def page_recall(dim, seed, trials, precision, feature_map, override):
    st.title("Experiment 1 — High-Density Associative Recall")
    st.caption("Tests H1: increasing binding density may produce retrieval interference.")
    st.caption(SOFTMAX_NOTE)
    seed = int(st.number_input("Random Seed", value=int(seed), step=1,
                               key="recall_seed",
                               help="Same seed replays identical curves."))
    m_options = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    m_sel = st.multiselect("Binding counts (m)", m_options, default=[2, 4, 8, 16, 32, 64, 128])
    dims_sel = st.multiselect("State dimensions to compare (auto-overlayed)",
                              [8, 16, 32, 64], default=[8, 16, 32, 64])
    if st.button("Run associative recall"):
        if not m_sel:
            st.error("Select at least one binding count.")
            return
        if not dims_sel:
            st.error("Select at least one state dimension.")
            return
        if not safety_check(max(dims_sel), trials, allow_override=override):
            st.error("Adjust configuration or enable override.")
            return
        prog = st.progress(0.0)
        all_rows = run_dimensionality_sweep(
            sorted(m_sel), dim_values=sorted(dims_sel), trials=trials,
            seed=seed, feature_map=feature_map, dtype=DTYPE_MAP[precision],
            progress_cb=lambda f: prog.progress(f))
        for r in all_rows:
            r["experiment"] = "Associative Recall"
        curves = {}
        for d in sorted(dims_sel):
            curves[f"linear d={d}"] = [r["linear_mean"] for r in all_rows
                                       if r["dim"] == d]
        df = pd.DataFrame(all_rows)
        results_table(df, "Associative Recall")
        # plot per-dim comparison (linear) + first dim with softmax overlay
        fig = plot_multi_curve(sorted(m_sel), curves,
                               "Number of stored bindings (m)", "Retrieval accuracy",
                               "Accuracy vs bindings (linear, by dim)", xscale="log")
        st.pyplot(fig)
        st.session_state["_figs_to_save"] = [fig]
        first_d = dims_sel[0]
        sub = df[df["dim"] == first_d].sort_values("m")
        fig2 = plot_accuracy_vs_bindings(sub["m"].tolist(), sub["linear_mean"].tolist(),
                                         sub["softmax_mean"].tolist(),
                                         title=f"Linear vs softmax (d={first_d})")
        st.pyplot(fig2)
        st.session_state["_figs_to_save"].append(fig2)
        config = {"experiment_id": "associative_recall", "dims": dims_sel,
                  "m_values": sorted(m_sel), "trials": trials, "seed": seed,
                  "precision": precision, "feature_map": feature_map}
        outdir = save_run("associative_recall", config, df)
        st.success(f"Run saved to {outdir}")
        export_buttons(df, config, "recall")
        repro_panel(config)
        first = sub.iloc[0]["linear_mean"]
        last = sub.iloc[-1]["linear_mean"]
        interpretation_panel(
            [f"retrieval accuracy changed from {first:.3f} to {last:.3f} as binding "
             f"count increased (d={first_d})."],
            x_vals=sub["m"].tolist(), y_vals=sub["linear_mean"].tolist())
    maybe_export_video_assets()


def page_length(dim, seed, trials, precision, feature_map, override):
    st.title("Experiment 2 — Sequence-Length Scaling")
    st.caption("Tests H2: performance may degrade beyond the represented length regime. "
               "64k is a stress-test region, not a proven universal breakpoint.")
    seed = int(st.number_input("Random Seed", value=int(seed), step=1,
                               key="length_seed",
                               help="Same seed replays identical curves."))
    n_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64000]
    n_sel = st.multiselect("Sequence lengths (N)", n_options, default=[128, 512, 1024, 4096])
    pos_map = {"Early (10%)": 0.1, "Middle (50%)": 0.5, "Recent (90%)": 0.9}
    pos_sel = st.multiselect("Marker positions", list(pos_map.keys()), default=list(pos_map.keys()))
    if n_sel and max(n_sel) >= 64000:
        st.warning("64k+ runs are a stress-test region. Only proceed if the machine is stable.")
    if st.button("Run length sweep"):
        if not n_sel or not pos_sel:
            st.error("Select at least one length and one position.")
            return
        if not safety_check(dim, trials, seq_len=max(n_sel), allow_override=override):
            st.error("Adjust configuration or enable override.")
            return
        est = estimate_memory_bytes(max(n_sel), dim, precision)
        st.caption(f"Estimated working memory: ~{est / 1e6:.1f} MB (streaming state only).")
        pfs = [pos_map[p] for p in pos_sel]
        prog = st.progress(0.0)
        rows = run_length_sweep(sorted(n_sel), dim=dim, trials=trials, seed=seed,
                                position_fracs=pfs, feature_map=feature_map,
                                dtype=DTYPE_MAP[precision],
                                progress_cb=lambda f: prog.progress(f))
        for r in rows:
            r["experiment"] = "Sequence Length"
        df = pd.DataFrame(rows)
        results_table(df, "Sequence Length")
        figs = []
        for pf, pname in zip(pfs, pos_sel):
            sub = df[df["position_frac"] == pf].sort_values("N")
            fig = plot_simple(sub["N"].tolist(), sub["accuracy"].tolist(),
                              "Sequence length (N)", "Retrieval accuracy",
                              f"Length vs accuracy — {pname}")
            st.pyplot(fig)
            figs.append(fig)
            fig2 = plot_simple(sub["N"].tolist(), sub["state_norm_mean"].tolist(),
                               "Sequence length (N)", "State norm",
                               f"Length vs state norm — {pname}")
            st.pyplot(fig2)
            figs.append(fig2)
        st.session_state["_figs_to_save"] = figs
        config = {"experiment_id": "context_length", "N_values": sorted(n_sel),
                  "positions": pos_sel, "dim": dim, "trials": trials, "seed": seed,
                  "precision": precision, "feature_map": feature_map}
        outdir = save_run("context_length", config, df)
        st.success(f"Run saved to {outdir}")
        export_buttons(df, config, "length")
        repro_panel(config)
        sub = df.sort_values("N")
        interpretation_panel(
            [f"accuracy ranged from {sub['accuracy'].min():.3f} to {sub['accuracy'].max():.3f} "
             f"across tested lengths; state norm ranged from {sub['state_norm_mean'].min():.2f} "
             f"to {sub['state_norm_mean'].max():.2f}."])
    maybe_export_video_assets()


def page_numerical(dim, seed, trials, precision, feature_map, override):
    st.title("Experiment 3 — Numerical / Floating-Point Saturation")
    st.caption("Tests H3: finite-precision accumulation may introduce information loss.")
    seed = int(st.number_input("Random Seed", value=int(seed), step=1,
                               key="numerical_seed",
                               help="Same seed replays identical curves."))
    precs = st.multiselect("Precisions", ["float64", "float32", "float16"],
                           default=["float64", "float32", "float16"])
    n_steps = st.slider("Shrinking-stream steps", 10, 200, 60)
    decay = st.slider("Update decay factor", 0.1, 0.95, 0.5)
    gammas_txt = st.text_input("Gamma values (comma-separated)", "0.99, 0.999, 0.9999")
    if st.button("Run numerical experiments"):
        try:
            gammas = [float(g.strip()) for g in gammas_txt.split(",") if g.strip()]
        except ValueError:
            st.error("Could not parse gamma values.")
            return
        rows = shrinking_update_experiment(n_steps=n_steps, start_magnitude=1.0,
                                           decay=decay, dim=min(dim, 16),
                                           precisions=precs, seed=seed)
        df = pd.DataFrame(rows)
        results_table(df, "Numerical Saturation")
        figs = []
        curves = {}
        for prec in df["precision"].unique():
            sub = df[df["precision"] == prec].sort_values("timestep")
            curves[prec] = sub["effective_change"].tolist()
        fig = plot_multi_curve(df["timestep"].unique().tolist(), curves,
                               "Timestep", "Effective state change",
                               "Update magnitude vs effective change")
        st.pyplot(fig)
        figs.append(fig)
        for prec in df["precision"].unique():
            sub = df[df["precision"] == prec].sort_values("timestep")
            figm = plot_simple(sub["timestep"].tolist(), sub["state_magnitude"].tolist(),
                               "Timestep", "State magnitude", f"State magnitude — {prec}")
            st.pyplot(figm)
            figs.append(figm)
        # decay part
        drows = decay_experiment(gamma_values=gammas, dim=min(dim, 16),
                                 precision=precision, seed=seed)
        ddf = pd.DataFrame(drows)
        st.subheader("Decay experiment (S_t = γ·S_{t-1} + u_t)")
        st.dataframe(ddf)
        figd = plot_simple(ddf["gamma"].tolist(), ddf["final_state_magnitude"].tolist(),
                           "Gamma", "Final state magnitude", "Gamma vs state retention")
        st.pyplot(figd)
        figs.append(figd)
        st.session_state["_figs_to_save"] = figs
        config = {"experiment_id": "numerical_saturation", "precisions": precs,
                  "n_steps": n_steps, "decay": decay, "gammas": gammas,
                  "dim": min(dim, 16), "seed": seed}
        outdir = save_run("numerical_saturation", config, df)
        ddf.to_csv(outdir / "decay.csv", index=False)
        st.success(f"Run saved to {outdir}")
        export_buttons(df, config, "numerical")
        repro_panel(config)
        lines = []
        for prec in df["precision"].unique():
            sub = df[df["precision"] == prec]
            n_ineff = int(sub["ineffective"].sum())
            lines.append(f"precision {prec}: {n_ineff}/{len(sub)} updates numerically ineffective.")
        interpretation_panel(lines)
    maybe_export_video_assets()


def page_compare():
    st.title("Compare Runs")
    st.caption("Overlay curves from two or more saved runs.")
    configs = sorted(RESULTS_DIR.rglob("config.json"))
    if len(configs) < 1:
        st.info("No saved runs yet.")
        return
    labels = [str(c.parent.relative_to(RESULTS_DIR)) for c in configs]
    sel = st.multiselect("Select runs", labels, default=labels[:min(2, len(labels))])
    figs = []
    for label in sel:
        csv = RESULTS_DIR / label / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        st.subheader(label)
        st.dataframe(df.head(20))
        # generic overlay: pick first numeric x-like col and accuracy-like col
        xcol = next((c for c in ["m", "N", "timestep", "gamma"] if c in df.columns), df.columns[0])
        ycol = next((c for c in ["linear_mean", "accuracy", "effective_change",
                                 "state_norm_mean", "state_magnitude"] if c in df.columns), df.columns[-1])
        if "position_frac" in df.columns:
            for pf, sub in df.groupby("position_frac"):
                fig = plot_simple(sub[xcol].tolist(), sub[ycol].tolist(),
                                  xcol, f"{ycol} (pos={pf})", f"{label} pos={pf}")
                st.pyplot(fig)
                figs.append(fig)
        else:
            fig = plot_simple(df[xcol].tolist(), df[ycol].tolist(), xcol, ycol, label)
            st.pyplot(fig)
            figs.append(fig)
    st.session_state["_figs_to_save"] = figs


def page_results():
    st.title("Results")
    configs = sorted(RESULTS_DIR.rglob("config.json"))
    if not configs:
        st.info("No completed runs yet.")
        return
    rows = []
    for c in configs:
        cfg = json.loads(c.read_text())
        rows.append({"run": str(c.parent.relative_to(RESULTS_DIR)),
                     "experiment": cfg.get("experiment_id", "?"),
                     "timestamp": cfg.get("timestamp", "?")})
    st.dataframe(pd.DataFrame(rows))


def page_hypotheses():
    st.title("Paper Hypotheses")
    st.markdown("### Hypothesis H1")
    st.markdown("Increasing associative recall density may produce state collisions in fixed-capacity linear recurrent states.")
    if st.button("Test this hypothesis (H1)"):
        st.info("Open the Associative Recall page from the sidebar.")
    st.markdown("### Hypothesis H2")
    st.markdown("Performance may degrade when sequence length moves sufficiently beyond the regime represented by the model/experiment.")
    if st.button("Test this hypothesis (H2)"):
        st.info("Open the Sequence Length page from the sidebar.")
    st.markdown("### Hypothesis H3")
    st.markdown("Finite-precision recurrent accumulation may introduce additional numerical information loss or ineffective updates.")
    if st.button("Test this hypothesis (H3)"):
        st.info("Open the Numerical Saturation page from the sidebar.")


def page_about():
    st.title("About / Methodology")
    st.markdown("**Theory:** fixed-size recurrent state has bounded representational capacity.")
    st.markdown("**Prediction:** high-density associative recall may exhibit a sharp degradation region.")
    st.markdown("**Observation:** reported per experiment, e.g. accuracy decreased from X to Y.")
    st.markdown("**Interpretation:** performed by the researcher; consistency with the collision mechanism is not proof.")
    st.warning("This experiment does not prove the theory.")
    st.info(METHODOLOGY)
    st.caption(SOFTMAX_NOTE)
    st.json(env_info())


# ---------------------------------------------------------------- paper registry
# Extensible paper metadata. To add another paper, append a dict with the same
# keys — no component changes needed. pdf_url must be a stable public file URL
# (GitHub raw). No API keys, tokens, or backends: the repository is public and
# the manuscript PDFs are intentionally public.
GITHUB_RAW = "https://raw.githubusercontent.com/Geeko001/stress-lab-proof/main"

PAPERS = [
    {
        "id": "paper-01",
        "title": "Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes",
        "pdf_url": f"{GITHUB_RAW}/Sharma_Stress-Testing_Linear_Attention_Manuscript.pdf",
        "filename": "Sharma_Stress-Testing_Linear_Attention_Manuscript.pdf",
        "note": "Final manuscript (canonical paper source).",
    },
]


def get_paper(paper_id):
    for paper in PAPERS:
        if paper["id"] == paper_id:
            return paper
    return PAPERS[0]


@st.cache_data(show_spinner=False)
def fetch_pdf_bytes(url, timeout=60):
    """Fetch the public GitHub-hosted PDF bytes. Raises on failure."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "stress-lab-paper-reader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("URL did not return a PDF document.")
    return data


def build_viewer_html(pdf_bytes):
    """Inject PDF bytes into the viewer template. Raises if template is broken."""
    import base64
    tpl_path = Path(__file__).parent / "assets" / "paper_viewer.html"
    if not tpl_path.exists():
        raise FileNotFoundError("Viewer template not found at assets/paper_viewer.html.")
    html = tpl_path.read_text(encoding="utf-8").replace(
        "__PDF_DATA__",
        base64.b64encode(pdf_bytes).decode("ascii"))
    if "__PDF_DATA__" in html:
        raise ValueError("Viewer template placeholder was not fully replaced.")
    return html


def page_paper():
    st.title("Research Papers")
    st.caption("Papers load directly from the public GitHub repository — no login, "
               "no backend. Select a paper, read it, or download the PDF.")
    options = {paper["title"]: paper["id"] for paper in PAPERS}
    title = st.selectbox("Paper", list(options.keys()))
    paper = get_paper(options[title])
    st.markdown(f"**{paper['title']}**")
    st.caption(paper.get("note", ""))
    st.link_button("Read Paper (new tab)", paper["pdf_url"])
    try:
        with st.spinner("Loading paper from GitHub…"):
            pdf_bytes = fetch_pdf_bytes(paper["pdf_url"])
    except Exception:
        st.error("Could not load the paper from GitHub. Check your connection, "
                 "or open it directly:")
        st.code(paper["pdf_url"])
        st.link_button("Open PDF on GitHub", paper["pdf_url"])
        return
    st.download_button("Download PDF", pdf_bytes, paper["filename"], "application/pdf")
    try:
        st.components.v1.html(build_viewer_html(pdf_bytes), height=950, scrolling=False)
    except Exception:
        st.error("Reader failed to start, but the PDF itself loaded fine — "
                 "use Read Paper or Download PDF above.")
    st.caption("Source: papers/paper-01-stress-testing-linear-attention.md · "
               "Working draft: Paper_1_Draft_v3.md · Evidence: public_evidence/")


# ---------------------------------------------------------------- main
page = sidebar_nav()
dim, seed, trials, precision, feature_map, override = global_settings()

if page == "Dashboard":
    page_dashboard()
elif page == "Paper":
    page_paper()
elif page == "Associative Recall":
    page_recall(dim, seed, trials, precision, feature_map, override)
elif page == "Sequence Length":
    page_length(dim, seed, trials, precision, feature_map, override)
elif page == "Numerical Saturation":
    page_numerical(dim, seed, trials, precision, feature_map, override)
elif page == "Compare Runs":
    page_compare()
elif page == "Results":
    page_results()
elif page == "Paper Hypotheses":
    page_hypotheses()
elif page == "About / Methodology":
    page_about()
