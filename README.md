# Linear Attention Stress-Test Lab

Local experimental validation environment for predicted linear-attention
architectural breakpoints. See `app.py` header and About page for the
methodology disclaimer: results are observed measurements, never proof.

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Tests

```bash
python -m pytest tests/ -v
```

## First-run plan

1. Experiment 1: d=16, m ≤ 128, trials=5
2. Experiment 2: d=16, N ≤ 4096
3. Experiment 3: small numerical sweep

Only attempt larger runs after confirming machine stability.
