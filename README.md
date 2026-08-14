# SME Customer Intelligence & Retention Framework

Refactored codebase, regenerated dataset, synchronized dashboard, and rewritten academic paper for the Telkom SME Retention project (Universitas Lampung — Konversi Magang PT Telkom Indonesia).

## Directory Structure

```
sme_retention_project/
├── data_generator_v2.py          # Phase 1 — refactored synthetic data generator
├── scoring_engine.py             # Phase 2 — refactored Triple-Scoring Engine (baseline)
├── dashboard.py                  # Phase 4 — synchronized Streamlit dashboard
├── .streamlit/config.toml        # Streamlit theme config (forced light)
├── data/
│   └── sme_dummy/                # 7 canonical CSVs (regenerated, anomaly-free)
│       ├── sme_customers.csv         (1,000 rows)
│       ├── subscription_history.csv  (1,386 rows)
│       ├── usage_metrics.csv         (9,844 rows)
│       ├── payment_records.csv       (8,165 rows)
│       ├── support_interactions.csv  (2,809 rows)
│       ├── external_enrichment.csv   (1,000 rows, with enrichment_tier marker)
│       └── intervention_log.csv      (1,230 rows, follow_up >= intervention)
├── output/
│   ├── customer_scores.csv       # Phase 3 output (1,000 rows, 417 unique priority_score)
│   └── scoring_validation_charts.png
├── SME_Retention_Academic_Paper_ID.docx  # Phase 5 — formal Bahasa Indonesia (48 pages)
└── SME_Retention_Academic_Paper_ID.pdf   # PDF rendering of the paper
```

## How to Run

### 1. Regenerate the dataset
```bash
cd sme_retention_project
python3 data_generator_v2.py
```
Post-generation assertions verify: no date paradox, no expired non-auto-renewal contracts, MRR scale alignment.

### 2. Run the scoring engine
```bash
python3 scoring_engine.py
```
Outputs `output/customer_scores.csv` (1,000 rows) and `output/scoring_validation_charts.png`. Prints full numerical audit (score stats per segment, 9-category counts, top-10 priority queue, granularity check).

### 3. Launch the dashboard
```bash
pip install streamlit plotly pandas numpy
streamlit run dashboard.py
```
Opens 5-tab dashboard: Ringkasan, Pelanggan, Antrean Intervensi, Langganan, Simulasi What-If.

### 4. Read the academic paper
Open `SME_Retention_Academic_Paper_ID.docx` (editable) or `SME_Retention_Academic_Paper_ID.pdf` (read-only) — 48 pages, 7 chapters, formal Bahasa Indonesia.

## Verification Assertions (all PASS)

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | No Date Paradox | PASS | 0/1,230 rows have follow_up < intervention (was 187/369 = 50.7%) |
| 2 | No Granularity Collapse | PASS | 417 unique priority_score values (was 7 discrete) |
| 3 | No Data Discrepancy | PASS | All Bab 5 numbers match customer_scores.csv with zero variance |
| 4 | Language Consistency | PASS | Code is 100% English; paper is 100% formal Bahasa Indonesia |

## Key Refactoring Changes

### data_generator_v2.py
- Self-contained (no dependency on legacy `./data_v1/`)
- `follow_up_date = intervention_date + random(7..30) days` — strict temporal integrity
- Non-auto-renewal expired contracts rolled forward (418 contracts no longer lost from expiry queue)
- Gaussian noise injection: `epsilon ~ N(0, 2.0)` on usage trajectories and payment amounts
- "At-risk loyal" sub-pattern (15% of Pelanggan Tetap with declining trajectory) for the At Risk within Loyal insight
- MRR ranges calibrated to hit ~Rp 3.94 Miliar cohort aggregate (target Rp 3.95 Miliar)
- Hierarchical Fallback enrichment tier marker (Tier-1 entity lookup vs Tier-2 sector-regional proxy)

### scoring_engine.py
- `first_last3()` trend delta with explicit branches: `len(g)==1 -> 0.0`, `len(g)==2 -> row[1]-row[0]`, `len(g)>=3 -> mean(last 3) - mean(first 3)`
- `calculate_confidence()` now uses pre-imputation snapshot (eliminates dead-code on missingness penalty)
- Urgency fix for Pelanggan Baru: `urgency = potential_score / 100` (high-potential → fast-track)
- `priority_score` exported at 3-decimal precision (never collapsed to 1 decimal)
- Modular interface hooks: `fit_supervised_baseline()` and `compute_shap_importance()` stubs for ML/XAI production roadmap
- Explicit positioning as Explainable Rule-Based Baseline Engine

### dashboard.py
- Tab 3 formula label synchronized with scoring_engine.py implementation
- Tab 1 "MRR Berisiko" now includes Pelanggan Baru Low Potential (silent churn risk)
- Tab 4 Contract Expiry Queue handles both auto-renewal (roll-forward) and non-auto-renewal (end_date actual)

## Financial Scale Harmonization

- **Macro (National Portfolio)**: PT Telkom Indonesia SME Division manages ~Rp 2 Triliun/month MRR across 500,000+ SME accounts.
- **Micro (Sample Cohort)**: 1,000 synthetic SME records (1:500 sampling slice) with aggregate MRR ~Rp 3.94 Miliar/month and average ~Rp 4.06 Juta/account/month — aligned with realistic Telkom SME product pricing (Dedicated Internet, Cloud, IoT, Broadband, Managed WiFi, Unified Communication, Bundle Business).
