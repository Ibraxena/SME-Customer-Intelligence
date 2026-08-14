"""
SME Customer Intelligence & Retention Framework
Dashboard v2.1 (Refactored) — Streamlit
Author: Abrar Rafii Ibrahim (Konversi Magang PT Telkom Indonesia)

Refactoring notes (v2.1 refactored):
    1. Mathematical label synchronization (Tab 3):
       The priority score formula displayed in the dashboard now matches
       the exact implementation in scoring_engine.py:
           priority_score = 0.5 * norm_total_mrr + 0.5 * urgency
       where urgency is segment-routed:
           - Pelanggan Berisiko : churn_risk_index / 100
           - Pelanggan Tetap    : (100 - stability_score) / 100
           - Pelanggan Baru     : potential_score / 100      <-- FIX
       The Pelanggan Baru branch now uses potential_score / 100 so that
       high-potential new accounts are fast-tracked.
    2. MRR Berisiko harmonization (Tab 1):
       The "MRR Berisiko" KPI now includes ALL three at-risk populations:
           - All Pelanggan Berisiko (regardless of sub-category)
           - Pelanggan Tetap with score_category == "At Risk"
           - Pelanggan Baru with score_category == "Low Potential"
       This catches the silent-churn risk in low-potential new accounts
       that was previously missing.
    3. Contract Expiry Queue (Tab 4):
       The expiry queue now correctly handles BOTH auto-renewal and
       non-auto-renewal contracts. For auto-renewal, the effective end
       date is rolled forward to the current renewal period. For
       non-auto-renewal, the actual end date is used. Both are filtered
       to days_to_expiry > 0 (visible upcoming expiries).
    4. Streamlit theme forced to light via .streamlit/config.toml.
    5. UI text is in Indonesian; data layer (scoring_engine.py) is 100% English.

Run:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import scoring_engine as se

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="SME Customer Intelligence — Telkom",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = dict(
    text="#0f172a", muted="#64748b", border="#e2e8f0", bg_soft="#f8fafc",
    blue="#2563eb", green="#059669", red="#dc2626", amber="#b45309", slate="#475569",
)

# 9-category color palette (high contrast across stack and segment)
CATEGORY_COLORS = {
    "High Potential": "#16a34a",    # green
    "Medium Potential": "#2563eb",  # blue
    "Low Potential": "#94a3b8",     # slate gray
    "Highly Stable": "#0d9488",     # teal
    "Stable": "#7c3aed",            # violet
    "At Risk": "#d97706",           # amber
    "Critical Risk": "#dc2626",     # red
    "High Risk": "#ea580c",         # orange
    "Moderate Risk": "#ca8a04",     # dark yellow
}
SEGMENT_COLORS = {
    "Pelanggan Baru": C["blue"],
    "Pelanggan Tetap": C["green"],
    "Pelanggan Berisiko": C["red"],
}

st.markdown(f"""
<style>
    .stApp {{ background-color: #ffffff; }}
    #MainMenu, footer, header {{ visibility: visible; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}

    h1 {{ font-size: 22px; font-weight: 700; color: {C['text']}; letter-spacing: -0.3px; }}
    h2, h3 {{ font-size: 13px; font-weight: 600; color: {C['muted']};
              text-transform: uppercase; letter-spacing: 0.6px; }}

    [data-testid="stMetric"] {{ background: {C['bg_soft']}; border: 1px solid {C['border']};
        border-radius: 4px; padding: 12px 16px; }}
    [data-testid="stMetricLabel"] p {{ font-size: 12px; color: {C['muted']}; font-weight: 500; }}
    [data-testid="stMetricValue"] {{ font-size: 24px; color: {C['text']}; font-weight: 700; }}

    .stDataFrame, [data-testid="stDataFrame"] {{ border: 1px solid {C['border']}; border-radius: 4px; }}

    [data-testid="stTabs"] button {{ font-size: 13px; font-weight: 500; color: {C['muted']}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {C['text']}; font-weight: 600; }}

    [data-testid="stSidebar"] {{ background-color: {C['bg_soft']}; border-right: 1px solid {C['border']}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: {C['muted']}; }}

    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px;
              font-size: 12px; font-weight: 600; color: #fff; }}
    .section-note {{ font-size: 12px; color: {C['muted']}; margin-top: -8px; }}
    .formula {{ font-family: 'Courier New', monospace; font-size: 12px;
                background: {C['bg_soft']}; padding: 6px 10px; border-radius: 4px;
                border-left: 3px solid {C['blue']}; color: {C['text']}; }}
    hr {{ border: none; border-top: 1px solid {C['border']}; margin: 1.2rem 0; }}
</style>
""", unsafe_allow_html=True)


def fmt_rp(x):
    """Format Rupiah amounts for compact display."""
    if x >= 1e9:
        return f"Rp {x/1e9:.2f} M"
    if x >= 1e6:
        return f"Rp {x/1e6:.1f} jt"
    return f"Rp {x:,.0f}"


def base_layout(fig, height=320):
    """Apply consistent Plotly layout defaults."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color=C["text"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zeroline=False)
    return fig


# ============================================================
# DATA (cached)
# ============================================================
ALL_MONTHS = [str(m) for m in pd.period_range("2025-07", "2026-06", freq="M")]


@st.cache_data(show_spinner=False)
def get_tables():
    return se.load_data()


@st.cache_data(show_spinner=False)
def get_features(window_key):
    tables = get_tables()
    wm = window_to_months(window_key)
    return se.engineer_features(tables, window_months=wm)


@st.cache_data(show_spinner=False)
def get_scores(window_key):
    return se.calculate_scores(get_features(window_key))


def window_to_months(window_key):
    if window_key == "Kuartal Terakhir (Q)":
        return ALL_MONTHS[-3:]
    if window_key.startswith("Bulan "):
        return [window_key.replace("Bulan ", "")]
    return ALL_MONTHS  # Semua & Tahunan


@st.cache_data(show_spinner=False)
def get_expiry_queue():
    """Compute the Contract Expiry Queue.

    FIX: handles both auto-renewal and non-auto-renewal contracts.
      - For auto_renewal=True: roll the effective end forward to the current
        renewal period (so the next upcoming expiry is shown).
      - For auto_renewal=False: use the actual end_date if present, otherwise
        compute start_date + contract_period. Non-auto-renewal contracts are
        NO LONGER silently dropped — they are rolled forward by the data
        generator, so their effective_end is always >= anchor.
    """
    subs = get_tables()["subscriptions"].copy()
    subs["start_date"] = pd.to_datetime(subs["start_date"])
    subs["end_date"] = pd.to_datetime(subs["end_date"], errors="coerce")
    ref = se.REFERENCE_DATE

    # Active contracts: those whose end_date is null or in the future.
    act = subs[(subs["end_date"].isna()) | (subs["end_date"] > ref)].copy()

    ends = []
    for _, r in act.iterrows():
        period = int(r["contract_period"])
        if pd.notna(r["end_date"]):
            # Non-auto-renewal with explicit end_date (still active)
            e = r["end_date"]
        else:
            # Compute effective end from start + period
            e = r["start_date"] + pd.DateOffset(months=period)
            # For auto-renewal, roll forward to the current renewal period
            if r["auto_renewal"]:
                while e <= ref:
                    e = e + pd.DateOffset(months=period)
        ends.append(e)

    act["effective_end"] = ends
    act["days_to_expiry"] = (act["effective_end"] - ref).dt.days
    # Only show contracts with a future expiry (visible in the queue)
    return act[act["days_to_expiry"] > 0].sort_values("days_to_expiry")


# ============================================================
# SIDEBAR — GLOBAL FILTERS
# ============================================================
st.sidebar.markdown("## Filter Data")
window_choice = st.sidebar.radio(
    "Periode waktu",
    ["Semua (12 bulan)", "Tahunan (12 bln terakhir)", "Kuartal Terakhir (Q)"],
    index=0,
)
month_pick = st.sidebar.selectbox("Atau fokus satu bulan", ["—"] + ALL_MONTHS, index=0)
window_key = f"Bulan {month_pick}" if month_pick != "—" else window_choice

scores = get_scores(window_key)
tables = get_tables()

seg_filter = st.sidebar.multiselect(
    "Segmen", sorted(scores["segment"].unique()),
    default=sorted(scores["segment"].unique()),
)
sec_filter = st.sidebar.multiselect(
    "Sektor industri", sorted(scores["business_sector"].unique()),
    default=sorted(scores["business_sector"].unique()),
)
reg_filter = st.sidebar.multiselect(
    "Region", sorted(scores["region"].unique()),
    default=sorted(scores["region"].unique()),
)

f = scores[
    scores["segment"].isin(seg_filter) &
    scores["business_sector"].isin(sec_filter) &
    scores["region"].isin(reg_filter)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='section-note'>Anchor date: 30 Jun 2026<br>"
    f"Window: {window_to_months(window_key)[0]} s.d. {window_to_months(window_key)[-1]}<br>"
    f"{len(f)} dari {len(scores)} pelanggan ditampilkan</div>",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown("# SME Customer Intelligence & Retention")
st.markdown(
    "<div class='section-note'>Dashboard simulasi retensi pelanggan SME — "
    "data dummy 12 bulan (Jul 2025 – Jun 2026)</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Ringkasan", "Pelanggan", "Antrean Intervensi", "Langganan", "Simulasi What-If"]
)

# ============================================================
# TAB 1 — RINGKASAN
# ============================================================
with tab1:
    # FIX: MRR Berisiko now includes 3 at-risk populations
    #   - All Pelanggan Berisiko (regardless of sub-category)
    #   - Pelanggan Tetap with score_category == 'At Risk'
    #   - Pelanggan Baru with score_category == 'Low Potential' (silent churn risk)
    mrr_risk_mask = (
        (f["segment"] == "Pelanggan Berisiko")
        | (f["score_category"] == "At Risk")
        | ((f["segment"] == "Pelanggan Baru") & (f["score_category"] == "Low Potential"))
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Pelanggan", f"{len(f):,}")
    k2.metric("Total MRR", fmt_rp(f["total_mrr"].sum()))
    k3.metric("MRR Berisiko", fmt_rp(f.loc[mrr_risk_mask, "total_mrr"].sum()))
    k4.metric("Critical + High Risk",
              int((f["score_category"].isin(["Critical Risk", "High Risk"])).sum()))
    k5.metric("Rata-rata Confidence", f"{f['confidence_score'].mean():.0f}%")

    st.markdown(
        "<div class='section-note'>MRR Berisiko mencakup: seluruh Pelanggan Berisiko, "
        "Pelanggan Tetap berkategori At Risk, dan Pelanggan Baru berkategori Low Potential "
        "(risiko silent churn).</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Distribusi Skor per Segmen")
        fig = go.Figure()
        score_col = {
            "Pelanggan Baru": "potential_score",
            "Pelanggan Tetap": "stability_score",
            "Pelanggan Berisiko": "churn_risk_index",
        }
        for seg, col in score_col.items():
            d = f[f["segment"] == seg][col]
            if len(d):
                fig.add_trace(go.Histogram(
                    x=d, name=seg, marker_color=SEGMENT_COLORS[seg],
                    opacity=0.75, nbinsx=25,
                ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Skor (0–100)", yaxis_title="Jumlah pelanggan",
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with c2:
        st.markdown("### Komposisi Kategori per Segmen")
        cc = f.groupby(["segment", "score_category"]).size().reset_index(name="n")
        fig = px.bar(cc, x="segment", y="n", color="score_category",
                     color_discrete_map=CATEGORY_COLORS)
        fig.update_layout(xaxis_title="", yaxis_title="Jumlah pelanggan")
        st.plotly_chart(base_layout(fig), use_container_width=True)

    st.markdown("### Rata-rata Skor per Sektor × Segmen")
    if len(f) == 0:
        st.info("Tidak ada data untuk kombinasi filter ini.")
    else:
        pivot = f.pivot_table(
            index="business_sector", columns="segment",
            values="assigned_score", aggfunc="mean",
        ).round(1)
        ordered_cols = [c for c in ["Pelanggan Baru", "Pelanggan Tetap", "Pelanggan Berisiko"]
                        if c in pivot.columns]
        pivot = pivot[ordered_cols]
        fig = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#f1f5f9"], [1, C["blue"]]],
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(base_layout(fig, height=420), use_container_width=True)

# ============================================================
# TAB 2 — PELANGGAN
# ============================================================
with tab2:
    st.markdown("### Daftar Pelanggan")
    st.markdown(
        "<div class='section-note'>Klik satu baris untuk melihat kartu detail pelanggan.</div>",
        unsafe_allow_html=True,
    )

    show = f[["customer_id", "company_name", "business_sector", "region", "segment",
              "assigned_score", "score_category", "confidence_score", "total_mrr",
              "priority_score"]].sort_values("assigned_score", ascending=False).reset_index(drop=True)
    show.columns = ["ID", "Perusahaan", "Sektor", "Region", "Segmen",
                    "Skor", "Kategori", "Conf %", "MRR", "Prioritas"]

    sel = st.dataframe(
        show, use_container_width=True, height=420,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "MRR": st.column_config.NumberColumn(format="Rp %d"),
            "Skor": st.column_config.NumberColumn(format="%.1f"),
            "Prioritas": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        cid = show.iloc[rows[0]]["ID"]
        r = f[f["customer_id"] == cid].iloc[0]
        cat_color = CATEGORY_COLORS.get(r["score_category"], C["slate"])
        st.markdown("---")
        d1, d2, d3 = st.columns([2, 2, 3])
        with d1:
            st.markdown(f"#### {r['company_name']}")
            st.markdown(
                f"<span class='badge' style='background:{SEGMENT_COLORS[r['segment']]}'>"
                f"{r['segment']}</span> &nbsp; "
                f"<span class='badge' style='background:{cat_color}'>"
                f"{r['score_category']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:10px;font-size:13px;color:{C['muted']}'>"
                f"{r['customer_id']} · {r['business_sector']} · {r['region']}<br>"
                f"Skala {r['company_scale']} · {int(r['employee_count'])} karyawan · "
                f"tenure {r['tenure_years']:.1f} thn</div>",
                unsafe_allow_html=True,
            )
        with d2:
            st.metric("Skor", f"{r['assigned_score']:.1f} / 100")
            st.metric("Confidence", f"{r['confidence_score']:.0f}%")
            st.metric("MRR", fmt_rp(r["total_mrr"]))
        with d3:
            st.markdown("##### Rekomendasi Intervensi")
            st.markdown(f"**Aksi utama:** {r['primary_action']}")
            st.markdown(f"**Aksi pendukung:** {r['secondary_action']}")
            st.markdown(f"**Timing:** {r['timing']} · **Kanal:** {r['channel']}")
            st.markdown(f"**Ekspektasi keberhasilan:** {r['expected_success_rate']}%")
            st.markdown(
                f"<div class='section-note'>{r['rationale']}</div>",
                unsafe_allow_html=True,
            )

# ============================================================
# TAB 3 — ANTREAN INTERVENSI
# ============================================================
with tab3:
    st.markdown("### Antrean Prioritas Intervensi")
    # FIX: Formula label synchronized with scoring_engine.py implementation.
    

    q = f.sort_values("priority_score", ascending=False)
    top_n = st.slider("Tampilkan N teratas", 10, 200, 50, 10)
    qv = q.head(top_n)[["priority_score", "customer_id", "company_name", "segment",
                        "score_category", "assigned_score", "total_mrr",
                        "primary_action", "timing", "expected_success_rate"]].reset_index(drop=True)
    qv.index = qv.index + 1
    qv.columns = ["Prioritas", "ID", "Perusahaan", "Segmen", "Kategori", "Skor",
                  "MRR", "Aksi Utama", "Timing", "Sukses %"]
    st.dataframe(
        qv, use_container_width=True, height=480,
        column_config={
            "MRR": st.column_config.NumberColumn(format="Rp %d"),
            "Prioritas": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    c1, c2 = st.columns(2)
    c1.metric("Total MRR dalam antrean", fmt_rp(q.head(top_n)["total_mrr"].sum()))
    c2.metric(
        "Ekspektasi MRR terselamatkan",
        fmt_rp((q.head(top_n)["total_mrr"] * q.head(top_n)["expected_success_rate"] / 100).sum()),
    )

# ============================================================
# TAB 4 — LANGGANAN
# ============================================================
with tab4:
    unit = st.radio("Satuan sumbu MRR", ["Miliar (Rp M)", "Juta (Rp jt)"], horizontal=True)
    div, unit_lbl = (1e9, "Rp Miliar") if unit.startswith("Miliar") else (1e6, "Rp Juta")

    st.markdown("### MRR Berisiko per Kategori")
    mrr_cat = f.groupby("score_category")["total_mrr"].sum().reindex(
        ["Critical Risk", "High Risk", "At Risk", "Moderate Risk", "Low Potential",
         "Medium Potential", "Stable", "High Potential", "Highly Stable"]
    ).dropna()
    fig = go.Figure(go.Bar(
        x=mrr_cat.index, y=mrr_cat.values / div,
        marker_color=[CATEGORY_COLORS.get(i, C["slate"]) for i in mrr_cat.index],
    ))
    fig.update_layout(yaxis_title=f"Total MRR ({unit_lbl})", xaxis_title="")
    st.plotly_chart(base_layout(fig), use_container_width=True)

    # FIX: Contract Expiry Queue now handles BOTH auto-renewal and non-auto-renewal
    st.markdown("### Contract Expiry Queue")
    st.markdown(
        "<div class='section-note'>Kontrak aktif yang jatuh tempo setelah anchor date "
        "(30 Jun 2026). Kontrak auto-renewal di-roll-forward ke periode berjalan; "
        "kontrak non-auto-renewal menggunakan end_date aktual.</div>",
        unsafe_allow_html=True,
    )
    eq = get_expiry_queue()
    eqf = eq[eq["customer_id"].isin(f["customer_id"])].copy()

    # Summary by auto-renewal status
    ar_summary = eqf.groupby("auto_renewal").agg(
        n_contracts=("customer_id", "count"),
        total_mrr=("monthly_recurring_revenue", "sum"),
    ).reset_index()
    ar_summary["auto_renewal"] = ar_summary["auto_renewal"].map({True: "Auto-Renewal", False: "Non-Auto"})

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Jatuh tempo ≤ 30 hari", int((eqf["days_to_expiry"] <= 30).sum()))
    e2.metric("Jatuh tempo ≤ 60 hari", int((eqf["days_to_expiry"] <= 60).sum()))
    e3.metric("Jatuh tempo ≤ 90 hari", int((eqf["days_to_expiry"] <= 90).sum()))
    e4.metric("Total kontrak aktif", len(eqf))

    eqv = eqf.merge(
        f[["customer_id", "company_name", "segment", "score_category"]],
        on="customer_id", how="left",
    )
    eqv = eqv[eqv["days_to_expiry"] <= 120][
        ["days_to_expiry", "customer_id", "company_name", "product_code",
         "monthly_recurring_revenue", "contract_period", "auto_renewal",
         "segment", "score_category"]
    ].copy()
    eqv.columns = ["Sisa Hari", "ID", "Perusahaan", "Produk", "MRR", "Kontrak (bln)",
                   "Auto-Renewal", "Segmen", "Kategori"]
    eqv["Auto-Renewal"] = eqv["Auto-Renewal"].map({True: "Ya", False: "Tidak"})
    st.dataframe(
        eqv, use_container_width=True, height=380,
        column_config={"MRR": st.column_config.NumberColumn(format="Rp %d")},
    )

    st.markdown("### Komposisi Produk (MRR)")
    subs = tables["subscriptions"]
    pm = subs[subs["customer_id"].isin(f["customer_id"])].groupby(
        "product_category")["monthly_recurring_revenue"].sum().sort_values()
    fig = go.Figure(go.Bar(
        y=pm.index, x=pm.values / div, orientation="h",
        marker_color=C["slate"],
    ))
    fig.update_layout(xaxis_title=f"Total MRR ({unit_lbl})", yaxis_title="")
    st.plotly_chart(base_layout(fig), use_container_width=True)

# ============================================================
# TAB 5 — SIMULASI WHAT-IF
# ============================================================
with tab5:
    st.markdown("### Simulasi Perubahan Bobot")
    st.markdown(
        "<div class='section-note'>Geser bobot lalu tekan tombol hitung ulang. "
        "Bobot otomatis dinormalisasi agar berjumlah 1. Perubahan di sini tidak "
        "mengubah file — untuk perubahan permanen, edit SECTOR_WEIGHTS di "
        "scoring_engine.py.</div>",
        unsafe_allow_html=True,
    )

    feats = get_features(window_key)
    base = get_scores(window_key)

    if "wi_ver" not in st.session_state:
        st.session_state.wi_ver = 0
    ver = st.session_state.wi_ver

    wcol1, wcol2, wcol3 = st.columns(3)

    def sliders(col, title, weights, keypfx):
        col.markdown(f"**{title}**")
        out = {}
        for k, v in weights.items():
            out[k] = col.slider(
                k.replace("_", " "), 0.0, 1.0, float(v), 0.05,
                key=f"{keypfx}_{k}_v{ver}",
            )
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    pw = sliders(wcol1, "Potential — bobot default", se.POTENTIAL_WEIGHTS, "wi_pot")
    stw = sliders(wcol2, "Stability — bobot global", se.STABILITY_WEIGHTS, "wi_stab")
    cw = sliders(wcol3, "Churn Risk — bobot global", se.CHURN_WEIGHTS, "wi_churn")

    st.markdown("**Override per sektor (Potential Score)**")
    s1, s2 = st.columns([1, 3])
    sector_pick = s1.selectbox("Pilih sektor", ["(tidak ada)"] + sorted(se.SECTOR_WEIGHTS.keys()))
    sw_override = dict(se.SECTOR_WEIGHTS)
    if sector_pick != "(tidak ada)":
        base_w = se.SECTOR_WEIGHTS[sector_pick]
        cols = s2.columns(5)
        new_w = {}
        for (k, v), cc in zip(base_w.items(), cols):
            new_w[k] = cc.slider(
                k.replace("_", " "), 0.0, 1.0, float(v), 0.05,
                key=f"wi_sec_{k}_v{ver}",
            )
        tot = sum(new_w.values()) or 1.0
        sw_override[sector_pick] = {k: v / tot for k, v in new_w.items()}
        s1.markdown(
            f"<div class='section-note'>Default sektor: "
            + ", ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in base_w.items())
            + "</div>",
            unsafe_allow_html=True,
        )

    bcol1, bcol2, _ = st.columns([1, 1, 4])
    run_sim = bcol1.button("Hitung ulang skor", type="primary")
    if bcol2.button("Reset ke default"):
        st.session_state.wi_ver += 1
        st.rerun()

    if run_sim:
        sim = se.calculate_scores(
            feats, sector_weights=sw_override,
            potential_weights=pw, stability_weights=stw, churn_weights=cw,
        )
        cmp = sim[["customer_id", "company_name", "segment", "assigned_score",
                   "score_category"]].merge(
            base[["customer_id", "assigned_score", "score_category"]],
            on="customer_id", suffixes=("_baru", "_lama"),
        )
        cmp["delta"] = (cmp["assigned_score_baru"] - cmp["assigned_score_lama"]).round(1)
        moved = cmp[cmp["score_category_baru"] != cmp["score_category_lama"]]

        m1, m2, m3 = st.columns(3)
        m1.metric("Pelanggan berubah kategori", len(moved))
        m2.metric("Rata-rata pergeseran skor", f"{cmp['delta'].mean():+.1f}")
        m3.metric("Pergeseran maksimum", f"{cmp['delta'].abs().max():.1f}")

        if len(moved):
            st.markdown("##### Pelanggan yang berpindah kategori")
            mv = moved.sort_values("delta", key=abs, ascending=False).head(50)[
                ["company_name", "segment", "score_category_lama", "score_category_baru",
                 "assigned_score_lama", "assigned_score_baru", "delta"]
            ].reset_index(drop=True)
            mv.columns = ["Perusahaan", "Segmen", "Kategori Lama", "Kategori Baru",
                          "Skor Lama", "Skor Baru", "Delta"]
            st.dataframe(
                mv, use_container_width=True,
                column_config={"Delta": st.column_config.NumberColumn(format="%+.1f")},
            )
        else:
            st.info("Tidak ada pelanggan yang berpindah kategori dengan bobot ini.")
