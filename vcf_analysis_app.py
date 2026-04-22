import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VCF Variant Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light/white theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
        color: #1A1A2E;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #DEE2E6;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 10px;
        padding: 15px;
        border-left: 3px solid #00ADB5;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6C757D;
        border-radius: 6px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
        font-weight: 600;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #DEE2E6;
        border-radius: 8px;
    }

    /* Section headers */
    .section-header {
        color: #00ADB5;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #DEE2E6;
    }

    /* Info boxes */
    .info-box {
        background-color: #F0FAFA;
        border: 1px solid #B2E0E3;
        border-left: 4px solid #00ADB5;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #1A1A2E;
    }

    /* Warning box */
    .warning-box {
        background-color: #FFFBF0;
        border: 1px solid #FFD991;
        border-left: 4px solid #F0A500;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #1A1A2E;
    }

    /* Success box */
    .success-box {
        background-color: #F0FFF4;
        border: 1px solid #A3D9B1;
        border-left: 4px solid #28A745;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #1A1A2E;
    }

    /* Buttons */
    .stButton > button {
        background-color: #00ADB5;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #008C94;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,173,181,0.3);
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #F8F9FA;
        border: 2px dashed #DEE2E6;
        border-radius: 10px;
        padding: 20px;
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #00ADB5;
    }

    /* Horizontal rule */
    hr {
        border-color: #DEE2E6;
    }

    /* Titles */
    h1 { color: #1A1A2E; }
    h2 { color: #2D2D44; }
    h3 { color: #3D3D5C; }

    /* Plot backgrounds */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #FFFFFF;
        color: #28A745;
        border: 1px solid #28A745;
        border-radius: 6px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background-color: #28A745;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SHARED PLOT COLOURS
# ─────────────────────────────────────────────
BG        = "#FFFFFF"
PLOT_BG   = "#FFFFFF"
FONT_CLR  = "#1A1A2E"
GRID_CLR  = "#E9ECEF"
BORDER_CLR= "#DEE2E6"
LEGEND_BG = "#F8F9FA"

def _base_layout(**overrides):
    """Return a consistent white-theme layout dict for every Plotly figure."""
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER_CLR,
                   tickcolor=BORDER_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER_CLR,
                   tickcolor=BORDER_CLR, zerolinecolor=GRID_CLR),
        legend=dict(bgcolor=LEGEND_BG, bordercolor=BORDER_CLR, borderwidth=1),
    )
    base.update(overrides)
    return base


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def load_vcf(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read().decode("utf-8", errors="replace")
    lines = content.splitlines()

    data_lines = []
    column_header = None

    for line in lines:
        if line.startswith("##"):
            continue
        elif line.startswith("#CHROM"):
            column_header = line.lstrip("#").strip().split("\t")
        elif line.strip():
            data_lines.append(line.strip().split("\t"))

    if not data_lines:
        raise ValueError("No variant data found in the VCF file.")

    if column_header is None:
        column_header = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

    num_cols = max(len(row) for row in data_lines)
    if len(column_header) < num_cols:
        column_header += [f"COL_{i}" for i in range(len(column_header), num_cols)]

    df = pd.DataFrame(data_lines, columns=column_header[:num_cols])

    required = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    for col in required:
        if col not in df.columns:
            df[col] = "."

    df["POS"]  = pd.to_numeric(df["POS"], errors="coerce").fillna(0).astype(int)
    df["QUAL"] = pd.to_numeric(df["QUAL"].replace(".", np.nan), errors="coerce")

    def extract_dp(info_str):
        if pd.isna(info_str) or info_str == ".":
            return np.nan
        for token in str(info_str).split(";"):
            if token.startswith("DP="):
                try:
                    return float(token.split("=")[1])
                except (ValueError, IndexError):
                    return np.nan
        return np.nan

    df["DP"]      = df["INFO"].apply(extract_dp)
    df["REF_len"] = df["REF"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df["ALT_len"] = df["ALT"].apply(lambda x: len(str(x).split(",")[0]) if pd.notna(x) else 0)

    def classify_variant(row):
        ref = str(row["REF"]).strip()
        alt = str(row["ALT"]).strip().split(",")[0]
        if len(ref) == 1 and len(alt) == 1 and ref != "." and alt != ".":
            return "SNP"
        return "INDEL"

    df["Variant_Type"] = df.apply(classify_variant, axis=1)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[["POS", "QUAL", "DP", "REF_len", "ALT_len"]].copy()
    for col in features.columns:
        median_val = features[col].median()
        features[col] = features[col].fillna(0 if pd.isna(median_val) else median_val)
    for col in ["QUAL", "DP"]:
        features[col] = features[col].clip(upper=features[col].quantile(0.99))
    return features


def generate_synthetic_training_data(n_samples: int = 2000) -> tuple:
    np.random.seed(42)
    n_benign = int(n_samples * 0.6)
    benign = {
        "POS":     np.random.randint(1_000_000, 250_000_000, n_benign),
        "QUAL":    np.random.normal(120, 35, n_benign).clip(10, 500),
        "DP":      np.random.normal(55, 18, n_benign).clip(5, 200),
        "REF_len": np.random.choice([1,1,1,2,3], n_benign, p=[0.6,0.2,0.1,0.05,0.05]),
        "ALT_len": np.random.choice([1,1,1,2,3], n_benign, p=[0.6,0.2,0.1,0.05,0.05]),
    }
    n_path = n_samples - n_benign
    pathogenic = {
        "POS":     np.random.randint(1_000_000, 250_000_000, n_path),
        "QUAL":    np.random.normal(65, 28, n_path).clip(5, 300),
        "DP":      np.random.normal(28, 15, n_path).clip(2, 150),
        "REF_len": np.random.choice([1,2,3,4,5], n_path, p=[0.3,0.25,0.2,0.15,0.1]),
        "ALT_len": np.random.choice([1,2,3,5,8], n_path, p=[0.3,0.2,0.2,0.2,0.1]),
    }
    X = pd.concat([pd.DataFrame(benign), pd.DataFrame(pathogenic)], ignore_index=True)
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_path, dtype=int)])
    X = (X + np.random.normal(0, 0.5, X.shape)).clip(lower=0)
    return X, y


def train_model() -> tuple:
    X, y = generate_synthetic_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), X_test, y_test, y_pred


def predict_variants(pipeline, features_df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(features_df)


# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────

def plot_variants_per_chromosome(df: pd.DataFrame):
    counts = df["CHROM"].value_counts().reset_index()
    counts.columns = ["Chromosome", "Count"]
    counts["sort_key"] = counts["Chromosome"].apply(
        lambda x: int(x.replace("chr","").replace("X","23").replace("Y","24")
                       .replace("M","25").replace("MT","25"))
        if x.replace("chr","").replace("X","23").replace("Y","24")
            .replace("M","25").replace("MT","25").isdigit() else 99
    )
    counts = counts.sort_values("sort_key").drop("sort_key", axis=1)

    fig = px.bar(counts, x="Chromosome", y="Count",
                 title="<b>Variant Distribution Across Chromosomes</b>",
                 color="Count", color_continuous_scale=["#00ADB5", "#7C3AED"])
    fig.update_layout(**_base_layout(
        coloraxis_showscale=False, title_font_size=14,
        xaxis_title="Chromosome", yaxis_title="Variant Count"
    ))
    fig.update_traces(marker_line_color=BORDER_CLR, marker_line_width=0.5)
    return fig


def plot_variant_type_pie(df: pd.DataFrame):
    type_counts = df["Variant_Type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    fig = px.pie(type_counts, names="Type", values="Count",
                 title="<b>SNP vs INDEL Distribution</b>",
                 color_discrete_sequence=["#00ADB5", "#7C3AED"], hole=0.55)
    fig.update_layout(**_base_layout(title_font_size=14))
    fig.update_traces(
        textposition="outside", textinfo="percent+label",
        marker=dict(line=dict(color="#FFFFFF", width=2))
    )
    return fig


def plot_position_histogram(df: pd.DataFrame):
    fig = px.histogram(df, x="POS", nbins=50,
                       title="<b>Variant Position Distribution</b>",
                       color_discrete_sequence=["#00ADB5"])
    fig.update_layout(**_base_layout(
        title_font_size=14, xaxis_title="Genomic Position",
        yaxis_title="Frequency", bargap=0.05
    ))
    fig.update_traces(marker_line_color=BORDER_CLR, marker_line_width=0.3)
    return fig


def plot_qual_boxplot(df: pd.DataFrame):
    qual_data = df["QUAL"].dropna()
    if qual_data.empty:
        return None

    type_styles = {
        "SNP":   {"color": "#7C3AED", "fill": "rgba(124,58,237,0.12)"},
        "INDEL": {"color": "#F59E0B", "fill": "rgba(245,158,11,0.12)"},
    }
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=qual_data, name="QUAL Score",
        marker_color="#00ADB5", line_color="#00ADB5",
        fillcolor="rgba(0,173,181,0.12)",
        boxmean="sd", boxpoints="outliers",
        marker=dict(size=3, opacity=0.5)
    ))
    for vtype, style in type_styles.items():
        subset = df[df["Variant_Type"] == vtype]["QUAL"].dropna()
        if not subset.empty:
            fig.add_trace(go.Box(
                y=subset, name=vtype,
                marker_color=style["color"], line_color=style["color"],
                fillcolor=style["fill"],
                boxmean="sd", boxpoints="outliers",
                marker=dict(size=3, opacity=0.5)
            ))
    fig.update_layout(**_base_layout(
        title="<b>Quality Score Distribution by Variant Type</b>",
        title_font_size=14, yaxis_title="QUAL Score", showlegend=True
    ))
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    labels = ["Benign", "Pathogenic"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#EBF9FA"], [0.5, "#00ADB5"], [1, "#7C3AED"]],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18, color="white"), showscale=True
    ))
    fig.update_layout(**_base_layout(
        title="<b>Confusion Matrix – Test Set</b>",
        title_font_size=14,
        xaxis_title="Predicted Label", yaxis_title="True Label"
    ))
    fig.update_xaxes(side="bottom")
    return fig


def plot_feature_importance(pipeline, feature_names: list):
    rf = pipeline.named_steps["classifier"]
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 title="<b>Feature Importance – Random Forest</b>",
                 color="Importance", color_continuous_scale=["#E9ECEF", "#00ADB5"])
    fig.update_layout(**_base_layout(title_font_size=14, coloraxis_showscale=False))
    fig.update_traces(marker_line_color=BORDER_CLR, marker_line_width=0.5)
    return fig


def plot_impact_distribution(df: pd.DataFrame):
    if "Impact" not in df.columns:
        return None
    counts = df["Impact"].value_counts().reset_index()
    counts.columns = ["Impact", "Count"]
    fig = px.bar(counts, x="Impact", y="Count",
                 title="<b>Predicted Impact Distribution</b>",
                 color="Impact",
                 color_discrete_map={"Benign": "#10B981", "Pathogenic": "#EF4444"},
                 text="Count")
    fig.update_layout(**_base_layout(title_font_size=14, showlegend=False))
    fig.update_traces(textposition="outside",
                      marker_line_color=BORDER_CLR, marker_line_width=0.5)
    return fig


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

for key, default in [
    ("vcf_df", None), ("pipeline", None), ("model_accuracy", None),
    ("confusion_mat", None), ("predictions_done", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size: 2.5rem;'>🧬</div>
        <div style='color: #00ADB5; font-weight: 700; font-size: 1rem; letter-spacing: 0.1em;'>VCF ANALYSIS</div>
        <div style='color: #6C757D; font-size: 0.75rem; letter-spacing: 0.05em;'>PLATFORM v2.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📁 Data Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload VCF File", type=["vcf", "txt"],
        help="Upload a standard VCF (Variant Call Format) file. Supports VCF 4.x format."
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>File loaded</strong><br>
            <span style='color:#6C757D; font-size:0.85rem;'>{uploaded_file.name}</span><br>
            <span style='color:#6C757D; font-size:0.85rem;'>{uploaded_file.size/1024:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color: #6C757D; font-size: 0.8rem; line-height: 1.6;'>
        <strong style='color: #1A1A2E;'>About</strong><br>
        AI-assisted VCF variant analysis tool for MSc Bioinformatics research.<br><br>
        <strong style='color: #1A1A2E;'>Features</strong><br>
        • VCF parsing & annotation<br>
        • Interactive visualizations<br>
        • ML impact prediction<br>
        • CSV export
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown("""
<div style='padding: 1rem 0 0.5rem 0;'>
    <h1 style='color: #00ADB5; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem;'>
        🧬 AI-Assisted VCF Variant Analysis & Impact Prediction
    </h1>
    <p style='color: #6C757D; font-size: 0.9rem; margin: 0;'>
        MSc Bioinformatics · Variant Call Format Processing 
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load & parse VCF ──
if uploaded_file and st.session_state.vcf_df is None:
    try:
        with st.spinner("🔬 Parsing VCF file..."):
            st.session_state.vcf_df = load_vcf(uploaded_file)
            st.session_state.predictions_done = False
    except Exception as e:
        st.error(f"❌ **VCF Parse Error:** {str(e)}")
        st.stop()

elif uploaded_file and st.session_state.vcf_df is not None:
    if st.session_state.get("last_filename") != uploaded_file.name:
        try:
            with st.spinner("🔬 Parsing VCF file..."):
                st.session_state.vcf_df = load_vcf(uploaded_file)
                st.session_state.predictions_done = False
                st.session_state.last_filename = uploaded_file.name
        except Exception as e:
            st.error(f"❌ **VCF Parse Error:** {str(e)}")
            st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_overview, tab_variants, tab_viz = st.tabs([
    "  📊  Overview  ",
    "  🔬  Variants  ",
    "  📈  Visualization",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab_overview:
    if st.session_state.vcf_df is None:
        st.markdown("""
        <div class="info-box" style='text-align: center; padding: 40px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📂</div>
            <h3 style='color: #1A1A2E; margin-bottom: 0.5rem;'>No VCF File Loaded</h3>
            <p style='color: #6C757D;'>Upload a VCF file using the sidebar to begin analysis.</p>
            <p style='color: #6C757D; font-size: 0.85rem;'>Supports standard VCF 4.0, 4.1, 4.2, 4.3 formats</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Expected VCF Format")
        st.code("""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
#CHROM  POS      ID        REF  ALT  QUAL  FILTER  INFO
chr1    925952   rs3131972  G    A    50    PASS    DP=35
chr1    1047122  .          C    T    29.5  PASS    DP=12
chr2    21234567 rs234567   AT   A    100   PASS    DP=78""", language="text")

    else:
        df = st.session_state.vcf_df

        total       = len(df)
        snp_count   = (df["Variant_Type"] == "SNP").sum()
        indel_count = (df["Variant_Type"] == "INDEL").sum()
        snp_pct     = snp_count / total * 100 if total > 0 else 0
        indel_pct   = indel_count / total * 100 if total > 0 else 0
        mean_qual   = df["QUAL"].mean()
        mean_dp     = df["DP"].mean()
        n_chroms    = df["CHROM"].nunique()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Total Variants", f"{total:,}")
        with col2: st.metric("SNP Count", f"{snp_count:,}", f"{snp_pct:.1f}%")
        with col3: st.metric("INDEL Count", f"{indel_count:,}", f"{indel_pct:.1f}%")
        with col4: st.metric("Mean QUAL", f"{mean_qual:.1f}" if not pd.isna(mean_qual) else "N/A")
        with col5: st.metric("Chromosomes", f"{n_chroms}")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_variants_per_chromosome(df), use_container_width=True,
                            config={"displayModeBar": False})
        with col_b:
            st.plotly_chart(plot_variant_type_pie(df), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("---")
        st.markdown('<div class="section-header">📋 File Summary</div>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Property": [
                "File Name", "Total Variants", "Chromosomes Covered",
                "SNP Count", "INDEL Count", "SNP Percentage", "INDEL Percentage",
                "Mean Quality Score", "Mean Read Depth (DP)",
                "Columns Detected", "Variants with DP", "Variants Missing QUAL"
            ],
            "Value": [
                uploaded_file.name if uploaded_file else "N/A",
                f"{total:,}",
                ", ".join(sorted(df["CHROM"].unique()[:10].tolist())) + ("..." if n_chroms > 10 else ""),
                f"{snp_count:,}", f"{indel_count:,}",
                f"{snp_pct:.2f}%", f"{indel_pct:.2f}%",
                f"{mean_qual:.2f}" if not pd.isna(mean_qual) else "Not available",
                f"{mean_dp:.1f}"  if not pd.isna(mean_dp)   else "Not in INFO",
                str(len(df.columns)),
                f"{df['DP'].notna().sum():,} ({df['DP'].notna().mean()*100:.1f}%)",
                f"{df['QUAL'].isna().sum():,}"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Property": st.column_config.TextColumn("Property", width=200),
                         "Value":    st.column_config.TextColumn("Value")
                     })


# ══════════════════════════════════════════════
# TAB 2 — VARIANTS TABLE
# ══════════════════════════════════════════════

with tab_variants:
    if st.session_state.vcf_df is None:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No data loaded.</strong> Please upload a VCF file in the sidebar.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.vcf_df
        st.markdown('<div class="section-header">🔍 Filter Variants</div>', unsafe_allow_html=True)

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            selected_chrom = st.selectbox("Chromosome",
                ["All"] + sorted(df["CHROM"].unique().tolist()), index=0)
        with fcol2:
            selected_type = st.selectbox("Variant Type", ["All","SNP","INDEL"], index=0)
        with fcol3:
            selected_filter = st.selectbox("Filter Status",
                ["All"] + sorted(df["FILTER"].unique().tolist()), index=0)

        pos_min, pos_max = int(df["POS"].min()), int(df["POS"].max())
        pos_range = st.slider("Position Range", pos_min, pos_max, (pos_min, pos_max),
                              format="%d") if pos_min < pos_max else (pos_min, pos_max)

        qual_threshold = 0.0
        if df["QUAL"].notna().any():
            qual_threshold = st.slider("Minimum QUAL Score", 0.0,
                                       float(df["QUAL"].max()), 0.0, step=1.0)

        fdf = df.copy()
        if selected_chrom  != "All": fdf = fdf[fdf["CHROM"]        == selected_chrom]
        if selected_type   != "All": fdf = fdf[fdf["Variant_Type"] == selected_type]
        if selected_filter != "All": fdf = fdf[fdf["FILTER"]       == selected_filter]
        fdf = fdf[(fdf["POS"] >= pos_range[0]) & (fdf["POS"] <= pos_range[1])]
        if qual_threshold > 0:
            fdf = fdf[fdf["QUAL"].isna() | (fdf["QUAL"] >= qual_threshold)]

        st.markdown(f"""
        <div class="info-box">
            📊 Showing <strong style='color:#00ADB5;'>{len(fdf):,}</strong> of
            <strong>{len(df):,}</strong> variants ({len(fdf)/len(df)*100:.1f}% of total)
        </div>
        """, unsafe_allow_html=True)

        display_cols = ["CHROM","POS","ID","REF","ALT","QUAL","FILTER",
                        "Variant_Type","DP","REF_len","ALT_len"]
        if "Impact" in fdf.columns:
            display_cols.append("Impact")
        available_cols = [c for c in display_cols if c in fdf.columns]

        st.dataframe(fdf[available_cols].reset_index(drop=True),
                     use_container_width=True, height=480,
                     column_config={
                         "CHROM":        st.column_config.TextColumn("Chromosome", width=100),
                         "POS":          st.column_config.NumberColumn("Position", format="%d"),
                         "QUAL":         st.column_config.NumberColumn("Quality",  format="%.1f"),
                         "DP":           st.column_config.NumberColumn("Depth",    format="%.0f"),
                         "Variant_Type": st.column_config.TextColumn("Type",   width=80),
                         "Impact":       st.column_config.TextColumn("Impact", width=100),
                     })

        st.markdown("---")
        st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button("⬇️ Download Filtered Variants (CSV)",
                               data=fdf[available_cols].to_csv(index=False),
                               file_name="filtered_variants.csv", mime="text/csv")
        with dcol2:
            all_cols = available_cols if all(c in df.columns for c in available_cols) else list(df.columns)
            st.download_button("⬇️ Download All Variants (CSV)",
                               data=df[all_cols].to_csv(index=False),
                               file_name="all_variants.csv", mime="text/csv")


# ══════════════════════════════════════════════
# TAB 3 — VISUALIZATION
# ══════════════════════════════════════════════

with tab_viz:
    if st.session_state.vcf_df is None:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No data loaded.</strong> Please upload a VCF file in the sidebar.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.vcf_df
        st.markdown('<div class="section-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.plotly_chart(plot_variants_per_chromosome(df), use_container_width=True)
        with vcol2:
            st.plotly_chart(plot_variant_type_pie(df), use_container_width=True)

        vcol3, vcol4 = st.columns(2)
        with vcol3:
            st.plotly_chart(plot_position_histogram(df), use_container_width=True)
        with vcol4:
            qual_fig = plot_qual_boxplot(df)
            if qual_fig:
                st.plotly_chart(qual_fig, use_container_width=True)
            else:
                st.markdown('<div class="info-box">ℹ️ Quality scores not available in this VCF.</div>',
                            unsafe_allow_html=True)

        if df["DP"].notna().sum() > 0:
            st.markdown("---")
            dp_fig = px.histogram(
                df.dropna(subset=["DP"]), x="DP", color="Variant_Type", nbins=40,
                title="<b>Read Depth (DP) Distribution by Variant Type</b>",
                color_discrete_map={"SNP": "#00ADB5", "INDEL": "#7C3AED"},
                barmode="overlay"
            )
            dp_fig.update_layout(**_base_layout())
            dp_fig.update_traces(opacity=0.75)
            st.plotly_chart(dp_fig, use_container_width=True)

        if df["QUAL"].notna().any() and df["DP"].notna().any():
            scatter_df = df[df["QUAL"].notna() & df["DP"].notna()].copy()
            if len(scatter_df) > 0:
                scatter_fig = px.scatter(
                    scatter_df, x="DP", y="QUAL", color="Variant_Type",
                    title="<b>Quality Score vs Read Depth</b>",
                    color_discrete_map={"SNP": "#00ADB5", "INDEL": "#F59E0B"},
                    opacity=0.6,
                    hover_data=["CHROM","POS","REF","ALT"]
                )
                scatter_fig.update_layout(**_base_layout(
                    xaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER_CLR, title="Read Depth (DP)"),
                    yaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER_CLR, title="Quality Score")
                ))
                st.plotly_chart(scatter_fig, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #ADB5BD; font-size: 0.8rem; padding: 10px 0;'>
    🧬 VCF Variant Analysis Platform · MSc Bioinformatics · Built with Streamlit & scikit-learn<br>
    <span style='color: #CED4DA;'>For research use only. Not intended for clinical diagnostic purposes.</span>
</div>
""", unsafe_allow_html=True)
