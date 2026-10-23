"""NetSentinel dashboard for the project's bundled NSL-KDD experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="NetSentinel Dashboard", page_icon="🛡️", layout="wide")

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "KDDTrain+.txt"
TEST_FILE = ROOT / "data" / "KDDTest+.txt"

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
DATA_COLUMNS = FEATURES + ["attack", "difficulty"]

# Values are the executed notebook's held-out evaluation results.
MODEL_METRICS = pd.DataFrame(
    [
        ("Decision Tree", .8003, .7991),
        ("Random Forest", .7690, .7657),
        ("Logistic Regression", .7520, .7500),
        ("Linear SVM", .7443, .7416),
        ("Naive Bayes", .3549, .2308),
    ], columns=["Model", "Accuracy", "Weighted F1"],
)
# Rows are actual normal/attack; columns are predicted normal/attack.
CONFUSION = [[9431, 280], [4221, 8612]]
FEATURE_IMPORTANCE = pd.DataFrame(
    [
        ("src_bytes", .7272), ("dst_host_same_srv_rate", .0288),
        ("wrong_fragment", .0260), ("hot", .0211),
        ("dst_host_srv_diff_host_rate", .0126), ("dst_bytes", .0122),
        ("dst_host_same_src_port_rate", .0120), ("dst_host_count", .0033),
        ("duration", .0027), ("dst_host_rerror_rate", .0026),
    ], columns=["Feature", "Importance"],
)

BG = "#0A101B"
PANEL = "#111B29"
GRID = "#283447"
TEXT = "#E4EAF2"
MUTED = "#91A0B4"
TEAL = "#55C2B1"
BLUE = "#7897D4"
AMBER = "#D9A85E"
RED = "#D97979"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family:Manrope,sans-serif; }}
    .stApp {{ background:radial-gradient(ellipse at 80% -25%,#1b2b41 0%,transparent 42%),{BG}; color:{TEXT}; }}
    .block-container {{ max-width:1500px; padding:.85rem 1.45rem 1.25rem; }}
    [data-testid="stSidebar"] {{ background:#0D1623; border-right:1px solid #263244; }}
    [data-testid="stSidebar"] > div {{ padding-top:1rem; }}
    h1,h2,h3,h4 {{ color:#F1F4F8; letter-spacing:-.025em; }}
    h1 {{ margin:.12rem 0 .1rem; }}
    h2 {{ font-size:1.12rem; margin:.05rem 0 .08rem; }}
    p, label {{ color:#BBC5D2; }}
    div[data-testid="stMetric"] {{ background:linear-gradient(145deg,#142133,#101926); border:1px solid #29374A; border-radius:12px; padding:8px 13px; }}
    div[data-testid="stMetricLabel"] {{ color:#9AA9BA; font-size:.76rem; }}
    div[data-testid="stMetricValue"] {{ color:{TEAL}; font-weight:750; font-size:1.65rem; }}
    div[data-testid="stTabs"] button {{ font-weight:700; }}
    div[data-testid="stDataFrame"] {{ border:1px solid #28364A; border-radius:10px; overflow:hidden; }}
    .eyebrow {{ color:{TEAL}; font:500 .69rem 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }}
    .subtitle {{ color:{MUTED}; font-size:.84rem; margin-bottom:.35rem; }}
    .card {{ background:linear-gradient(145deg,#142133,#101926); border:1px solid #29374A; border-radius:12px; padding:10px 13px; min-height:70px; }}
    .card-label {{ color:#9AA9BA; font-size:.76rem; }}
    .card-value {{ font:600 1.35rem 'DM Mono',monospace; color:{TEXT}; margin-top:4px; }}
    .section-line {{ height:1px; background:linear-gradient(90deg,#344359,transparent); margin:.6rem 0 .8rem; }}
    hr {{ border-color:#28364A; margin:.6rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the project datasets…")
def load_data(path: str, modified: float):
    del modified
    frame = pd.read_csv(path, header=None, names=DATA_COLUMNS, skipinitialspace=True)
    frame["attack"] = frame["attack"].astype(str).str.strip().str.rstrip(".")
    return frame


def theme(ax, grid_axis="y"):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9, length=0, pad=6)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=.65, alpha=.8)
    ax.set_axisbelow(True)


def render(fig):
    fig.tight_layout(pad=.45)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def heading(kicker, title, subtitle=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    st.error("The project's NSL-KDD training and test files are missing from the data folder.")
    st.stop()

train = load_data(str(TRAIN_FILE), TRAIN_FILE.stat().st_mtime)
test = load_data(str(TEST_FILE), TEST_FILE.stat().st_mtime)

st.markdown('<div class="eyebrow">NETWORK ANALYTICS &nbsp; / &nbsp; NSL-KDD</div>', unsafe_allow_html=True)
st.title("NetSentinel")
st.markdown('<div class="subtitle">Traffic overview · model training · held-out testing · classification results</div>', unsafe_allow_html=True)

page = st.radio(
    "Dashboard section",
    ["Dashboard", "Dataset", "Training", "Testing"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

train_normal = int(train["attack"].eq("normal").sum())
train_attack = len(train) - train_normal
test_normal = int(test["attack"].eq("normal").sum())
test_attack = len(test) - test_normal


if page == "Dashboard":
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Training records", f"{len(train):,}")
    k2.metric("Test records", f"{len(test):,}")
    k3.metric("Training attacks", f"{train_attack:,}", f"{train_attack / len(train):.1%} of train")
    k4.metric("Test attacks", f"{test_attack:,}", f"{test_attack / len(test):.1%} of test")
    k5.metric("Model features", "41")

    st.markdown("")
    left, right = st.columns([1.05, .95], gap="small")
    with left:
        heading("MODEL EVALUATION", "Five model comparison", "Accuracy and weighted F1 · notebook test results")
        # The donut is rendered close to a square by Matplotlib's equal aspect.
        # Match its on-screen height so this comparison panel does not look empty.
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        metrics = MODEL_METRICS.sort_values("Weighted F1")
        y = list(range(len(metrics)))
        ax.barh([v + .16 for v in y], metrics["Accuracy"], height=.29, color=TEAL, label="Accuracy")
        ax.barh([v - .16 for v in y], metrics["Weighted F1"], height=.29, color=BLUE, label="Weighted F1")
        ax.set_yticks(y, metrics["Model"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
        ax.set_xlabel("Score")
        theme(ax, "x")
        ax.legend(frameon=False, ncol=2, loc="lower right", labelcolor=TEXT, fontsize=8)
        render(fig)
        st.caption("Decision Tree ranks highest: 80.03% accuracy and 79.91% weighted F1.")

    with right:
        heading("DATASET OVERVIEW", "Normal and attack traffic", "Bundled NSL-KDD train and test splits")
        fig, ax = plt.subplots(figsize=(5.2, 2.15))
        values = [train_normal + test_normal, train_attack + test_attack]
        wedges, _ = ax.pie(
            values, startangle=90, counterclock=False, colors=[TEAL, AMBER],
            wedgeprops={"width": .31, "edgecolor": PANEL, "linewidth": 2},
        )
        ax.text(0, .07, f"{sum(values):,}", ha="center", va="center", color=TEXT, fontsize=17, fontweight="bold")
        ax.text(0, -.15, "connections", ha="center", va="center", color=MUTED, fontsize=8)
        ax.legend(wedges, [f"Normal · {values[0]:,}", f"Attack · {values[1]:,}"], loc="lower center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, labelcolor=TEXT, fontsize=8)
        ax.set_facecolor(PANEL)
        fig.set_facecolor(PANEL)
        render(fig)

    left, right = st.columns([.95, 1.05], gap="small")
    with left:
        heading("TEST RESULTS", "Decision Tree confusion matrix", "Rows are actual classes; columns are predicted classes")
        fig, ax = plt.subplots(figsize=(5.0, 2.35))
        sns.heatmap(
            CONFUSION, annot=True, fmt=",d", cmap=sns.color_palette(["#172334", TEAL], as_cmap=True),
            cbar=False, linewidths=2, linecolor=BG, square=False,
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
            annot_kws={"color": "#F1F4F8", "size": 13, "weight": "bold"}, ax=ax,
        )
        ax.set_xlabel("Predicted", color=TEXT, labelpad=4)
        ax.set_ylabel("Actual", color=TEXT, labelpad=4)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_facecolor(PANEL)
        ax.figure.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, length=0)
        render(fig)
    with right:
        heading("DECISION TREE", "Top feature importance", "Relative importance from the fitted tree")
        ranked = FEATURE_IMPORTANCE.sort_values("Importance")
        fig, ax = plt.subplots(figsize=(6.1, 2.65))
        bars = ax.barh(ranked["Feature"], ranked["Importance"], color=[BLUE] * (len(ranked)-1) + [TEAL], height=.68)
        ax.set_xlim(0, .82)
        ax.set_xlabel("Importance")
        theme(ax, "x")
        ax.tick_params(axis="y", labelsize=8)
        ax.text(.79, bars[-1].get_y() + bars[-1].get_height()/2, "72.72%", va="center", ha="right", color=BG, fontsize=8, fontweight="bold")
        render(fig)


elif page == "Dataset":
    heading("DATASET", "Explore the bundled records", "Choose the project's training or test split and inspect the original NSL-KDD fields.")
    selected = st.radio("Split", ["Training", "Test"], horizontal=True, label_visibility="collapsed")
    frame = train if selected == "Training" else test
    normal_count = train_normal if selected == "Training" else test_normal
    attack_count = train_attack if selected == "Training" else test_attack

    a, b, c, d = st.columns(4)
    a.metric("Total records", f"{len(frame):,}")
    b.metric("Normal", f"{normal_count:,}", f"{normal_count / len(frame):.1%}")
    c.metric("Attack", f"{attack_count:,}", f"{attack_count / len(frame):.1%}")
    d.metric("Fields", "43 · 41 features + label + difficulty")

    tab_profile, tab_attacks, tab_records = st.tabs(["Traffic profile", "Attack types", "Dataset records"])
    with tab_profile:
        left, right = st.columns(2, gap="small")
        with left:
            heading("CLASS BALANCE", f"{selected} class distribution")
            fig, ax = plt.subplots(figsize=(6, 3.0))
            parts = [normal_count, attack_count]
            bars = ax.bar(["Normal", "Attack"], parts, color=[TEAL, AMBER], width=.55)
            ax.set_ylabel("Records")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
            theme(ax)
            for bar, val in zip(bars, parts):
                ax.text(bar.get_x()+bar.get_width()/2, val+max(parts)*.018, f"{val:,}", ha="center", color=TEXT, fontsize=9)
            ax.set_ylim(0, max(parts)*1.16)
            render(fig)
        with right:
            heading("CONNECTION CATEGORIES", "Protocol and service mix")
            protocol = frame["protocol_type"].value_counts().rename_axis("Protocol").reset_index(name="Records")
            protocol["Share"] = protocol["Records"] / len(frame)
            st.dataframe(protocol.style.format({"Records": "{:,.0f}", "Share": "{:.1%}"}), use_container_width=True, hide_index=True)
            services = frame["service"].value_counts().head(8).rename_axis("Service").reset_index(name="Records")
            st.dataframe(services, use_container_width=True, hide_index=True)
    with tab_attacks:
        attack_counts = frame.loc[frame["attack"] != "normal", "attack"].value_counts().head(18).sort_values()
        fig, ax = plt.subplots(figsize=(10, 5.0))
        bars = ax.barh(attack_counts.index, attack_counts.values, color=AMBER, height=.72)
        ax.set_xlabel("Records")
        theme(ax, "x")
        ax.tick_params(axis="y", labelsize=9)
        for bar, value in zip(bars, attack_counts.values):
            ax.text(value + max(attack_counts.values)*.012, bar.get_y()+bar.get_height()/2, f"{value:,}", va="center", color=TEXT, fontsize=8)
        ax.set_xlim(0, max(attack_counts.values)*1.18)
        render(fig)
    with tab_records:
        labels = ["All records", "Normal only", "Attack only"]
        filter_choice = st.selectbox("Show", labels)
        if filter_choice == "Normal only":
            view = frame[frame["attack"] == "normal"]
        elif filter_choice == "Attack only":
            view = frame[frame["attack"] != "normal"]
        else:
            view = frame
        st.dataframe(view.head(1000), use_container_width=True, hide_index=True, height=510)
        st.caption(f"Showing up to 1,000 of {len(view):,} matching project records.")


elif page == "Training":
    heading("TRAINING", "Training data and model comparison", "Training split: 125,973 connections · binary target: normal (0) or attack (1)")
    a, b, c, d = st.columns(4)
    a.metric("Training records", f"{len(train):,}")
    b.metric("Normal connections", f"{train_normal:,}", f"{train_normal/len(train):.1%}")
    c.metric("Attack connections", f"{train_attack:,}", f"{train_attack/len(train):.1%}")
    d.metric("Categorical features", "3 of 41")
    left, right = st.columns([1, 1], gap="small")
    with left:
        heading("LABEL DISTRIBUTION", "Training classes")
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        vals = [train_normal, train_attack]
        bars = ax.bar(["Normal", "Attack"], vals, color=[TEAL, AMBER], width=.5)
        ax.set_ylabel("Connections")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        theme(ax)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+1500, f"{val:,}\n{val/len(train):.1%}", ha="center", color=TEXT, fontsize=9)
        ax.set_ylim(0, max(vals)*1.18)
        render(fig)
    with right:
        heading("MODEL SELECTION", "Reported training workflow")
        st.markdown("**Categorical inputs**  \n`protocol_type` · `service` · `flag`")
        st.markdown("**Numeric inputs**  \n38 connection, content, and traffic-window measurements")
        st.markdown("**Pipeline**  \nString index → one-hot encode → assemble features → classifier")
        st.markdown("**Saved classifiers**  \nDecision Tree · Random Forest · Logistic Regression · Linear SVM · Naive Bayes")
    heading("MODEL COMPARISON", "Five classifiers · held-out test metrics")
    ordered = MODEL_METRICS.sort_values("Weighted F1", ascending=False).copy()
    st.dataframe(ordered.style.format({"Accuracy": "{:.2%}", "Weighted F1": "{:.2%}"}).highlight_max(subset=["Accuracy", "Weighted F1"], color="#24423F"), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    y = list(range(len(ordered)))
    ax.barh([v+.16 for v in y], ordered["Accuracy"], height=.3, color=TEAL, label="Accuracy")
    ax.barh([v-.16 for v in y], ordered["Weighted F1"], height=.3, color=BLUE, label="Weighted F1")
    ax.set_yticks(y, ordered["Model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    theme(ax, "x")
    ax.legend(frameon=False, ncol=2, labelcolor=TEXT)
    render(fig)


elif page == "Testing":
    heading("TESTING", "Held-out test results", "Metrics below are from the notebook's executed evaluation on 22,544 test records.")
    best = MODEL_METRICS.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Test records", f"{len(test):,}")
    a2, a3 = CONFUSION[0]
    b.metric("Decision Tree accuracy", f"{best['Accuracy']:.2%}")
    c.metric("Weighted F1", f"{best['Weighted F1']:.2%}")
    d.metric("Correct test predictions", f"{CONFUSION[0][0] + CONFUSION[1][1]:,}")

    st.markdown("")
    left, right = st.columns([.88, 1.12], gap="small")
    with left:
        heading("DECISION TREE", "Confusion matrix", "Actual class by predicted class")
        fig, ax = plt.subplots(figsize=(5.4, 2.65))
        sns.heatmap(
            CONFUSION, annot=True, fmt=",d", cmap=sns.color_palette(["#182638", TEAL], as_cmap=True),
            cbar=False, linewidths=2, linecolor=BG, square=False,
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
            annot_kws={"color": "#F1F4F8", "size": 13, "weight": "bold"}, ax=ax,
        )
        ax.set_xlabel("Predicted", color=TEXT, labelpad=3)
        ax.set_ylabel("Actual", color=TEXT, labelpad=3)
        ax.tick_params(colors=TEXT, length=0)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_facecolor(PANEL)
        ax.figure.set_facecolor(PANEL)
        render(fig)
    with right:
        heading("RESULT CARDS", "Classification outcomes", "TN · FP · FN · TP")
        cards = st.columns(2, gap="small")
        card_values = [
            ("TN · Correct normal", CONFUSION[0][0], TEAL),
            ("FP · False alarm", CONFUSION[0][1], AMBER),
            ("FN · Missed attack", CONFUSION[1][0], RED),
            ("TP · Correct attack", CONFUSION[1][1], BLUE),
        ]
        for col, (label, value, color) in zip([cards[0], cards[1], cards[0], cards[1]], card_values):
            with col:
                st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="color:{color}">{value:,}</div></div>', unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="card-label">Test-set class counts</div><div class="card-value" style="font-size:1rem">Normal {test_normal:,} &nbsp; · &nbsp; Attack {test_attack:,}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="small")
    with left:
        heading("ERROR PROFILE", "Decision Tree outcomes")
        outcomes = pd.Series({"Correct normal": CONFUSION[0][0], "False alarm": CONFUSION[0][1], "Missed attack": CONFUSION[1][0], "Correct attack": CONFUSION[1][1]})
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        bars = ax.barh(outcomes.index[::-1], outcomes.values[::-1], color=[BLUE, RED, AMBER, TEAL])
        ax.set_xlabel("Test records")
        theme(ax, "x")
        for bar, value in zip(bars, outcomes.values[::-1]):
            ax.text(value+100, bar.get_y()+bar.get_height()/2, f"{value:,}", va="center", color=TEXT, fontsize=8)
        ax.set_xlim(0, max(outcomes)*1.2)
        render(fig)
    with right:
        heading("MODEL INTERPRETATION", "Most influential inputs")
        ranked = FEATURE_IMPORTANCE.sort_values("Importance")
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        ax.barh(ranked["Feature"], ranked["Importance"], color=[BLUE]*(len(ranked)-1)+[TEAL], height=.68)
        ax.set_xlim(0, .82)
        ax.set_xlabel("Relative importance")
        theme(ax, "x")
        ax.tick_params(axis="y", labelsize=8)
        render(fig)


st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NETSENTINEL &nbsp; · &nbsp; NSL-KDD &nbsp; · &nbsp; SPARK MLLIB</div>', unsafe_allow_html=True)
