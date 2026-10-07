"""
Predictd Dashboard - Multi-League with 1X2 + Double Chance + Adaptive Alerts
Reads from dashboard_data/ when available (for cloud), falls back to live folders.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import json
from datetime import datetime

st.set_page_config(page_title="Predictd", layout="wide", page_icon="⚽")
st.title("⚽ Predictd - Sports Prediction Lab")


# ==============================================================================
# PATH RESOLUTION (cloud vs local)
# ==============================================================================

def data_path(filename, live_subfolder=None):
    """
    Prefer dashboard_data/<filename> if it exists (cloud deployment).
    Otherwise fall back to live_subfolder/<filename> (local run).
    """
    snapshot = os.path.join("dashboard_data", filename)
    if os.path.exists(snapshot):
        return snapshot
    if live_subfolder:
        return os.path.join(live_subfolder, filename)
    return snapshot


def find_latest_live_file():
    """
    Return the latest live_*.csv from dashboard_data/ (if present),
    otherwise from predictions/.
    """
    snapshot = sorted(glob.glob("dashboard_data/live_*.csv"))
    if snapshot:
        return snapshot[-1]
    live = sorted(glob.glob("predictions/live_*.csv"))
    if live:
        return live[-1]
    return None


# ==============================================================================
# DATA LOADERS
# ==============================================================================

@st.cache_data(ttl=60)
def load_latest_predictions():
    latest = find_latest_live_file()
    if latest is None:
        return pd.DataFrame()

    df = pd.read_csv(latest)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['source_file'] = os.path.basename(latest)

    # Normalize prob_* → pred_*
    rename_map = {}
    for std, alt in [('pred_home', 'prob_home'),
                     ('pred_draw', 'prob_draw'),
                     ('pred_away', 'prob_away')]:
        if std not in df.columns and alt in df.columns:
            rename_map[alt] = std
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ['pred_home', 'pred_draw', 'pred_away']:
        if col not in df.columns:
            df[col] = 0.333

    # Ensure DC columns exist
    if 'dc_prediction' not in df.columns:
        ph = df['pred_home'].fillna(0.333)
        pd_ = df['pred_draw'].fillna(0.333)
        pa = df['pred_away'].fillna(0.333)
        df['prob_1x'] = ph + pd_
        df['prob_x2'] = pd_ + pa
        df['prob_12'] = ph + pa
        dc_probs = df[['prob_1x', 'prob_x2', 'prob_12']].values
        import numpy as np
        df['dc_prediction'] = np.array(['1X', 'X2', '12'])[np.argmax(dc_probs, axis=1)]

    if 'League' not in df.columns:
        df['League'] = 'Unknown'

    return df


@st.cache_data(ttl=60)
def load_accuracy_summary():
    try:
        p = data_path("accuracy_summary.csv", "logs")
        df = pd.read_csv(p)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_accuracy_log():
    try:
        p = data_path("accuracy_log.csv", "logs")
        df = pd.read_csv(p)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for std, alt in [('pred_home', 'prob_home'),
                         ('pred_draw', 'prob_draw'),
                         ('pred_away', 'prob_away')]:
            if std not in df.columns and alt in df.columns:
                df[std] = df[alt]
        for col in ['pred_home', 'pred_draw', 'pred_away']:
            if col not in df.columns:
                df[col] = 0.333
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_adaptive_status():
    try:
        p = data_path("adaptive_status.json", "logs")
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header("📊 Navigation")
option = st.sidebar.radio(
    "Choose a view:",
    ["🎯 Upcoming Predictions", "📈 Accuracy Tracker", "🧠 Model Insights", "📖 About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

preds = load_latest_predictions()
if not preds.empty:
    st.sidebar.success(f"✓ {len(preds)} predictions loaded")
    st.sidebar.caption(f"Latest: {preds['source_file'].iloc[0]}")
    if 'League' in preds.columns and preds['League'].nunique() > 1:
        st.sidebar.caption(f"{preds['League'].nunique()} leagues")
else:
    st.sidebar.warning("No predictions yet")

acc = load_accuracy_summary()
if not acc.empty:
    st.sidebar.success(f"✓ {len(acc)} accuracy runs logged")
    latest_acc = acc.iloc[-1].get('accuracy', None)
    if latest_acc is not None:
        st.sidebar.metric("1X2 Accuracy", f"{latest_acc*100:.1f}%")
    latest_dc_acc = acc.iloc[-1].get('dc_accuracy', None)
    if latest_dc_acc is not None and pd.notna(latest_dc_acc):
        st.sidebar.metric("DC Accuracy", f"{latest_dc_acc*100:.1f}%")
else:
    st.sidebar.info("No accuracy data yet")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Alerts")

adaptive = load_adaptive_status()
if adaptive:
    overall = adaptive.get('overall_status', 'no_data')
    trend = adaptive.get('trend', 'unknown')
    status_emoji = {
        'excellent': '🟢', 'good': '🟢', 'warning': '🟡',
        'critical': '🔴', 'no_data': '⚪',
    }.get(overall, '⚪')
    trend_emoji = {
        'improving': '📈', 'worsening': '📉',
        'stable': '➡️', 'insufficient_data': '⏳',
    }.get(trend, '❓')
    st.sidebar.markdown(f"{status_emoji} Status: **{overall.upper()}**")
    st.sidebar.markdown(f"{trend_emoji} Trend: **{trend}**")
    alerts = adaptive.get('alerts', [])
    if alerts:
        for alert in alerts:
            level = alert.get('level', 'INFO')
            msg = alert.get('message', '')
            if level == 'CRITICAL':
                st.sidebar.error(msg)
            elif level == 'WARNING':
                st.sidebar.warning(msg)
            elif level == 'EXCELLENT':
                st.sidebar.success(msg)
            else:
                st.sidebar.info(msg)
    else:
        st.sidebar.info("No alerts")
else:
    st.sidebar.info("Monitor not run yet")


# ==============================================================================
# VIEW 1: UPCOMING PREDICTIONS
# ==============================================================================

if option == "🎯 Upcoming Predictions":
    st.subheader("🎯 Upcoming Match Predictions")

    if preds.empty:
        st.warning("No predictions available.")
    else:
        today = pd.Timestamp.now().normalize()
        upcoming = preds[preds['Date'] >= today].copy()

        st.metric("Total Upcoming Matches", len(upcoming))

        if upcoming.empty:
            st.info("No upcoming matches.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("1X2: Home Picks", (upcoming['prediction'] == 'H').sum())
            col2.metric("1X2: Draw Picks", (upcoming['prediction'] == 'D').sum())
            col3.metric("1X2: Away Picks", (upcoming['prediction'] == 'A').sum())

            st.markdown("---")
            st.markdown("### 🔍 Filters")

            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                league_filter = st.multiselect(
                    "Leagues",
                    options=sorted(upcoming['League'].unique()),
                    default=sorted(upcoming['League'].unique())
                )
            with fcol2:
                market_view = st.selectbox(
                    "Market",
                    options=['Both (1X2 + DC)', '1X2 Only', 'DC Only']
                )
            with fcol3:
                min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

            upcoming['max_conf'] = upcoming[['pred_home', 'pred_draw', 'pred_away']].max(axis=1)
            filtered = upcoming[
                (upcoming['max_conf'] >= min_confidence) &
                (upcoming['League'].isin(league_filter))
            ].sort_values(['Date', 'max_conf'], ascending=[True, False])

            st.caption(f"Showing {len(filtered)} of {len(upcoming)} matches")

            for league in sorted(filtered['League'].unique()):
                league_df = filtered[filtered['League'] == league]
                st.markdown(f"### {league} ({len(league_df)} matches)")

                for _, row in league_df.iterrows():
                    with st.container():
                        c1, c2, c3, c4 = st.columns([1.2, 2, 1.5, 3])
                        c1.markdown(f"**{row['Date'].strftime('%b %d, %Y')}**")
                        c2.markdown(f"**{row['HomeTeam']}** vs **{row['AwayTeam']}**")

                        if market_view == 'Both (1X2 + DC)':
                            c3.markdown(
                                f"1X2: **`{row['prediction']}`**<br>"
                                f"DC: **`{row.get('dc_prediction', '?')}`**",
                                unsafe_allow_html=True
                            )
                        elif market_view == '1X2 Only':
                            c3.markdown(f"### `{row['prediction']}`")
                        else:
                            c3.markdown(f"### `{row.get('dc_prediction', '?')}`")

                        if market_view == 'Both (1X2 + DC)':
                            probs = [
                                row['pred_home'], row['pred_draw'], row['pred_away'],
                                row.get('prob_1x', 0), row.get('prob_x2', 0), row.get('prob_12', 0)
                            ]
                            labels = ['H', 'D', 'A', '1X', 'X2', '12']
                            colors = ['#2E86C1', '#F1C40F', '#E74C3C', '#7FB3D5', '#F7DC6F', '#EC7063']
                        elif market_view == '1X2 Only':
                            probs = [row['pred_home'], row['pred_draw'], row['pred_away']]
                            labels = ['Home', 'Draw', 'Away']
                            colors = ['#2E86C1', '#F1C40F', '#E74C3C']
                        else:
                            probs = [
                                row.get('prob_1x', 0),
                                row.get('prob_x2', 0),
                                row.get('prob_12', 0)
                            ]
                            labels = ['1X', 'X2', '12']
                            colors = ['#7FB3D5', '#F7DC6F', '#EC7063']

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=labels, y=probs, marker_color=colors,
                            text=[f"{p*100:.0f}%" for p in probs],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            height=180,
                            margin=dict(l=0, r=0, t=0, b=0),
                            yaxis_range=[0, 1],
                            showlegend=False,
                            yaxis=dict(showticklabels=False)
                        )
                        c4.plotly_chart(fig, use_container_width=True, key=f"chart_{row.name}")
                    st.markdown("---")


# ==============================================================================
# VIEW 2: ACCURACY TRACKER
# ==============================================================================

elif option == "📈 Accuracy Tracker":
    st.subheader("📈 Prediction Accuracy")

    summary = load_accuracy_summary()
    log = load_accuracy_log()
    adaptive = load_adaptive_status()

    if summary.empty:
        st.info("No accuracy data yet. Predictions are logged; results pending.")
    else:
        latest = summary.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1X2 Accuracy", f"{latest['accuracy']*100:.2f}%")
        col2.metric("1X2 Correct", f"{int(latest['correct'])}/{int(latest['total_matches'])}")
        if 'dc_accuracy' in latest and pd.notna(latest['dc_accuracy']):
            col3.metric("DC Accuracy", f"{latest['dc_accuracy']*100:.2f}%")
            col4.metric("DC Correct", f"{int(latest['dc_correct'])}/{int(latest['total_matches'])}")
        else:
            col3.metric("DC Accuracy", "N/A")
            col4.metric("DC Correct", "N/A")

        if len(summary) > 1 and 'dc_accuracy' in summary.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=summary['timestamp'], y=summary['accuracy'],
                mode='lines+markers', name='1X2'
            ))
            if 'dc_accuracy' in summary.columns:
                fig.add_trace(go.Scatter(
                    x=summary['timestamp'], y=summary['dc_accuracy'],
                    mode='lines+markers', name='Double Chance'
                ))
            fig.update_layout(title="Accuracy Over Time", yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

        if adaptive:
            st.markdown("---")
            st.markdown("### 🚨 Adaptive Monitor")

            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric("Overall Status", adaptive.get('overall_status', 'no_data').upper())
            with status_col2:
                st.metric("Trend", adaptive.get('trend', 'unknown'))

            windows = adaptive.get('windows', {})
            if windows:
                st.markdown("#### Rolling Windows")
                for wname, w in windows.items():
                    if w:
                        st.write(f"**{wname}**: {w['accuracy']*100:.1f}% ({w['correct']}/{w['matches']})")

            leagues_stats = adaptive.get('leagues', {})
            if leagues_stats:
                st.markdown("#### Per-League")
                for lg, s in leagues_stats.items():
                    st.write(f"**{lg}**: {s['accuracy']*100:.1f}% ({s['correct']}/{s['matches']})")

            alerts = adaptive.get('alerts', [])
            if alerts:
                st.markdown("#### Alerts")
                for a in alerts:
                    lvl, msg, act = a.get('level'), a.get('message'), a.get('action', '')
                    if lvl == 'CRITICAL':
                        st.error(f"🔴 {msg} — {act}")
                    elif lvl == 'WARNING':
                        st.warning(f"⚠️ {msg} — {act}")
                    elif lvl == 'EXCELLENT':
                        st.success(f"🟢 {msg} — {act}")
                    else:
                        st.info(f"ℹ️ {msg} — {act}")

        if not log.empty and 'correct' in log.columns and 'FTR' in log.columns:
            st.markdown("---")
            st.markdown("### Confusion Matrix (1X2)")
            cm = pd.crosstab(log['prediction'], log['FTR'])
            fig_cm = px.imshow(cm, labels=dict(x="Actual", y="Predicted", color="Count"),
                                color_continuous_scale='Blues', text_auto=True)
            st.plotly_chart(fig_cm, use_container_width=True)

            if 'dc_correct' in log.columns and 'dc_prediction' in log.columns:
                st.markdown("### DC Accuracy by Prediction Type")
                for pred in ['1X', 'X2', '12']:
                    sub = log[log['dc_prediction'] == pred]
                    if not sub.empty:
                        st.metric(f"DC {pred}", f"{sub['dc_correct'].mean()*100:.1f}%",
                                  delta=f"{len(sub)} matches")

        if not log.empty:
            st.markdown("### Recent Match Results")
            cols = ['Date', 'League', 'HomeTeam', 'AwayTeam',
                    'prediction', 'dc_prediction', 'FTR', 'correct', 'dc_correct']
            available = [c for c in cols if c in log.columns]
            st.dataframe(log[available].sort_values('Date', ascending=False).head(50),
                         use_container_width=True)


# ==============================================================================
# VIEW 3: MODEL INSIGHTS
# ==============================================================================

elif option == "🧠 Model Insights":
    st.subheader("🧠 Model Performance & Insights")

    if preds.empty:
        st.warning("No predictions available.")
    else:
        preds['max_conf'] = preds[['pred_home', 'pred_draw', 'pred_away']].max(axis=1)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Confidence Distribution")
            fig = px.histogram(preds, x='max_conf', nbins=20)
            fig.update_layout(xaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 1X2 Distribution")
            dist = preds['prediction'].value_counts().reset_index()
            dist.columns = ['Outcome', 'Count']
            fig = px.pie(dist, names='Outcome', values='Count', color='Outcome',
                         color_discrete_map={'H': '#2E86C1', 'D': '#F1C40F', 'A': '#E74C3C'})
            st.plotly_chart(fig, use_container_width=True)

        if 'dc_prediction' in preds.columns:
            st.markdown("### Double Chance Distribution")
            dc_dist = preds['dc_prediction'].value_counts().reset_index()
            dc_dist.columns = ['DC Outcome', 'Count']
            fig = px.bar(dc_dist, x='DC Outcome', y='Count', color='DC Outcome',
                         color_discrete_map={'1X': '#7FB3D5', 'X2': '#F7DC6F', '12': '#EC7063'})
            st.plotly_chart(fig, use_container_width=True)

        if 'League' in preds.columns and preds['League'].nunique() > 1:
            st.markdown("### Predictions by League")
            fig = px.histogram(preds, x='League', color='prediction',
                               color_discrete_map={'H': '#2E86C1', 'D': '#F1C40F', 'A': '#E74C3C'},
                               barmode='stack')
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VIEW 4: ABOUT
# ==============================================================================

else:
    st.subheader("📖 About Predictd")
    st.markdown("### What is Predictd?")
    st.markdown("A sports prediction lab using market odds across 7 top European competitions.")

    st.markdown("### Markets Covered")
    st.markdown("- **1X2** — Home / Draw / Away")
    st.markdown("- **Double Chance** — 1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away)")

    st.markdown("### Competitions")
    st.markdown("- Premier League, La Liga, Bundesliga, Serie A, Ligue 1")
    st.markdown("- Champions League, Europa League")

    st.markdown("### Model: V20")
    st.markdown("- **Odds source:** Pinnacle (preferred), Betfair Exchange (fallback)")
    st.markdown("- **Data:** 20,013 matches across 5 domestic leagues")
    st.markdown("- **Live updates:** Daily at 13:10 via Windows Task Scheduler")

    st.markdown("### System Components")
    st.markdown("1. **Data Pipeline** — 12 seasons × 5 leagues from Football-Data.co.uk")
    st.markdown("2. **Live Predictions** — Pinnacle odds, 1X2 + Double Chance")
    st.markdown("3. **Accuracy Tracking** — Weekly comparison vs actuals")
    st.markdown("4. **Adaptive Monitor** — Alerts on performance issues")
    st.markdown("5. **Dashboard** — This view")