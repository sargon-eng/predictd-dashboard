"""
Predictd Dashboard - Multi-League with 1X2 + DC + Bet Slips + Accuracy Tracking
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
# PATH RESOLUTION
# ==============================================================================

def data_path(filename, live_subfolder=None):
    snapshot = os.path.join("dashboard_data", filename)
    if os.path.exists(snapshot):
        return snapshot
    if live_subfolder:
        return os.path.join(live_subfolder, filename)
    return snapshot


def find_latest_live_file():
    s = sorted(glob.glob("dashboard_data/live_*.csv"))
    if s:
        return s[-1]
    l = sorted(glob.glob("predictions/live_*.csv"))
    return l[-1] if l else None


def find_latest_slip_file():
    s = sorted(glob.glob("dashboard_data/bet_slips_*.csv"))
    if s:
        return s[-1]
    l = sorted(glob.glob("predictions/bet_slips_*.csv"))
    return l[-1] if l else None


def find_slip_accuracy_file():
    s = os.path.join("dashboard_data", "bet_slip_accuracy.csv")
    if os.path.exists(s):
        return s
    l = os.path.join("logs", "bet_slip_accuracy.csv")
    return l if os.path.exists(l) else None


# ==============================================================================
# LOADERS
# ==============================================================================

@st.cache_data(ttl=60)
def load_latest_predictions():
    latest = find_latest_live_file()
    if latest is None:
        return pd.DataFrame()
    df = pd.read_csv(latest)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['source_file'] = os.path.basename(latest)

    for std, alt in [('pred_home', 'prob_home'),
                     ('pred_draw', 'prob_draw'),
                     ('pred_away', 'prob_away')]:
        if std not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: std})

    for col in ['pred_home', 'pred_draw', 'pred_away']:
        if col not in df.columns:
            df[col] = 0.333

    if 'dc_prediction' not in df.columns:
        ph = df['pred_home'].fillna(0.333)
        pd_ = df['pred_draw'].fillna(0.333)
        pa = df['pred_away'].fillna(0.333)
        df['prob_1x'] = ph + pd_
        df['prob_x2'] = pd_ + pa
        df['prob_12'] = ph + pa
        import numpy as np
        dc_probs = df[['prob_1x', 'prob_x2', 'prob_12']].values
        df['dc_prediction'] = np.array(['1X', 'X2', '12'])[np.argmax(dc_probs, axis=1)]

    if 'League' not in df.columns:
        df['League'] = 'Unknown'

    return df


@st.cache_data(ttl=60)
def load_bet_slips():
    latest = find_latest_slip_file()
    if latest is None:
        return pd.DataFrame()
    df = pd.read_csv(latest)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


@st.cache_data(ttl=60)
def load_slip_accuracy():
    path = find_slip_accuracy_file()
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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
    ["🎯 Upcoming Predictions", "🎫 Bet Slips", "📈 Accuracy Tracker",
     "🧠 Model Insights", "📖 About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

preds = load_latest_predictions()
if not preds.empty:
    st.sidebar.success(f"✓ {len(preds)} predictions loaded")
    if 'League' in preds.columns and preds['League'].nunique() > 1:
        st.sidebar.caption(f"{preds['League'].nunique()} leagues")

slips = load_bet_slips()
if not slips.empty:
    st.sidebar.success(f"✓ {slips['slip_id'].nunique()} bet slips")

acc = load_accuracy_summary()
if not acc.empty:
    latest_acc = acc.iloc[-1].get('accuracy', None)
    if latest_acc is not None:
        st.sidebar.metric("1X2 Accuracy", f"{latest_acc*100:.1f}%")
    latest_dc_acc = acc.iloc[-1].get('dc_accuracy', None)
    if latest_dc_acc is not None and pd.notna(latest_dc_acc):
        st.sidebar.metric("DC Accuracy", f"{latest_dc_acc*100:.1f}%")

slip_acc = load_slip_accuracy()
if not slip_acc.empty:
    wins = (slip_acc['outcome'] == 'WIN').sum()
    st.sidebar.metric("Slip Win Rate", f"{wins}/{len(slip_acc)} = {wins/len(slip_acc)*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Alerts")

adaptive = load_adaptive_status()
if adaptive:
    overall = adaptive.get('overall_status', 'no_data')
    trend = adaptive.get('trend', 'unknown')
    status_emoji = {'excellent': '🟢', 'good': '🟢', 'warning': '🟡',
                    'critical': '🔴', 'no_data': '⚪'}.get(overall, '⚪')
    trend_emoji = {'improving': '📈', 'worsening': '📉',
                   'stable': '➡️', 'insufficient_data': '⏳'}.get(trend, '❓')
    st.sidebar.markdown(f"{status_emoji} Status: **{overall.upper()}**")
    st.sidebar.markdown(f"{trend_emoji} Trend: **{trend}**")
    alerts = adaptive.get('alerts', [])
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
            col1.metric("Home Picks", (upcoming['prediction'] == 'H').sum())
            col2.metric("Draw Picks", (upcoming['prediction'] == 'D').sum())
            col3.metric("Away Picks", (upcoming['prediction'] == 'A').sum())

            st.markdown("---")
            st.markdown("### 🔍 Filters")
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                league_filter = st.multiselect("Leagues",
                    options=sorted(upcoming['League'].unique()),
                    default=sorted(upcoming['League'].unique()))
            with fcol2:
                market_view = st.selectbox("Market",
                    options=['Both (1X2 + DC)', '1X2 Only', 'DC Only'])
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
                                unsafe_allow_html=True)
                        elif market_view == '1X2 Only':
                            c3.markdown(f"### `{row['prediction']}`")
                        else:
                            c3.markdown(f"### `{row.get('dc_prediction', '?')}`")

                        if market_view == 'Both (1X2 + DC)':
                            probs = [row['pred_home'], row['pred_draw'], row['pred_away'],
                                    row.get('prob_1x', 0), row.get('prob_x2', 0), row.get('prob_12', 0)]
                            labels = ['H', 'D', 'A', '1X', 'X2', '12']
                            colors = ['#2E86C1', '#F1C40F', '#E74C3C', '#7FB3D5', '#F7DC6F', '#EC7063']
                        elif market_view == '1X2 Only':
                            probs = [row['pred_home'], row['pred_draw'], row['pred_away']]
                            labels = ['Home', 'Draw', 'Away']
                            colors = ['#2E86C1', '#F1C40F', '#E74C3C']
                        else:
                            probs = [row.get('prob_1x', 0), row.get('prob_x2', 0), row.get('prob_12', 0)]
                            labels = ['1X', 'X2', '12']
                            colors = ['#7FB3D5', '#F7DC6F', '#EC7063']

                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=labels, y=probs, marker_color=colors,
                                             text=[f"{p*100:.0f}%" for p in probs],
                                             textposition='outside'))
                        fig.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0),
                                          yaxis_range=[0, 1], showlegend=False,
                                          yaxis=dict(showticklabels=False))
                        c4.plotly_chart(fig, width='stretch', key=f"chart_{row.name}")
                    st.markdown("---")


# ==============================================================================
# VIEW 2: BET SLIPS
# ==============================================================================

elif option == "🎫 Bet Slips":
    st.subheader("🎫 Learning-Optimized Bet Slips")

    if slips.empty:
        st.warning("No bet slips available. Run `python src\\bet_slip_builder.py` first.")
    else:
        # Filter controls
        st.markdown("### 🔍 Filters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            strategy_filter = st.multiselect(
                "Strategy",
                options=sorted(slips['strategy'].unique()),
                default=sorted(slips['strategy'].unique())
            )
        with fcol2:
            market_filter = st.multiselect(
                "Market",
                options=sorted(slips['market'].unique()),
                default=sorted(slips['market'].unique())
            )

        filtered_slips = slips[
            slips['strategy'].isin(strategy_filter) &
            slips['market'].isin(market_filter)
        ]

        # Summary metrics
        n_slips = filtered_slips['slip_id'].nunique() if not filtered_slips.empty else 0
        st.metric("Showing Slips", n_slips)

        st.markdown("---")

        if filtered_slips.empty:
            st.info("No slips match your filters.")
        else:
            # Group by strategy + market
            for strat in sorted(filtered_slips['strategy'].unique()):
                strat_data = filtered_slips[filtered_slips['strategy'] == strat]
                strat_market = strat_data['market'].iloc[0]

                # Strategy header with description
                strategy_desc = {
                    'top_mixed': '🏆 Best overall picks across H/D/A',
                    'home_best': '🏠 Best home-win picks only',
                    'away_best': '✈️ Best away-win picks only',
                    'dc_best': '🛡️ Best Double Chance picks (safest)',
                }.get(strat, '')

                st.markdown(f"## {strategy_desc} — `{strat}`")

                slip_ids = sorted(strat_data['slip_id'].unique())

                for slip_id in slip_ids:
                    slip_data = strat_data[strat_data['slip_id'] == slip_id]
                    if slip_data.empty:
                        continue

                    size = int(slip_data['size'].iloc[0])
                    combined = float(slip_data['combined_prob'].iloc[0])
                    leagues = sorted(slip_data['League'].unique())

                    if combined >= 0.5:
                        status = "🟢"
                    elif combined >= 0.25:
                        status = "🟡"
                    else:
                        status = "🔴"

                    with st.expander(
                        f"{status} **Slip #{slip_id}** — {size}-fold | "
                        f"Combined: {combined*100:.1f}% | "
                        f"Odds: {1/combined:.2f}x | "
                        f"Leagues: {', '.join(leagues)}",
                        expanded=False
                    ):
                        for _, row in slip_data.sort_values('pick_num').iterrows():
                            cols = st.columns([0.4, 1.2, 1.8, 0.7, 0.7, 0.7])
                            cols[0].markdown(f"**{int(row['pick_num'])}**")
                            cols[1].markdown(f"{row['Date'].strftime('%b %d')} • {row['League']}")
                            cols[2].markdown(f"**{row['HomeTeam']}** vs {row['AwayTeam']}")
                            cols[3].markdown(f"**`{row['prediction']}`**")
                            cols[4].markdown(f"{row['probability']*100:.1f}%")
                            cols[5].markdown(f"*{row['score']:.2f}*")

                        st.markdown(f"**Combined probability:** {combined*100:.2f}% "
                                    f"| **Implied odds:** {1/combined:.2f}x")

                st.markdown("---")

        # ============================================================
        # HISTORICAL BET SLIP ACCURACY
        # ============================================================
        st.markdown("## 📊 Historical Bet Slip Accuracy")

        if slip_acc.empty:
            st.info("No bet slips have been fully played yet. "
                    "Once all picks in a slip resolve, accuracy will appear here.")
        else:
            wins = (slip_acc['outcome'] == 'WIN').sum()
            total_slips = len(slip_acc)
            win_rate = wins / total_slips

            # Top metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Slips Evaluated", total_slips)
            mc2.metric("Wins", int(wins))
            mc3.metric("Win Rate", f"{win_rate*100:.1f}%")
            avg_pred = slip_acc['combined_prob'].mean()
            mc4.metric("Avg Predicted", f"{avg_pred*100:.1f}%")

            # Calibration
            gap = (avg_pred - win_rate) * 100
            if gap > 5:
                st.warning(f"⚠️ Model overconfident by {gap:.1f} pp "
                           f"(predicted {avg_pred*100:.1f}%, actually won {win_rate*100:.1f}%)")
            elif gap < -5:
                st.success(f"✅ Model under-confident by {abs(gap):.1f} pp "
                           f"(predicted {avg_pred*100:.1f}%, actually won {win_rate*100:.1f}%)")
            else:
                st.success(f"✅ Well-calibrated (gap: {gap:.1f} pp)")

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("### Win Rate by Strategy")
                strat_stats = []
                for strat in ['top_mixed', 'home_best', 'away_best', 'dc_best']:
                    sub = slip_acc[slip_acc['strategy'] == strat]
                    if len(sub) == 0:
                        continue
                    w = (sub['outcome'] == 'WIN').sum()
                    strat_stats.append({
                        'strategy': strat,
                        'matches': len(sub),
                        'wins': int(w),
                        'win_rate': w / len(sub),
                    })
                if strat_stats:
                    strat_df = pd.DataFrame(strat_stats)
                    fig = px.bar(strat_df, x='strategy', y='win_rate',
                                text=strat_df['win_rate'].apply(lambda x: f"{x*100:.0f}%"),
                                color='strategy',
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_layout(yaxis_tickformat='.0%', showlegend=False,
                                     yaxis_title="Win Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, width='stretch')

            with chart_col2:
                st.markdown("### Win Rate by Slip Size")
                size_stats = []
                for sz in sorted(slip_acc['size'].unique()):
                    sub = slip_acc[slip_acc['size'] == sz]
                    if len(sub) == 0:
                        continue
                    w = (sub['outcome'] == 'WIN').sum()
                    size_stats.append({
                        'size': f"{int(sz)}-fold",
                        'matches': len(sub),
                        'wins': int(w),
                        'win_rate': w / len(sub),
                    })
                if size_stats:
                    size_df = pd.DataFrame(size_stats)
                    fig = px.bar(size_df, x='size', y='win_rate',
                                text=size_df['win_rate'].apply(lambda x: f"{x*100:.0f}%"),
                                color='size',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(yaxis_tickformat='.0%', showlegend=False,
                                     yaxis_title="Win Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, width='stretch')

            # Detail table
            st.markdown("### All Evaluated Slips")
            display = slip_acc.sort_values('slip_date', ascending=False).copy()
            display['outcome_icon'] = display['outcome'].map({'WIN': '✅', 'LOSS': '❌'})
            display['combined_prob_pct'] = (display['combined_prob'] * 100).round(1).astype(str) + '%'
            st.dataframe(
                display[['slip_date', 'strategy', 'market', 'size',
                         'outcome_icon', 'combined_prob_pct', 'outcome']],
                width='stretch'
            )


# ==============================================================================
# VIEW 3: ACCURACY TRACKER
# ==============================================================================

elif option == "📈 Accuracy Tracker":
    st.subheader("📈 Prediction Accuracy")

    summary = load_accuracy_summary()
    log = load_accuracy_log()
    adaptive = load_adaptive_status()

    if summary.empty:
        st.info("No accuracy data yet.")
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
            fig.add_trace(go.Scatter(x=summary['timestamp'], y=summary['accuracy'],
                                     mode='lines+markers', name='1X2'))
            if 'dc_accuracy' in summary.columns:
                fig.add_trace(go.Scatter(x=summary['timestamp'], y=summary['dc_accuracy'],
                                         mode='lines+markers', name='Double Chance'))
            fig.update_layout(title="Accuracy Over Time", yaxis_tickformat='.1%')
            st.plotly_chart(fig, width='stretch')

        if adaptive:
            st.markdown("---")
            st.markdown("### 🚨 Adaptive Monitor")
            c1, c2 = st.columns(2)
            c1.metric("Overall Status", adaptive.get('overall_status', 'no_data').upper())
            c2.metric("Trend", adaptive.get('trend', 'unknown'))

            windows = adaptive.get('windows', {})
            if windows:
                st.markdown("#### Rolling Windows")
                for wname, w in windows.items():
                    if w:
                        st.write(f"**{wname}**: {w['accuracy']*100:.1f}% ({w['correct']}/{w['matches']})")

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
            st.plotly_chart(fig_cm, width='stretch')

        if not log.empty:
            st.markdown("### Recent Match Results")
            cols = ['Date', 'League', 'HomeTeam', 'AwayTeam',
                    'prediction', 'dc_prediction', 'FTR', 'correct', 'dc_correct']
            available = [c for c in cols if c in log.columns]
            st.dataframe(log[available].sort_values('Date', ascending=False).head(50),
                         width='stretch')


# ==============================================================================
# VIEW 4: MODEL INSIGHTS
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
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("### 1X2 Distribution")
            dist = preds['prediction'].value_counts().reset_index()
            dist.columns = ['Outcome', 'Count']
            fig = px.pie(dist, names='Outcome', values='Count', color='Outcome',
                         color_discrete_map={'H': '#2E86C1', 'D': '#F1C40F', 'A': '#E74C3C'})
            st.plotly_chart(fig, width='stretch')

        if 'dc_prediction' in preds.columns:
            st.markdown("### Double Chance Distribution")
            dc_dist = preds['dc_prediction'].value_counts().reset_index()
            dc_dist.columns = ['DC Outcome', 'Count']
            fig = px.bar(dc_dist, x='DC Outcome', y='Count', color='DC Outcome',
                         color_discrete_map={'1X': '#7FB3D5', 'X2': '#F7DC6F', '12': '#EC7063'})
            st.plotly_chart(fig, width='stretch')

        if 'League' in preds.columns and preds['League'].nunique() > 1:
            st.markdown("### Predictions by League")
            fig = px.histogram(preds, x='League', color='prediction',
                               color_discrete_map={'H': '#2E86C1', 'D': '#F1C40F', 'A': '#E74C3C'},
                               barmode='stack')
            st.plotly_chart(fig, width='stretch')


# ==============================================================================
# VIEW 5: ABOUT
# ==============================================================================

else:
    st.subheader("📖 About Predictd")
    st.markdown("### What is Predictd?")
    st.markdown("A sports prediction lab using market odds across 7 top European competitions.")

    st.markdown("### Markets Covered")
    st.markdown("- **1X2** — Home / Draw / Away")
    st.markdown("- **Double Chance** — 1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away)")
    st.markdown("- **Bet Slips** — Learning-optimized combinations (4 strategies × 5 sizes = 20 slips)")

    st.markdown("### Bet Slip Strategies")
    st.markdown("- **top_mixed** — Best overall picks across H/D/A")
    st.markdown("- **home_best** — Best home-win picks only")
    st.markdown("- **away_best** — Best away-win picks only")
    st.markdown("- **dc_best** — Best Double Chance picks (highest combined probability)")

    st.markdown("### Competitions")
    st.markdown("- Premier League, La Liga, Bundesliga, Serie A, Ligue 1")
    st.markdown("- Champions League, Europa League")

    st.markdown("### Model: V20")
    st.markdown("- **Odds source:** Pinnacle (preferred), Betfair Exchange (fallback)")
    st.markdown("- **Learning:** Weights update from historical accuracy per league/type/confidence")

    st.markdown("### Commands")
    st.code(
        "python src/predict_live.py        # Generate predictions\n"
        "python src/bet_slip_builder.py    # Build 20 bet slips\n"
        "python src/track_accuracy.py      # Track accuracy + slips\n"
        "streamlit run dashboard.py        # Launch dashboard",
        language="bash"
    )