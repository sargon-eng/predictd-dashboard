"""
Predictd - Bet Slip Builder (v4)
- 4 strategies × 5 sizes = 20 slips
- All picks within a single 8-day week window
- Enforced league diversity per slip
- Learning weights from historical accuracy
- Jitter shuffle for variation between strategies
- Confidence floor: only picks with confidence >= MIN_PICK_CONFIDENCE
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime


PREDICTIONS_DIR = "predictions"
LOGS_DIR = "logs"
MIN_MATCHES_FOR_LEARNING = 10

# Week window: 8 days from today (covers full weekend + mid-week UCL/UEL)
WEEK_WINDOW_DAYS = 8

# Confidence floor: skip picks whose confidence is below this threshold
# Rationale: low-confidence picks (0.4-0.5) historically underperform
MIN_PICK_CONFIDENCE = 0.50

# Strategy seeds (different seeds = different picks)
STRATEGY_SEEDS = {
    'top_mixed': 100,
    'home_best': 200,
    'away_best': 300,
    'dc_best': 400,
}


# ==============================================================================
# LOADERS
# ==============================================================================

def load_latest_predictions():
    files = sorted(glob.glob(f"{PREDICTIONS_DIR}/live_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
    return df


def load_accuracy_log():
    path = f"{LOGS_DIR}/accuracy_log.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ==============================================================================
# WEEK WINDOW
# ==============================================================================

def get_week_window(df, window_days=WEEK_WINDOW_DAYS):
    today = pd.Timestamp.now().normalize()
    end = today + pd.Timedelta(days=window_days)
    return df[(df['Date'] >= today) & (df['Date'] < end)].copy()


# ==============================================================================
# LEARNING WEIGHTS
# ==============================================================================

def compute_learning_weights(acc_log):
    weights = {'league': {}, 'type_1x2': {}, 'type_dc': {}, 'confidence': {}}
    if acc_log.empty or 'correct' not in acc_log.columns:
        return weights

    def acc_to_weight(acc):
        w = 0.7 + (acc - 0.5) * 3
        return max(0.7, min(1.3, float(w)))

    if 'League' in acc_log.columns:
        for league in acc_log['League'].dropna().unique():
            sub = acc_log[acc_log['League'] == league]
            if len(sub) >= MIN_MATCHES_FOR_LEARNING:
                weights['league'][league] = acc_to_weight(sub['correct'].mean())

    if 'prediction' in acc_log.columns:
        for pred in ['H', 'D', 'A']:
            sub = acc_log[acc_log['prediction'] == pred]
            if len(sub) >= MIN_MATCHES_FOR_LEARNING:
                weights['type_1x2'][pred] = acc_to_weight(sub['correct'].mean())

    if 'dc_prediction' in acc_log.columns and 'dc_correct' in acc_log.columns:
        for pred in ['1X', 'X2', '12']:
            sub = acc_log[acc_log['dc_prediction'] == pred]
            if len(sub) >= MIN_MATCHES_FOR_LEARNING and sub['dc_correct'].notna().any():
                weights['type_dc'][pred] = acc_to_weight(sub['dc_correct'].mean())

    if 'max_conf' not in acc_log.columns:
        conf_cols = [c for c in ['prob_home', 'prob_draw', 'prob_away'] if c in acc_log.columns]
        if conf_cols:
            acc_log = acc_log.copy()
            acc_log['max_conf'] = acc_log[conf_cols].max(axis=1)

    if 'max_conf' in acc_log.columns:
        for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            sub = acc_log[(acc_log['max_conf'] >= lo) & (acc_log['max_conf'] < hi)]
            if len(sub) >= MIN_MATCHES_FOR_LEARNING:
                weights['confidence'][(lo, hi)] = acc_to_weight(sub['correct'].mean())

    return weights


# ==============================================================================
# SCORING
# ==============================================================================

def get_confidence_bucket(conf):
    if conf < 0.4:
        return (0.0, 0.4)
    elif conf < 0.5:
        return (0.4, 0.5)
    elif conf < 0.6:
        return (0.5, 0.6)
    elif conf < 0.7:
        return (0.6, 0.7)
    return (0.7, 1.0)


def score_picks(df, weights):
    df = df.copy()

    for std, alt in [('pred_home', 'prob_home'),
                     ('pred_draw', 'prob_draw'),
                     ('pred_away', 'prob_away')]:
        if std not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: std})

    for col in ['pred_home', 'pred_draw', 'pred_away']:
        if col not in df.columns:
            df[col] = 0.333

    df['confidence_1x2'] = df[['pred_home', 'pred_draw', 'pred_away']].max(axis=1)

    if 'prob_1x' not in df.columns:
        df['prob_1x'] = df['pred_home'] + df['pred_draw']
        df['prob_x2'] = df['pred_draw'] + df['pred_away']
        df['prob_12'] = df['pred_home'] + df['pred_away']

    if 'dc_prediction' not in df.columns:
        dc_probs = df[['prob_1x', 'prob_x2', 'prob_12']].values
        df['dc_prediction'] = np.array(['1X', 'X2', '12'])[np.argmax(dc_probs, axis=1)]

    dc_lookup = {'1X': 'prob_1x', 'X2': 'prob_x2', '12': 'prob_12'}
    df['dc_confidence'] = df.apply(
        lambda r: r[dc_lookup[r['dc_prediction']]], axis=1
    )

    df['league_weight'] = df['League'].map(weights['league']).fillna(1.0)
    df['type_weight_1x2'] = df['prediction'].map(weights['type_1x2']).fillna(1.0)
    df['type_weight_dc'] = df['dc_prediction'].map(weights['type_dc']).fillna(1.0)

    df['conf_bucket'] = df['confidence_1x2'].apply(get_confidence_bucket)
    df['conf_weight'] = df['conf_bucket'].map(weights['confidence']).fillna(1.0)

    df['score_1x2'] = (
        df['confidence_1x2'] * df['league_weight'] *
        df['type_weight_1x2'] * df['conf_weight']
    )
    df['score_dc'] = (
        df['dc_confidence'] * df['league_weight'] *
        df['type_weight_dc'] * df['conf_weight']
    )

    return df


# ==============================================================================
# LEAGUE DIVERSITY RULE
# ==============================================================================

def max_per_league_for_size(size):
    if size <= 3:
        return 1
    elif size <= 5:
        return 2
    else:
        return 3


# ==============================================================================
# SLIP CONSTRUCTION
# ==============================================================================

def build_slips_for_strategy(week_df, market, strategy, sizes):
    """
    Build slips using only the week window.
    Applies MIN_PICK_CONFIDENCE filter to exclude weak picks.
    Jitter shuffle ensures different strategies pick different games.
    """
    if week_df.empty:
        return []

    if market == '1x2':
        score_col = 'score_1x2'
        prob_col = 'confidence_1x2'
        pred_col = 'prediction'
    else:
        score_col = 'score_dc'
        prob_col = 'dc_confidence'
        pred_col = 'dc_prediction'

    filtered = week_df.copy()

    # ---- CONFIDENCE FLOOR ----
    # Skip any pick below MIN_PICK_CONFIDENCE threshold.
    # The 0.4-0.5 bucket historically underperforms, so we exclude it.
    filtered = filtered[filtered[prob_col] >= MIN_PICK_CONFIDENCE]

    # ---- STRATEGY FILTER ----
    if strategy == 'home_best':
        filtered = filtered[filtered['prediction'] == 'H']
    elif strategy == 'away_best':
        filtered = filtered[filtered['prediction'] == 'A']

    if filtered.empty:
        return []

    # ---- JITTER SHUFFLE ----
    seed = STRATEGY_SEEDS.get(strategy, 42)
    rng = np.random.default_rng(seed)
    filtered['jittered_score'] = filtered[score_col] + rng.normal(0, 0.01, len(filtered))
    filtered = filtered.sort_values('jittered_score', ascending=False)

    slips = []

    for size in sizes:
        max_per_league = max_per_league_for_size(size)
        picked = []
        league_counts = {}
        used_matches = set()

        for _, row in filtered.iterrows():
            if len(picked) >= size:
                break
            league = row['League']
            if league_counts.get(league, 0) >= max_per_league:
                continue
            match_key = (row['Date'], row['HomeTeam'], row['AwayTeam'])
            if match_key in used_matches:
                continue

            picked.append({
                'Date': row['Date'],
                'League': league,
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
                'prediction': row[pred_col],
                'probability': float(row[prob_col]),
                'score': float(row[score_col]),
            })
            league_counts[league] = league_counts.get(league, 0) + 1
            used_matches.add(match_key)

        if len(picked) < size:
            continue

        combined = float(np.prod([p['probability'] for p in picked]))

        slips.append({
            'market': market,
            'strategy': strategy,
            'size': size,
            'picks': picked,
            'combined_prob': combined,
            'leagues': list(league_counts.keys()),
        })

    return slips


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("PREDICTD - BET SLIP BUILDER v4 (CONFIDENCE-FLOORED)")
    print("=" * 70)

    preds = load_latest_predictions()
    if preds.empty:
        print("[!] No predictions found. Run predict_live.py first.")
        return

    print(f"[OK] Loaded {len(preds)} predictions")
    print(f"[CFG] MIN_PICK_CONFIDENCE = {MIN_PICK_CONFIDENCE}")

    acc_log = load_accuracy_log()
    print(f"[OK] Loaded {len(acc_log)} historical results for learning")

    weights = compute_learning_weights(acc_log)

    def clean(w):
        return {k: round(v, 2) for k, v in w.items()}

    print("\nLearned weights:")
    print(f"  Leagues:   {clean(weights['league']) if weights['league'] else '(neutral)'}")
    print(f"  1X2 types: {clean(weights['type_1x2']) if weights['type_1x2'] else '(neutral)'}")
    print(f"  DC types:  {clean(weights['type_dc']) if weights['type_dc'] else '(neutral)'}")
    print(f"  Confidence buckets: {clean(weights['confidence']) if weights['confidence'] else '(neutral)'}")

    df = score_picks(preds, weights)
    print(f"\n[OK] Scored {len(df)} picks")

    # Apply week window
    today = pd.Timestamp.now().normalize()
    week_df = get_week_window(df, WEEK_WINDOW_DAYS)
    week_end = today + pd.Timedelta(days=WEEK_WINDOW_DAYS - 1)

    print(f"\n[WEEK WINDOW] {today.date()} to {week_end.date()} ({WEEK_WINDOW_DAYS} days)")
    print(f"[OK] {len(week_df)} matches in window:")

    # Show confidence filter impact per league
    for league in sorted(week_df['League'].unique()):
        league_df = week_df[week_df['League'] == league]
        total = len(league_df)
        filtered_1x2 = len(league_df[league_df['confidence_1x2'] >= MIN_PICK_CONFIDENCE])
        filtered_dc = len(league_df[league_df['dc_confidence'] >= MIN_PICK_CONFIDENCE])
        print(f"  {league:20}: {total} matches "
              f"(1X2: {filtered_1x2} pass, DC: {filtered_dc} pass)")

    if week_df.empty:
        print("[!] No matches in week window.")
        return

    # Total usable picks after filter
    usable_1x2 = len(week_df[week_df['confidence_1x2'] >= MIN_PICK_CONFIDENCE])
    usable_dc = len(week_df[week_df['dc_confidence'] >= MIN_PICK_CONFIDENCE])
    print(f"\n[FILTER] {usable_1x2} 1X2 picks and {usable_dc} DC picks pass "
          f"confidence >= {MIN_PICK_CONFIDENCE}")

    # Generate slips
    sizes = [3, 4, 5, 6, 7]
    all_slips = []
    all_slips += build_slips_for_strategy(week_df, '1x2', 'top_mixed', sizes)
    all_slips += build_slips_for_strategy(week_df, '1x2', 'home_best', sizes)
    all_slips += build_slips_for_strategy(week_df, '1x2', 'away_best', sizes)
    all_slips += build_slips_for_strategy(week_df, 'dc', 'dc_best', sizes)

    if not all_slips:
        print("\n[!] No slips could be built after confidence filter.")
        print(f"     Try lowering MIN_PICK_CONFIDENCE below {MIN_PICK_CONFIDENCE}")
        return

    for i, s in enumerate(all_slips, 1):
        s['slip_id'] = i

    print("\n" + "=" * 70)
    print(f"GENERATED {len(all_slips)} BET SLIPS (all resolve by {week_end.date()})")
    print("=" * 70)

    for slip in all_slips:
        print(f"\n--- SLIP #{slip['slip_id']} [{slip['strategy']}] "
              f"({slip['market'].upper()}, {slip['size']}-fold) ---")
        print(f"Combined probability: {slip['combined_prob']*100:.2f}% | "
              f"Odds: {1/slip['combined_prob']:.2f}x")
        print(f"Leagues: {', '.join(slip['leagues'])}")
        for pick in slip['picks']:
            print(f"  {pick['Date'].strftime('%a %m-%d')} | {pick['League']:16} | "
                  f"{pick['HomeTeam']:22} vs {pick['AwayTeam']:22} | "
                  f"{pick['prediction']} | {pick['probability']*100:.1f}% | score: {pick['score']:.3f}")

    # Save flat CSV
    rows = []
    for slip in all_slips:
        for i, pick in enumerate(slip['picks'], 1):
            rows.append({
                'slip_id': slip['slip_id'],
                'strategy': slip['strategy'],
                'market': slip['market'],
                'size': slip['size'],
                'combined_prob': round(slip['combined_prob'], 4),
                'pick_num': i,
                'Date': pick['Date'],
                'League': pick['League'],
                'HomeTeam': pick['HomeTeam'],
                'AwayTeam': pick['AwayTeam'],
                'prediction': pick['prediction'],
                'probability': round(pick['probability'], 4),
                'score': round(pick['score'], 4),
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    output_file = f"{PREDICTIONS_DIR}/bet_slips_{datetime.now().strftime('%Y%m%d')}.csv"
    out_df.to_csv(output_file, index=False)

    print(f"\n[OK] Saved {len(all_slips)} slips to: {output_file}")
    print("\nBreakdown by strategy:")
    for strat in ['top_mixed', 'home_best', 'away_best', 'dc_best']:
        count = sum(1 for s in all_slips if s['strategy'] == strat)
        print(f"  {strat:12}: {count} slips")

    print("\nLeague coverage per strategy (unique leagues used):")
    for strat in ['top_mixed', 'home_best', 'away_best', 'dc_best']:
        strat_slips = [s for s in all_slips if s['strategy'] == strat]
        all_leagues = set()
        for s in strat_slips:
            all_leagues.update(s['leagues'])
        print(f"  {strat:12}: {sorted(all_leagues)}")


if __name__ == "__main__":
    main()