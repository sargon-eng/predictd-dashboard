"""
Predictd - Accuracy Tracker (Multi-League, 1X2 + Double Chance + Bet Slips)
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from team_mapper import normalize_team


PREDICTIONS_DIR = "predictions"
DATA_DIR = "data/raw/multi_league"
LOGS_DIR = "logs"

LEAGUE_TO_CODE = {
    'Premier League': 'E0',
    'La Liga': 'SP1',
    'Bundesliga': 'D1',
    'Serie A': 'I1',
    'Ligue 1': 'F1',
}

CURRENT_SEASON = '2627'


# ==============================================================================
# LOADERS
# ==============================================================================

def load_predictions():
    files = sorted(glob.glob(f"{PREDICTIONS_DIR}/live_*.csv"))
    if not files:
        print("[!] No prediction files found.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[!] Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    preds = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Loaded {len(preds)} prediction rows from {len(dfs)} files")

    preds['Date'] = pd.to_datetime(preds['Date'], errors='coerce').dt.normalize()
    preds['HomeTeam'] = preds['HomeTeam'].apply(normalize_team)
    preds['AwayTeam'] = preds['AwayTeam'].apply(normalize_team)
    preds = preds.dropna(subset=['Date'])
    preds = preds.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
    print(f"[OK] After deduplication: {len(preds)} unique matches")

    # ============================================================
    # DERIVE DC PREDICTIONS — always fill any NaN rows
    # ============================================================
    if 'dc_prediction' not in preds.columns:
        preds['dc_prediction'] = np.nan

    # Identify probability source columns
    conf_cols = None
    if 'prob_home' in preds.columns and preds['prob_home'].notna().any():
        conf_cols = ('prob_home', 'prob_draw', 'prob_away')
    elif 'pred_home' in preds.columns and preds['pred_home'].notna().any():
        conf_cols = ('pred_home', 'pred_draw', 'pred_away')

    if conf_cols:
        # Build per-row ph/pd/pa, preferring prob_* but falling back to pred_* per row
        ph = preds[conf_cols[0]].copy()
        pd_ = preds[conf_cols[1]].copy()
        pa = preds[conf_cols[2]].copy()

        # For any NaN in prob_* rows, backfill from pred_*
        if 'pred_home' in preds.columns and conf_cols[0] == 'prob_home':
            ph = ph.fillna(preds['pred_home'])
            pd_ = pd_.fillna(preds['pred_draw'])
            pa = pa.fillna(preds['pred_away'])

        # Final fallback to uniform
        ph = ph.fillna(0.333)
        pd_ = pd_.fillna(0.333)
        pa = pa.fillna(0.333)

        prob_1x = ph + pd_
        prob_x2 = pd_ + pa
        prob_12 = ph + pa
        dc_probs = np.column_stack([prob_1x.values, prob_x2.values, prob_12.values])
        derived_dc = np.array(['1X', 'X2', '12'])[np.argmax(dc_probs, axis=1)]

        # Fill any NaN in dc_prediction (regardless of how many)
        nan_mask = preds['dc_prediction'].isna()
        n_filled = int(nan_mask.sum())
        if n_filled > 0:
            preds.loc[nan_mask, 'dc_prediction'] = derived_dc[nan_mask.values]
            print(f"[OK] Filled {n_filled} missing DC predictions")

    return preds


def load_league_actuals(league_name):
    code = LEAGUE_TO_CODE.get(league_name)
    if code is None:
        return pd.DataFrame()

    filepath = os.path.join(DATA_DIR, 'by_league', f"{code}_{CURRENT_SEASON}.csv")
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    except Exception as e:
        print(f"[!] Failed to read {filepath}: {e}")
        return pd.DataFrame()

    if 'Date' not in df.columns:
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    df['HomeTeam'] = df['HomeTeam'].apply(normalize_team)
    df['AwayTeam'] = df['AwayTeam'].apply(normalize_team)
    df = df.dropna(subset=['Date', 'FTR'])
    df['League'] = league_name
    return df[['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]


def load_all_actuals():
    dfs = []
    for league in LEAGUE_TO_CODE.keys():
        df = load_league_actuals(league)
        if not df.empty:
            print(f"   {league:20}: {len(df)} matches")
            dfs.append(df)
        else:
            print(f"   {league:20}: (no data)")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def actual_dc_outcome(ftr):
    if pd.isna(ftr):
        return []
    if ftr == 'H':
        return ['1X', '12']
    elif ftr == 'D':
        return ['1X', 'X2']
    elif ftr == 'A':
        return ['X2', '12']
    return []


def dc_correct(row):
    actual_set = actual_dc_outcome(row['FTR'])
    if not actual_set:
        return 0
    dc = row.get('dc_prediction')
    if pd.isna(dc):
        return 0
    return int(dc in actual_set)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("PREDICTD - ACCURACY TRACKER (1X2 + DC + BET SLIPS)")
    print("=" * 70)

    preds = load_predictions()
    if preds.empty:
        print("[!] No predictions to check.")
        return

    has_dc = 'dc_prediction' in preds.columns and preds['dc_prediction'].notna().any()

    print(f"\n[..] Loading actual results for {CURRENT_SEASON} season:")
    actuals = load_all_actuals()
    if actuals.empty:
        print("[!] No actual results found.")
        return

    print(f"\n[OK] Total actuals: {len(actuals)} matches")

    merged = pd.merge(
        preds.drop(columns=['League'], errors='ignore'),
        actuals[['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTR']],
        on=['Date', 'HomeTeam', 'AwayTeam'],
        how='inner'
    )

    today = pd.Timestamp.now().normalize()
    future_preds = preds[preds['Date'] > today]
    if len(future_preds) > 0:
        next_match = future_preds['Date'].min()
        days_until = (next_match - today).days
        print(f"\n[..] {len(future_preds)} predictions waiting for future matches")
        print(f"     Next match: {next_match.date()} ({days_until} days from now)")

    if merged.empty:
        print("\n[!] No overlapping matches found.")
        return

    merged['correct'] = (merged['prediction'] == merged['FTR']).astype(int)

    if has_dc and 'dc_prediction' in merged.columns:
        merged['dc_correct'] = merged.apply(dc_correct, axis=1)
    else:
        has_dc = False

    total = len(merged)
    correct = merged['correct'].sum()
    accuracy = merged['correct'].mean()

    print("\n" + "=" * 70)
    print(f"1X2 ACCURACY: {correct}/{total} correct  =  {accuracy * 100:.2f}%")

    dc_accuracy = None
    if has_dc:
        dc_correct_count = int(merged['dc_correct'].sum())
        dc_accuracy = merged['dc_correct'].mean()
        print(f"DOUBLE CHANCE ACCURACY: {dc_correct_count}/{total} correct  =  {dc_accuracy * 100:.2f}%")

    print("=" * 70)

    print("\nAccuracy by league:")
    for league in LEAGUE_TO_CODE.keys():
        sub = merged[merged['League'] == league]
        if len(sub) == 0:
            continue
        acc = sub['correct'].mean()
        dc_str = ""
        if has_dc:
            dc_acc = sub['dc_correct'].mean()
            dc_str = f" | DC: {dc_acc*100:5.1f}% ({int(sub['dc_correct'].sum())}/{len(sub)})"
        print(f"  {league:20}: {acc * 100:5.1f}% ({int(sub['correct'].sum())}/{len(sub)}){dc_str}")

    print("\n1X2 Accuracy by prediction type:")
    for pred in ['H', 'D', 'A']:
        sub = merged[merged['prediction'] == pred]
        if len(sub) == 0:
            print(f"  Predicted {pred}: (none)")
            continue
        acc = sub['correct'].mean()
        print(f"  Predicted {pred}: {acc * 100:5.1f}%  ({int(sub['correct'].sum())}/{len(sub)})")

    if has_dc:
        print("\nDouble Chance Accuracy by prediction type:")
        for pred in ['1X', 'X2', '12']:
            sub = merged[merged['dc_prediction'] == pred]
            if len(sub) == 0:
                print(f"  Predicted {pred}: (none)")
                continue
            acc = sub['dc_correct'].mean()
            print(f"  Predicted {pred}: {acc * 100:5.1f}%  ({int(sub['dc_correct'].sum())}/{len(sub)})")

    conf_cols = None
    if 'prob_home' in merged.columns and merged['prob_home'].notna().any():
        conf_cols = ['prob_home', 'prob_draw', 'prob_away']
    elif 'pred_home' in merged.columns and merged['pred_home'].notna().any():
        conf_cols = ['pred_home', 'pred_draw', 'pred_away']

    if conf_cols:
        merged['max_conf'] = merged[conf_cols].max(axis=1)
        print("\n1X2 Accuracy by confidence:")
        for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            sub = merged[(merged['max_conf'] >= lo) & (merged['max_conf'] < hi)]
            if len(sub) == 0:
                continue
            acc = sub['correct'].mean()
            print(f"  Confidence {lo:.1f}-{hi:.1f}: {acc * 100:5.1f}%  ({int(sub['correct'].sum())}/{len(sub)})")

    print("\n" + "=" * 70)
    print("MATCH-BY-MATCH (first 30)")
    print("=" * 70)
    for _, r in merged.sort_values('Date').head(30).iterrows():
        status_1x2 = "OK  " if r['correct'] else "MISS"
        league = r.get('League', 'Unknown')
        dc_status = ""
        if has_dc:
            dc_pred = r.get('dc_prediction', '?')
            dc_ok = "OK" if r.get('dc_correct', 0) else "MISS"
            dc_status = f" | DC:{dc_pred} {dc_ok}"
        print(f"  [{status_1x2}] {r['Date'].strftime('%Y-%m-%d')} | "
              f"{str(league):15} | {r['HomeTeam']:20} vs {r['AwayTeam']:20} | "
              f"1X2:{r['prediction']} Actual:{r['FTR']}{dc_status}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    merged.to_csv(f"{LOGS_DIR}/accuracy_log.csv", index=False)

    summary_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_matches': total,
        'correct': int(correct),
        'accuracy': round(accuracy, 4),
    }
    if has_dc:
        summary_row['dc_correct'] = int(merged['dc_correct'].sum())
        summary_row['dc_accuracy'] = round(dc_accuracy, 4)

    summary_file = f"{LOGS_DIR}/accuracy_summary.csv"
    if os.path.exists(summary_file):
        existing = pd.read_csv(summary_file)
        summary = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary = pd.DataFrame([summary_row])
    summary.to_csv(summary_file, index=False)

    print(f"\n[OK] Saved match log to {LOGS_DIR}/accuracy_log.csv")
    print(f"[OK] Saved summary to {LOGS_DIR}/accuracy_summary.csv")

    # ============================================================
    # BET SLIP ACCURACY
    # ============================================================
    print("\n" + "=" * 70)
    print("BET SLIP ACCURACY")
    print("=" * 70)

    try:
        slip_files = sorted(glob.glob(f"{PREDICTIONS_DIR}/bet_slips_*.csv"))
        if not slip_files:
            print("[..] No bet slip files found.")
        else:
            print(f"[OK] Found {len(slip_files)} bet slip snapshot(s)")
            all_slips = []
            for f in slip_files:
                base = os.path.basename(f)
                date_str = base.replace('bet_slips_', '').replace('.csv', '')
                try:
                    slip_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                except Exception:
                    continue
                sdf = pd.read_csv(f)
                if sdf.empty:
                    continue
                sdf['slip_date'] = slip_date
                all_slips.append(sdf)

            if all_slips:
                slips_df = pd.concat(all_slips, ignore_index=True)
                slips_df['Date'] = pd.to_datetime(slips_df['Date'], errors='coerce').dt.normalize()
                slips_df['HomeTeam'] = slips_df['HomeTeam'].apply(normalize_team)
                slips_df['AwayTeam'] = slips_df['AwayTeam'].apply(normalize_team)

                results = []
                group_cols = ['slip_date', 'strategy', 'market', 'size', 'slip_id']

                for key, grp in slips_df.groupby(group_cols):
                    slip_date, strategy, market, size, slip_id = key
                    grp = grp.sort_values('pick_num')

                    all_played = True
                    all_correct = True

                    for _, pick in grp.iterrows():
                        mask = (
                            (actuals['Date'] == pick['Date']) &
                            (actuals['HomeTeam'] == pick['HomeTeam']) &
                            (actuals['AwayTeam'] == pick['AwayTeam'])
                        )
                        m = actuals[mask]
                        if m.empty:
                            all_played = False
                            break
                        ftr = m.iloc[0]['FTR']

                        if market == '1x2':
                            correct_pick = (pick['prediction'] == ftr)
                        else:
                            correct_pick = pick['prediction'] in actual_dc_outcome(ftr)

                        if not correct_pick:
                            all_correct = False

                    if not all_played:
                        continue

                    results.append({
                        'slip_date': slip_date,
                        'strategy': strategy,
                        'market': market,
                        'size': int(size),
                        'slip_id': int(slip_id),
                        'combined_prob': float(grp['combined_prob'].iloc[0]),
                        'outcome': 'WIN' if all_correct else 'LOSS',
                    })

                if results:
                    results_df = pd.DataFrame(results)
                    slip_accuracy_file = f"{LOGS_DIR}/bet_slip_accuracy.csv"
                    results_df.to_csv(slip_accuracy_file, index=False)

                    wins = (results_df['outcome'] == 'WIN').sum()
                    total_slips = len(results_df)
                    win_rate = wins / total_slips

                    print(f"\n[OK] Evaluated {total_slips} bet slips (all picks resolved)")
                    print(f"     Wins:     {wins}")
                    print(f"     Losses:   {total_slips - wins}")
                    print(f"     Win rate: {win_rate*100:.2f}%")

                    print("\nWin rate by strategy:")
                    for strat in ['top_mixed', 'home_best', 'away_best', 'dc_best']:
                        sub = results_df[results_df['strategy'] == strat]
                        if len(sub) == 0:
                            continue
                        w = (sub['outcome'] == 'WIN').sum()
                        print(f"  {strat:12}: {w}/{len(sub)} = {w/len(sub)*100:.1f}%")

                    print("\nWin rate by slip size:")
                    for sz in sorted(results_df['size'].unique()):
                        sub = results_df[results_df['size'] == sz]
                        w = (sub['outcome'] == 'WIN').sum()
                        print(f"  {sz}-fold: {w}/{len(sub)} = {w/len(sub)*100:.1f}%")

                    print("\nWin rate by market:")
                    for mk in results_df['market'].unique():
                        sub = results_df[results_df['market'] == mk]
                        w = (sub['outcome'] == 'WIN').sum()
                        print(f"  {mk:6}: {w}/{len(sub)} = {w/len(sub)*100:.1f}%")

                    avg_pred = results_df['combined_prob'].mean()
                    print(f"\nCalibration check:")
                    print(f"  Avg predicted probability: {avg_pred*100:.2f}%")
                    print(f"  Actual win rate:           {win_rate*100:.2f}%")
                    gap = (avg_pred - win_rate) * 100
                    if gap > 5:
                        print(f"  [!] Overconfident by {gap:.1f} percentage points")
                    elif gap < -5:
                        print(f"  [OK] Under-confident by {abs(gap):.1f} pp (good news)")
                    else:
                        print(f"  [OK] Well-calibrated (gap: {gap:.1f} pp)")

                else:
                    print("[..] No bet slips with fully-played matches yet.")
                    print("     Slips will be evaluated once all their matches are played.")

    except Exception as e:
        print(f"[!] Bet slip tracking failed: {e}")

    # ============================================================
    # ADAPTIVE MONITOR
    # ============================================================
    print("\n" + "=" * 70)
    print("Running adaptive monitor...")
    print("=" * 70)
    try:
        import adaptive_monitor
        adaptive_monitor.main()
    except Exception as e:
        print(f"[!] Adaptive monitor failed: {e}")


if __name__ == "__main__":
    main()