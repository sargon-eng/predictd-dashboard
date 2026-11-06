"""
Predictd - Adaptive Monitor
Analyzes recent accuracy performance and logs alerts.
Does NOT change predictions. Observation only.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


LOGS_DIR = "logs"
ACCURACY_LOG = f"{LOGS_DIR}/accuracy_log.csv"
STATUS_FILE = f"{LOGS_DIR}/adaptive_status.json"


# ==============================================================================
# THRESHOLDS
# ==============================================================================

THRESHOLDS = {
    'excellent': 0.55,
    'good': 0.50,
    'warning': 0.45,
    'critical': 0.40,
}

# Calibration: if high-conf accuracy is more than this below expected, warn
CALIBRATION_GAP_WARNING = 0.15

# Minimum matches for a group to be evaluated
MIN_MATCHES_FOR_SIGNAL = 10

# Trend thresholds: require this much change to call it improving/worsening
# Wider than before to avoid false alarms from weekend clustering
TREND_DELTA_IMPROVING = 0.08
TREND_DELTA_WORSENING = -0.08


# ==============================================================================
# HELPERS
# ==============================================================================

def compute_window_accuracy(df, window):
    if len(df) < window:
        return None
    recent = df.tail(window)
    return {
        'matches': len(recent),
        'correct': int(recent['correct'].sum()),
        'accuracy': float(recent['correct'].mean()),
    }


def status_from_accuracy(acc):
    if acc is None:
        return 'no_data'
    if acc >= THRESHOLDS['excellent']:
        return 'excellent'
    if acc >= THRESHOLDS['good']:
        return 'good'
    if acc >= THRESHOLDS['warning']:
        return 'warning'
    return 'critical'


def detect_trend(df, window=30):
    """
    Detect if accuracy is improving, worsening, or stable.
    Uses wider thresholds (+/- 8pp) to avoid false alarms from
    weekend-clustered match data.
    """
    if len(df) < window * 2:
        return 'insufficient_data'

    first_half = df.iloc[-window * 2:-window]['correct'].mean()
    second_half = df.iloc[-window:]['correct'].mean()
    delta = second_half - first_half

    if delta > TREND_DELTA_IMPROVING:
        return 'improving'
    elif delta < TREND_DELTA_WORSENING:
        return 'worsening'
    return 'stable'


def detect_calibration_issues(df):
    if 'max_conf' not in df.columns:
        return None

    issues = []
    for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        sub = df[(df['max_conf'] >= lo) & (df['max_conf'] < hi)]
        if len(sub) < 5:
            continue
        acc = sub['correct'].mean()
        midpoint = (lo + hi) / 2
        gap = midpoint - acc
        if gap > CALIBRATION_GAP_WARNING:
            issues.append({
                'bucket': f"{lo:.1f}-{hi:.1f}",
                'predicted': round(midpoint, 3),
                'actual': round(acc, 3),
                'gap': round(gap, 3),
                'matches': len(sub),
            })
    return issues


def per_league_accuracy(df):
    if 'League' not in df.columns:
        return {}
    result = {}
    for league in df['League'].dropna().unique():
        sub = df[df['League'] == league]
        if len(sub) < 5:
            continue
        result[league] = {
            'matches': len(sub),
            'correct': int(sub['correct'].sum()),
            'accuracy': round(float(sub['correct'].mean()), 4),
        }
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("PREDICTD - ADAPTIVE MONITOR")
    print("=" * 70)

    if not os.path.exists(ACCURACY_LOG):
        print("[!] No accuracy log found. Run track_accuracy.py first.")
        return

    df = pd.read_csv(ACCURACY_LOG)
    if 'correct' not in df.columns or len(df) == 0:
        print("[!] No accuracy data to analyze.")
        return

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)

    df['correct'] = df['correct'].astype(int)

    # Rolling windows
    windows = {
        'last_10': compute_window_accuracy(df, 10),
        'last_30': compute_window_accuracy(df, 30),
        'last_100': compute_window_accuracy(df, 100),
        'all_time': {
            'matches': len(df),
            'correct': int(df['correct'].sum()),
            'accuracy': float(df['correct'].mean()),
        },
    }

    # Determine primary status
    primary = None
    for key in ['last_10', 'last_30', 'last_100', 'all_time']:
        if windows[key] is not None and windows[key]['matches'] >= MIN_MATCHES_FOR_SIGNAL:
            primary = key
            break

    overall_status = 'no_data'
    if primary is not None:
        overall_status = status_from_accuracy(windows[primary]['accuracy'])

    trend = detect_trend(df)
    calibration_issues = detect_calibration_issues(df) or []
    leagues = per_league_accuracy(df)

    # Build alerts
    alerts = []

    if overall_status == 'critical':
        alerts.append({
            'level': 'CRITICAL',
            'message': f"Recent accuracy is below {THRESHOLDS['critical']*100:.0f}% — model is failing",
            'action': 'Consider pausing predictions or retraining',
        })
    elif overall_status == 'warning':
        alerts.append({
            'level': 'WARNING',
            'message': f"Recent accuracy is below {THRESHOLDS['good']*100:.0f}% — model underperforming",
            'action': 'Monitor closely for next 20 matches',
        })
    elif overall_status == 'excellent':
        alerts.append({
            'level': 'EXCELLENT',
            'message': f"Recent accuracy above {THRESHOLDS['excellent']*100:.0f}% — model performing strongly",
            'action': 'No action needed',
        })
    elif overall_status == 'good':
        alerts.append({
            'level': 'GOOD',
            'message': "Recent accuracy is in a healthy range",
            'action': 'No action needed',
        })

    if trend == 'worsening':
        alerts.append({
            'level': 'WARNING',
            'message': 'Accuracy is trending downward over the last 30 matches',
            'action': 'Look for systematic issues (league, time of day)',
        })
    elif trend == 'improving':
        alerts.append({
            'level': 'INFO',
            'message': 'Accuracy is trending upward',
            'action': 'No action needed',
        })

    for issue in calibration_issues:
        alerts.append({
            'level': 'WARNING',
            'message': f"Calibration issue: bucket {issue['bucket']} predicted ~{issue['predicted']*100:.0f}% but got {issue['actual']*100:.0f}%",
            'action': 'Model may be overconfident in this range',
        })

    for league, stats in leagues.items():
        if stats['accuracy'] < 0.40 and stats['matches'] >= 10:
            alerts.append({
                'level': 'WARNING',
                'message': f"{league} accuracy is low: {stats['accuracy']*100:.1f}% ({stats['matches']} matches)",
                'action': f'Consider excluding {league} until accuracy recovers',
            })

    # Save status
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_status': overall_status,
        'primary_window': primary,
        'trend': trend,
        'windows': windows,
        'leagues': leagues,
        'calibration_issues': calibration_issues,
        'alerts': alerts,
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2, default=str)

    # Print summary
    print(f"\n[OVERALL STATUS] {overall_status.upper()}")
    print(f"[TREND]          {trend}")
    print(f"[PRIMARY]        {primary if primary else 'none'}")

    print("\nRolling windows:")
    for name, w in windows.items():
        if w is None:
            print(f"  {name:12}: insufficient data")
        else:
            print(f"  {name:12}: {w['accuracy']*100:5.1f}% ({w['correct']}/{w['matches']})")

    print("\nPer-league accuracy:")
    for league, stats in leagues.items():
        print(f"  {league:20}: {stats['accuracy']*100:5.1f}% ({stats['correct']}/{stats['matches']})")

    if calibration_issues:
        print("\nCalibration issues detected:")
        for issue in calibration_issues:
            print(f"  Bucket {issue['bucket']}: predicted {issue['predicted']*100:.0f}% → actual {issue['actual']*100:.0f}%")

    print("\nAlerts:")
    if not alerts:
        print("  (none)")
    else:
        for a in alerts:
            print(f"  [{a['level']:9}] {a['message']}")

    print(f"\n[OK] Status saved to: {STATUS_FILE}")


if __name__ == "__main__":
    main()