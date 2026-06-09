from pathlib import Path
import csv
import math
import statistics

ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / 'demo_videos' / '箱子1_joint_coords.csv'
OUT_CSV = ROOT / 'demo_videos' / '箱子1_angle_diffs_per_frame.csv'

def safe_f(v):
    try:
        return float(v)
    except Exception:
        return None

def main():
    rows = {}
    with open(IN_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            frame = int(r['frame'])
            backend = r['backend']
            if frame not in rows:
                rows[frame] = {}
            rows[frame][backend] = r

    metrics = []
    per_frame = []
    for frame, pair in sorted(rows.items()):
        if 'MEDIAPIPE' not in pair or 'RTMW3D' not in pair:
            continue
        a = pair['MEDIAPIPE']
        b = pair['RTMW3D']
        ul_a = safe_f(a.get('upper_arm_left_deg'))
        ur_a = safe_f(a.get('upper_arm_right_deg'))
        neck_a = safe_f(a.get('neck_angle_signed_deg'))
        ul_b = safe_f(b.get('upper_arm_left_deg'))
        ur_b = safe_f(b.get('upper_arm_right_deg'))
        neck_b = safe_f(b.get('neck_angle_signed_deg'))

        if None in (ul_a, ul_b, ur_a, ur_b, neck_a, neck_b):
            continue

        diff_ul = abs(ul_a - ul_b)
        diff_ur = abs(ur_a - ur_b)
        diff_neck = abs(neck_a - neck_b)

        per_frame.append({'frame': frame, 'diff_ul': diff_ul, 'diff_ur': diff_ur, 'diff_neck': diff_neck,
                          'ul_mp': ul_a, 'ul_rtm': ul_b, 'ur_mp': ur_a, 'ur_rtm': ur_b, 'neck_mp': neck_a, 'neck_rtm': neck_b})
        metrics.append(('ul', diff_ul))
        metrics.append(('ur', diff_ur))
        metrics.append(('neck', diff_neck))

    # compute stats per-metric
    def stats(values):
        return {
            'count': len(values),
            'mean': statistics.mean(values) if values else 0,
            'stdev': statistics.pstdev(values) if values else 0,
            'median': statistics.median(values) if values else 0,
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
        }

    ul_vals = [p['diff_ul'] for p in per_frame]
    ur_vals = [p['diff_ur'] for p in per_frame]
    neck_vals = [p['diff_neck'] for p in per_frame]

    s_ul = stats(ul_vals)
    s_ur = stats(ur_vals)
    s_neck = stats(neck_vals)

    def pct_over(vals, thr):
        if not vals: return 0
        return sum(1 for v in vals if v > thr) / len(vals) * 100

    print('Frames compared:', len(per_frame))
    print('\nUpper arm left diffs:')
    print(s_ul)
    print('>5°:', pct_over(ul_vals,5), '%  >10°:', pct_over(ul_vals,10), '%')

    print('\nUpper arm right diffs:')
    print(s_ur)
    print('>5°:', pct_over(ur_vals,5), '%  >10°:', pct_over(ur_vals,10), '%')

    print('\nNeck diffs:')
    print(s_neck)
    print('>5°:', pct_over(neck_vals,5), '%  >10°:', pct_over(neck_vals,10), '%  >20°:', pct_over(neck_vals,20), '%')

    # top frames by neck diff
    top_neck = sorted(per_frame, key=lambda x: x['diff_neck'], reverse=True)[:20]
    print('\nTop 10 frames by neck diff:')
    for p in top_neck[:10]:
        print(p['frame'], 'diff_neck=', round(p['diff_neck'],3), ' mp=', round(p['neck_mp'],3), ' rtm=', round(p['neck_rtm'],3))

    # write per-frame diffs csv
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame','diff_ul','diff_ur','diff_neck','ul_mp','ul_rtm','ur_mp','ur_rtm','neck_mp','neck_rtm'])
        for p in per_frame:
            w.writerow([p['frame'], p['diff_ul'], p['diff_ur'], p['diff_neck'], p['ul_mp'], p['ul_rtm'], p['ur_mp'], p['ur_rtm'], p['neck_mp'], p['neck_rtm']])

    print('\nWrote per-frame diffs to', OUT_CSV)

if __name__ == '__main__':
    main()
