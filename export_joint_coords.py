from pathlib import Path
import sys
import csv
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rula_realtime_app.core.pose_detector import PoseDetector
from rula_realtime_app.core.utils import safe_angle

VIDEO = ROOT / 'demo_videos' / '箱子1.MOV'
OUT_CSV = ROOT / 'demo_videos' / '箱子1_joint_coords.csv'
FRAME_INTERVAL = 5

KEY_INDICES = {
    'SH_L': 11, 'SH_R': 12,
    'EL_L': 13, 'EL_R': 14,
    'HIP_L': 23, 'HIP_R': 24,
    'EAR_L': 7, 'EAR_R': 8,
}

def centroid(a,b):
    return (a + b) / 2.0

def process_backend(cap, backend, writer, frame_idx):
    det = PoseDetector(backend_mode=backend)
    # seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        det.close(); return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected = det.process_frame(rgb)
    pose = det.get_landmarks_array() if detected else None
    if pose is None:
        det.close(); return True

    pose = np.array(pose)
    # extract points
    SH_L = pose[KEY_INDICES['SH_L'], :3]
    SH_R = pose[KEY_INDICES['SH_R'], :3]
    EL_L = pose[KEY_INDICES['EL_L'], :3]
    EL_R = pose[KEY_INDICES['EL_R'], :3]
    HIP_L = pose[KEY_INDICES['HIP_L'], :3]
    HIP_R = pose[KEY_INDICES['HIP_R'], :3]
    EAR_L = pose[KEY_INDICES['EAR_L'], :3]
    EAR_R = pose[KEY_INDICES['EAR_R'], :3]

    SHO_C = centroid(SH_L, SH_R)
    HIP_C = centroid(HIP_L, HIP_R)
    HEAD_C = centroid(EAR_L, EAR_R)

    # upper arm angles
    v_body = HIP_C - SHO_C
    v_sh_el_L = EL_L - SH_L
    v_sh_el_R = EL_R - SH_R
    theta_upper_L = safe_angle(v_sh_el_L, v_body)
    theta_upper_R = safe_angle(v_sh_el_R, v_body)

    # neck angle (signed as in rula_score_side)
    v_u = SHO_C - HIP_C
    v_hip_lr = HIP_R - HIP_L
    v_f = np.cross(v_u, v_hip_lr)
    P_s = np.cross(v_u, v_f)
    v_neck = HEAD_C - SHO_C
    P_s_norm = np.linalg.norm(P_s)
    if P_s_norm > 1e-6:
        P_s_hat = P_s / P_s_norm
        v_neck_proj = v_neck - np.dot(v_neck, P_s_hat) * P_s_hat
    else:
        v_neck_proj = v_neck
    theta_neck = safe_angle(v_neck_proj, v_u)
    v_f_norm = np.linalg.norm(v_f)
    if v_f_norm > 1e-6:
        v_f_hat = v_f / v_f_norm
        neck_forward = np.dot(v_neck_proj, v_f_hat) >= 0
    else:
        neck_forward = True
    signed_neck = theta_neck if neck_forward else -theta_neck

    row = [frame_idx, backend]
    for p in [SH_L, EL_L, SH_R, EL_R, HIP_L, HIP_R, EAR_L, EAR_R]:
        row += [float(np.round(x,6)) for x in p.tolist()]
    for p in [SHO_C, HIP_C, HEAD_C]:
        row += [float(np.round(x,6)) for x in p.tolist()]
    row += [float(np.round(theta_upper_L,6)), float(np.round(theta_upper_R,6)), float(np.round(signed_neck,6))]

    writer.writerow(row)
    det.close()
    return True

def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print('cannot open video', VIDEO); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frames:', total_frames)

    header = ['frame','backend']
    names = ['SH_L','EL_L','SH_R','EL_R','HIP_L','HIP_R','EAR_L','EAR_R']
    for n in names:
        header += [f'{n}_x', f'{n}_y', f'{n}_z']
    for n in ['SHO_C','HIP_C','HEAD_C']:
        header += [f'{n}_x', f'{n}_y', f'{n}_z']
    header += ['upper_arm_left_deg','upper_arm_right_deg','neck_angle_signed_deg']

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % FRAME_INTERVAL == 0:
                # process both backends
                for backend in ['MEDIAPIPE', 'RTMW3D']:
                    success = process_backend(cap, backend, writer, frame_idx)
                    if not success:
                        print('stopping at frame', frame_idx); break
                # after processing, continue reading from next frame (process_backend used cap.set)
            frame_idx += 1

    cap.release()
    print('wrote', OUT_CSV)

if __name__ == '__main__':
    main()
