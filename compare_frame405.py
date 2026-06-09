from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from rula_realtime_app.core.pose_detector import PoseDetector
from rula_realtime_app.core.utils import safe_angle
import cv2

VIDEO = ROOT / 'demo_videos' / '箱子1.MOV'
FRAME = 405

def sample(backend):
    det = PoseDetector(backend_mode=backend)
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = cap.read()
    if not ok:
        det.close(); cap.release(); return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected = det.process_frame(rgb)
    pose = det.get_landmarks_array() if detected else None
    det.close(); cap.release()
    return np.array(pose) if pose is not None else None

mp = sample('MEDIAPIPE')
rtm = sample('RTMW3D')

if mp is None or rtm is None:
    print('missing pose for one backend')
    sys.exit(1)

def centroid(a,i,j):
    return (a[i,:3]+a[j,:3])/2.0

SHO_mp = centroid(mp,11,12)
SHO_r = centroid(rtm,11,12)
HIP_mp = centroid(mp,23,24)
HIP_r = centroid(rtm,23,24)
HEAD_mp = centroid(mp,7,8)
HEAD_r = centroid(rtm,7,8)

v_u_mp = SHO_mp - HIP_mp
v_u_r = SHO_r - HIP_r
v_neck_mp = HEAD_mp - SHO_mp
v_neck_r = HEAD_r - SHO_r

v_f_mp = np.cross(v_u_mp, (mp[12,:3]-mp[23,:3]))
P_s_mp = np.cross(v_u_mp, v_f_mp)
P_s_norm = np.linalg.norm(P_s_mp)
if P_s_norm>1e-6:
    P_s_hat = P_s_mp / P_s_norm
    v_neck_proj_mp = v_neck_mp - np.dot(v_neck_mp,P_s_hat)*P_s_hat
else:
    v_neck_proj_mp = v_neck_mp

v_f_r = np.cross(v_u_r, (rtm[12,:3]-rtm[23,:3]))
P_s_r = np.cross(v_u_r, v_f_r)
P_s_norm_r = np.linalg.norm(P_s_r)
if P_s_norm_r>1e-6:
    P_s_hat_r = P_s_r / P_s_norm_r
    v_neck_proj_r = v_neck_r - np.dot(v_neck_r,P_s_hat_r)*P_s_hat_r
else:
    v_neck_proj_r = v_neck_r

theta_mp = safe_angle(v_neck_proj_mp, v_u_mp)
theta_r = safe_angle(v_neck_proj_r, v_u_r)

print('Frame', FRAME)
print('mediapipe:')
print('  SHO_C', np.round(SHO_mp,4).tolist())
print('  HIP_C', np.round(HIP_mp,4).tolist())
print('  HEAD_C', np.round(HEAD_mp,4).tolist())
print('  theta_neck', round(float(theta_mp),4))

print('rtmw3d:')
print('  SHO_C', np.round(SHO_r,4).tolist())
print('  HIP_C', np.round(HIP_r,4).tolist())
print('  HEAD_C', np.round(HEAD_r,4).tolist())
print('  theta_neck', round(float(theta_r),4))

print('\nDiff norms:')
print('  SHO_C diff', np.linalg.norm(SHO_mp-SHO_r))
print('  HIP_C diff', np.linalg.norm(HIP_mp-HIP_r))
print('  HEAD_C diff', np.linalg.norm(HEAD_mp-HEAD_r))
print('  v_u diff', np.linalg.norm(v_u_mp - v_u_r))
print('  v_neck diff', np.linalg.norm(v_neck_mp - v_neck_r))
print('  v_neck_proj diff', np.linalg.norm(v_neck_proj_mp - v_neck_proj_r))

# Upper arm (left/right) - follow rula_score_side: v_sh_el vs v_body (HIP_C - SHO_C)
SH_L_mp = np.array([mp[11,0], mp[11,1], mp[11,2]])
EL_L_mp = np.array([mp[13,0], mp[13,1], mp[13,2]])
SH_R_mp = np.array([mp[12,0], mp[12,1], mp[12,2]])
EL_R_mp = np.array([mp[14,0], mp[14,1], mp[14,2]])

SH_L_r = np.array([rtm[11,0], rtm[11,1], rtm[11,2]])
EL_L_r = np.array([rtm[13,0], rtm[13,1], rtm[13,2]])
SH_R_r = np.array([rtm[12,0], rtm[12,1], rtm[12,2]])
EL_R_r = np.array([rtm[14,0], rtm[14,1], rtm[14,2]])

v_body_mp = HIP_mp - SHO_mp  # HIP_C - SHO_C as in rula_score_side
v_body_r = HIP_r - SHO_r

v_sh_el_L_mp = EL_L_mp - SH_L_mp
v_sh_el_R_mp = EL_R_mp - SH_R_mp
v_sh_el_L_r = EL_L_r - SH_L_r
v_sh_el_R_r = EL_R_r - SH_R_r

theta_upper_L_mp = safe_angle(v_sh_el_L_mp, v_body_mp)
theta_upper_R_mp = safe_angle(v_sh_el_R_mp, v_body_mp)
theta_upper_L_r = safe_angle(v_sh_el_L_r, v_body_r)
theta_upper_R_r = safe_angle(v_sh_el_R_r, v_body_r)

print('\nUpper arm angles (deg):')
print('  mediapipe left', round(float(theta_upper_L_mp),4), ' right', round(float(theta_upper_R_mp),4))
print('  rtmw3d   left', round(float(theta_upper_L_r),4), ' right', round(float(theta_upper_R_r),4))

print('\nUpper arm diffs:')
print('  left abs diff', abs(float(theta_upper_L_mp)-float(theta_upper_L_r)))
print('  right abs diff', abs(float(theta_upper_R_mp)-float(theta_upper_R_r)))
print('  v_sh_el left diff norm', np.linalg.norm(v_sh_el_L_mp - v_sh_el_L_r))
print('  v_sh_el right diff norm', np.linalg.norm(v_sh_el_R_mp - v_sh_el_R_r))
