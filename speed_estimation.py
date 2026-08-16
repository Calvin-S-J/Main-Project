from config import *

def compute_relative_speed(obj_id, y_bottom_warped, warped_h, dt, prev_states, lane_length_m=LANE_VISIBLE_LENGTH_M, smooth_alpha=SMOOTH_ALPHA):
    
    H_px = warped_h
    m_per_px = lane_length_m / float(H_px)
    d_m = (H_px - float(y_bottom_warped)) * m_per_px

    prev = prev_states.get(obj_id, None)
    if prev is None:
        prev_states[obj_id] = {'d': d_m, 'v': 0.0}
        return 0.0, d_m
    d_prev = prev['d']
    v_raw = (d_prev - d_m) / float(dt) if dt > 0 else 0.0
    v_prev = prev['v']
    v_smooth = smooth_alpha * v_prev + (1.0 - smooth_alpha) * v_raw
    prev_states[obj_id] = {'d': d_m, 'v': v_smooth}
    return v_smooth, d_m


