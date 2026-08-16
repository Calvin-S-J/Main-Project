import cv2
import numpy as np
from config import *

def binary_from_warp(warped_bgr,
                     use_color=True,
                     use_sobel=True,
                     s_thresh=S_THRESH,
                     sx_thresh=SX_THRESH,
                     morph_kernel=MORPH_KERNEL):
    if len(warped_bgr.shape) == 2:
        b = warped_bgr.copy()
        return (b > 0).astype(np.uint8) * 255

    img = warped_bgr.copy().astype(np.uint8)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    L = hls[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(gray, dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_y = np.array([15, 60, 60], dtype=np.uint8)
    upper_y = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_y, upper_y)

    white_mask1 = np.zeros_like(gray)
    white_mask1[(L >= 200)] = 255
    white_mask2 = np.zeros_like(gray)
    white_mask2[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 255

    if use_color:
        color_mask = (yellow_mask > 0) | (white_mask1 > 0) | (white_mask2 > 0)
        combined[color_mask] = 255

    if use_sobel:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sx = np.absolute(sobelx)
        maxv = np.max(abs_sx) if np.max(abs_sx) != 0 else 1.0
        scaled = np.uint8(255 * abs_sx / maxv)
        sx_binary = np.zeros_like(scaled)
        sx_binary[(scaled >= sx_thresh[0]) & (scaled <= sx_thresh[1])] = 255
        combined[(sx_binary > 0)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined


def compute_histogram(binary_warped):
    h = binary_warped.shape[0]
    bottom_half = binary_warped[h//2:,:]
    hist = np.sum(bottom_half > 0, axis=0)
    midpoint = hist.shape[0] // 2
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    return hist, leftx_base, rightx_base

def find_lane_pixels_sliding_window(binary_warped, leftx_base, rightx_base,
                                    nwindows=NWINDOWS, margin=MARGIN, minpix=MINPIX):
    binary = (binary_warped > 0).astype(np.uint8)
    h, w = binary.shape
    window_height = int(h // nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = int(leftx_base)
    rightx_current = int(rightx_base)
    left_inds_list = []
    right_inds_list = []
    window_centers = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_inds_list.append(good_left_inds)
        right_inds_list.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

        window_centers.append(((leftx_current, (win_y_low + win_y_high)//2),
                               (rightx_current, (win_y_low + win_y_high)//2)))

    left_inds = np.concatenate([a for a in left_inds_list if a.size > 0]) if any(a.size>0 for a in left_inds_list) else np.array([], dtype=np.int64)
    right_inds = np.concatenate([a for a in right_inds_list if a.size > 0]) if any(a.size>0 for a in right_inds_list) else np.array([], dtype=np.int64)

    leftx = nonzerox[left_inds] if left_inds.size else np.array([])
    lefty = nonzeroy[left_inds] if left_inds.size else np.array([])
    rightx = nonzerox[right_inds] if right_inds.size else np.array([])
    righty = nonzeroy[right_inds] if right_inds.size else np.array([])

    return leftx, lefty, rightx, righty, window_centers


def fit_polynomial_from_pixels(leftx, lefty, rightx, righty):
    left_fit = None
    right_fit = None
    if leftx.size and lefty.size:
        left_fit = np.polyfit(lefty, leftx, 2)
    if rightx.size and righty.size:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def generate_fit_points_and_curvature(binary_warped, left_fit, right_fit,
                                     ym_per_pix=30/720.0, xm_per_pix=3.7/700.0):
    h = binary_warped.shape[0]
    ploty = np.linspace(0, h-1, h)
    out = {'ploty': ploty}

    left_fitx = None
    right_fitx = None
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out['left_fitx'] = left_fitx
    out['right_fitx'] = right_fitx

    left_curverad = None
    right_curverad = None
    if left_fitx is not None:
        leftx_m = left_fitx * xm_per_pix
        y_m = ploty * ym_per_pix
        left_fit_cr = np.polyfit(y_m, leftx_m, 2)
        A, B = left_fit_cr[0], left_fit_cr[1]
        left_curverad = ((1 + (2*A*y_m[-1] + B)**2)**1.5) / np.abs(2*A)
    if right_fitx is not None:
        rightx_m = right_fitx * xm_per_pix
        y_m = ploty * ym_per_pix
        right_fit_cr = np.polyfit(y_m, rightx_m, 2)
        A, B = right_fit_cr[0], right_fit_cr[1]
        right_curverad = ((1 + (2*A*y_m[-1] + B)**2)**1.5) / np.abs(2*A)

    out['left_curverad_m'] = left_curverad
    out['right_curverad_m'] = right_curverad

    if left_fitx is not None and right_fitx is not None:
        lane_center_x = (left_fitx[-1] + right_fitx[-1]) / 2.0
        image_center_x = binary_warped.shape[1] / 2.0
        offset_pixels = image_center_x - lane_center_x
        out['lane_center_offset_m'] = offset_pixels * xm_per_pix
    else:
        out['lane_center_offset_m'] = None

    return out


def draw_lane_on_original(original_img, Minv, ploty, left_fitx, right_fitx,
                          left_curverad_m=None, right_curverad_m=None,
                          lane_center_offset_m=None, alpha=0.6):
    h, w = original_img.shape[:2]
    if ploty is None or left_fitx is None or right_fitx is None:
        return original_img.copy(), None

    color_warp = np.zeros((h, w, 3), dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, pts, (0,255,0))
    cv2.polylines(color_warp, pts_left, isClosed=False, color=(0,255,0), thickness=6)
    cv2.polylines(color_warp, pts_right, isClosed=False, color=(0,0,255), thickness=6)

    lane_unwarped = cv2.warpPerspective(color_warp, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    result = cv2.addWeighted(original_img.copy(), 1.0, lane_unwarped, alpha, 0.0)

    
    y0 = 40; dy = 30; font = cv2.FONT_HERSHEY_SIMPLEX
    if left_curverad_m is not None:
        cv2.putText(result, f"Left curvature: {left_curverad_m:.1f} m", (30, y0), font, 0.8, (255,255,255), 2, cv2.LINE_AA); y0 += dy
    if right_curverad_m is not None:
        cv2.putText(result, f"Right curvature: {right_curverad_m:.1f} m", (30, y0), font, 0.8, (255,255,255), 2, cv2.LINE_AA); y0 += dy
    if lane_center_offset_m is not None:
        cv2.putText(result, f"Center offset: {lane_center_offset_m:.3f} m", (30, y0), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return result, lane_unwarped

