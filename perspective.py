import cv2
import numpy as np

# def default_src_dst(img_shape):
#     h, w = img_shape[:2]
#     base_src = np.float32([[605, 460], [203, 720], [1127, 720], [725, 460]])
#     base_dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
#     sx = w / 1280.0
#     sy = h / 720.0
#     src = (base_src * np.array([sx, sy])).astype(np.float32)
#     dst = (base_dst * np.array([sx, sy])).astype(np.float32)
#     return src, dst

def default_src_dst(img_shape):
    h, w = img_shape[:2]

    # Source points (lane trapezoid from your 4K frame)
    src = np.float32([
        [1764, 1440],  # top left
        [1072, 1900],  # bottom left
        [2432, 1900],  # bottom right
        [2020, 1440]   # top right
    ])

    # Destination points (rectangle for bird's eye view)
    dst = np.float32([
        [w*0.25, 0],
        [w*0.25, h],
        [w*0.75, h],
        [w*0.75, 0]
    ])

    return src, dst

def get_transform_matrices(img_shape, src=None, dst=None):
    if src is None or dst is None:
        src, dst = default_src_dst(img_shape)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst

def warp_img(img, M, dst_size=None, border_mode=cv2.BORDER_REPLICATE):
    if dst_size is None:
        h, w = img.shape[:2]
        dst_size = (w, h)
    warped = cv2.warpPerspective(img, M, dst_size, flags=cv2.INTER_LINEAR, borderMode=border_mode)
    return warped
