import cv2
from perspective import warp_img
from lane_detection import *
from speed_estimation import compute_relative_speed


def process_frame_and_detect(frame, model, sign_model,
                             M, Minv, src_pts, dst_pts,
                             prev_states, dt):
    h, w = frame.shape[:2]
    # 1) warp for lane processing
    warped_color = warp_img(frame, M, dst_size=(w, h))
    binary = binary_from_warp(warped_color)
    hist, leftx_base, rightx_base = compute_histogram(binary)
    leftx, lefty, rightx, righty, window_centers = find_lane_pixels_sliding_window(binary, leftx_base, rightx_base)
    left_fit, right_fit = fit_polynomial_from_pixels(leftx, lefty, rightx, righty)
    fit_info = generate_fit_points_and_curvature(binary, left_fit, right_fit)

    # 2) draw lane on original
    overlay, lane_unwarped = None, None
    if fit_info.get('left_fitx') is not None and fit_info.get('right_fitx') is not None:
        overlay, lane_unwarped = draw_lane_on_original(frame, Minv,
                                                       fit_info['ploty'],
                                                       fit_info['left_fitx'],
                                                       fit_info['right_fitx'],
                                                       fit_info['left_curverad_m'],
                                                       fit_info['right_curverad_m'],
                                                       fit_info['lane_center_offset_m'])
    else:
        overlay = frame.copy()

    # 3) detect vehicles using YOLO track (on original frame)
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  # car, motorbike, bus, truck
    annotated = overlay.copy()

    if len(results) > 0 and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            obj_id = int(ids[i])
            x1, y1, x2, y2 = map(int, box)

            
            bottom_pt = np.array([[[ (x1+x2)/2.0, y2 ]]], dtype=np.float32)  # shape (1,1,2)

            
            warped_pt = cv2.perspectiveTransform(bottom_pt, M)  
            wx, wy = warped_pt[0,0]

          
            speed_m_s, d_m = compute_relative_speed(obj_id, wy, binary.shape[0], dt, prev_states)

           
            if speed_m_s > FAST_APPROACH:
                color = (0,0,255)   # red
                label = f"+{speed_m_s:.1f} m/s"
            elif speed_m_s > APPROACH:
                color = (0,128,255) # orange
                label = f"+{speed_m_s:.1f} m/s"
            elif speed_m_s < AWAY:
                color = (255,255,0) # cyan-ish
                label = f"{speed_m_s:.1f} m/s"
            else:
                color = (0,255,0)   # green
                label = f"{speed_m_s:.1f} m/s"

            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(15,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # 4) detect traffic signs
    # sign_results = sign_model(frame)

    # if len(sign_results) > 0 and sign_results[0].boxes is not None:

    #     boxes = sign_results[0].boxes.xyxy.cpu().numpy()
    #     classes = sign_results[0].boxes.cls.cpu().numpy()

    #     for i, box in enumerate(boxes):

    #         x1, y1, x2, y2 = map(int, box)
    #         cls = int(classes[i])

    #         label = sign_model.names[cls]

    #         color = (255, 0, 0)  # blue box for traffic signs

    #         cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    #         cv2.putText(
    #             annotated,
    #             label,
    #             (x1, max(15, y1 - 10)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             color,
    #             2
    #         )
   
    return {
        'overlay': annotated,
        'binary_warped': binary,
        'fit_info': fit_info,
        'M': M,
        'Minv': Minv,
        'src_pts': src_pts,
        'dst_pts': dst_pts
    }
