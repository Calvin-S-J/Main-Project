import cv2
import time
from ultralytics import YOLO

from config import *
from perspective import get_transform_matrices
from pipeline import process_frame_and_detect


def main():

    model = YOLO(VEHICLE_MODEL)
    sign_model = YOLO(TRAFFIC_SIGN_MODEL)

    cap = cv2.VideoCapture(VIDEO_IN)

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1/30

    ret, frame0 = cap.read()
    if not ret:
        print("Unable to read video")
        return

    h, w = frame0.shape[:2]

    M, Minv, src_pts, dst_pts = get_transform_matrices((h, w))

    prev_states = {}

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_overlay = cv2.VideoWriter(
        "output_overlay.mp4",
        fourcc,
        fps,
        (w, h)
    )

    out_binary = cv2.VideoWriter(
        "output_binary.mp4",
        fourcc,
        fps,
        (w, h),
        False  # grayscale
    )
    

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        res = process_frame_and_detect(
            frame, model, sign_model,
            M, Minv,
            src_pts, dst_pts,
            prev_states, dt
        )

        out_frame = res['overlay']
        binary = res['binary_warped']

        # write videos
        out_overlay.write(out_frame)
        out_binary.write(binary)

        cv2.imshow("Output", out_frame)
        cv2.imshow("Binary", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_overlay.release()
    out_binary.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()