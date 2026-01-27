import cv2
import time
from detection import FaceFeatureDetector
from alerts import play_alert_sound
from collections import deque
import numpy as np


# smoothing window (adjust 4–8 as needed)
EAR_WINDOW = 6
left_buf = deque(maxlen=EAR_WINDOW)
right_buf = deque(maxlen=EAR_WINDOW)

def main():
    detector = FaceFeatureDetector(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(1)

    # --- startup calibration: measure baseline open-eye EAR for 2 seconds ---
    calib_samples = []
    calib_duration = 2.0   # seconds to sample
    calib_end = time.time() + calib_duration
    print("Calibration: please look at camera with eyes open for 2 seconds...")
    while time.time() < calib_end:
        ret, f = cap.read()
        if not ret:
            continue
        r = detector.process(f)
        if r.get("faceDetected", False):
            le = r.get("left_ear", r.get("ear", 0.0))
            re = r.get("right_ear", r.get("ear", 0.0))
            calib_samples.append((le + re) / 2.0)
    if len(calib_samples) == 0:
        baseline_open = 0.28   # fallback default
        print("Calibration failed (no face). Using default baseline:", baseline_open)
    else:
        baseline_open = float(np.median(calib_samples))
        print(f"Calibration done. baseline_open EAR = {baseline_open:.3f}")

    # thresholds / params (dynamic EAR threshold from calibration)
    EAR_THRESHOLD = max(0.12, baseline_open * 0.7)  # floor at 0.12 so not too low
    MAR_THRESHOLD = 0.60
    ALERT_COOLDOWN = 5.0
    ALERT_SECONDS = 0.8


    closed_time = 0.0
    yawn_frames = 0
    last_alert_time = 0
    total_alerts = 0

    prev_time = time.time()
    DROWSY_DISPLAY_SEC = 2.0
    show_drowsy_until = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.process(frame)
        faceDetected = res.get("faceDetected", False)

        # --- smoothing & per-eye median ---
        if not faceDetected:
            left_buf.clear(); right_buf.clear()
            s_left = s_right = 0.0
        else:
            left_ear = res.get("left_ear", res.get("ear", 0.0))
            right_ear = res.get("right_ear", res.get("ear", 0.0))

            left_buf.append(left_ear)
            right_buf.append(right_ear)

            s_left = float(np.median(list(left_buf))) if len(left_buf) > 0 else left_ear
            s_right = float(np.median(list(right_buf))) if len(right_buf) > 0 else right_ear

        # averaged (smoothed) EAR for display/overlay
        ear = (s_left + s_right) / 2.0
        mar = res.get("mar", 0.0)

        # timing for frame-independent logic
        now = time.time()
        dt = now - prev_time
        prev_time = now

        alert_trigger = False
        if faceDetected:
            # require both eyes closed (robust to glance)
            if (s_left < EAR_THRESHOLD) and (s_right < EAR_THRESHOLD):
                closed_time += dt
            else:
                closed_time = 0.0

            # yawn logic (frame-count based, keep as fallback)
            if mar > MAR_THRESHOLD:
                yawn_frames += 1
            else:
                yawn_frames = 0

            # trigger if closed long enough OR sustained yawn
            if closed_time >= ALERT_SECONDS or yawn_frames >= 15:
                alert_trigger = True

        # cooldown and alert action
        if alert_trigger and (now - last_alert_time) > ALERT_COOLDOWN:
            print("[ALERT] Drowsiness detected!")
            play_alert_sound()
            last_alert_time = now
            total_alerts += 1
            # reset timers to avoid rapid repeats
            closed_time = 0.0
            yawn_frames = 0
            show_drowsy_until = max(show_drowsy_until, now + DROWSY_DISPLAY_SEC)
        else:
            if alert_trigger:
                show_drowsy_until = max(show_drowsy_until, now + DROWSY_DISPLAY_SEC)   

        # overlay & HUD: pass entire res to draw_overlay
        frame = detector.draw_overlay(frame, {**res, "ear": ear, "mar": mar, "faceDetected": faceDetected})

        # auxiliary HUD texts (some duplicative, but helpful)
        cv2.putText(frame, f"ClosedSec:{closed_time:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"YawnFrames:{yawn_frames}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"TotalAlerts:{total_alerts}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if time.time() < show_drowsy_until:
            cv2.putText(
                frame,
                "DROWSY!!",
                (400, 100),                 # position (x,y)
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,                        # font scale (big text)
                (0, 0, 255),                # RED (BGR)
                4,                          # thickness (bold)
                cv2.LINE_AA
            )

        cv2.imshow("Driver Drowsiness - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
