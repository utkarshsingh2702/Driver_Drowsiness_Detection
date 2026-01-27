import math
import cv2
import mediapipe as mp
import numpy as np

class FaceFeatureDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # indices used earlier
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 380, 373]
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

    def _landmark_to_point(self, lm, w, h):
        return np.array([int(lm.x * w), int(lm.y * h)])

    def process(self, frame):
        """
        Returns dict with:
         - faceDetected: bool
         - ear, left_ear, right_ear, mar (floats)
         - bbox: (x1,y1,x2,y2) in pixel coords for face bounding box
         - left_eye_center: (x,y)
         - right_eye_center: (x,y)
         - mouth_center: (x,y)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"faceDetected": False, "ear": 0.0, "mar": 0.0}

        face_lms = results.multi_face_landmarks[0].landmark

        # helper to get point
        def P(i): return self._landmark_to_point(face_lms[i], w, h)

        # compute EAR
        def eye_aspect_ratio(idxs):
            p1, p2, p3, p4, p5, p6 = [P(i) for i in idxs]
            A = np.linalg.norm(p2 - p6)
            B = np.linalg.norm(p3 - p5)
            C = np.linalg.norm(p1 - p4)
            return (A + B) / (2.0 * C) if C != 0 else 0.0

        left_ear = eye_aspect_ratio(self.LEFT_EYE)
        right_ear = eye_aspect_ratio(self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        # MAR
        top, bottom = P(self.MOUTH_TOP), P(self.MOUTH_BOTTOM)
        left, right = P(self.MOUTH_LEFT), P(self.MOUTH_RIGHT)
        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        mar = (vertical / horizontal) if horizontal != 0 else 0.0

        # compute bounding box from all landmarks (face area)
        pts = np.array([self._landmark_to_point(lm, w, h) for lm in face_lms])
        x_min = int(np.min(pts[:, 0]))
        x_max = int(np.max(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        y_max = int(np.max(pts[:, 1]))

        # clamp to frame
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(w - 1, x_max); y_max = min(h - 1, y_max)

        # compute centers for small indicator boxes
        left_eye_pts = np.array([P(i) for i in self.LEFT_EYE])
        right_eye_pts = np.array([P(i) for i in self.RIGHT_EYE])
        left_eye_center = tuple(np.mean(left_eye_pts, axis=0).astype(int))
        right_eye_center = tuple(np.mean(right_eye_pts, axis=0).astype(int))
        mouth_center = tuple(((top + bottom + left + right) / 4.0).astype(int))

        return {
            "faceDetected": True,
            "ear": float(ear),
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "mar": float(mar),
            "bbox": (x_min, y_min, x_max, y_max),
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
            "mouth_center": mouth_center
        }

    def draw_overlay(self, frame, res):
        """
        Draws:
         - light green rectangle around face (bbox)
         - three small light green boxes inside (left eye, right eye, mouth)
         - EAR/MAR text
        """
        ear = res.get("ear", 0.0)
        mar = res.get("mar", 0.0)
        faceDetected = res.get("faceDetected", False)

        text = f"EAR: {ear:.3f}  MAR: {mar:.3f}"
        color_text = (0,255,0) if ear > 0.22 else (0,0,255)
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)

        if not faceDetected:
            return frame

          # light green color
        box_color = (144, 238, 144)  # light green (BGR)
        
        # big face bbox
        x1, y1, x2, y2 = res.get("bbox", (0,0,0,0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # size of small boxes
        lw = 18  

        # left eye
        lx, ly = res.get("left_eye_center", (0,0))
        cv2.rectangle(frame, (lx - lw, ly - lw), (lx + lw, ly + lw), box_color, 2)

        # right eye
        rx, ry = res.get("right_eye_center", (0,0))
        cv2.rectangle(frame, (rx - lw, ry - lw), (rx + lw, ry + lw), box_color, 2)

        # mouth
        mx, my = res.get("mouth_center", (0,0))
        cv2.rectangle(frame, (mx - lw, my - lw), (mx + lw, my + lw), box_color, 2)

        return frame
