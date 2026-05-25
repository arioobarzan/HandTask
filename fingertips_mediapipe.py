"""
Fingertip Detection with MediaPipe
===================================
Detects 5 fingertips per hand using MediaPipe Hand Landmarker
and draws colored circles on each fingertip.
When both hands are detected, matching fingertips are connected
with gradient lines (thick & bright in the middle, thin & saturated at the ends).

Landmark indices for fingertips:
    4  = Thumb tip
    8  = Index finger tip
    12 = Middle finger tip
    16 = Ring finger tip
    20 = Pinky tip

Requires: hand_landmarker.task in the same directory.
Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# ===== Clear caches safely =====
try:
    mp_cache = os.path.join(os.path.expanduser("~"), ".mediapipe")
    if os.path.exists(mp_cache):
        import shutil
        shutil.rmtree(mp_cache, ignore_errors=True)
except Exception:
    pass

# Suppress MediaPipe GPU warnings on Windows
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
print("Ready.\n")

# ===== Model path =====
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at:\n  {MODEL_PATH}")
    print("Download from:")
    print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    sys.exit(1)

# ===== MediaPipe setup =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ===== Fingertip landmark indices and their colors (BGR) =====
FINGERTIPS = {
    4:  (255, 0, 0),     # Blue   — Thumb
    8:  (0, 255, 0),     # Green  — Index
    12: (0, 0, 255),     # Red    — Middle
    16: (255, 255, 0),   # Cyan   — Ring
    20: (255, 0, 255),   # Magenta — Pinky
}


def draw_gradient_line(frame, p1, p2, color, max_thickness=14, min_thickness=3, num_segments=50):
    """
    Draw a line with smooth gradient:
    - Middle: thick, bright (glowing)
    - Ends:   thin, saturated (original color)
    """
    x1, y1 = p1
    x2, y2 = p2

    b, g, r = color

    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 1) / num_segments
        mid = (t1 + t2) / 2

        # Cosine curve: 0 at ends, 1 at middle (smooth bell shape)
        glow = np.sin(np.pi * mid)

        # ---- Color: ends = saturated, middle = bright (closer to white) ----
        # Use a power curve for more dramatic center glow
        color_factor = glow ** 1.5 * 0.60
        seg_color = (
            int(b + (255 - b) * color_factor),
            int(g + (255 - g) * color_factor),
            int(r + (255 - r) * color_factor),
        )

        # Thickness: sin(0)=0 (thin), sin(pi/2)=1 (thick), sin(pi)=0 (thin)
        thick_factor = np.sin(np.pi * mid)
        seg_thick = int(min_thickness + (max_thickness - min_thickness) * thick_factor)

        sx1 = int(x1 + (x2 - x1) * t1)
        sy1 = int(y1 + (y2 - y1) * t1)
        sx2 = int(x1 + (x2 - x1) * t2)
        sy2 = int(y1 + (y2 - y1) * t2)

        cv2.line(frame, (sx1, sy1), (sx2, sy2), seg_color, seg_thick)


def draw_fingertips(frame, result):
    """Draw colored circles on fingertips + gradient lines between matching fingers."""
    if not result.hand_landmarks:
        return

    h, w = frame.shape[:2]

    # Collect fingertip coordinates for each detected hand
    all_hands_tips = []

    for hand_landmarks in result.hand_landmarks:
        tips = {}
        for tip_idx in FINGERTIPS:
            lm = hand_landmarks[tip_idx]
            x, y = int(lm.x * w), int(lm.y * h)
            tips[tip_idx] = (x, y)
        all_hands_tips.append(tips)

    # Connect matching fingertips between two hands with gradient lines
    if len(all_hands_tips) == 2:
        for tip_idx, color in FINGERTIPS.items():
            p1 = all_hands_tips[0][tip_idx]
            p2 = all_hands_tips[1][tip_idx]
            draw_gradient_line(frame, p1, p2, color, max_thickness=14, min_thickness=3)

    # Draw fingertip circles
    for tips in all_hands_tips:
        for tip_idx, (x, y) in tips.items():
            color = FINGERTIPS[tip_idx]
            # Outer glow ring
            cv2.circle(frame, (x, y), 16, (255, 255, 255), 1)
            # Solid colored circle
            cv2.circle(frame, (x, y), 13, color, -1)
            # White border
            cv2.circle(frame, (x, y), 13, (255, 255, 255), 2)


# ===== Main =====
def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("=" * 50)
    print("Fingertip Detection — MediaPipe")
    print("Colored circles on fingertips + gradient lines")
    print("Press 'q' to quit.")
    print("=" * 50)

    frame_idx = 0
    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = detector.detect_for_video(mp_image, frame_idx)
                frame_idx += 1

                draw_fingertips(frame, result)

                cv2.imshow("Fingertips - MediaPipe", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
