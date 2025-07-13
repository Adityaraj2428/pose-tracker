import cv2
import mediapipe as mp
import numpy as np

# Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# === Configurable Settings ===
max_trail_length = 8
pulse_intensity = 0.8
thickness = 6
model_complexity = 0
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

trail_mode = 0      # 0 = circles, 1 = lines, 2 = triangle
color_mode = 2      # 1 = static yellow, 2 = fade red-yellow, 3 = rainbow

# Initialize Pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pose_trails = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)
    current_landmarks = []

    if results.pose_landmarks:
        current_landmarks = [
            (int((1 - lm.x) * width), int(lm.y * height))
            for lm in results.pose_landmarks.landmark
        ]

        # Draw Pose
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        pose_trails.append(current_landmarks)
        if len(pose_trails) > max_trail_length:
            pose_trails.pop(0)

    # === Trail Drawing ===
    for t_index, trail in enumerate(pose_trails):
        alpha = (t_index + 1) / max_trail_length * pulse_intensity
        overlay = frame.copy()

        # 🎨 Color Modes
        if color_mode == 1:
            trail_color = (0, 255, 255)  # Static Yellow

        elif color_mode == 2:
            r = 255
            g = int(255 * (t_index + 1) / max_trail_length)
            b = 0
            trail_color = (b, g, r)

        elif color_mode == 3:
            hue = int((t_index * 25) % 180)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            trail_color = tuple(int(c) for c in bgr)

        # Trail Shape Modes
        if trail_mode == 0:
            for point in trail:
                cv2.circle(overlay, point, thickness, trail_color, -1)

        elif trail_mode == 1:
            for i in range(1, len(trail)):
                cv2.line(overlay, trail[i - 1], trail[i], trail_color, 2)

        elif trail_mode == 2 and len(trail) >= 16:
            try:
                p1 = trail[11]
                p2 = trail[13]
                p3 = trail[15]
                pts = np.array([p1, p2, p3], np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=trail_color, thickness=2)
                cv2.fillPoly(overlay, [pts], trail_color)
            except:
                pass

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display Window
    cv2.imshow('Duffer Project - Clone Tracker', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('m'):  # Switch trail shape
        trail_mode = (trail_mode + 1) % 3
    elif key == ord('1'):
        color_mode = 1  # Static Yellow
    elif key == ord('2'):
        color_mode = 2  # Fade Red → Yellow
    elif key == ord('3'):
        color_mode = 3  # Rainbow Cycling

cap.release()
cv2.destroyAllWindows()
