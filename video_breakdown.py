import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

VIDEO = "gang3.mp4"      # skift til den video, du vil analysere

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=vision.RunningMode.VIDEO,
)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

with vision.PoseLandmarker.create_from_options(options) as landmarker, \
     open("landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_s", "nose_x", "nose_y",
                     "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y",
                     "l_heel_y", "r_heel_y"])
    frame_nr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_nr / fps * 1000))

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            nose = lm[0]
            la, ra = lm[27], lm[28]       # ankler
            lh, rh = lm[29], lm[30]       # hæle
            writer.writerow([frame_nr, frame_nr / fps,
                             nose.x * w, nose.y * h,
                             la.x * w, la.y * h, ra.x * w, ra.y * h,
                             lh.y * h, rh.y * h])
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_nr += 1

cap.release()
cv2.destroyAllWindows()