import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"

# Navn -> indeks i MediaPipes pose-model
LED = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_hip": 23,      "r_hip": 24,
    "l_knee": 25,     "r_knee": 26,
    "l_ankle": 27,    "r_ankle": 28,
    "l_heel": 29,     "r_heel": 30,
}


def find_leds(video_path, model_path=MODEL_PATH, vis=True):
    """
    Finder kroppens led i hver frame af en video.

    Parametre:
        video_path: sti til videofilen, fx "gang2.mp4"
        model_path: sti til MediaPipe-modellen (.task)
        vis:        True viser videoen med prikker på leddene (tryk q for at stoppe)

    Returnerer:
        pandas DataFrame med én række pr. frame, hvor personen blev fundet.
        Kolonner: frame, time_s samt <led>_x og <led>_y i pixels.
    """
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Kunne ikke åbne videoen: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    rows = []
    frame_nr = 0
    missed = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:  # videoen er slut
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(frame_nr / fps * 1000))

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                row = {"frame": frame_nr, "time_s": frame_nr / fps}
                for navn, idx in LED.items():
                    row[f"{navn}_x"] = lm[idx].x * w
                    row[f"{navn}_y"] = lm[idx].y * h
                rows.append(row)

                if vis:
                    for p in lm:
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
            else:
                missed += 1

            if vis:
                cv2.imshow("Pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_nr += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"{len(rows)} af {frame_nr} frames havde en person ({missed} uden).")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = find_leds("gang3.mp4")
    df.to_csv("landmarks.csv", index=False)