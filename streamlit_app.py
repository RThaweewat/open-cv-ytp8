import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd
import time
from functools import reduce
from os.path import exists as file_exists

# from playsound import playsound
import gspread

now = datetime.now().time()
now = str(now).replace(".", "_").replace(":", "_")

# run = st.checkbox('Start')

st.title("Webcam Live Feed")
run_no = st.text_input("Run name")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

## Setup mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(f".\output\output_{str(now)}.mp4", fourcc, 20.0, (640, 480))


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


# Curl counter variables
counter = 0
stage = None
i = 0
start_time = time.time()
time_spare = 0
test = True
prev_frame_time = 0
new_frame_time = 0
sit_count = []
angle_sit = []
angle_stand = []


# Using NumPy
if file_exists("data.csv"):
    df = pd.read_csv("data.csv", index_col=0)
else:
    dtypes = np.dtype(
        [
            ("Name", str),
            ("Angle_sit_avg", float),
            ("Angle_stand_avg", float),
            ("Total", str),
            ("Sit Total", str),
            ("Time", str)
        ]
    )
    df = pd.DataFrame(np.empty(0, dtype=dtypes))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while camera.isOpened():

        ret, frame = camera.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = True

        # Make detection
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (int(image_width * (480 / image_height)), 480))

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(
                image,
                str(angle),
                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Curl counter logic

            if angle > 30:
                stage = "sit"
                sit_count.append(1)
                angle_sit.append(angle)

            if stage == "sit" and angle < 15:
                stage = "stand"
                counter = time.time() - start_time
                counter = float("{0:.2f}".format(counter))
                print(counter, "seconds")
                sit_to_stand = float("{0:.2f}".format(counter - time_spare))
                print("Sit to stand", sit_to_stand, "seconds")
                time_spare = counter
                angle_stand.append(angle)
                sit_count.append(0)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(
            image,
            "REPS",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(counter),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stage data
        cv2.putText(
            image,
            "STAGE",
            (65, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            stage,
            (60, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        out.write(image)
        FRAME_WINDOW.image(image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            total_time = float("{0:.2f}".format(time.time() - start_time))
            print("Total", total_time, "seconds")
            break

        elif sit_count[-2:] == [0, 1]:
            total_time = float("{0:.2f}".format(time.time() - start_time))
            print("Total", total_time, "seconds")

            camera.release()
            out.release()
            break

st.write("Stopped")

# append dataframe
df = df.append(
    {
        "No_Name": f"output_{str(now)}_run_no{run_no}.mp4",
        "Name": run_no,
        "Sit Total": round(sit_to_stand),
        "Total": str(now),
        "Time": round(total_time),
        "Angle_stand_avg": Average(angle_stand),
        "Angle_sit_avg": Average(angle_sit),
    },
    ignore_index=True,
)


gc = gspread.service_account(filename='./secret/secret_new.json')
sh = gc.open_by_url('https://docs.google.com/spreadsheets/d/1US-cmh4EL_Kps9dNcOT2y1YFls1dWui3ag7ZlEzr1V4/edit#gid=0')
worksheet = sh.get_worksheet(0)
worksheet.update([df.columns.values.tolist()] + df.values.tolist())


st.dataframe(df)
df.to_csv("data.csv")


