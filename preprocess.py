import xml.etree.ElementTree as ET
from pathlib import Path
from sys import exit
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("models/yolo11n-pose.pt")

def interpolate_keypoints(keypoints):
    keypoints = keypoints.copy()

    for joint in range(17):
        valid = (
            np.isfinite(keypoints[:, joint, 0]) &
            np.isfinite(keypoints[:, joint, 1]) &
            (keypoints[:, joint, 2] > 0.3)
        )

        indices = np.where(valid)[0]

        if len(indices) < 2:
            continue

        for left, right in zip(indices[:-1], indices[1:]):
            if right - left <= 1:
                continue

            for coordinate in (0, 1):
                a = keypoints[left, joint, coordinate]
                b = keypoints[right, joint, coordinate]

                keypoints[left + 1:right, joint, coordinate] = (
                    a + 0.5 * (b - a)
                )

            keypoints[left + 1:right, joint, 2] = 0.0

    return keypoints

def pose_extractor(frame):
    results = model.predict(
        frame, 
        verbose=False,
            conf=0.15
            )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Pose", annotated_frame)
    cv2.waitKey(100)

    result = results[0]

    if len(result.boxes) == 0:
        return None

    confidences = result.boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)

    keypoints = result.keypoints.data.cpu().numpy()[best_idx]

    return keypoints

def process_xml(xml):
    if not xml.is_file():
        print(f"Нет разметки для {video.name}")
        return

    tree = ET.parse(xml)
    root = tree.getroot()

    annotations = {}

    for image in root.findall("image"):
        form_id = int(image.get("id"))
        labels = [
            tag.get("label")
            for tag in image.findall("tag")
        ]
        if len(labels) == 0:
            print(f"Не полная разметка в {xml.name} фрагмент видео пропущен")
            return
        annotations[form_id] = labels
    return annotations
def normalize(keypoints):
    main = [keypoints[0][1]] * len(keypoints[0])

    for i in range(len(keypoints)):
        keypoints[i] -= main

    return keypoints

def process_video(video):
    print(video.name)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output = processed_dir / f"{video.stem}.npz"

    if output.exists():
        print(f"Видео {video.name} уже обработано")
        return



    xml = video.with_suffix(".xml")

    annotations = process_xml(xml)

    if not annotations:
        print(f"XML {xml.name} не содержит разметки")
        return

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"Не удалось открыть {video.name}")
        return

    keypoints = []

    missing = np.full((17, 3), np.nan, dtype=np.float32)
    missing[:, 2] = 0

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read frame")
            break

        result = pose_extractor(frame)

        if result is None:
            result = missing

        keypoints.append(result)

    keypoints = np.asarray(keypoints, dtype=np.float32)
    keypoints_interpolated = interpolate_keypoints(keypoints)

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if total_frames != len(keypoints_interpolated) or len(keypoints_interpolated) != len(annotations):
        print(f"Количество кадров в {video.name} не совпадает с разметкой")
        return

    np.savez_compressed(
        output,
        keypoints=normalize(keypoints_interpolated),
        labels=annotations,
        fps=fps
    )
    print(f"Видео {video.name} обработано")

if __name__ == "__main__":
    raw_dir = Path("data/raw")

    videos = sorted(raw_dir.glob("*.mp4"))

    for video in videos:
        process_video(video)