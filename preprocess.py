import xml.etree.ElementTree as ET
from pathlib import Path
from sys import exit
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo11n-pose_ncnn_model")

def pose_extractor(frame):
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        imgsz=320,
        verbose=False
    )

    result = results[0]
    return result

def process_xml(xml):
    if not xml.is_file():
        print(f"Нет разметки для {video.name}")
        return

    tree = ET.parse(xml)
    root = tree.getroot()

    annotations = []

    for image in root.findall("image"):
        labels = [
            tag.get("label")
            for tag in image.findall("tag")
        ]
        if len(labels) == 0:
            print(f"Не полная разметка в {xml.name} фрагмент видео пропущен")
            return
        annotations.append(labels)
    return annotations

def normalize(keypoints):
    first = keypoints[0].copy()

    keypoints[0] = np.zeros_like(first)

    for i in range(1, len(keypoints)):
        keypoints[i] -= first

    return keypoints

def process_video(video):
    print(video.name)

    xml = video.with_suffix(".xml")

    annotations = process_xml(xml)

    if not annotations:
        print(f"XML {xml.name} не содержит разметки")
        return

    cap = cv2.VideoCapture(str(video))

    keypoints = []

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read frame")
            break

        result = pose_extractor(frame)

        keypoints.append(result.keypoints)

    cap.release()

    output = raw_dir / f"{video.stem}.npz"

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if total_frames != len(keypoints) and len(keypoints) != len(annotations):
        print(f"Количество кадров в {video.name} не совпадает с разметкой")
        return

    np.savez_compressed(
        output,
        keypoints=normalize(keypoints),
        labels=annotations,
        fps=fps
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output = processed_dir / f"{video.stem}.npz"

    if output.exists():
        return

if __name__ == "__main__":
    raw_dir = Path("data/raw")

    videos = sorted(raw_dir.glob("*.mp4"))

    for video in videos:
        process_video(video)