"""
Extract every Nth frame from a video file.

Usage (in terminal from this folder):
    python extractFrames.py
"""

import os
import cv2

# ==== CONFIGURE THESE ====
VIDEO_PATH = "/Users/chasecarson/Desktop/ROVlandfootage/landVal.mp4"   # path to your video file
OUTPUT_DIR = "landValFrames"      # folder where frames will be saved
N = 20                           # save every Nth frame
# ==========================


def extract_every_nth_frame(video_path: str, output_dir: str, n: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % n == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames to '{output_dir}'.")


if __name__ == "__main__":
    extract_every_nth_frame(VIDEO_PATH, OUTPUT_DIR, N)
