import cv2
import numpy as np
from ultralytics import YOLO

from core.video_loader import VideoLoader


def crop_person(frame, box):
    """
    Crop person region from frame
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return frame[y1:y2, x1:x2]


def visualize_crop(video_path):
    model = YOLO("yolov8n.pt")
    loader = VideoLoader(target_fps=15)

    frames, _ = loader.load_video(video_path)
    frame = frames[0]  # test first frame

    results = model(frame)[0]

    # Find first detected person
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # class 0 = person
            cropped = crop_person(frame, box)

            # Draw box on original
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            original_with_box = frame.copy()
            cv2.rectangle(original_with_box, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            break
    else:
        print("No person detected")
        return

    # Convert to BGR for display
    original_with_box = cv2.cvtColor(original_with_box, cv2.COLOR_RGB2BGR)
    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

    # Show side-by-side
    cv2.imshow("Original + Box", original_with_box)
    cv2.imshow("Cropped Person", cropped_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_crop("IMG_6123 (1).mov")

