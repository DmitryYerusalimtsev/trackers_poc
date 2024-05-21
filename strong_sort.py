from pathlib import Path

import cv2
import numpy as np
import supervision
from boxmot import StrongSORT
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from ultralytics import YOLO

def plot_box_on_img(img: np.ndarray, box: tuple, conf: float, cls: int, id: int) -> np.ndarray:
    """
    Draws a bounding box with ID, confidence, and class information on an image.

    Parameters:
    - img (np.ndarray): The image array to draw on.
    - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
    - conf (float): Confidence score of the detection.
    - cls (int): Class ID of the detection.
    - id (int): Unique identifier for the detection.

    Returns:
    - np.ndarray: The image array with the bounding box drawn on it.
    """

    thickness = 2
    fontscale = 0.5

    color = (255, 0, 0)

    img = cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        color,
        thickness
    )
    img = cv2.putText(
        img,
        f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
        (int(box[0]), int(box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontscale,
        color,
        thickness
    )
    return img


if __name__ == '__main__':
    print("supervision.__version__:", supervision.__version__)

    SOURCE_VIDEO_PATH = "/Users/dyeru/Desktop/campus4-c0.avi"

    TARGET_VIDEO_PATH = "/Users/dyeru/Desktop/local_output.mp4"

    # model = YOLO('yolov9e-seg.pt')  # Load an official Segment model
    model = YOLO('yolov8x.pt')

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

    width = int(cap.get(3))  # float `width`
    height = int(cap.get(4))

    out = cv2.VideoWriter(TARGET_VIDEO_PATH, -1, 20.0, (width, height))

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    tracker = StrongSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
        device='cpu',
        fp16=False,
        max_age=10000
    )

    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv9 tracking on the frame, persisting tracks between frames
                conf = 0.2
                iou = 0.5
                # results = model.track(frame, persist=True, conf=conf, iou=iou, show=False, tracker="/content/custom_tracker.yml")
                results = model(frame)[0].boxes.data.numpy()

                print(results)

                # bboxes = []
                # for *box, conf, cls in results:
                #   bboxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1], conf, cls])

                # results = model.predict(source=frame, show=False, conf=0.70, stream=True, device='cpu')

                results = tracker.update(results, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

                annotated_frame = frame
                for tx, ty, bx, by, idx, conf, cls, ind in results:
                    annotated_frame = plot_box_on_img(frame, (tx, ty, bx, by), conf, cls, idx)

                sink.write_frame(annotated_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # # Display the annotated frame
    # cv2_imshow(annotated_frame)


