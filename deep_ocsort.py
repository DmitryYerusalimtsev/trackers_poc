import cv2
from ultralytics import YOLO
import os
import supervision

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from boxmot import DeepOCSORT
from pathlib import Path

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

    tracker = DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
        device='cpu',
        fp16=False,
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

                tracker.update(results, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)
                tracker.plot_results(frame, show_trajectories=False)

                # Visualize the results on the frame
                # annotated_frame = results[0].plot()

                # print(type(annotated_frame))
                # print(annotated_frame.shape)
                # out.write(annotated_frame)
                # cv2_imshow(annotated_frame)
                sink.write_frame(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # # Display the annotated frame
    # cv2_imshow(annotated_frame)


