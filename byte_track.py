import cv2
from ultralytics import YOLO
import supervision
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink

if __name__ == '__main__':
    print("supervision.__version__:", supervision.__version__)

    SOURCE_VIDEO_PATH = "/Users/dyeru/Desktop/campus4-c0.avi"

    TARGET_VIDEO_PATH = "/Users/dyeru/Desktop/local_output.mp4"

    model = YOLO('yolov9e-seg.pt')  # Load an official Segment model

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

    width = int(cap.get(3))  # float `width`
    height = int(cap.get(4))

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv9 tracking on the frame, persisting tracks between frames
                conf = 0.2
                iou = 0.5
                results = model.track(frame, persist=True, conf=conf, iou=iou, show=False, tracker="custom_tracker.yml")

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                sink.write_frame(frame)

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()



