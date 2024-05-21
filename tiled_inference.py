from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cpu',
    fp16=False,
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt',
    confidence_threshold=0.5,
    device="cpu",  # or 'cuda:0'
)

SOURCE_VIDEO_PATH = "/Users/dyeru/Desktop/campus4-c0.avi"

TARGET_VIDEO_PATH = "/Users/dyeru/Desktop/local_output.mp4"

vid = cv2.VideoCapture(SOURCE_VIDEO_PATH)

color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    while True:
        ret, im = vid.read()

        # get sliced predictions
        result = get_sliced_prediction(
            im,
            detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        num_predictions = len(result.object_prediction_list)
        dets = np.zeros([num_predictions, 6], dtype=np.float32)
        for ind, object_prediction in enumerate(result.object_prediction_list):
            dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
            dets[ind, 4] = object_prediction.score.value
            dets[ind, 5] = object_prediction.category.id

        tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind)

        tracker.plot_results(im, show_trajectories=False)

        sink.write_frame(im)

vid.release()
cv2.destroyAllWindows()