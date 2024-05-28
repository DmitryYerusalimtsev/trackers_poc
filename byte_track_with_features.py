import cv2
import numpy as np
from ultralytics import YOLO
import supervision
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from feature_extraction import FeatureExtraction
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import distance


@dataclass
class KnownObject:
    id: int
    features: np.ndarray


def crop_object(src_image, points):
    mask = np.zeros(src_image.shape[:2], dtype=np.uint8)  # Create a blank mask
    pts = points.astype(np.int32)
    mask = cv2.fillPoly(mask, [pts], color=(255, 0, 0))

    cropped = cv2.bitwise_and(src_image, src_image, mask=mask)

    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(pts)

    # Crop the region of interest
    cropped = cropped[y:y + h, x:x + w]

    return cropped


def compare_object_with_known(obj_features):
    for known_object in known_objects:
        dist, _, _ = distance.directed_hausdorff(known_object.features, obj_features)

        if dist < 12.5:  # the same
            return known_object.id

    if len(known_objects) == 0:
        new_id = 1
    else:
        new_id = known_objects[-1].id + 1

    known_objects.append(KnownObject(new_id, obj_features))
    return new_id


def plot_box_on_img(img: np.ndarray, box: tuple, conf: float, cls: int, id: int) -> np.ndarray:
    thickness = 2
    font_scale = 0.5

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
        font_scale,
        color,
        thickness
    )
    return img


if __name__ == '__main__':
    print("supervision.__version__:", supervision.__version__)

    SOURCE_VIDEO_PATH = "/Users/dyeru/Desktop/campus4-c0.avi"

    TARGET_VIDEO_PATH = "/Users/dyeru/Desktop/local_output.mp4"

    model = YOLO('yolov9e-seg.pt')  # Load an official Segment model

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

    width = int(cap.get(3))  # float `width`
    height = int(cap.get(4))

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    feature_extraction = FeatureExtraction()

    known_objects = []
    current_id = 1

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
                # annotated_frame = results[0].plot()

                boxes = results[0].boxes
                masks = results[0].masks

                for i in range(len(boxes)):
                    conf = boxes.conf[i].tolist()
                    cls = boxes.cls[i].tolist()
                    tx, ty, bx, by = boxes.xyxy[i].tolist()

                    if conf > 0.83:
                        seg_points = masks.xy[i]

                        # Get the bounding box coordinates
                        # x, y, w, h = map(lambda f: int(f), boxes.xywh[i].tolist())

                        # Crop the region of interest
                        # cropped_frame = frame[y:y + h, x:x + w]

                        cropped_object = crop_object(frame, seg_points)

                        # plt.imshow(cropped_object), plt.show()

                        # Detected features
                        object_features = feature_extraction.predict_img(cropped_object)[0]

                        object_id = compare_object_with_known(object_features)

                        plot_box_on_img(frame, (tx, ty, bx, by), conf, cls, object_id)

                sink.write_frame(frame)

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
