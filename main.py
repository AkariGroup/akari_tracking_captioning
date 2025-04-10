import argparse
import datetime
import cv2
import threading
from queue import Queue
from lib.akari_yolo_lib.util import download_file
from lib.akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo
from lib.tracking_captioning import FrameData, TrackingCaptioning


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    parser.add_argument(
        "--vlm_host",
        help="Host name of local VLM server",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--vlm_port",
        help="Port number of local VLM server",
        type=str,
        default="10020",
    )
    args = parser.parse_args()
    model_path = "model/human_parts.blob"
    config_path = "config/human_parts.json"
    download_file(
        model_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.blob",
    )
    download_file(
        config_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.json",
    )
    tracking_data_queue = Queue(maxsize=1)
    tracking_captioning = TrackingCaptioning(
        queue=tracking_data_queue, host=args.vlm_host, port=args.vlm_port
    )
    tracking_caption_thread = threading.Thread(target=tracking_captioning.run)
    tracking_caption_thread.start()

    end = False
    while not end:
        oakd_tracking_yolo = OakdTrackingYolo(
            config_path="config/human_parts.json",
            model_path="model/human_parts.blob",
            fps=8,
            robot_coordinate=args.robot_coordinate,
            track_targets=["person"],
            show_bird_frame=True,
            show_orbit=False,
        )
        oakd_tracking_yolo.update_bird_frame_distance(10000)
        while True:
            frame = None
            detections = []
            try:
                frame, detections, tracklets = oakd_tracking_yolo.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Please lower FPS.")
                print("==================")
                break
            if frame is not None:
                # tracking_data_queueにデータが何もない時のみput
                if tracking_data_queue.empty():
                    tracking_data_queue.put(
                        FrameData(
                            image=frame,
                            tracklets=tracklets,
                            timestamp=datetime.datetime.now(),
                        )
                    )
                oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            if cv2.waitKey(1) == ord("q"):
                end = True
                break
        oakd_tracking_yolo.close()


if __name__ == "__main__":
    main()
