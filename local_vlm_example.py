import argparse
import base64
import grpc
import os
import sys
import time
import datetime
import cv2
import numpy as np
import threading
import dataclasses
from queue import Queue
from typing import List, Any, Optional, Tuple, Union
from lib.akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo
from lib.akari_yolo_lib.util import download_file

sys.path.append(os.path.join(os.path.dirname(__file__), "lib/grpc"))
import local_vlm_server_pb2
import local_vlm_server_pb2_grpc


@dataclasses.dataclass
class TrackingData(object):
    """トラッキングデータを保持するクラス"""

    image: np.ndarray
    tracklets: List[Any]
    timestamp: datetime.datetime

def vlm_caption(queue: Queue[TrackingData]) -> None:
    """VLMのキャプションを取得する関数"""
    # gRPCチャネルを作成
    channel = grpc.insecure_channel("localhost:10020")
    stub = local_vlm_server_pb2_grpc.LocalVlmServerServiceStub(channel)

    while True:
        image, tracklet,timestamp = queue.get()

        encoded_images = []

        # 画像をbase64エンコード
        for image_path in :
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                continue
            try:
                with open(image_path, "rb") as image_file:
                    # 画像データをbase64エンコード（プレフィックスなし）
                    encoded = base64.b64encode(image_file.read()).decode("utf-8")
                    encoded_images.append(encoded)
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                continue

        if not encoded_images:
            print("No valid images to process")
            return
        start_time = time.time()
        # 画像とプロンプトを送信
        request = local_vlm_server_pb2.SendImageRequest(
            images=["base64_encoded_image"],
            prompt="What is this?",
        )
        try:
            response = stub.SendImage(request)
            print(f"Response: {response.response}")
            print(f"Time taken: {time.time() - start_time:.2f}s")
        except grpc.RpcError as e:
            print(f"RPC error: {e}")

def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
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

    # gRPCチャネルを作成
    channel = grpc.insecure_channel("localhost:10020")
    stub = local_vlm_server_pb2_grpc.LocalVlmServerServiceStub(channel)

    tracking_data_queue: Queue[TrackingData] = Queue()
    vlm_caption_thread = threading.Thread(target=vlm_caption, args=(tracking_data_queue,))
    vlm_caption_thread.start()

    end = False
    while not end:
        oakd_tracking_yolo = OakdTrackingYolo(
            config_path="config/human_parts.json",
            model_path="model/human_parts.blob",
            fps=args.fps,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
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
                        TrackingData(image=frame, tracklets=tracklets, timestamp=datetime.datetime.now())
                    )
                oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            if cv2.waitKey(1) == ord("q"):
                end = True
                break
        oakd_tracking_yolo.close()


if __name__ == "__main__":
    main()
