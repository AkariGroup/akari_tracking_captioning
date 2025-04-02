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
from lib.akari_chatgpt_bot.lib.chat_akari import ChatStreamAkari

sys.path.append(os.path.join(os.path.dirname(__file__), "lib/grpc"))
import local_vlm_server_pb2
import local_vlm_server_pb2_grpc

ACT_LOG_SUMMARISING_PROMPT = """
与えられた文章を元に、各行動とその時刻を抽出し、要約してください。
"""

@dataclasses.dataclass
class FrameData(object):
    """トラッキングデータを保持するクラス"""

    image: np.ndarray
    tracklets: List[Any]
    timestamp: datetime.datetime

@dataclasses.dataclass
class CaptionInfo(object):
    """キャプション情報を保持するクラス"""
    caption: str
    timestamp: datetime.datetime

class PersonLog(object):
    """トラッキング対象の人物情報を保持するクラス"""

    def __init__(self, id: int, name: str, start_time: datetime.datetime) -> None:
        self.id: int = id
        self.appearance_log: str = None
        self.act_log: Queue[TrackingCaptioning.CaptionInfo] = Queue()
        self.tmp_summarized_act_log: Queue[str] = Queue()
        self.start_time: datetime.datetime = start_time
        self.end_time: Optional[datetime.datetime] = None

class TrackingCaptioning(object):
    """トラッキングキャプショニングクラス"""
    def __init__(self, queue: Queue[FrameData], port: str, host: str = "10020") -> None:
        self.MAX_ACT_LOG_SIZE = 120
        self.queue = queue
        # gRPCチャネルを作成
        channel = grpc.insecure_channel(f"{host}:{port}")
        self.vlm_stub = local_vlm_server_pb2_grpc.LocalVlmServerServiceStub(channel)
        self.cur_tracking_person_list: List[TrackingCaptioning.PersonLog] = []
        self.track_finish_id_queue: Queue[int] = Queue()
        self.chat_stream = ChatStreamAkari()
        now = datetime.datetime.now()
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(
            log_dir,
            f"{now.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.log_control_thread = threading.Thread(
            target=self.log_control,
            args=(self.log_path,),
            daemon=True,
        )
        self.log_control_thread.start()

    def log_control(self, log_path: str) -> None:
        while True:
            # actrion_logが指定サイズ以上になったら要約を生成
            for person in self.cur_tracking_person_list:
                self.summarize_act_log(person)
            # track_finish_id_queueが空でなければ、まとめてログを保存
            if not self.track_finish_id_queue.empty():
                # track_finish_id_queueからIDを取得
                id = self.track_finish_id_queue.get()
                person = self.get_person_log(id)
                if person is not None:
                    # 要約を作成
                    self.summary_act_log(person)
                    # ログを保存
                    with open(log_path, "a") as f:
                        f.write(f"{id},{person.start_time},{person.end_time},{person.appearance_log}\n")
                    # 人物ログを削除
                    self.cur_tracking_person_list.remove(person)



    def run(self) -> None:
        while True:
            tracking_data: FrameData = self.queue.get()
            if tracking_data is None:
                continue
            image = tracking_data.image
            tracklets = tracking_data.tracklets
            timestamp = tracking_data.timestamp
            for tracklet in tracklets:
                target_image = self.get_target_image_from_tracklet(image, tracklet)
                # OpenCV画像をbase64エンコード
                base64_image = self.cv_to_base64(target_image)
                if not self.is_tracking_person(tracklet):
                    # VLMに送信
                    request = local_vlm_server_pb2.SendImageRequest(
                        images=base64_image,
                        prompt="Explain this person's gender, age, appearance briefly.",
                    )
                    try:
                        response = self.vlm_stub.SendImage(request)
                        new_person = TrackingCaptioning.PersonLog(
                            id=tracklet.id,
                            appearance_log=response,
                            start_time=timestamp,
                        )
                    except grpc.RpcError as e:
                        print(f"RPC error: {e}")
                        continue
                else:
                    # VLMに送信
                    request = local_vlm_server_pb2.SendImageRequest(
                        images=base64_image,
                        prompt="Explain this person's action briefly.",
                    )
                    try:
                        response = self.vlm_stub.SendImage(request)
                        person_log = self.get_person_log(tracklet.id)
                        person_log.act_log.put(
                            self.CaptionInfo(
                                caption=response, timestamp=timestamp
                            )
                        )
                    except grpc.RpcError as e:
                        print(f"RPC error: {e}")
                        continue

    def is_tracking_person(self, tracklet: Any) -> bool:
        """トラッキング中の人物idかどうかを判定する

        Args:
            tracklet (Any): トラッキング情報

        Returns:
            bool: トラッキング対象の人物であればTrue, それ以外はFalse
        """
        for person in self.cur_tracking_person_list:
            if person.id == tracklet.id:
                return True
        return False

    def cv_to_base64(self, image: np.ndarray) -> str:
        """OpenCV画像をbase64エンコードした文字列に変換する
        Args:
            image (np.ndarray): OpenCV画像データ

        Returns:
            str: base64エンコードされた画像データ

        """
        _, encoded = cv2.imencode(".jpg", image)
        return base64.b64encode(encoded).decode("ascii")

    def get_target_image_from_tracklet(
        self, frame: np.ndarray, tracklet: Any
    ) -> np.ndarray:
        """
        トラッキング対象の画像を切り出す

        Args:
            frame (np.ndarray): カメラ画像
            tracklet (Any): トラッキング情報

        Returns:
            np.ndarray: 切り出したトラッキング対象の画像
        """
        # TODO: 対象以外の人の顔をマスクする処理の追加
        roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
        x1 = int(roi.topLeft().x)
        y1 = int(roi.topLeft().y)
        x2 = int(roi.bottomRight().x)
        y2 = int(roi.bottomRight().y)
        return frame[y1:y2, x1:x2]

    def get_person_log(self, id: int) -> Optional["PersonLog"]:
        """指定されたIDの人物ログを取得する

        Args:
            id (int): 人物ID

        Returns:
            Optional[TrackingCaptioning.PersonLog]: 指定されたIDの人物ログ
        """
        for person in self.cur_tracking_person_list:
            if person.id == id:
                return person
        return None

    def tmp_summary_act_log(self, person_log: PersonLog) -> None:
        """アクションログを要約するメソッド"""
        if person_log.act_log.qsize() < self.MAX_ACT_LOG_SIZE:
            return
        messages = [self.chat_stream.create_message(text=ACT_LOG_SUMMARISING_PROMPT, role="system")]
        # self.MAX_ACT_LOG_SIZE回分のログをqueueから取得
        act_info = ""
        for _ in range(self.MAX_ACT_LOG_SIZE):
            act_data = person_log.act_log.get()
            act_info += f"{act_data.timestamp.strftime('%H:%M:%S')}: {act_data.caption}\n"
        messages.append(
            self.chat_stream.create_message(
                text=act_info,
                role="user",
            )
        )
        response = ""
        # 要約を生成
        for sentence in self.chat_stream.chat(
            messages=act_info,
            model="gemini-2.0-flash",
            temperature=0.0,
            stream_per_sentence=True,
        ):
            response += sentence
        # 要約をキューに追加
        person_log.tmp_summarized_act_log.put(response)

    def summary_act_log(self, person_log: PersonLog) -> str:
        """アクションログを要約するメソッド"""
        tmp_summary = ""
        # tmp_summarized_act_logが空でなければ、要約を取得
        if not person_log.tmp_summarized_act_log.empty():
            tmp_summary += f"{person_log.tmp_summarized_act_log.get()}\n"



def vlm_caption(queue: Queue[FrameData]) -> None:
    """VLMのキャプションを取得する関数"""

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

    tracking_data_queue: Queue[FrameData] = Queue()
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
            track_targets=["person"],
            show_bird_frame=True,
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
                        FrameData(image=frame, tracklets=tracklets, timestamp=datetime.datetime.now())
                    )
                oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            if cv2.waitKey(1) == ord("q"):
                end = True
                break
        oakd_tracking_yolo.close()


if __name__ == "__main__":
    main()
