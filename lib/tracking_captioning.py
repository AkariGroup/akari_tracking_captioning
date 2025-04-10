import base64
import grpc
import json
import os
import sys
import datetime
import cv2
import numpy as np
import threading
import dataclasses
from queue import Queue
from typing import List, Any, Optional
from lib.chat_akari_captioning import ChatCaptioning

sys.path.append(os.path.join(os.path.dirname(__file__), "local_vlm_server/lib/grpc"))
import local_vlm_server_pb2
import local_vlm_server_pb2_grpc

LOCAL_VLM_APPEARANCE_PROMPT = "Explain this person's gender, age, appearance briefly."
LOCAL_VLM_ACTION_PROMPT = "Explain this person's action briefly."
ACT_LOG_SUMMARISING_PROMPT = """
あなたは時系列データを元に、ユーザーの行動記録をまとめています。
与えられた記録を元に、各行動とその時刻を抽出し、特筆すべき行動のみを簡潔に要約してください。
回答だけを出力してください。
"""

LOG_SUMMARISING_PROMPT = """
あなたは時系列データを元に、ユーザーの記録をまとめています。
与えられた文章を元に、最初にユーザーの年齢、性別、外観の特徴をまとめた後、その人の特筆すべき行動のみを時系列に沿って説明してください。
"gender": 想定される性別。日本語で。
"age" : 想定される年齢。日本語で。
"appearance": 外見の特徴。日本語で。
"act_summary": 行動の要約を文章形式で。
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

    def __init__(self, id: int, start_time: datetime.datetime) -> None:
        self.id: int = id
        self.appearance_log: str = None
        self.act_log: Queue[TrackingCaptioning.CaptionInfo] = Queue()
        self.tmp_summarized_act_log: Queue[str] = Queue()
        self.start_time: datetime.datetime = start_time
        self.end_time: Optional[datetime.datetime] = None


class TrackingCaptioning(object):
    """トラッキングキャプショニングクラス"""

    def __init__(self, queue: Queue[FrameData], host: str, port: str = "10020") -> None:
        """
        コンストラクタ

        Args:
            queue (Queue[FrameData]): トラッキングデータのキュー
            host (str): gRPCサーバーのホスト名
            port (str): gRPCサーバーのポート番号
        """
        self.MAX_ACT_LOG_SIZE = 120
        self.ROI_EXPANSION_PIXEL = 30
        self.lock = threading.Lock()
        self.queue = queue
        # gRPCチャネルを作成
        channel = grpc.insecure_channel(f"{host}:{port}")
        self.vlm_stub = local_vlm_server_pb2_grpc.LocalVlmServerServiceStub(channel)
        self.cur_tracking_person_list: List[PersonLog] = []
        self.track_finish_id_queue: Queue[int] = Queue()
        self.chat_captioning = ChatCaptioning()
        now = datetime.datetime.now()
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{now.strftime('%Y%m%d_%H%M%S')}.csv")
        self.log_control_thread = threading.Thread(
            target=self.log_control,
            daemon=True,
        )
        self.log_control_thread.start()

    def log_control(self) -> None:
        """ログ制御スレッド"""
        while True:
            # action_logが指定サイズ以上になったら要約を生成
            for person in self.cur_tracking_person_list:
                if person.act_log.qsize() > self.MAX_ACT_LOG_SIZE:
                    self.tmp_summary_act_log(person, data_size=self.MAX_ACT_LOG_SIZE)
            # track_finish_id_queueが空でなければ、まとめてログを保存
            if not self.track_finish_id_queue.empty():
                id = self.track_finish_id_queue.get()
                person = self.get_person_log(id)
                if person is not None:
                    # 要約を生成し、csvに記録後、トラッキングリストから削除
                    summary = self.summary_act_log(person)
                    self.add_log_to_csv(person, summary)
                    self.lock.acquire()
                    self.cur_tracking_person_list.remove(person)
                    self.lock.release()

    def run(self) -> None:
        """トラッキングキャプショニングを実行するメソッド"""
        while True:
            tracking_data: FrameData = self.queue.get()
            if tracking_data is None:
                continue
            image = tracking_data.image
            tracklets = tracking_data.tracklets
            timestamp = tracking_data.timestamp
            self.lock.acquire()
            for person in self.cur_tracking_person_list:
                # トラッキング対象の人物がトラッキング外になった場合
                # かつ、self.track_finish_id_queueにIDが存在しない場合追加
                if not self.is_available_in_tracklets(
                    person.id, tracklets
                ) and person.id not in list(self.track_finish_id_queue.queue):
                    person.end_time = timestamp
                    self.track_finish_id_queue.put(person.id)
            for tracklet in tracklets:
                if not tracklet.status.name == "TRACKED":
                    continue
                target_image = self.get_target_image_from_tracklet(
                    frame=image,
                    tracklet=tracklet,
                    expantion_pixel=self.ROI_EXPANSION_PIXEL,
                )
                # OpenCV画像をbase64エンコード
                base64_image = self.cv_to_base64(target_image)
                if not self.is_tracking_person(tracklet):
                    # VLMに送信
                    request = local_vlm_server_pb2.SendImageRequest(
                        images=[base64_image],
                        prompt=LOCAL_VLM_APPEARANCE_PROMPT,
                    )
                    try:
                        response = self.vlm_stub.SendImage(request)
                        new_person = PersonLog(
                            id=tracklet.id,
                            start_time=timestamp,
                        )
                        print(
                            f"New person detected: {tracklet.id}, appearance: {response}"
                        )
                        new_person.appearance_log = response
                        self.cur_tracking_person_list.append(new_person)
                    except grpc.RpcError as e:
                        print(f"Send image RPC error: {e}")
                        continue
                else:
                    # VLMに送信
                    request = local_vlm_server_pb2.SendImageRequest(
                        images=[base64_image],
                        prompt=LOCAL_VLM_ACTION_PROMPT,
                    )
                    try:
                        response = self.vlm_stub.SendImage(request)
                        person_log = self.get_person_log(tracklet.id)
                        person_log.act_log.put(
                            CaptionInfo(caption=response, timestamp=timestamp)
                        )
                        print(f"id: {tracklet.id}, action: {response}")
                    except grpc.RpcError as e:
                        print(f"Send image RPC error: {e}")
                        continue
            self.lock.release()

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

    def is_available_in_tracklets(self, id: int, tracklets: List[Any]) -> bool:
        """トラッキング情報の中に指定したIDが存在するかどうかを判定する

        Args:
            id (int): トラッキング対象の人物ID
            tracklets (List[Any]): トラッキング情報

        Returns:
            bool: トラッキング対象の人物であればTrue, それ以外はFalse
        """
        for tracklet in tracklets:
            if tracklet.id == id:
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
        self, frame: np.ndarray, tracklet: Any, expantion_pixel: int = 0
    ) -> np.ndarray:
        """
        トラッキング対象の画像を切り出す

        Args:
            frame (np.ndarray): カメラ画像
            tracklet (Any): トラッキング情報
            expantion_pixel (int): ここで指定した値だけ、ROIの範囲を拡大する
                デフォルトは0で、拡大しない

        Returns:
            np.ndarray: 切り出したトラッキング対象の画像
        """
        # TODO: 対象以外の人の顔をマスクする処理の追加
        roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
        x1 = int(roi.topLeft().x) - expantion_pixel
        if x1 < 0:
            x1 = 0
        y1 = int(roi.topLeft().y) - expantion_pixel
        if y1 < 0:
            y1 = 0
        x2 = int(roi.bottomRight().x) + expantion_pixel
        if x2 > frame.shape[1]:
            x2 = frame.shape[1]
        y2 = int(roi.bottomRight().y) + expantion_pixel
        if y2 > frame.shape[0]:
            y2 = frame.shape[0]
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

    def tmp_summary_act_log(
        self, person_log: PersonLog, data_size: Optional[int] = None
    ) -> None:
        """アクションログを要約するメソッド

        Args:
            person_log (PersonLog): 人物ログ
            data_size (Optional[int]): 要約するデータのサイズ
                デフォルトはNoneで、MAX_ACT_LOG_SIZEを使用
        """
        if person_log.act_log.qsize() == 0:
            return
        if data_size is not None:
            if data_size > person_log.act_log.qsize():
                raise ValueError("data_size must be less than act_log length.")
        else:
            data_size = person_log.act_log.qsize()
        messages = [
            self.chat_captioning.create_message(
                text=ACT_LOG_SUMMARISING_PROMPT, role="system"
            )
        ]
        # data_size回分のログをqueueから取得
        query = ""
        for _ in range(data_size):
            act_data = person_log.act_log.get()
            query += f"{act_data.timestamp.strftime('%H:%M:%S')}: {act_data.caption}\n"
        messages.append(
            self.chat_captioning.create_message(
                text=query,
                role="user",
            )
        )
        response = ""
        # 要約を生成
        print(f"tmp summarizing action log... query: {query}")
        for sentence in self.chat_captioning.chat(
            messages=messages,
            model="gemini-2.0-flash",
            temperature=0.0,
            stream_per_sentence=True,
        ):
            response += sentence
        print(f"tmp summarizing action log... response: {response}")
        # 要約をキューに追加
        person_log.tmp_summarized_act_log.put(response)

    def summary_act_log(self, person_log: PersonLog) -> dict:
        """アクションログを要約するメソッド

        Args:
            person_log (PersonLog): 人物ログ

        Returns:
            dict: 要約されたログ
        """
        self.tmp_summary_act_log(person_log=person_log)
        messages = [
            self.chat_captioning.create_message(
                text=LOG_SUMMARISING_PROMPT, role="system"
            )
        ]
        query = f"#外観\n {person_log.appearance_log}\n"
        query += f"#行動履歴 \n"
        # tmp_summarized_act_logが空でなければ、要約を取得
        if not person_log.tmp_summarized_act_log.empty():
            query += f" {person_log.tmp_summarized_act_log.get()}\n"
        messages.append(
            self.chat_captioning.create_message(
                text=query,
                role="user",
            )
        )
        response = ""
        # 要約を生成
        print(f"summarizing log... query: {query}")
        for sentence in self.chat_captioning.captioning_gemini(
            messages=messages,
            model="gemini-2.0-flash",
            temperature=0.0,
            stream_per_sentence=True,
        ):
            response += sentence
        response = json.loads(response)
        print(f"summarizing log... response: {response}")
        return response

    def add_log_to_csv(self, person_log: PersonLog, summary: dict) -> None:
        """要約をCSVに追加するメソッド

        Args:
            person_log (PersonLog): 人物ログ
            summary (str): 要約
        """
        age = summary.get("age", "")
        gender = summary.get("gender", "")
        appearance = summary.get("appearance", "")
        act_summary = summary.get("act_summary", "")
        act_summary = act_summary.replace("\n", " ")
        act_summary = act_summary.replace(",", "")
        with open(self.log_path, "a") as f:
            f.write(
                f"{person_log.id},{person_log.start_time},{person_log.end_time},{age},{gender},{appearance},{act_summary}\n"
            )
