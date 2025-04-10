import copy
from typing import Generator
from google import genai
from google.genai import types

from .akari_chatgpt_bot.lib.chat_akari import (
    ChatStreamAkari,
)
from .akari_chatgpt_bot.lib.conf import GEMINI_APIKEY


class ChatCaptioning(ChatStreamAkari):
    """Geminiでキャプショニングを行うためのクラス。"""

    def __init__(self):
        super().__init__()

    def captioning_gemini(
        self,
        messages: list,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        stream_per_sentence: bool = True,
    ) -> Generator[str, None, None]:
        """Geminiを使用して行動情報のキャプショニングを行う。

        Args:
            messages (list): 会話のメッセージリスト。
            model (str, optional): モデル。デフォルトは"gemini-2.0-flash"。
            temperature (float, optional): 生成の多様性。デフォルトは0.7。
            stream_per_sentence (bool, optional): 文ごとにストリームを生成するかどうか。デフォルトはTrue。
        Yields:
            str: チャット応答のジェネレータ。

        """
        if GEMINI_APIKEY is None:
            print("Gemini API key is not set.")
            return
        (
            system_instruction,
            history,
            cur_message,
        ) = self.convert_messages_from_gpt_to_gemini(copy.deepcopy(messages))
        print(cur_message)
        print(history)
        chat = self.gemini_client.chats.create(
            model=model,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["gender", "age", "appearance", "act_summary"],
                    properties={
                        "gender": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                        "age": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                        "appearance": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                        "act_summary": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    },
                ),
            ),
        )
        try:
            responses = chat.send_message_stream(cur_message)
        except BaseException as e:
            raise ValueError(f"Geminiレスポンスエラー: {e}")
        yield from self.parse_output_stream_gemini(responses, stream_per_sentence)
