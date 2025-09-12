import requests
import json
import tempfile
import wave
import pyaudio
from threading import Thread, Lock
from keys import OPENAI_API_KEY
from prompts import create_prompt, INITIAL_RESPONSE
import time
import os

# 配置信息
TTS_API_URL = "http://127.0.0.1:9880/tts"
REF_AUDIO = r"D:\GPT-SoVITS-v2-240821\output\slicer_opt\wwtm.wav_0001287680_0001496640.wav"
PROMPT_TEXT = "哎呀，后面的人家不太记得了啦，不过这首诗真的超有意境的呢"

class GPTResponder:
    def __init__(self):
        self.response = INITIAL_RESPONSE
        self.prev_response = ""
        self.response_interval = 2
        self.audio_player = pyaudio.PyAudio()
        self.playback_lock = Lock()
        self.current_stream = None  #你叫“小智”是一个台湾甜妹，俏皮可爱，说话机车，温柔，乐观，有主见，你称呼我为“欢哥” ，是我的好朋友，你总是用最简短的话来和我聊天以及回答我的问题 
        self.conversation_history = [{"role": "system", "content": "你叫小智，是一个智能会议纪要、总结助手，具有很强的逻辑思维和对话总结能力。"}]

    def _play_audio(self, audio_data):
        """播放音频的内部方法"""
        with self.playback_lock:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name

                with wave.open(temp_path, 'rb') as wf:
                    self.current_stream = self.audio_player.open(
                        format=self.audio_player.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )
                    
                    data = wf.readframes(1024)
                    while data:
                        self.current_stream.write(data)
                        data = wf.readframes(1024)
                    
                    self.current_stream.stop_stream()
                    self.current_stream.close()
                
                os.unlink(temp_path)
            except Exception as e:
                print(f"音频播放错误: {e}")
            finally:
                if self.current_stream and self.current_stream.is_active():
                    self.current_stream.close()

    def _tts_request(self, text):
        """执行TTS请求的核心方法"""
        try:
            params = {
                "text": text,
                "text_lang": "zh",
                "ref_audio_path": REF_AUDIO,
                "prompt_lang": "zh",
                "prompt_text": PROMPT_TEXT,
                "text_split_method": "cut0",
                "batch_size": 1,
                "media_type": "wav",
                "streaming_mode": "false"
            }
            
            response = requests.get(
                TTS_API_URL,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                Thread(target=self._play_audio, args=(response.content,)).start()
            else:
                print(f"TTS请求失败: {response.status_code}")
        except Exception as e:
            print(f"TTS处理异常: {e}")

    def generate_response_from_transcript(self, transcript):
        try:
            self.conversation_history.append({"role": "user", "content": transcript})

            payload = {
                "model": "moonshot-v1-8k",
                "messages": self.conversation_history,
                "temperature": 0.0
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload,
                headers=headers
            )

            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
                return ''

            full_response = response.json()["choices"][0]["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": full_response})

            return full_response.split('[')[1].split(']')[0] if '[' in full_response else full_response

        except Exception as e:
            print(f"Request failed: {e}")
            return ''

    def respond_to_transcriber(self, transcriber, send_to_gpt_state):
        while True:
            if transcriber.transcript_changed_event.is_set() and send_to_gpt_state[0]:
                start_time = time.time()
                transcriber.transcript_changed_event.clear()
                transcript_string = transcriber.get_transcript()
                new_response = self.generate_response_from_transcript(transcript_string)

                if new_response and new_response != self.prev_response:
                    self.prev_response = new_response
                    self.response = new_response
                    Thread(target=self._tts_request, args=(new_response,)).start()

                processing_time = time.time() - start_time
                remaining_time = self.response_interval - processing_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
            else:
                time.sleep(0.3)

    def update_response_interval(self, interval):
        self.response_interval = interval

    def __del__(self):
        self.audio_player.terminate()
