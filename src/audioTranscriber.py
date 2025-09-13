# AudioTranscriber.py
import os
import sys
import io
import wave
import tempfile
import threading
from datetime import datetime
import pytz                          # ← 补上

import custom_speech_recognition as sr
import pyaudiowpatch as pyaudio


PHRASE_TIMEOUT = 3.05
MAX_PHRASES    = 4


class AudioTranscriber:
    def __init__(self, mic_source, speaker_source, asr_model, translator=None):
        self.asr_model = asr_model
        self.translator = translator  # 新增：翻译模块（可选）
        self.transcript_data = {"You": [], "Speaker": []}
        self.transcript_changed_event = threading.Event()

        # —— 初始化音源状态
        self.audio_sources = {
            "You": {
                "sample_rate": mic_source.SAMPLE_RATE,
                "sample_width": mic_source.SAMPLE_WIDTH,
                "channels":    mic_source.channels,
                "last_sample": bytes(),
                "last_spoken": None,
                "new_phrase":  True,
                "process_data_func": self.process_mic_data
            }
        }
        if speaker_source:
            self.audio_sources["Speaker"] = {
                "sample_rate": speaker_source.SAMPLE_RATE,
                "sample_width": speaker_source.SAMPLE_WIDTH,
                "channels":    speaker_source.channels,
                "last_sample": bytes(),
                "last_spoken": None,
                "new_phrase":  True,
                "process_data_func": self.process_speaker_data
            }

    def transcribe_audio_queue(self, audio_queue):
        while True:
            who, data, time_spoken = audio_queue.get()
            if who not in self.audio_sources:
                continue

            # —— 更新缓存
            self._update_audio_buffer(who, data, time_spoken)

            # —— 写 WAV 做 ASR
            orig_text, tmp = "", None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                    tmp = fp.name
                    self.audio_sources[who]["process_data_func"](
                        self.audio_sources[who]["last_sample"], tmp
                    )
                orig_text = self.asr_model.transcribe(tmp, language="en")
            except Exception as e:
                print("ASR Error:", e)
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)

            if not orig_text:
                continue

            # —— 翻译（可选）
            trans_text = ""
            if self.translator:
                trans_text = self.translator.translate(orig_text)

            # —— 更新 transcript
            ts_str = time_spoken.strftime("%H:%M:%S")
            self.update_transcript(who, orig_text, trans_text, ts_str)

    def _update_audio_buffer(self, who, data, time_spoken):
        src = self.audio_sources[who]
        if (src["last_spoken"] and
            (time_spoken - src["last_spoken"]).total_seconds() > PHRASE_TIMEOUT):
            src["last_sample"] = bytes()  # 新段清空缓存
            src["new_phrase"]  = True
        else:
            src["new_phrase"]  = False
        src["last_sample"] += data
        src["last_spoken"]  = time_spoken

    def process_mic_data(self, data, temp_file_name):
        audio_data = sr.AudioData(
            data,
            self.audio_sources["You"]["sample_rate"],
            self.audio_sources["You"]["sample_width"]
        )
        wav = io.BytesIO(audio_data.get_wav_data())
        with open(temp_file_name, "wb") as f:
            f.write(wav.read())

    def process_speaker_data(self, data, temp_file_name):
        with wave.open(temp_file_name, "wb") as wf:
            wf.setnchannels(self.audio_sources["Speaker"]["channels"])
            p = pyaudio.PyAudio()
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.audio_sources["Speaker"]["sample_rate"])
            wf.writeframes(data)

    def update_transcript(self, who, orig_text, trans_text, ts_str):
        lst = self.transcript_data[who]
        entry = (orig_text, trans_text, ts_str)
        if self.audio_sources[who]["new_phrase"] or not lst:
            lst.append(entry)
        else:
            lst[-1] = entry

        if len(lst) > MAX_PHRASES:
            lst[:] = lst[-MAX_PHRASES:]

        self.transcript_changed_event.set()

    def get_transcript_entries(self):
        merged = []
        for who, lst in self.transcript_data.items():
            for orig, trans, ts in lst:
                merged.append((who, orig, trans, ts))
        merged.sort(key=lambda x: x[3])  # 按时间排序
        return merged[-MAX_PHRASES:]
    
    def get_transcript(self) -> str:
        entries = self.get_transcript_entries()
        lines = [f"[{ts}] {who}: {orig}" for who, orig, _, ts in entries]
        return "\n".join(lines)

    def clear_transcript_data(self):
        self.transcript_data = {"You": [], "Speaker": []}
        for src in self.audio_sources.values():
            src["last_sample"] = bytes()
            src["last_spoken"] = None
            src["new_phrase"]  = True
