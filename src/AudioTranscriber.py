# AudioTranscriber.py

import os
import sys
import io
import wave
import tempfile
import threading
from datetime import datetime

import custom_speech_recognition as sr
import pyaudiowpatch as pyaudio
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer
)

PHRASE_TIMEOUT = 3.05
MAX_PHRASES    = 4


def resource_path(relative_path):
    """PyInstaller 打包时使用 sys._MEIPASS，否则取当前路径"""
    base = getattr(sys, '_MEIPASS',
                   os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, relative_path)


class AudioTranscriber:
    def __init__(self, mic_source, speaker_source, asr_model,
                 mt_backend, mt_model_name):
        self.asr_model = asr_model
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

        # —— 初始化翻译模型
        self.mt_backend = mt_backend.lower()
        if self.mt_backend == "helsinki":
            path = resource_path(
                os.path.join("models", "Helsinki-NLP", f"opus-mt-{mt_model_name}")
            )
            self.tok = MarianTokenizer.from_pretrained(path)
            self.model = MarianMTModel.from_pretrained(path)
        else:  # m2m100
            self.src_lang, self.tgt_lang = mt_model_name.split("-")
            path = resource_path(os.path.join("models", "m2m100_418M"))
            self.tok = M2M100Tokenizer.from_pretrained(path)
            self.model = M2M100ForConditionalGeneration.from_pretrained(path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device.type == "cuda":  # 仅 GPU 下启用半精度
            self.model.half()

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

            # —— 翻译
            trans_text = ""
            try:
                if self.mt_backend == "helsinki":
                    inputs = self.tok([orig_text],
                                      return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    gen = self.model.generate(**inputs)
                    trans_text = self.tok.decode(gen[0],
                                                skip_special_tokens=True)
                else:
                    self.tok.src_lang = self.src_lang
                    inputs = self.tok([orig_text],
                                      return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    gen = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tok.get_lang_id(self.tgt_lang)
                    )
                    trans_text = self.tok.batch_decode(
                        gen, skip_special_tokens=True
                    )[0]
            except Exception as e:
                print("Translation warning:", e)

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
        """
        新段：append；同一段：覆盖最后一条
        """
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
        """
        合并双方的 transcript，并按时间排序，取最新 MAX_PHRASES 条
        """
        merged = []
        for who, lst in self.transcript_data.items():
            for orig, trans, ts in lst:
                merged.append((who, orig, trans, ts))
        merged.sort(key=lambda x: x[3])  # 按时间排序
        return merged[-MAX_PHRASES:]
    
    def get_transcript(self) -> str:
        """
        兼容旧接口：返回最新 MAX_PHRASES 条原文字幕，供 GPTResponder 调用。
        """
        entries = self.get_transcript_entries()
        lines = [f"[{ts}] {who}: {orig}" for who, orig, _, ts in entries]
        return "\n".join(lines)


    def clear_transcript_data(self):
        self.transcript_data = {"You": [], "Speaker": []}
        for src in self.audio_sources.values():
            src["last_sample"] = bytes()
            src["last_spoken"] = None
            src["new_phrase"]  = True
