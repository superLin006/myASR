# transcriberModels.py
import os
import sys
import torch
import whisper
from funasr import AutoModel
# from funasr.utils.postprocess_utils import rich_transcription_postprocess

def resource_path(relative_path):
    """
    获取资源文件的绝对路径： 
    """
    # PyInstaller 创建临时文件夹，会把路径存到 _MEIPASS
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)

class BaseASRModel:
    """ASR 模型统一接口"""
    def transcribe(self, file_path, language="auto"):
        raise NotImplementedError

class WhisperASR(BaseASRModel):
    def __init__(self, model_name="small", device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 从 models/whisper/*.pt 加载
        model_rel = os.path.join("models", "whisper", f"{model_name}.pt")
        model_path = resource_path(model_rel)
        self.model = whisper.load_model(model_path, device=device)

    def transcribe(self, file_path, language="auto"):
        result = self.model.transcribe(file_path, language=language, task="transcribe")
        return result.get("text", "").strip()

class FunASR(BaseASRModel):
    def __init__(self, model_name="paraformer-speech_68m", device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 模型目录：models/funasr/<model_name>/
        base_rel = os.path.join("models", "funasr", model_name)
        base_path = resource_path(base_rel)

        self.model = AutoModel(
            model=base_path,  # Directly use base_path without appending 'speech_seaco'
            vad_model=resource_path(os.path.join("models", "funasr", "speech_fsmn_vad")),
            punc_model=resource_path(os.path.join("models", "funasr", "punc_ct-transformer_291m")),
            device=device,
            disable_update=True,
            trust_remote_code=True
        )

    def transcribe(self, file_path, language="auto"):
        # funasr 不需要 language 参数，直接 decode
        res = self.model.generate(input=file_path)
        if res and "text" in res[0]:
            text = res[0]["text"].strip()
            # 如果需要后处理：
            # text = rich_transcription_postprocess(text)
            return text
        return ""

def load_asr_model(series, model_name, device=None):
    """
    工厂方法：根据系列名加载模型
    """
    series = series.lower()
    if series == "whisper":
        return WhisperASR(model_name, device)
    elif series == "funasr":
        return FunASR(model_name, device)
    else:
        raise ValueError(f"Unknown ASR series: {series}")
