# download_all_offline.py
import os
from pathlib import Path
import whisper
from transformers import (
    M2M100Tokenizer, M2M100ForConditionalGeneration,
    MarianTokenizer, MarianMTModel,
)
from funasr import AutoModel

# ========== 1. 翻译模型 ==========
def download_mt():
    cache = Path("../models")
    # 1.1 M2M100-418M
    print("↓ M2M100-418M")
    M2M100Tokenizer.from_pretrained("facebook/m2m100_418m", cache_dir=cache / "m2m100_418m")
    M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=cache / "m2m100_418m")

    # 1.2 opus-mt-zh-en
    print("↓ opus-mt-zh-en")
    MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir=cache / "opus-mt-zh-en")
    MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir=cache / "opus-mt-en-zh")

# ========== 2. ASR 系列 ==========
def download_asr():
    cache = Path("../models")
    # 2.1 Paraformer-zh + 配套 vad/punc/spk（FunASR 会自动缓存）
    print("↓ Paraformer-zh 及配套 vad/punc/spk")
    _ = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device="cpu"          # 仅下载，不占用 GPU
    )

    # 2.2 SenseVoiceSmall
    print("↓ SenseVoiceSmall")
    _ = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        device="cpu"
    )

    # 2.3 Whisper 全尺寸（openai/whisper 仓库权重）
    print("↓ Whisper 全系列")
    for sz in ("tiny", "tiny.en", "base", "base.en",
               "small", "small.en", "medium", "medium.en",
               "large", "large-v2", "large-v3", "turbo"):
        whisper.load_model(sz, download_root=str(cache / "whisper"))   # 仅下载

# ========== 主入口 ==========
if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "0"   # 强制在线拉取
    download_mt()
    download_asr()
    print("► 全部模型已缓存到 ../models/ ，可离线使用。")
