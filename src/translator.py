# translator.py
import os
import sys
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer
)


def resource_path(relative_path):
    """PyInstaller 打包时使用 sys._MEIPASS，否则取当前路径"""
    base = getattr(sys, '_MEIPASS',
                   os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, relative_path)


class Translator:
    def __init__(self, mt_backend, mt_model_name):
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

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        try:
            if self.mt_backend == "helsinki":
                inputs = self.tok([text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                gen = self.model.generate(**inputs)
                return self.tok.decode(gen[0], skip_special_tokens=True)
            else:
                self.tok.src_lang = self.src_lang
                inputs = self.tok([text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                gen = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tok.get_lang_id(self.tgt_lang)
                )
                return self.tok.batch_decode(gen, skip_special_tokens=True)[0]
        except Exception as e:
            print("Translation warning:", e)
            return ""
