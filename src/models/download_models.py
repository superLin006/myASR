# paraformer模型下载
from funasr import AutoModel
model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad",  punc_model="ct-punc", 
                   spk_model="cam++", 
                  )
res = model.generate(input=f"{model.model_path}/example/asr_example.wav", 
                     batch_size_s=300, 
                     hotword='魔搭')
print(res)

"""
#下载whisper系列模型
import whisper
model = whisper.load_model("tiny")
model = whisper.load_model("tiny.en")
model = whisper.load_model("base")
model = whisper.load_model("base.en")
model = whisper.load_model("small")
model = whisper.load_model("small.en")
model = whisper.load_model("medium")
model = whisper.load_model("medium.en")
model = whisper.load_model("large")
model = whisper.load_model("turbo")

"""
"""
下载测试senseVoiceSmall模型
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
"""