from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import psutil

# 函数：获取当前内存使用情况（以 GB 为单位）
def get_memory_usage():
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3)

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 打印初始内存使用情况
print(f"Initial Memory Usage: {get_memory_usage():.2f} GB")

# 指定模型路径
model_path = "../models/m2m100_418M"

# 加载分词器和模型
tokenizer = M2M100Tokenizer.from_pretrained(model_path)
model = M2M100ForConditionalGeneration.from_pretrained(model_path)

# 打印内存使用情况
print(f"After loading model, Memory Usage: {get_memory_usage():.2f} GB")

# 将模型移动到 GPU 并转换为半精度（可选）
model.to(device)
model.half()
print(f"After moving to GPU and half precision, Memory Usage: {get_memory_usage():.2f} GB")

# 设置源语言和目标语言
tokenizer.src_lang = "en"
target_language = "ja"

# 输入文本
text = "Strolling along the forest path, you can feel the fresh air and the sound of birdsong, accompanied by the leaves swaying gently in the wind, and the breeze brushing your face, bringing bursts of floral fragrance, which makes you feel relaxed and happy. In the sound of flowing mountain streams, you seem to hear the whispers of nature. It is a beauty beyond language that can allow people to escape from the hustle and bustle of the city and regain their inner peace. Whether sitting quietly under the mottled walls of an ancient temple or wandering through the busy streets of a city, you can experience the colorful life and the impermanence of the world in different scenes. Life is like a river, flowing endlessly in an ever-changing trajectory, and we are travelers floating on it. On the road to pursuing ideals and happiness, our steps may be slow or fast, light or heavy, but we are always full of hope, because every fall makes us stronger, and every time we look up, we see the light. As long as there is love in our hearts, we can shine our own light no matter where we are."

# 编码输入文本并移动到 GPU
encoded = tokenizer(text, return_tensors="pt").to(device)

# 打印内存使用情况
print(f"Before generation, Memory Usage: {get_memory_usage():.2f} GB")

# 生成翻译
with torch.no_grad():
    generated_tokens = model.generate(
        **encoded, 
        forced_bos_token_id=tokenizer.get_lang_id(target_language)
    )

# 将生成的翻译移动到 CPU 并解码
translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(translated_text)

# 打印最终内存使用情况
print(f"After generation, Memory Usage: {get_memory_usage():.2f} GB")
