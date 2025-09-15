什么是模型量化，有什么用，要怎么做？
从FP32计算精度变成更小的精度，加快推理速度。
尝试拿一些小的模型量化，看看效果


# 🎙️ myASR —— 你的实时语音“万能小助手”
![demo](https://user-images.githubusercontent.com/xxx/xxx.svg)  
（点我 2× 速度看 30s 演示，音量预警⚠️）

---

## 🌟 一句话能干嘛？
| 场景 | 秒变现实 |
|---|---|
| 英文网课直播 | 实时双语字幕 + 中文总结，一键复制到笔记 |
| 日文生肉番剧 | 本地离线字幕，再也不等字幕组 |
| 2 小时会议录音 | 10 秒生成「待办清单」+ 关键结论 |
| 低配置老电脑 | 无需显卡，CPU 也能跑！ |

---

## 🚀 3 种打开方式，0 门槛上车

| 方式 | 适合人群 | 耗时 | 教程定位 |
|---|---|---|---|
| 🥇 Windows 一键包 | 纯小白 | 2 min | 见下方「绿色按钮」 |
| 🥈 pip 安装 | Python 玩家 | 5 min | 见「命令行」 |
| 🥉 Docker | 极客 / NAS | 8 min | 见「高级玩法」 |

---

## 🥇 绿色按钮 · Windows 一键包
1. 下载 ⬇️  
   [📦 myASR_v1.2_win.zip](https://drive.google.com/file/d/1N_j7x-Sa8gCPG1tfyJW0UDHBrBzyyHU7/view?usp=drive_link)  
   ![download](https://img.shields.io/badge/size-400MB-00c853?style=flat-square)

2. 解压后双击 ▶️  
   - 中文会议 → 双击 `funasr.bat`  
   - 英文影视 → 双击 `whisper.bat`  
   弹出黑框即开始实时字幕！  
   ![win](docs/win.gif)

3. 字幕窗口置顶，字体大小 `Ctrl + 滚轮` 调；结束按 `Esc` 自动保存 Markdown 总结到桌面。

---

## 🥈 命令行 · 5 分钟自建
&lt;details&gt;
&lt;summary&gt;📖 展开 / 收起&lt;/summary&gt;

### ① 克隆 + 创建虚拟环境
```bash
git clone https://github.com/yourname/myASR.git && cd myASR
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

### ① 克隆 + 创建虚拟环境
```bash
git clone https://github.com/yourname/myASR.git && cd myASR
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### ② 一键装依赖（自动下模型）
```bash
pip install -r requirements.txt
python download_models.py   # 可选：预先拉取模型，断网也能跑
```

### ③ 启动！
```bash
# 例 1：中文会议 → 英文字幕 + 总结
python src/main.py funasr paraformer-speech_68m m2m100 zh-en

# 例 2：英文播客 → 中文字幕
python src/main.py whisper small helsinki en-zh

# 例 3：自己微调过的 large-v3 模型
python src/main.py whisper /path/to/large-v3 m2m100 en-zh
```
参数速查表  
| 参数 | 可选值 | 说明 |
|---|---|---|
| ASR 系列 | `whisper` / `funasr` | 前者多语言强，后者中文更快 |
| ASR 模型 | `tiny` `base` `small` `medium` `large-v3` … | 越大越准，越小越快 |
| 翻译 | `m2m100` `helsinki` | M2M100 质量高，Helsinki 速度狂魔 |
| 语向 | `zh-en` `en-zh` `ja-zh` … | 用 `-` 连接，方向别反 |

</details>

---

## 🐳 高级玩法 · Docker（NAS、服务器）
```bash
docker run -d --gpus all -p 7860:7860 \
  -v ./output:/app/output \
  ghcr.io/yourname/myasr:latest
```
浏览器打开 `http://localhost:7860` 即可获得 Web 字幕控制台，支持扫码手机看字幕。

---

## 🛠️ 模型清单（自动下载，可手动替换）
| 类型 | 名称 | 大小 | 下载地址 |
|---|---|---|---|
| ASR | `whisper-small` | 466 MB | [HuggingFace](https://huggingface.co/openai/whisper-small) |
| ASR | `paraformer-speech-68m` | 272 MB | [ModelScope](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |
| 翻译 | `m2m100-418M-zh-en` | 1.7 GB | [HF](https://huggingface.co/facebook/m2m100_418M) |
| 翻译 | `Helsinki-opus-mt-zh-en` | 298 MB | [HF](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) |

> 全部放在 `models/` 目录即可识别，支持软链省磁盘。

---

## 📊 性能参考（笔记本 i5-1240P）
| 模型组合 | 占用 RAM | 实时倍速 | GPU 要求 |
|---|---|---|---|
| funasr-small + helsinki | 1.2 GB | 1.4× | 无 |
| whisper-small + m2m100 | 3.8 GB | 0.9× | 可选 |
| whisper-large-v3 + m2m100 | 6.5 GB | 0.5× | 推荐 4G 显存 |

---

## 🧩 二次开发 API
启动时加 `--api` 即暴露本地 REST：  
```bash
curl -X POST http://127.0.0.1:8000/asr \
  -F "audio=@test.wav" \
  -F "src_lang=zh" \
  -F "tgt_lang=en"
```
返回 JSON：
```json
{"text": "你好世界", "translation": "Hello world", "summary": "一句简单的问候"}
```

---

## 🙋‍♂️ 常见问题
<details>
<summary>Q1：Mac 能跑吗？</summary>
> 可以！M1/M2 用 `whisper.cpp` 后端，已集成，只需 `pip install -r requirements_mac.txt`
</details>

<details>
<summary>Q2：断网环境下能用吗？</summary>
> 提前运行 `python download_models.py` 把模型拉到本地即可 100% 离线。
</details>

<details>
<summary>Q3：字幕延迟高怎么办？</summary>
> 换 `tiny/base` 模型，或加 `--chunk 1` 把分片调到 1 秒。
</details>

---

## 🤝 贡献 & 鸣谢
[![contributors](https://img.shields.io/github/contributors/yourname/myASR.svg)](https://github.com/yourname/myASR/graphs/contributors)  
感谢 [OpenAI Whisper](https://github.com/openai/whisper)、[FunASR](https://github.com/alibaba-damo-academy/FunASR)、[Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT) 等开源项目。

---

## 📄 许可证
MIT © 2024 yourname  
**Star ⭐ 一下， Issue / PR 随时欢迎！**
```