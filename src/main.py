# main.py

import threading
import queue
import time
import subprocess
import argparse
import tkinter as tk
import customtkinter as ctk

from GPTResponder import GPTResponder
from AudioRecorder import DefaultMicRecorder, DefaultSpeakerRecorder
from AudioTranscriber import AudioTranscriber
from TranscriberModels import load_asr_model

def write_in_textbox(textbox, text):
    textbox.delete("1.0", "end")
    textbox.insert("1.0", text)

def update_transcript_UI(transcriber, textbox):
    """
    每 300ms 刷新转写区：原文默认色，译文黄色。
    """
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")
    # 配置译文高亮 tag
    textbox.tag_configure("trans", foreground="yellow")
    entries = transcriber.get_transcript_entries()
    for who, orig, trans, ts in entries:
        textbox.insert("end", f"[{ts}] {who}: {orig}\n")
        if trans.strip():
            textbox.insert("end", f"    → {trans}\n\n", "trans")
    textbox.configure(state="disabled")
    textbox.after(300, update_transcript_UI, transcriber, textbox)

def update_response_UI(responder, textbox,
                       slider_label, slider,
                       freeze_state, send_to_gpt_state):
    if not freeze_state[0]:
        resp = responder.response
        textbox.configure(state="normal")
        write_in_textbox(textbox, resp)
        textbox.configure(state="disabled")
        interval = int(slider.get())
        responder.response_interval = interval
        slider_label.configure(text=f"Update interval: {interval} seconds")
        if send_to_gpt_state[0] and resp and resp != responder.prev_response:
            responder.prev_response = resp
            responder._tts_request(resp)
    textbox.after(300, update_response_UI,
                   responder, textbox,
                   slider_label, slider,
                   freeze_state, send_to_gpt_state)

def clear_context(transcriber, mic_queue, speaker_queue, responder):
    transcriber.clear_transcript_data()
    with mic_queue.mutex:
        mic_queue.queue.clear()
    with speaker_queue.mutex:
        speaker_queue.queue.clear()
    responder.history.clear()

def create_ui_components(root):
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.title("小智")
    root.configure(bg='#252422')
    root.geometry("1000x600")

    font_size = 20

    # 转写区：使用 tk.Text 支持 tag_configure
    transcript_tb = tk.Text(
        root,
        width=40, height=20,
        font=("Arial", font_size),
        fg='#FFFCF2',
        bg='#1f1f1d',
        wrap="word"
    )
    transcript_tb.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")

    # GPT 回复区继续用 CTkTextbox
    response_tb = ctk.CTkTextbox(
        root, width=300, font=("Arial", font_size),
        text_color='#639cdc', wrap="word"
    )
    response_tb.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")

    # 行1按钮
    send_btn   = ctk.CTkButton(root, text="Send to GPT: ON")
    freeze_btn = ctk.CTkButton(root, text="Freeze")
    mic_btn    = ctk.CTkButton(root, text="Mic: ON")
    spkr_btn   = ctk.CTkButton(root, text="Spkr: ON")
    send_btn.grid(row=1, column=0, padx=5, pady=3, sticky="nsew")
    freeze_btn.grid(row=1, column=1, padx=5, pady=3, sticky="nsew")
    mic_btn.grid(row=1, column=2, padx=5, pady=3, sticky="nsew")
    spkr_btn.grid(row=1, column=3, padx=5, pady=3, sticky="nsew")

    # 行2：清空按钮 & 滑条标签
    clear_btn    = ctk.CTkButton(root, text="Clear Transcript")
    slider_label = ctk.CTkLabel(root, text="", font=("Arial",12), text_color="#FFFCF2")
    clear_btn.grid(row=2, column=0, padx=10, pady=3, sticky="nsew")
    slider_label.grid(row=2, column=1, padx=10, pady=3, sticky="nsew")

    # 行3：滑条
    slider = ctk.CTkSlider(root, from_=15, to=30, width=300, height=20, number_of_steps=9)
    slider.set(20)
    slider.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

    # 布局权重
    root.grid_rowconfigure(0, weight=100)
    for r in [1,2,3]:
        root.grid_rowconfigure(r, weight=1)
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=2)
    root.grid_columnconfigure(2, weight=1)
    root.grid_columnconfigure(3, weight=1)

    return (transcript_tb, response_tb,
            slider, slider_label,
            send_btn, freeze_btn,
            mic_btn, spkr_btn,
            clear_btn)

def main():
    # 检查 ffmpeg
    try:
        subprocess.run(["ffmpeg","-version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("请先安装 ffmpeg")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("series", help="ASR 系列 (whisper, funasr, wenet)")
    parser.add_argument("model", help="具体模型 (如 small, paraformer, ...)")
    parser.add_argument("mt_backend",
                        choices=["helsinki","m2m100"],
                        help="翻译后端: helsinki 或 m2m100")
    parser.add_argument("mt_model_name",
                        help="语言对 (如 en-zh, zh-en, en-ja 等)")
    args = parser.parse_args()

    # 加载 ASR 模型
    asr_model = load_asr_model(args.series, args.model)

    # 准备录音队列
    mic_queue     = queue.Queue()
    speaker_queue = queue.Queue()
    audio_queue   = queue.Queue()

    mic_rec = DefaultMicRecorder()
    mic_rec.record_into_queue(mic_queue)
    time.sleep(0.5)
    spk_rec = DefaultSpeakerRecorder()
    spk_rec.record_into_queue(speaker_queue)

    # 合并音频数据到 audio_queue
    mic_enabled     = [True]
    speaker_enabled = [True]
    def audio_merger():
        while True:
            try:
                who, data, ts = speaker_queue.get(timeout=0.05)
                if speaker_enabled[0]:
                    audio_queue.put((who, data, ts))
            except queue.Empty:
                pass
            try:
                who, data, ts = mic_queue.get(timeout=0.05)
                if mic_enabled[0]:
                    audio_queue.put((who, data, ts))
            except queue.Empty:
                pass

    threading.Thread(target=audio_merger, daemon=True).start()

    # 初始化转写器
    transcriber = AudioTranscriber(
        mic_rec.source,
        spk_rec.source,
        asr_model,
        args.mt_backend,
        args.mt_model_name
    )
    threading.Thread(
        target=transcriber.transcribe_audio_queue,
        args=(audio_queue,),
        daemon=True
    ).start()

    # 初始化 GPTResponder
    responder = GPTResponder()
    send_to_gpt_state = [True]
    threading.Thread(
        target=responder.respond_to_transcriber,
        args=(transcriber, send_to_gpt_state),
        daemon=True
    ).start()

    print("READY")

    # 构建 UI 并绑定事件
    root = ctk.CTk()
    (trans_tb, resp_tb,
     slider, slider_label,
     send_btn, freeze_btn,
     mic_btn, spkr_btn,
     clear_btn) = create_ui_components(root)

    clear_btn.configure(command=lambda:
        clear_context(transcriber, mic_queue, speaker_queue, responder)
    )

    # Freeze 按钮
    freeze_state = [False]
    def toggle_freeze():
        freeze_state[0] = not freeze_state[0]
        freeze_btn.configure(text="Unfreeze" if freeze_state[0] else "Freeze")
    freeze_btn.configure(command=toggle_freeze)

    # Send to GPT 按钮
    def toggle_gpt():
        send_to_gpt_state[0] = not send_to_gpt_state[0]
        send_btn.configure(text=f"Send to GPT: {'ON' if send_to_gpt_state[0] else 'OFF'}")
    send_btn.configure(command=toggle_gpt)

    # Mic 开关
    def toggle_mic():
        mic_enabled[0] = not mic_enabled[0]
        mic_btn.configure(text="Mic: ON" if mic_enabled[0] else "Mic: OFF")
    mic_btn.configure(command=toggle_mic)

    # Speaker 开关
    def toggle_speaker():
        speaker_enabled[0] = not speaker_enabled[0]
        spkr_btn.configure(text="Spkr: ON" if speaker_enabled[0] else "Spkr: OFF")
    spkr_btn.configure(command=toggle_speaker)

    # 初始化滑条标签
    slider_label.configure(text=f"Update interval: {int(slider.get())} seconds")

    # 启动 UI 更新循环
    update_transcript_UI(transcriber, trans_tb)
    update_response_UI(responder, resp_tb,
                       slider_label, slider,
                       freeze_state, send_to_gpt_state)

    root.mainloop()

if __name__ == "__main__":
    main()
