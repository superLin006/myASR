# custom_speech_recognition/__init__.py
# 最小化版本 —— 仅服务 AudioRecorder.py
import io
import math
import audioop
import collections
import threading
import time

# ---------------- 异常 ----------------
class WaitTimeoutError(Exception): pass
class UnknownValueError(Exception): pass



# ---------------- AudioData ----------------
class AudioData(object):
    """
    只保留 get_raw_data() —— AudioRecorder.record_into_queue() 会调用它。
    其余格式转换方法全部删除。
    """
    def __init__(self, frame_data, sample_rate, sample_width):
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    def get_raw_data(self, convert_rate=None, convert_width=None):
        """
        返回 PCM 原始字节；AudioRecorder 需要它。
        这里只处理最简单的 rate/width 转换，够用即可。
        """
        data = self.frame_data
        if convert_rate is not None and convert_rate != self.sample_rate:
            # 简单线性重采样，够用
            data, _ = audioop.ratecv(data, self.sample_width, 1,
                                     self.sample_rate, convert_rate, None)
        if convert_width is not None and convert_width != self.sample_width:
            data = audioop.lin2lin(data, self.sample_width, convert_width)
        return data


# ---------------- AudioSource 基类 ----------------
class AudioSource(object):
    def __enter__(self): raise NotImplementedError
    def __exit__(self, *args): raise NotImplementedError


# ---------------- Microphone ----------------
try:
    import pyaudiowpatch as _pyaudio
except ImportError:
    raise AttributeError("AudioRecorder 依赖 pyaudiowpatch，请先安装")

class Microphone(AudioSource):
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024,
                 speaker=False, channels=1):
        self.speaker = speaker
        self.format = _pyaudio.paInt16
        self.SAMPLE_WIDTH = _pyaudio.get_sample_size(self.format)
        self.CHUNK = chunk_size
        self.channels = channels

        audio = _pyaudio.PyAudio()
        try:
            if sample_rate is None:
                dev = audio.get_device_info_by_index(
                    device_index) if device_index is not None else audio.get_default_input_device_info()
                sample_rate = int(dev["defaultSampleRate"])
        finally:
            audio.terminate()

        self.SAMPLE_RATE = sample_rate
        self.device_index = device_index
        self.audio = None
        self.stream = None

    def __enter__(self):
        assert self.stream is None, "上下文管理器不可重入"
        self.audio = _pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK
            )
        except Exception:
            self.audio.terminate()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()


# ---------------- Recognizer （仅保留录音相关） ----------------
class Recognizer(AudioSource):
    def __init__(self):
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5
        self.pause_threshold = 0.8
        self.phrase_threshold = 0.3
        self.non_speaking_duration = 0.5
        self.operation_timeout = None

    # ---------- 以下四个方法是 AudioRecorder 真正会调用的 ----------
    def record(self, source, duration=None, offset=None):
        assert isinstance(source, AudioSource) and source.stream is not None
        frames = io.BytesIO()
        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        elapsed = offset_time = 0
        offset_reached = False
        while True:
            if offset and not offset_reached:
                offset_time += seconds_per_buffer
                if offset_time > offset:
                    offset_reached = True
            buf = source.stream.read(source.CHUNK)
            if not buf:
                break
            if offset_reached or not offset:
                elapsed += seconds_per_buffer
                if duration and elapsed > duration:
                    break
                frames.write(buf)
        return AudioData(frames.getvalue(), source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def adjust_for_ambient_noise(self, source, duration=1):
        assert isinstance(source, AudioSource) and source.stream is not None
        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        elapsed = 0
        while elapsed < duration:
            elapsed += seconds_per_buffer
            buf = source.stream.read(source.CHUNK)
            energy = audioop.rms(buf, source.SAMPLE_WIDTH)
            if self.dynamic_energy_threshold:
                damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                target = energy * self.dynamic_energy_ratio
                self.energy_threshold = self.energy_threshold * damping + target * (1 - damping)

    def listen(self, source, timeout=None, phrase_time_limit=None):
        assert isinstance(source, AudioSource) and source.stream is not None
        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))

        elapsed = 0
        while True:
            frames = collections.deque()
            # 1. 等待语音开始
            while True:
                elapsed += seconds_per_buffer
                if timeout and elapsed > timeout:
                    raise WaitTimeoutError("listening timed out")
                buf = source.stream.read(source.CHUNK)
                if not buf:
                    break
                frames.append(buf)
                if len(frames) > non_speaking_buffer_count:
                    frames.popleft()
                energy = audioop.rms(buf, source.SAMPLE_WIDTH)
                if energy > self.energy_threshold:
                    break
                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                    target = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target * (1 - damping)
            # 2. 录制直到停顿
            pause_count, phrase_count = 0, 0
            phrase_start = elapsed
            while True:
                elapsed += seconds_per_buffer
                if phrase_time_limit and elapsed - phrase_start > phrase_time_limit:
                    break
                buf = source.stream.read(source.CHUNK)
                if not buf:
                    break
                frames.append(buf)
                phrase_count += 1
                energy = audioop.rms(buf, source.SAMPLE_WIDTH)
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:
                    break
            phrase_count -= pause_count
            if phrase_count >= phrase_buffer_count or not buf:
                break
        # 去掉末尾多余静音
        for _ in range(pause_count - non_speaking_buffer_count):
            frames.pop()
        return AudioData(b"".join(frames), source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def listen_in_background(self, source, callback, phrase_time_limit=None):
        assert isinstance(source, AudioSource)
        running = [True]

        def threaded_listen():
            with source as s:
                while running[0]:
                    try:
                        audio = self.listen(s, timeout=1, phrase_time_limit=phrase_time_limit)
                    except WaitTimeoutError:
                        pass
                    else:
                        if running[0]:
                            callback(self, audio)

        listener_thread = threading.Thread(target=threaded_listen, daemon=True)
        listener_thread.start()

        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()
        return stopper