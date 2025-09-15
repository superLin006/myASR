"""
最小化 custom_speech_recognition
仅保留录音相关功能，删除所有 ASR 引擎。
"""
import io
import collections
import audioop
import math
import time
from .audio import AudioData
from .exceptions import WaitTimeoutError

__all__ = ['Microphone', 'AudioFile', 'Recognizer', 'AudioData', 'WaitTimeoutError']

# ====================== AudioSource 骨架 ======================
class AudioSource:
    def __enter__(self): raise NotImplementedError
    def __exit__(self, *args): raise NotImplementedError

# ====================== Microphone ======================
class Microphone(AudioSource):
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024,
                 speaker=False, channels=1):
        self.speaker = speaker
        self.pyaudio_module = self._get_pyaudio()
        audio = self.pyaudio_module.PyAudio()
        try:
            if device_index is None:
                info = audio.get_default_input_device_info()
            else:
                info = audio.get_device_info_by_index(device_index)
            if sample_rate is None:
                sample_rate = int(info['defaultSampleRate'])
        finally:
            audio.terminate()

        self.device_index = device_index
        self.format = self.pyaudio_module.paInt16
        self.SAMPLE_WIDTH = self.pyaudio_module.get_sample_size(self.format)
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = chunk_size
        self.channels = channels
        self.audio = None
        self.stream = None

    @staticmethod
    def _get_pyaudio():
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AttributeError("需要 pyaudiowpatch")
        return pyaudio

    def __enter__(self):
        assert self.stream is None
        self.audio = self.pyaudio_module.PyAudio()
        try:
            self.stream = self.audio.open(
                input_device_index=self.device_index,
                channels=self.channels,
                format=self.format,
                rate=self.SAMPLE_RATE,
                frames_per_buffer=self.CHUNK,
                input=True
            )
        except Exception:
            self.audio.terminate()
            raise
        return self

    def __exit__(self, *args):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()

# ====================== AudioFile（可选） ======================
class AudioFile(AudioSource):
    def __init__(self, filename_or_fileobject):
        self.filename_or_fileobject = filename_or_fileobject
        self.stream = None
        self.DURATION = None
        self.audio_reader = None
        self.SAMPLE_RATE = None
        self.SAMPLE_WIDTH = None
        self.CHUNK = 4096

    def __enter__(self):
        import wave, aifc, audioop
        try:
            self.audio_reader = wave.open(self.filename_or_fileobject, 'rb')
            little_endian = True
        except (wave.Error, EOFError):
            try:
                self.audio_reader = aifc.open(self.filename_or_fileobject, 'rb')
                little_endian = False
            except (aifc.Error, EOFError):
                raise ValueError("仅支持 WAV/AIFF")
        assert 1 <= self.audio_reader.getnchannels() <= 2
        self.SAMPLE_WIDTH = self.audio_reader.getsampwidth()
        self.SAMPLE_RATE = self.audio_reader.getframerate()
        self.stream = AudioFileStream(self.audio_reader, little_endian)
        self.DURATION = self.audio_reader.getnframes() / float(self.SAMPLE_RATE)
        return self

    def __exit__(self, *args):
        if hasattr(self.filename_or_fileobject, 'read'):
            self.audio_reader.close()
        self.stream = None
        self.DURATION = None

class AudioFileStream:
    def __init__(self, reader, little_endian):
        self.reader = reader
        self.little_endian = little_endian
    def read(self, size=-1):
        frames = self.reader.readframes(size)
        if not self.little_endian and self.reader.getsampwidth() != 1:
            frames = audioop.byteswap(frames, self.reader.getsampwidth())
        if self.reader.getnchannels() != 1:
            frames = audioop.tomono(frames, self.reader.getsampwidth(), 1, 1)
        return frames

# ====================== Recognizer（仅录音） ======================
class Recognizer:
    def __init__(self):
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5
        self.pause_threshold = 0.8
        self.non_speaking_duration = 0.5
        self.phrase_threshold = 0.3
        self.operation_timeout = None

    def adjust_for_ambient_noise(self, source, duration=1):
        assert isinstance(source, AudioSource) and source.stream is not None
        secs = source.CHUNK / float(source.SAMPLE_RATE)
        elapsed = 0
        while elapsed < duration:
            elapsed += secs
            buf = source.stream.read(source.CHUNK)
            energy = audioop.rms(buf, source.SAMPLE_WIDTH)
            damp = self.dynamic_energy_adjustment_damping ** secs
            target = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damp + target * (1 - damp)

    def listen(self, source, timeout=None, phrase_time_limit=None):
        assert isinstance(source, AudioSource) and source.stream is not None
        secs = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / secs))
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / secs))
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / secs))

        elapsed = 0
        frames = collections.deque()
        while True:
            elapsed += secs
            if timeout and elapsed > timeout:
                raise WaitTimeoutError("listening timed out")
            buf = source.stream.read(source.CHUNK)
            if len(buf) == 0: break
            frames.append(buf)
            if len(frames) > non_speaking_buffer_count:
                frames.popleft()
            energy = audioop.rms(buf, source.SAMPLE_WIDTH)
            if energy > self.energy_threshold:
                break
            if self.dynamic_energy_threshold:
                damp = self.dynamic_energy_adjustment_damping ** secs
                target = energy * self.dynamic_energy_ratio
                self.energy_threshold = self.energy_threshold * damp + target * (1 - damp)

        pause_count, phrase_count = 0, 0
        phrase_start = elapsed
        while True:
            elapsed += secs
            if phrase_time_limit and elapsed - phrase_start > phrase_time_limit:
                break
            buf = source.stream.read(source.CHUNK)
            if len(buf) == 0: break
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
        if phrase_count < phrase_buffer_count and len(buf) != 0:
            return self.listen(source, timeout, phrase_time_limit)

        for _ in range(pause_count - non_speaking_buffer_count):
            frames.pop()
        frame_data = b''.join(frames)
        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def listen_in_background(self, source, callback, phrase_time_limit=None):
        assert isinstance(source, AudioSource)
        running = [True]
        def threaded_listen():
            with source as s:
                while running[0]:
                    try:
                        audio = self.listen(s, 1, phrase_time_limit)
                    except WaitTimeoutError:
                        pass
                    else:
                        if running[0]:
                            callback(self, audio)
        import threading
        listener_thread = threading.Thread(target=threaded_listen, daemon=True)
        listener_thread.start()
        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()
        return stopper