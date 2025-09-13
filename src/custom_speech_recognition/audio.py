import io
import wave
import audioop

class AudioData:
    """
    最小化 AudioData，只保留 get_raw_data / get_wav_data，
    供 AudioTranscriber.py 生成 wav 文件。
    """
    def __init__(self, frame_data, sample_rate, sample_width):
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    def get_raw_data(self, convert_rate=None, convert_width=None):
        data = self.frame_data
        if convert_rate is not None and convert_rate != self.sample_rate:
            data, _ = audioop.ratecv(data, self.sample_width, 1,
                                       self.sample_rate, convert_rate, None)
        if convert_width is not None and convert_width != self.sample_width:
            data = audioop.lin2lin(data, self.sample_width, convert_width)
        return data

    def get_wav_data(self, convert_rate=None, convert_width=None):
        raw = self.get_raw_data(convert_rate, convert_width)
        rate = convert_rate or self.sample_rate
        width = convert_width or self.sample_width
        with io.BytesIO() as buf:
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(width)
                wf.setframerate(rate)
                wf.writeframes(raw)
            return buf.getvalue()