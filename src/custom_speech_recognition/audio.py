# audio_minimal.py  —— 够用版，直接替换原 audio.py 即可
import audioop
import io


class AudioData:
    """
    最小化 AudioData，只服务 AudioRecorder.py
    仅保留 get_raw_data()。
    """

    def __init__(self, frame_data: bytes, sample_rate: int, sample_width: int):
        assert sample_rate > 0, "Sample rate must be positive"
        assert 1 <= sample_width <= 4, "Sample width must be 1-4 bytes"
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    # ---------- AudioRecorder 唯一需要的方法 ----------
    def get_raw_data(self, convert_rate=None, convert_width=None) -> bytes:
        """
        返回 PCM 原始字节；可选项:
        - convert_rate: 重采样到此采样率
        - convert_width: 转换到此位宽(字节)
        """
        raw = self.frame_data

        # 8-bit 特殊处理: 先转有符号
        if self.sample_width == 1:
            raw = audioop.bias(raw, 1, -128)

        # 重采样
        if convert_rate and convert_rate != self.sample_rate:
            raw, _ = audioop.ratecv(raw, self.sample_width, 1,
                                     self.sample_rate, convert_rate, None)

        # 位宽转换
        if convert_width and convert_width != self.sample_width:
            raw = audioop.lin2lin(raw, self.sample_width, convert_width)

        # 若目标 8-bit 再转回无符号
        if convert_width == 1:
            raw = audioop.bias(raw, 1, 128)

        return raw
    

    