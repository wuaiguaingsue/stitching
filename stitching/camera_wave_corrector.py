from collections import OrderedDict

import cv2 as cv
import numpy as np


class WaveCorrector:
    """https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#ga8faf9588aebd5aeb6f8c649c82beb1fb
    WaveCorrector类用于波形校正。"""

    WAVE_CORRECT_CHOICES = OrderedDict()
    WAVE_CORRECT_CHOICES["horiz"] = cv.detail.WAVE_CORRECT_HORIZ
    WAVE_CORRECT_CHOICES["vert"] = cv.detail.WAVE_CORRECT_VERT
    WAVE_CORRECT_CHOICES["auto"] = cv.detail.WAVE_CORRECT_AUTO
    WAVE_CORRECT_CHOICES["no"] = None

    DEFAULT_WAVE_CORRECTION = list(WAVE_CORRECT_CHOICES.keys())[0]

    def __init__(self, wave_correct_kind=DEFAULT_WAVE_CORRECTION):
        """初始化WaveCorrector类。
        参数:
            wave_correct_kind (str): 波形校正类型，默认为"auto"。"""
        self.wave_correct_kind = WaveCorrector.WAVE_CORRECT_CHOICES[wave_correct_kind]

    def correct(self, cameras):
        """执行波形校正。
        参数:
            cameras (list): 相机参数列表。
        返回:
            cameras (list): 校正后的相机参数列表。"""
        if self.wave_correct_kind is not None:
            rmats = [np.copy(cam.R) for cam in cameras]
            rmats = cv.detail.waveCorrect(rmats, self.wave_correct_kind)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
            return cameras
        return cameras
