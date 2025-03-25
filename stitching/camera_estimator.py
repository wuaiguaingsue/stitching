from collections import OrderedDict

import cv2 as cv
import numpy as np

from .stitching_error import StitchingError


class CameraEstimator:
    """https://docs.opencv.org/4.x/df/d15/classcv_1_1detail_1_1Estimator.html
    CameraEstimator类用于估计相机参数。"""

    CAMERA_ESTIMATOR_CHOICES = OrderedDict()
    CAMERA_ESTIMATOR_CHOICES["homography"] = cv.detail_HomographyBasedEstimator
    CAMERA_ESTIMATOR_CHOICES["affine"] = cv.detail_AffineBasedEstimator

    DEFAULT_CAMERA_ESTIMATOR = list(CAMERA_ESTIMATOR_CHOICES.keys())[0]

    def __init__(self, estimator=DEFAULT_CAMERA_ESTIMATOR, **kwargs):
        """初始化CameraEstimator类。
        参数:
            estimator (str): 估计器类型，默认为"homography"。"""
        self.estimator = CameraEstimator.CAMERA_ESTIMATOR_CHOICES[estimator](**kwargs)

    def estimate(self, features, pairwise_matches):
        """估计相机参数。
        参数:
            features (list): 特征点列表。
            pairwise_matches (list): 成对匹配列表。
        返回:
            cameras (list): 估计的相机参数列表。"""
        b, cameras = self.estimator.apply(features, pairwise_matches, None)
        if not b:
            raise StitchingError("Homography estimation failed.")
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras
