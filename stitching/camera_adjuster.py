from collections import OrderedDict

import cv2 as cv
import numpy as np

from .stitching_error import StitchingError


class CameraAdjuster:
    """https://docs.opencv.org/4.x/d5/d56/classcv_1_1detail_1_1BundleAdjusterBase.html
    CameraAdjuster类用于调整相机参数。"""

    CAMERA_ADJUSTER_CHOICES = OrderedDict()
    CAMERA_ADJUSTER_CHOICES["ray"] = cv.detail_BundleAdjusterRay
    CAMERA_ADJUSTER_CHOICES["reproj"] = cv.detail_BundleAdjusterReproj
    CAMERA_ADJUSTER_CHOICES["affine"] = cv.detail_BundleAdjusterAffinePartial
    CAMERA_ADJUSTER_CHOICES["no"] = cv.detail_NoBundleAdjuster

    DEFAULT_CAMERA_ADJUSTER = list(CAMERA_ADJUSTER_CHOICES.keys())[0]
    DEFAULT_REFINEMENT_MASK = "xxxxx"

    def __init__(
        self,
        adjuster=DEFAULT_CAMERA_ADJUSTER,
        refinement_mask=DEFAULT_REFINEMENT_MASK,
        confidence_threshold=1.0,
    ):
        """初始化CameraAdjuster类。
        参数:
            adjuster (str): 调整器类型，默认为"ray"。
            refinement_mask (str): 精细化掩码，默认为"xxxxx"。
            confidence_threshold (float): 置信度阈值，默认为1.0。"""
        self.adjuster = CameraAdjuster.CAMERA_ADJUSTER_CHOICES[adjuster]()
        self.set_refinement_mask(refinement_mask)
        self.adjuster.setConfThresh(confidence_threshold)

    def set_refinement_mask(self, refinement_mask):
        """设置精细化掩码。
        参数:
            refinement_mask (str): 精细化掩码。"""
        mask_matrix = np.zeros((3, 3), np.uint8)
        if refinement_mask[0] == "x":
            mask_matrix[0, 0] = 1
        if refinement_mask[1] == "x":
            mask_matrix[0, 1] = 1
        if refinement_mask[2] == "x":
            mask_matrix[0, 2] = 1
        if refinement_mask[3] == "x":
            mask_matrix[1, 1] = 1
        if refinement_mask[4] == "x":
            mask_matrix[1, 2] = 1
        self.adjuster.setRefinementMask(mask_matrix)

    def adjust(self, features, pairwise_matches, estimated_cameras):
        """调整相机参数。
        参数:
            features (list): 特征点列表。
            pairwise_matches (list): 成对匹配列表。
            estimated_cameras (list): 估计的相机参数列表。
        返回:
            cameras (list): 调整后的相机参数列表。"""
        b, cameras = self.adjuster.apply(features, pairwise_matches, estimated_cameras)
        if not b:
            raise StitchingError("Camera parameters adjusting failed.")

        return cameras
