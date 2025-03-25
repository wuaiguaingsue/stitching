import cv2 as cv
import numpy as np


class Blender:
    """https://docs.opencv.org/4.x/d6/d4a/classcv_1_1detail_1_1Blender.html
    Blender类用于图像拼接中的图像融合。"""

    BLENDER_CHOICES = (
        "multiband",
        "feather",
        "no",
    )
    DEFAULT_BLENDER = "multiband"
    DEFAULT_BLEND_STRENGTH = 5

    def __init__(
        self, blender_type=DEFAULT_BLENDER, blend_strength=DEFAULT_BLEND_STRENGTH
    ):
        """初始化Blender类。
        参数:
            blender_type (str): 融合类型，默认为"multiband"。
            blend_strength (int): 融合强度，默认为5。"""
        self.blender_type = blender_type
        self.blend_strength = blend_strength
        self.blender = None

    def prepare(self, corners, sizes):
        """准备图像融合。
        参数:
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。"""
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100

        if self.blender_type == "no" or blend_width < 1:
            self.blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)

        elif self.blender_type == "multiband":
            self.blender = cv.detail_MultiBandBlender()
            self.blender.setNumBands(int((np.log(blend_width) / np.log(2.0) - 1.0)))

        elif self.blender_type == "feather":
            self.blender = cv.detail_FeatherBlender()
            self.blender.setSharpness(1.0 / blend_width)

        self.blender.prepare(dst_sz)

    def feed(self, img, mask, corner):
        """向Blender提供图像、掩码和角点。
        参数:
            img (numpy.ndarray): 输入图像。
            mask (numpy.ndarray): 图像掩码。
            corner (tuple): 图像的角点。"""
        self.blender.feed(cv.UMat(img.astype(np.int16)), mask, corner)

    def blend(self):
        """执行图像融合。
        返回:
            result (numpy.ndarray): 融合后的图像。
            result_mask (numpy.ndarray): 融合后的图像掩码。"""
        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)
        result = cv.convertScaleAbs(result)
        return result, result_mask

    @classmethod
    def create_panorama(cls, imgs, masks, corners, sizes):
        """创建全景图。
        参数:
            imgs (list): 输入图像列表。
            masks (list): 图像掩码列表。
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。
        返回:
            result (numpy.ndarray): 全景图。
            result_mask (numpy.ndarray): 全景图掩码。"""
        blender = cls("no")
        blender.prepare(corners, sizes)
        for img, mask, corner in zip(imgs, masks, corners):
            blender.feed(img, mask, corner)
        return blender.blend()
