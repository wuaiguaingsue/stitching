from collections import namedtuple

import cv2 as cv
import numpy as np

from .blender import Blender
from .stitching_error import StitchingError


class Rectangle(namedtuple("Rectangle", "x y width height")):
    __slots__ = ()

    @property
    def area(self):
        return self.width * self.height

    @property
    def corner(self):
        return (self.x, self.y)

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    def times(self, x):
        return Rectangle(*(int(round(i * x)) for i in self))

    def draw_on(self, img, color=(0, 0, 255), size=1):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        start_point = (self.x, self.y)
        end_point = (self.x2 - 1, self.y2 - 1)
        cv.rectangle(img, start_point, end_point, color, size)
        return img


class Cropper:
    """Cropper类用于裁剪图像。"""

    DEFAULT_CROP = True

    def __init__(self, crop=DEFAULT_CROP):
        """初始化Cropper类。
        参数:
            crop (bool): 是否进行裁剪，默认为True。"""
        self.do_crop = crop
        self.overlapping_rectangles = []
        self.cropping_rectangles = []

    def prepare(self, imgs, masks, corners, sizes):
        """准备裁剪操作。
        参数:
            imgs (list): 输入图像列表。
            masks (list): 图像掩码列表。
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。"""
        if self.do_crop:
            mask = self.estimate_panorama_mask(imgs, masks, corners, sizes)
            lir = self.estimate_largest_interior_rectangle(mask)
            corners = self.get_zero_center_corners(corners)
            rectangles = self.get_rectangles(corners, sizes)
            self.overlapping_rectangles = self.get_overlaps(rectangles, lir)
            self.intersection_rectangles = self.get_intersections(
                rectangles, self.overlapping_rectangles
            )

    def crop_images(self, imgs, aspect=1):
        """裁剪图像列表。
        参数:
            imgs (list): 输入图像列表。
            aspect (float): 缩放比例，默认为1。
        返回:
            generator: 裁剪后的图像生成器。"""
        for idx, img in enumerate(imgs):
            yield self.crop_img(img, idx, aspect)

    def crop_img(self, img, idx, aspect=1):
        """裁剪单个图像。
        参数:
            img (numpy.ndarray): 输入图像。
            idx (int): 图像索引。
            aspect (float): 缩放比例，默认为1。
        返回:
            numpy.ndarray: 裁剪后的图像。"""
        if self.do_crop:
            intersection_rect = self.intersection_rectangles[idx]
            scaled_intersection_rect = intersection_rect.times(aspect)
            cropped_img = self.crop_rectangle(img, scaled_intersection_rect)
            return cropped_img
        return img

    def crop_rois(self, corners, sizes, aspect=1):
        """裁剪感兴趣区域（ROI）。
        参数:
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。
            aspect (float): 缩放比例，默认为1。
        返回:
            tuple: 裁剪后的角点列表和尺寸列表。"""
        if self.do_crop:
            scaled_overlaps = [r.times(aspect) for r in self.overlapping_rectangles]
            cropped_corners = [r.corner for r in scaled_overlaps]
            cropped_corners = self.get_zero_center_corners(cropped_corners)
            cropped_sizes = [r.size for r in scaled_overlaps]
            return cropped_corners, cropped_sizes
        return corners, sizes

    @staticmethod
    def estimate_panorama_mask(imgs, masks, corners, sizes):
        """估计全景图掩码。
        参数:
            imgs (list): 输入图像列表。
            masks (list): 图像掩码列表。
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。
        返回:
            numpy.ndarray: 全景图掩码。"""
        _, mask = Blender.create_panorama(imgs, masks, corners, sizes)
        return mask

    def estimate_largest_interior_rectangle(self, mask):
        """估计最大内部矩形。
        参数:
            mask (numpy.ndarray): 全景图掩码。
        返回:
            Rectangle: 最大内部矩形。"""
        # largestinteriorrectangle is only imported if cropping
        # is explicitly desired (needs some time to compile at the first run!)
        import largestinteriorrectangle

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if not hierarchy.shape == (1, 1, 4) or not np.all(hierarchy == -1):
            raise StitchingError(
                "Invalid Contour. Run with --no-crop (using the stitch interface), crop=false (using the stitcher class) or Cropper(False) (using the cropper class)"  # noqa: E501
            )
        contour = contours[0][:, 0, :]

        lir = largestinteriorrectangle.lir(mask > 0, contour)
        lir = Rectangle(*lir)
        return lir

    @staticmethod
    def get_zero_center_corners(corners):
        """获取以零为中心的角点。
        参数:
            corners (list): 图像的角点列表。
        返回:
            list: 以零为中心的角点列表。"""
        min_corner_x = min([corner[0] for corner in corners])
        min_corner_y = min([corner[1] for corner in corners])
        return [(x - min_corner_x, y - min_corner_y) for x, y in corners]

    @staticmethod
    def get_rectangles(corners, sizes):
        """获取矩形列表。
        参数:
            corners (list): 图像的角点列表。
            sizes (list): 图像的尺寸列表。
        返回:
            list: 矩形列表。"""
        rectangles = []
        for corner, size in zip(corners, sizes):
            rectangle = Rectangle(*corner, *size)
            rectangles.append(rectangle)
        return rectangles

    @staticmethod
    def get_overlaps(rectangles, lir):
        """获取重叠矩形列表。
        参数:
            rectangles (list): 矩形列表。
            lir (Rectangle): 最大内部矩形。
        返回:
            list: 重叠矩形列表。"""
        return [Cropper.get_overlap(r, lir) for r in rectangles]

    @staticmethod
    def get_overlap(rectangle1, rectangle2):
        """获取两个矩形的重叠部分。
        参数:
            rectangle1 (Rectangle): 矩形1。
            rectangle2 (Rectangle): 矩形2。
        返回:
            Rectangle: 重叠部分矩形。"""
        x1 = max(rectangle1.x, rectangle2.x)
        y1 = max(rectangle1.y, rectangle2.y)
        x2 = min(rectangle1.x2, rectangle2.x2)
        y2 = min(rectangle1.y2, rectangle2.y2)
        if x2 < x1 or y2 < y1:
            raise StitchingError("Rectangles do not overlap!")
        return Rectangle(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def get_intersections(rectangles, overlapping_rectangles):
        """获取矩形与重叠矩形的交集。
        参数:
            rectangles (list): 矩形列表。
            overlapping_rectangles (list): 重叠矩形列表。
        返回:
            list: 交集矩形列表。"""
        return [
            Cropper.get_intersection(r, overlap_r)
            for r, overlap_r in zip(rectangles, overlapping_rectangles)
        ]

    @staticmethod
    def get_intersection(rectangle, overlapping_rectangle):
        """获取矩形与重叠矩形的交集。
        参数:
            rectangle (Rectangle): 矩形。
            overlapping_rectangle (Rectangle): 重叠矩形。
        返回:
            Rectangle: 交集矩形。"""
        x = abs(overlapping_rectangle.x - rectangle.x)
        y = abs(overlapping_rectangle.y - rectangle.y)
        width = overlapping_rectangle.width
        height = overlapping_rectangle.height
        return Rectangle(x, y, width, height)

    @staticmethod
    def crop_rectangle(img, rectangle):
        """裁剪矩形区域。
        参数:
            img (numpy.ndarray): 输入图像。
            rectangle (Rectangle): 矩形区域。
        返回:
            numpy.ndarray: 裁剪后的图像。"""
        return img[rectangle.y : rectangle.y2, rectangle.x : rectangle.x2]
