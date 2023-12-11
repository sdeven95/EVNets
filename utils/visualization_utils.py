import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np
from typing import Optional, Union, Tuple, List
from .logger import Logger

import os
import glob
import cv2
import copy

FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2


class Colormap(object):
    """
    Generate colormap for visualizing segmentation masks or bounding boxes.

    This is based on the MATLab code in the PASCAL VOC repository:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def __init__(self, n: Optional[int] = 256, normalized: Optional[bool] = False):
        super(Colormap, self).__init__()
        self.n = n
        self.normalized = normalized

    @staticmethod
    def get_bit_at_idx(val, idx):
        return (val & (1 << idx)) != 0

    def get_color_map(self) -> np.ndarray:

        dtype = "float32" if self.normalized else "uint8"
        color_map = np.zeros((self.n, 3), dtype=dtype)
        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3

            color_map[i] = np.array([r, g, b])
        color_map = color_map / 255 if self.normalized else color_map
        return color_map

    def get_box_color_codes(self) -> List:
        box_codes = []

        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3
            box_codes.append((int(r), int(g), int(b)))
        return box_codes

    def get_color_map_list(self) -> List:
        cmap = self.get_color_map()
        cmap = np.asarray(cmap).flatten()
        return list(cmap)


class Visualization:
    @staticmethod
    def visualize_boxes_xyxy(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Utility function to draw bounding boxes of objects on a given image"""
        boxes = boxes.astype(np.int)

        new_image = copy.deepcopy(image)
        for box_idx in range(boxes.shape[0]):
            coords = boxes[box_idx]
            r, g, b = 255, 0, 0  # red color
            # top -left corner
            start_coord = (coords[0], coords[1])
            # bottom-right corner
            end_coord = (coords[2], coords[3])
            cv2.rectangle(new_image, end_coord, start_coord, (r, g, b), thickness=1)
        return new_image

    @staticmethod
    def draw_bounding_boxes(
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        color_map: Optional = None,
        object_names: Optional[List] = None,
        is_bgr_format: Optional[bool] = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Utility function to draw bounding boxes of objects along with their labels and score on a given image"""
        boxes = boxes.astype(np.int)

        if is_bgr_format:
            # convert from BGR to RGB colorspace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if color_map is None:
            color_map = Colormap().get_box_color_codes()

        for label, score, coords in zip(labels, scores, boxes):
            r, g, b = color_map[label]
            c1 = (coords[0], coords[1])
            c2 = (coords[2], coords[3])

            cv2.rectangle(image, c1, c2, (r, g, b), thickness=RECT_BORDER_THICKNESS)
            if object_names is not None:
                label_text = "{label}: {score:.2f}".format(
                    label=object_names[label], score=score
                )
                t_size = cv2.getTextSize(label_text, FONT_SIZE, 1, TEXT_THICKNESS)[0]
                new_c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(image, c1, new_c2, (r, g, b), -1)
                cv2.putText(
                    image,
                    label_text,
                    (c1[0], c1[1] + t_size[1] + 4),
                    FONT_SIZE,
                    1,
                    LABEL_COLOR,
                    TEXT_THICKNESS,
                )

        if save_path is not None:
            cv2.imwrite(save_path, image)
            Logger.log(f"Detection results stored at: {save_path}")
        return image

    @staticmethod
    def convert_to_cityscape_format(img: Tensor) -> Tensor:
        """Utility to map predicted segmentation labels to cityscapes format"""
        img[img == 19] = 255
        img[img == 18] = 33
        img[img == 17] = 32
        img[img == 16] = 31
        img[img == 15] = 28
        img[img == 14] = 27
        img[img == 13] = 26
        img[img == 12] = 25
        img[img == 11] = 24
        img[img == 10] = 23
        img[img == 9] = 22
        img[img == 8] = 21
        img[img == 7] = 20
        img[img == 6] = 19
        img[img == 5] = 17
        img[img == 4] = 13
        img[img == 3] = 12
        img[img == 2] = 11
        img[img == 1] = 8
        img[img == 0] = 7
        img[img == 255] = 0
        return img

    @staticmethod
    # optional colors include: b-blue, g-green, r-red, c-cyan, m-magenta, y-yellow, k-black, w-white
    # line style: - -- -. : solid dashed dashdot dotted
    # marker: .  , o v ^ < > 1 2 3 4 8 s P p * h H + x X D d | _ generally used : o + x D
    def plot_list(x_list: list,
                  y_list: list,
                  label_list: list,
                  x_label: str,
                  y_label: str,
                  x_limit: Optional[tuple] = (0, 100),
                  y_limit: Optional[tuple] = (0, 100),
                  color_list: Optional[tuple] = ('r', 'g', 'm', 'b', 'c'),
                  marker_list: Optional[tuple] = ('*', 'o', '+', 'x', 'd'),
                  line_style_list: Optional[tuple] = ('solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1, 1, 1))),
                  legend_loc: Optional[str] = "lower right",
                  legend_font_size: Optional[Union[str, int]] = "large",
                  axes_label_font_size: Optional[Union[str, int]] = "large"):

        # plt.rcParams["font.sans-serif"] = ['SimHei']
        # plt.title("Title")
        # names = ['a', 'b', 'c']
        # plt.xticks(names)

        for idx in range(len(x_list)):
            plt.plot(x_list[idx], y_list[idx],
                     color=color_list[idx],
                     marker=marker_list[idx],
                     linestyle=line_style_list[idx],
                     label=label_list[idx])

        plt.xlim(x_limit)
        plt.ylim(y_limit)

        plt.legend(loc=legend_loc, fontsize=legend_font_size)
        plt.grid()
        plt.xlabel(xlabel=x_label, fontsize=axes_label_font_size)
        plt.ylabel(ylabel=y_label, fontsize=axes_label_font_size)

    @staticmethod
    def concat_images(
            image_path: Optional[str] = r'.\cam_results\*.jpg',
            size: Optional[Tuple] = None,
            num_column: Optional[int] = 10,
            padding: Optional[Union[int, tuple]] = None
    ):
        if padding is not None:
            if isinstance(padding, int):
                padding = (padding, padding, padding, padding)
            elif isinstance(padding, tuple) and len(padding) == 2:
                padding = (padding[0], padding[0], padding[1], padding[1])
            elif isinstance(padding, tuple) and len(padding) != 4:
                raise ValueError("padding must be an integer or a tuple with length 2 or 4.")

        cnt = -1
        col_images = []
        # iterate on images
        for file_name in glob.glob(image_path):
            cnt += 1
            # break line
            if cnt % num_column == 0:
                row_imgs = []
            img = cv2.imread(file_name)
            # resize
            if size is not None:
                img = cv2.resize(img, size)
            # make paddings
            if padding:
                # BORDER_CONSTANT BORDER_REFLECT BORDER_DEFAULT BORDER_REPLICATE BORDER_WRAP
                img = cv2.copyMakeBorder(img, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            row_imgs.append(img)
            # combine images in the same line
            if cnt % num_column == num_column - 1:
                col_images.append(cv2.hconcat(row_imgs))

        # combine images by column
        img_rel = cv2.vconcat(col_images)
        # making saved file name
        rel_path = os.path.split(image_path)[0] + "\\" + "combine_cam.jpg"
        # save the combined image
        cv2.imwrite(rel_path, img_rel)

    @staticmethod
    def plot_piecewise_function():
        def plot_soft_relu(lamda, line_style, color, label):

            x = np.linspace(-lamda, 6 - lamda, 100)
            y = x * (x + lamda) / 6

            x1 = np.linspace(6 - lamda,  6, 100)
            y1 = x1

            x2 = np.linspace(-3, -lamda, 100)
            y2 = np.sign(x2) + 1

            plt.plot(x, y, linestyle=line_style, color=color, label=label)
            plt.plot(x1, y1, linestyle=line_style, color=color)
            plt.plot(x2, y2, linestyle=line_style, color=color)

        plt.figure(figsize=(4, 3))
        plot_soft_relu(3, line_style="-.", color="b", label="HardSwish")
        plot_soft_relu(2, line_style="-", color="r", label="SoftReLU-2")
        plot_soft_relu(1, line_style="--", color="g", label="SoftReLU-1")
        plt.legend()
        plt.gcf()
        plt.savefig('soft_relu.svg', format="svg")
        plt.show()

