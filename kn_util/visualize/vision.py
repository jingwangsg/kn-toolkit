import os, os.path as osp
from PIL import Image
from dataclasses import dataclass
import cv2
import numpy as np
import copy
from enum import Enum
import os.path as osp
import ffmpeg
import numpy as np
from tempfile import NamedTemporaryFile
import io
import torch
from torchvision.utils import make_grid

from .text import draw_text, draw_text_line
from .utils import color_val
from ..data.video import read_frames_decord


def plt_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def draw_image(x, height=12):
    def _draw_from_file(filename):
        cmd = f"termvisage {filename} --query-timeout 1 -S iterm2 --force-style -H left --height {height}"
        os.system(cmd)

    if isinstance(x, str):
        _draw_from_file(x)
    elif isinstance(x, Image.Image):
        with NamedTemporaryFile(suffix=".jpg") as f:
            x.save(f.name)
            _draw_from_file(f.name)
    else:
        raise NotImplementedError


def merge_by_fid(bboxs_list, texts=None):
    """
    Return:
        ret: dict(fid: dict(xyxy: text))
    """
    ret = dict()
    for idx, bbox in enumerate(bboxs_list):
        xyxy = tuple(bbox["bbox"])
        fid = bbox["fid"]
        fid_dict = ret.get(fid, dict())
        if texts is not None:
            cur_text = texts[idx]
            if xyxy in fid_dict:
                fid_dict[xyxy] = f"{fid_dict[xyxy]}\n{cur_text}"
            else:
                fid_dict[xyxy] = cur_text
        else:
            fid_dict[xyxy] = None

        ret[fid] = fid_dict
    return ret


class VideoVisualizer:

    @classmethod
    def draw_bbox_on_video(cls, video, bboxs_list, texts=None, **bbox_args):
        """
        Args:
            video: list of images
            bboxs_list: list of dict(fid, bbox)
            texts: list of dict(fid, text)
        """
        video = copy.deepcopy(video)
        content_by_frame = merge_by_fid(bboxs_list, texts=texts)

        for fid, content in content_by_frame.items():
            cur_bboxes = []
            cur_texts = []
            for bbox, text in content.items():
                cur_bboxes.append(bbox)
                cur_texts.append(text)

            image = Visualizer.draw_bboxs_on_image(video[fid], bboxes=cur_bboxes, texts=cur_texts, **bbox_args)
            video[fid] = image

        return video

    @classmethod
    def output_array_to_video_cv2(cls, array, video_path, fps=2):
        #! Deprecated
        codec = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter(video_path, codec, fps, (array.shape[2], array.shape[1]), False)
        for img in array:
            video.write(img)
        video.release()

    @classmethod
    def output_array_to_images(cls, array, image_dir):
        # for idx, img in enumerate(array):
        #     img = Image.fromarray(img)
        #     img.save(osp.join(image_dir, f"{idx}.jpg"))
        from kn_util.utils.multiproc import map_async

        map_async(
            iterable=enumerate(array),
            func=lambda x: Image.fromarray(x[1]).save(osp.join(image_dir, f"{x[0]}.jpg")),
            verbose=False,
        )


class Visualizer:

    @classmethod
    def draw_bboxs_on_image(
        cls,
        image,
        bboxes,
        texts=None,
        thickness: int = 1,
        bbox_color: int = "blue",
        text_color: int = "blue",
        font_scale: float = 0.5,
        bg_color: str = "opaque",
    ):
        """from mmcv.visualization.image"""
        image = copy.deepcopy(image)

        bbox_color = color_val(bbox_color)
        text_color = color_val(text_color)

        def draw_single_bbox(image, bbox, text=None):
            image = copy.deepcopy(image)
            if not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox)

            assert len(bbox) == 4 and isinstance(bbox, np.ndarray)

            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            # draw bbox
            cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=thickness)
            # annotate
            if text is not None:
                image = draw_text_line(
                    img=image,
                    point=(bbox_int[0], bbox_int[1] - 2),
                    text_line=text,
                    font_scale=font_scale,
                    bg_color=bg_color,
                    text_color=text_color,
                )
            return image

        if texts is None:
            texts = [None] * len(bboxes)

        for bbox, text in zip(bboxes, texts):
            image = draw_single_bbox(image, bbox, text)

        return image

    def draw_arrow_on_image(
        self,
        image,
        points,
        arrow_color: str = "green",
        thickness: int = 1,
        arrow_length: int = 10,
        arrow_thickness: int = 1,
        texts=None,
    ):
        image = copy.deepcopy(image)
        arrow_color = color_val(arrow_color)

        def draw_arrow_on_image_single(start_point, end_point):
            start_point = tuple(start_point)
            end_point = tuple(end_point)
            cv2.arrowedLine(
                image,
                start_point,
                end_point,
                arrow_color,
                thickness=thickness,
                tipLength=arrow_length,
                line_type=cv2.LINE_AA,
                shift=0,
            )

        for start_point, end_point in points:
            draw_arrow_on_image_single(start_point, end_point)
        return image


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError("Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored.")

    assert img_array.shape[1] == 3, "Expecting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)


def _vis_image_tensors(image_tensors):

    with NamedTemporaryFile(suffix=".jpg") as f:
        Image.fromarray(image_grid.permute(1, 2, 0).numpy()).save(f.name)
        os.system(f"termvisage {f.name} -H left -S iterm2 --height 30 --force-style")


def vis_images(image_files):
    from torchvision.utils import make_grid

    if len(image_files) == 1:
        image = image_files[0]
        os.system(f"termvisage --query-timeout 1 {image} -H left --height 12")

    else:
        images = [Image.open(image).convert("RGB") for image in image_files]
        image_tensors = [torch.from_numpy(np.array(image)) for image in images]
        image_tensors = torch.stack(image_tensors)
        _vis_image_tensors(image_tensors)

def read_video_as_grid(video_file, num_frames=8):
    """
    @return: numpy array (c, h, w)
    """
    frames = read_frames_decord(video_file, num_frames=num_frames, output_format="tchw")
    frames = torch.from_numpy(frames)
    frames_grid = make_grid(frames)
    return frames_grid.numpy()

def vis_video(video_file, num_frames=8):
    frames_grid = read_video_as_grid(video_file, num_frames=num_frames)

    with NamedTemporaryFile(suffix=".jpg") as f:
        Image.fromarray(frames_grid.permute(1, 2, 0).numpy()).save(f.name)
        vis_images([f.name])