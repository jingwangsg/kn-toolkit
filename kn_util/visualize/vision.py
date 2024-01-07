from PIL import Image
from dataclasses import dataclass
import cv2
import numpy as np
import copy
from enum import Enum
import os.path as osp
import ffmpeg
from .text import draw_text, draw_text_line
from .utils import color_val


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
    def output_array_to_video_ffmpeg(cls, array, video_path, fps=2, vcodec="libx264", quiet=True):
        n, height, width, channels = array.shape
        process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24',
                                s='{}x{}'.format(width, height)).output(video_path, pix_fmt='yuv420p', vcodec=vcodec,
                                                                        r=fps).overwrite_output().run_async(pipe_stdin=True, quiet=quiet))
        for frame in array:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.close()
        process.wait()

    @classmethod
    def output_array_to_images(cls, array, image_dir):
        # for idx, img in enumerate(array):
        #     img = Image.fromarray(img)
        #     img.save(osp.join(image_dir, f"{idx}.jpg"))
        from kn_util.basic import map_async
        map_async(iterable=enumerate(array), func=lambda x: Image.fromarray(x[1]).save(osp.join(image_dir, f"{x[0]}.jpg")))


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

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)
