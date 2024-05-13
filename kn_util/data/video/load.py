"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""

import random
import io

try:
    import torch
except:
    pass

# import av
import cv2
import decord
import imageio
from decord import VideoReader
import numpy as np
import math
from loguru import logger
from einops import rearrange

from .download import download_youtube_as_bytes

from ...utils.download import MultiThreadDownloaderInMem
from ...utils.system import buffer_keep_open


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(video_stream.duration, video_stream.time_base, video_stream.start_time)
    return float(video_duration)


def fill_temporal_param(duration, num_frames=None, fps=None):
    if num_frames is None:
        assert fps is not None
        assert duration is not None
        num_frames = int(duration * fps)
    elif fps is None:
        assert num_frames is not None
        assert duration is not None
        fps = num_frames / duration
    # duration should always be given
    # elif duration is None:
    #     assert fps is not None
    #     assert num_frames is not None
    #     duration = num_frames / fps

    return num_frames, fps, duration


def get_frame_indices(
    num_frames,
    vlen,
    mode="rand",
    offset_from_start=None,
):

    assert mode in ["rand", "middle", "start", "round"]
    acc_samples = min(num_frames, vlen)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    if mode == "rand":
        try:
            frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
        except:
            frame_indices = np.random.permutation(vlen)[:acc_samples]
            frame_indices.sort()
            frame_indices = list(frame_indices)
    elif mode == "start":
        offset_from_start = offset_from_start if offset_from_start is not None else 0
        frame_indices = [np.minimum(x[0] + offset_from_start, x[1]) for x in ranges]
    elif mode == "middle":
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
    elif mode == "round":
        frame_indices = np.round(np.linspace(0, vlen - 1, num_frames)).astype(int)
    else:
        raise NotImplementedError

    return frame_indices


# def read_frames_av(video_path, num_frames, sample="rand", fix_start=None, max_num_frames=-1):
#     reader = av.open(video_path)
#     frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
#     vlen = len(frames)
#     duration = get_pyav_video_duration(reader)
#     fps = vlen / float(duration)
#     frame_indices = get_frame_indices(num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps, max_num_frames=max_num_frames)
#     frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
#     frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
#     return frames, frame_indices, duration


def read_frames_gif(
    video_path,
    num_frames,
    sample_mode="rand",
    fix_start=None,
    max_num_frames=-1,
):
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(num_frames, vlen, mode=sample_mode, fix_start=fix_start, max_num_frames=max_num_frames)
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, None


def read_frames_decord(
    video_path,
    # frame sampling
    num_frames=None,
    fps=None,
    frame_indices=None,
    sample_mode="round",
    offset_from_start=None,
    truncate_secs=None,
    # ----------------
    # decord kwargs
    size=None,
    max_size=None,
    bridge="native",
    # input format
    is_online_video=False,
    is_youtube_video=False,
    is_bytes=False,
    # output format
    output_format="tchw",
    # return
    return_reader=False,
    return_meta=False,
):
    assert (
        int(is_online_video) + int(is_youtube_video) + int(is_bytes) <= 1
    ), "Only one of is_online_video, is_youtube_video, is_bytes can be True"

    if is_youtube_video:
        download_format = "best"
        download_suffix = ""
        if size is not None:
            download_format = "worst"
            download_suffix += f"[height>={size}][width>={size}]"

        if max_size is not None:
            download_format = "worst"
            download_suffix += f"[height>={max_size}][width>={max_size}]"
        download_format += download_suffix

        video_bytes, error_code = download_youtube_as_bytes(video_path, video_format=download_format)
        if error_code != 0:
            logger.error(f"Error downloading youtube video: {video_path}")
            return None
        video_path = io.BytesIO(video_bytes)

    if is_online_video:
        downloader = MultiThreadDownloaderInMem(verbose=False)
        video_path = io.BytesIO(downloader.download(video_path))
    
    if is_bytes:
        video_path = io.BytesIO(video_path)

    decord.bridge.set_bridge(bridge)

    # calculate frame size according to size and max_size, [-1, -1] by default
    frame_size = [-1, -1]
    if size is not None:
        orig_size = VideoReader(video_path, num_threads=1)[0].shape[:2]
        argmin_size_dim = 0 if orig_size[0] < orig_size[1] else 1
        argmax_size_dim = 1 - argmin_size_dim
        frame_size[argmin_size_dim] = size
        frame_size[argmax_size_dim] = int(size * orig_size[argmax_size_dim] / orig_size[argmin_size_dim])
        if max_size is not None:
            frame_size[argmin_size_dim] = min(frame_size[argmin_size_dim], max_size)

    if is_online_video or is_youtube_video or is_bytes:
        video_path.seek(0)

    video_reader = VideoReader(
        video_path,
        num_threads=1,
        width=frame_size[1],
        height=frame_size[0],
    )

    # setup params
    vlen = len(video_reader)
    fps_orig = video_reader.get_avg_fps()
    duration = vlen / float(fps_orig)
    if num_frames is None and fps is None:
        num_frames = vlen

    num_frames, fps, duration = fill_temporal_param(duration=duration, num_frames=num_frames, fps=fps)

    # only truncate if duration is longer than truncate
    if truncate_secs is not None and duration > truncate_secs:
        duration = truncate_secs
        vlen = int(truncate_secs * float(fps))

    # calculate frame indices if not given
    if frame_indices is None:
        frame_indices = get_frame_indices(
            num_frames,
            vlen,
            mode=sample_mode,
            offset_from_start=offset_from_start,
        )

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

    # to expected output format
    output_format = " ".join(output_format)
    frames = rearrange(frames, f"t h w c -> {output_format}")

    if not return_reader and not return_meta:
        return frames

    ret = (frames,)

    if return_meta:
        meta = {
            "fps": fps,
            "vlen": vlen,
            "duration": duration,
            "frame_indices": frame_indices,
        }
        ret += (meta,)

    if return_reader:
        ret += (video_reader,)

    return ret


# ======================== Read Video Info ========================


class DecordVideoMeta:

    def __init__(self, video_path):
        # super().__init__(video_path)
        self.vr = VideoReader(video_path)

    @property
    def hw(self):
        return self.vr[0].shape[:2]

    @property
    def length(self):
        return len(self.vr)

    @property
    def fps(self):
        return self.vr.get_avg_fps()


VIDEO_READER_FUNCS = {
    # "av": read_frames_av,
    "decord": read_frames_decord,
    "gif": read_frames_gif,
}
