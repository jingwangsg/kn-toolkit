import numpy as np
import os.path as osp
import subprocess
import os
import glob
from ...utils.multiproc import map_async
import numpy as np
import io
from contextlib import redirect_stdout
import ffmpeg
import cv2
from ...utils.logger import FakeLogger, StorageLogger


class OpenCVVideoLoader:

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    @property
    def hw(self):
        cap = self.cap
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return height, width

    @property
    def fps(self):
        cap = self.cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

    @property
    def length(self):
        cap = self.cap
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return length

    def get_frames(self):
        imgs = []
        while True:
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs += [rgb_frame]
            else:
                break
        return np.stack(imgs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()


class PyAVVideoLoader(OpenCVVideoLoader):

    def __init__(self, video_path, multi_thread=True):
        super().__init__(video_path)
        file_obj = open(video_path, "rb")
        self.container = av.open(file_obj)

        if multi_thread:
            self.container.streams.video[0].thread_type = "AUTO"

    def get_frames(self):
        container = self.container

        total_frames = self.length

        i = 0
        imgs = []
        for frame in container.decode(video=0):
            if i >= total_frames:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1
        imgs = np.stack(imgs)
        return imgs

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.container.close()


class FFMPEGVideoLoader:
    def __init__(self, video_path):
        metadata = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in metadata["streams"] if stream["codec_type"] == "video"), None)

        self._fps = metadata["streams"][0]["r_frame_rate"]
        self._duration = metadata["streams"][0]["duration"]
        self._width = metadata["streams"][0]["width"]
        self._height = metadata["streams"][0]["height"]

    @property
    def hw(self):
        return self._height, self._width

    @property
    def length(self):
        return int(float(self._duration) * float(self._fps))

    @property
    def fps(self):
        # get fps from ffmpeg
        return self.fps

    def get_frames(self):
        h, w = self.hw

        # filter_kwargs = {}
        # if frame_ids is not None:
        #     select_filter = '+'.join([f'eq(n\,{frame_id+1})' for frame_id in frame_ids])
        #     filter_string = f'select={select_filter}'
        #     filter_kwargs = {"vf": filter_string}

        buffer, _ = (
            ffmpeg.input(self.video_path)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                # **filter_kwargs,
            )
            .run(
                capture_stdout=True,
                quiet=True,
            )
        )
        frames = np.frombuffer(buffer, np.uint8).reshape([-1, h, w, 3])
        return frames


class AVVideoLoader(OpenCVVideoLoader):

    def decode_frames(self):
        pass


import yt_dlp
import re


class YTDLPDownloader:

    @staticmethod
    def _maybe_youtube_id(url):
        if url.startswith("http"):
            return re.match(r".*v=([^&]+)", url).group(1)
        else:
            return url

    @classmethod
    def download(
        cls,
        youtube_id,
        video_path,
        video_format="worst[ext=mp4][height>=224]",
        quiet=True,
        logger=None,
    ):
        # scale should be conditon like "<=224" or ">=224"
        youtube_id = cls._maybe_youtube_id(youtube_id)
        ydl_opts = {
            "ignoreerrors": True,
            "format": video_format,
            "outtmpl": video_path,
            "quiet": quiet,
            "noprogress": quiet,
            "logger": logger,
        }
        if quiet and logger is None:
            # completely suppress yt-dlp output
            ydl_opts["logger"] = FakeLogger()

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)

        return error_code

    @staticmethod
    def _check_format_dict(
        format_dict, is_storyboad=False, has_video=True, has_audio=True, h_min=0, h_max=np.inf, w_min=0, w_max=np.inf, ext=None
    ):
        if is_storyboad:
            if format_dict["ext"] == "mhtml":
                return True
            else:
                return False

        if has_video and format_dict["vcodec"] == "none":
            return False

        if has_audio and format_dict["acodec"] == "none":
            return False

        if format_dict["height"] < h_min or format_dict["height"] > h_max:
            return False

        if format_dict["width"] < w_min or format_dict["width"] > w_max:
            return False

        if ext is not None and format_dict["ext"] != ext:
            return False

        return True

    @classmethod
    def extract_formats(cls, youtube_id, quiet=True, **format_kwargs):
        ydl_opts = {"ignoreerrors": True, "logtostderr": True, "quiet": quiet, "noprogress": quiet}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            infos = ydl.extract_info(youtube_id, download=False)
            formats = [_ for _ in infos["formats"] if cls._check_format_dict(_, **format_kwargs)]

        formats = sorted(formats, key=lambda _: _["height"])

        return formats

    @staticmethod
    def _download_ydl(youtube_id, buffer, video_format, quiet=True):
        ydl_opts = {"ignoreerrors": True, "format": video_format, "outtmpl": "-", "logtostderr": True, "quiet": quiet, "noprogress": quiet}
        with redirect_stdout(buffer), yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_id])
        return error_code

    @staticmethod
    def _download_async(youtubu_id, buffer, video_format, quiet):
        pass

    @classmethod
    def load_to_buffer(cls, youtube_id, video_format="worst[ext=mp4][height>=224]", quiet=True):
        # refer to https://github.com/yt-dlp/yt-dlp/issues/3298
        youtube_id = cls._maybe_youtube_id(youtube_id)

        buffer = io.BytesIO()
        error_code = cls._download_ydl(youtube_id=youtube_id, buffer=buffer, video_format=video_format, quiet=quiet)

        buffer.seek(0)

        # write out the buffer for demonstration purposes
        # Path(f"{youtube_id}.mp4").write_bytes(buffer.getvalue())

        return buffer, error_code == 0


import decord
from decord import VideoReader

# decord.bridge.set_bridge('numpy')


class DecordVideoLoader:

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

    def get_frames(self, frame_ids=None):
        if frame_ids is None:
            frame_ids = range(self.length)
        frames = self.vr.get_batch(frame_ids)
        return frames.asnumpy()


"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""
import random
import io
import cv2
import decord
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
from loguru import logger

decord.bridge.set_bridge("torch")


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


def get_frame_indices_by_fps():
    pass


def fill_temporal_param(num_frames=None, fps=None, duration=None):
    if num_frames is None:
        assert fps is not None
        assert duration is not None
        num_frames = int(duration * fps)
    elif fps is None:
        assert num_frames is not None
        assert duration is not None
        fps = num_frames / duration
    elif duration is None:
        assert fps is not None
        assert num_frames is not None
        duration = num_frames / fps

    return num_frames, fps, duration


def get_frame_indices(
    num_frames,
    vlen,
    sample="rand",
    offset_from_start=None,
    max_num_frames=-1,
):

    assert sample in ["rand", "middle", "start"]
    acc_samples = min(num_frames, vlen)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    if sample == "rand":
        try:
            frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
        except:
            frame_indices = np.random.permutation(vlen)[:acc_samples]
            frame_indices.sort()
            frame_indices = list(frame_indices)
    elif sample == "start":
        offset_from_start = offset_from_start if offset_from_start is not None else 0
        frame_indices = [np.minimum(x[0] + offset_from_start, x[1]) for x in ranges]
    elif sample == "middle":
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_indices


def read_frames_av(video_path, num_frames, sample="rand", fix_start=None, max_num_frames=-1):
    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps, max_num_frames=max_num_frames)
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


def read_frames_gif(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    max_num_frames=-1,
):
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(num_frames, vlen, sample=sample, fix_start=fix_start, max_num_frames=max_num_frames)
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
    num_frames=None,
    fps=None,
    sample="rand",
    offset_from_start=None,
    truncate_secs=None,
):

    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps_orig = video_reader.get_avg_fps()
    duration = vlen / float(fps_orig)
    num_frames, fps, duration = fill_temporal_param(num_frames, fps, duration)

    # only truncate if duration is longer than truncate
    if truncate_secs is not None and duration > truncate_secs:
        duration = truncate_secs
        vlen = int(truncate_secs * float(fps))

    frame_indices = get_frame_indices(
        num_frames,
        vlen,
        sample=sample,
        offset_from_start=offset_from_start,
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


VIDEO_READER_FUNCS = {
    "av": read_frames_av,
    "decord": read_frames_decord,
    "gif": read_frames_gif,
}
