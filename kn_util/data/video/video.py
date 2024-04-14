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
from ...utils.error import SuppressStdoutStderr
from ...data.video.video_utils import validate_bytes
from collections import defaultdict
import requests


def sub_to_dict(sub, dedupe=True, single=False) -> list:
    """Convert WebVTT to JSON, optionally removing duplicate lines"""
    import webvtt

    captions = webvtt.read_buffer(io.StringIO(sub))
    dicts = [{"start": c.start, "end": c.end, "lines": c.lines} for c in captions]
    if dedupe:
        dicts = []
        prev_line = None
        for c in captions:
            if any("<c>" in l for l in c.lines):
                continue
            # Collect lines that are not dupes
            not_dupe_lines = []
            for line in c.lines:
                if not line.strip():
                    continue
                if line != prev_line:
                    not_dupe_lines.append(line)
                prev_line = line
            if not_dupe_lines:
                dicts.append({"start": c.start, "end": c.end, "lines": not_dupe_lines})
    if single:
        for d in dicts:
            d["line"] = "\n".join(d.pop("lines"))
    return dicts


class FakeLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class StorageLogger:
    def __init__(self):
        self.storage = defaultdict(list)

    def debug(self, msg):
        self.storage["debug"].append(msg)

    def warning(self, msg):
        self.storage["warning"].append(msg)

    def error(self, msg):
        self.storage["error"].append(msg)


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


# class PyAVVideoLoader(OpenCVVideoLoader):

#     def __init__(self, video_path, multi_thread=True):
#         super().__init__(video_path)
#         file_obj = open(video_path, "rb")
#         self.container = av.open(file_obj)

#         if multi_thread:
#             self.container.streams.video[0].thread_type = "AUTO"

#     def get_frames(self):
#         container = self.container

#         total_frames = self.length

#         i = 0
#         imgs = []
#         for frame in container.decode(video=0):
#             if i >= total_frames:
#                 break
#             imgs.append(frame.to_rgb().to_ndarray())
#             i += 1
#         imgs = np.stack(imgs)
#         return imgs

#     def __exit__(self, exc_type, exc_value, traceback):
#         super().__exit__(exc_type, exc_value, traceback)
#         self.container.close()


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


import yt_dlp
import re


class YTDLPDownloader:
    DEFAULT_YTMETA_ARGS = {
        "writesubtitles": "first",
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": True,
    }
    DEFAULT_VIDEO_FORMAT = "worst[ext=mp4][height>=224]"

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
        video_format=DEFAULT_VIDEO_FORMAT,
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
    def get_yt_meta(
        cls,
        url,
        yt_metadata_args: dict = DEFAULT_YTMETA_ARGS,
    ):
        """Return yt meta dict with meta data and/or subtitles
        yt_metadata_args is a dict of follwing format:
        yt_metadata_args = {
            'writesubtitles': 'first',
            'subtitleslangs': ['en'],
            'writeautomaticsub': True,
            'get_info': True
        }

        writesubtitles:    Whether to write subtitles for each provided language or just the first present
        writeautomaticsub: Write the automatically generated subtitles to a file
        subtitleslangs:    List of languages of the subtitles to download.
        get_info:          Whether to add info (title, description, tags etc) to the output.
        """

        write_subs = yt_metadata_args.get("writesubtitles", None)

        yt_metadata_args["skip_download"] = True
        yt_metadata_args["ignoreerrors"] = True
        yt_metadata_args["quiet"] = True

        info_dict, full_sub_dict = None, None

        with yt_dlp.YoutubeDL(yt_metadata_args) as yt:
            info_dict = yt.extract_info(url, download=False)

        if write_subs:
            full_sub_dict = {}
            for lang in yt_metadata_args["subtitleslangs"]:
                # if info_dict["requested_subtitles"] is None:
                #     import ipdb; ipdb.set_trace()
                if info_dict["requested_subtitles"] is None:
                    break
                if lang not in info_dict["requested_subtitles"]:
                    continue
                sub_url = info_dict["requested_subtitles"][lang]["url"]
                res = requests.get(sub_url, timeout=10)
                sub = io.TextIOWrapper(io.BytesIO(res.content)).read()
                full_sub_dict[lang] = sub_to_dict(sub)

                if write_subs == "first":
                    break

        if yt_metadata_args["get_info"]:
            info_dict.pop("subtitles")
            info_dict.pop("requested_formats")
            info_dict.pop("formats")
            info_dict.pop("thumbnails")
            info_dict.pop("automatic_captions")
        else:
            info_dict = None

        yt_meta_dict = {"info": info_dict, "subtitles": full_sub_dict}

        return yt_meta_dict

    @staticmethod
    def _download_ydl(youtube_id, buffer, video_format=DEFAULT_VIDEO_FORMAT, quiet=True):
        ydl_opts = {"ignoreerrors": True, "format": video_format, "outtmpl": "-", "logtostderr": True, "quiet": quiet, "noprogress": quiet}
        with redirect_stdout(buffer), yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_id])
        return error_code

    @classmethod
    def download_bytes(cls, youtube_id, video_format=DEFAULT_VIDEO_FORMAT, quiet=True, logger=None):
        youtube_id = cls._maybe_youtube_id(youtube_id)

        buffer = io.BytesIO()

        # ! hack workaround to prevent buffer from closing
        # refer to https://github.com/yt-dlp/yt-dlp/issues/3298
        old_buffer_close = buffer.close
        buffer.close = lambda *_: ...

        with redirect_stdout(buffer):

            error_code = cls.download(
                youtube_id=youtube_id,
                video_path="-",
                video_format=video_format,
                quiet=quiet,
                logger=logger,
            )

        m = buffer.getvalue()
        old_buffer_close()

        # write out the buffer for demonstration purposes
        # Path(f"{youtube_id}.mp4").write_bytes(buffer.getvalue())
        return m, error_code


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
