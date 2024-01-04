import numpy as np
import os.path as osp
import subprocess
import os
import glob
from ..basic import map_async
import numpy as np
import io
from contextlib import redirect_stdout
import ffmpeg
import cv2
import av


class __FFMPEG:

    @classmethod
    def single_video_process(cls, video_path, output_video_path, frame_scale=None, fps=None, quiet=True):
        flags = ""
        if quiet:
            flags += " -hide_banner -loglevel error"

        vf_flags = []
        if frame_scale:
            vf_flags += [f"scale={frame_scale}"]
        if fps:
            vf_flags += [f"fps=fps={fps}"]

        if len(vf_flags) > 0:
            flags += " -vf " + ",".join(vf_flags)

        cmd = f"ffmpeg -i {video_path} {flags} {output_video_path}"
        if not quiet:
            print(cmd)

        subprocess.run(cmd, shell=True)

    @classmethod
    def single_video_to_image(cls, video_path, cur_image_dir, max_frame=None, frame_scale=None, fps=None, quiet=True, overwrite=False):
        # if not overwrite and osp.exists(osp.join(cur_image_dir, ".finish")):
        #     return
        os.makedirs(cur_image_dir, exist_ok=True)
        subprocess.run(f"rm -rf {cur_image_dir}/*", shell=True)
        flag = ""
        if quiet:
            flag += "-hide_banner -loglevel error "
        # filter:v
        filter_v = []
        if frame_scale:
            if isinstance(frame_scale, int):
                frame_scale = f"{frame_scale}:{frame_scale}"
            assert isinstance(frame_scale, str)
            filter_v += [f"scale={frame_scale}"]
        if fps:
            filter_v += [f"fps=fps={fps}"]
        if len(filter_v) > 0:
            flag += f" -filter:v {','.join(filter_v)}"
        if max_frame:
            flag += f"-frames:v {max_frame} "

        cmd = f"ffmpeg -i {video_path} {flag} {osp.join(cur_image_dir, '%04d.png')}"
        if not quiet:
            print(cmd)

        subprocess.run(
            cmd,
            shell=True,
        )
        # subprocess.run(f"cd {cur_image_dir} && touch .finish", shell=True)

    @classmethod
    def multiple_video_to_image_async(cls, video_dir, image_root, **kwargs):
        video_paths = glob.glob(video_dir + "*")

        def func_single(inputs):
            return cls.single_video_to_image(**inputs)

        args = []
        for video_path in video_paths:
            video_id = osp.basename(video_path)[:-4]
            cur_image_dir = osp.join(image_root, video_id)
            args += [dict(video_path=video_path, cur_image_dir=cur_image_dir, **kwargs)]

        map_async(args, func_single, num_process=64)


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
            self.container.streams.video[0].thread_type = 'AUTO'

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


class FFMPEGVideoLoader(OpenCVVideoLoader):

    def get_frames(self):
        h, w = self.hw

        # filter_kwargs = {}
        # if frame_ids is not None:
        #     select_filter = '+'.join([f'eq(n\,{frame_id+1})' for frame_id in frame_ids])
        #     filter_string = f'select={select_filter}'
        #     filter_kwargs = {"vf": filter_string}

        buffer, _ = (
            ffmpeg.input(self.video_path).output(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                # **filter_kwargs,
            ).run(
                capture_stdout=True,
                quiet=True,
            ))
        frames = np.frombuffer(buffer, np.uint8).reshape([-1, h, w, 3])
        return frames


class AVVideoLoader(OpenCVVideoLoader):

    def decode_frames(self):
        pass


import yt_dlp


class YTDLPDownloader:

    @staticmethod
    def _maybe_youtube_id(url):
        if url.startswith("http"):
            return osp.basename(url)
        else:
            return url

    @classmethod
    def download(cls, youtube_id, video_path, video_format="worst[ext=mp4][height>=224]", quiet=True):
        # scale should be conditon like "<=224" or ">=224"
        youtube_id = cls._maybe_youtube_id(youtube_id)
        ydl_opts = {'ignoreerrors': True, 'format': video_format, 'outtmpl': video_path, 'quiet': quiet, 'noprogress': quiet}

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)

        return error_code == 0

    @staticmethod
    def _check_format_dict(format_dict,
                           is_storyboad=False,
                           has_video=True,
                           has_audio=True,
                           h_min=0,
                           h_max=np.inf,
                           w_min=0,
                           w_max=np.inf,
                           ext=None):
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
        ydl_opts = {'ignoreerrors': True, 'logtostderr': True, 'quiet': quiet, 'noprogress': quiet}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            infos = ydl.extract_info(youtube_id, download=False)
            formats = [_ for _ in infos["formats"] if cls._check_format_dict(_, **format_kwargs)]

        formats = sorted(formats, key=lambda _: _["height"])

        return formats

    @staticmethod
    def _download_ydl(youtube_id, buffer, video_format, quiet=True):
        ydl_opts = {'ignoreerrors': True, 'format': video_format, 'outtmpl': "-", 'logtostderr': True, 'quiet': quiet, 'noprogress': quiet}
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


class DecordVideoLoader(OpenCVVideoLoader):

    def __init__(self, video_path):
        super().__init__(video_path)
        self.vr = VideoReader(video_path)

    @property
    def length(self):
        return len(self.vr)

    def get_frames(self, frame_ids=None):
        if frame_ids is None:
            frame_ids = range(self.length)
        frames = self.vr.get_batch(frame_ids)
        return frames.asnumpy()
