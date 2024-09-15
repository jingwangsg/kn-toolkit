
import cv2
import ffmpeg
import numpy as np



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


# ======================== Download Youtube Utility ========================
