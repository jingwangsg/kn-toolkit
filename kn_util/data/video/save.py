import numpy as np
import ffmpeg
import imageio
import torch
import torchvision
from einops import rearrange
import os, os.path as osp
from contextlib import nullcontext
from io import BytesIO
import av
from fractions import Fraction

from ...utils.error import SuppressStdoutStderr


def save_video_ffmpeg(array, video_path, fps=2, vcodec="libx264", input_format="thwc", crf=7, quiet=True):

    if input_format != "thwc":
        input_format = " ".join(input_format)
        array = rearrange(array, f"{input_format} -> t h w c")

    assert (
        array.ndim == 4 and array.shape[-1] == 3
    ), f"array should be 4D (T x H x W x C) and last dim should be 3 (RGB), but got {array.shape}"

    N, H, W, C = array.shape
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}")
        .output(video_path, vcodec=vcodec, r=fps, crf=crf)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=quiet)
    )
    for frame in array:
        process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()


def save_video_imageio(array, video_path, fps=2, input_format="thwc", vcodec="libx264", quality=9, quiet=True):
    # use imageio.mimsave to write video
    if input_format != "thwc":
        input_format = " ".join(input_format)
        array = rearrange(array, f"{input_format} -> t h w c")

    assert array.shape[-1] == 3, "Last dim of array should be 3 (RGB)"

    # suppress all error from opencv and ffmpeg
    maybe_quiet = nullcontext
    if quiet:
        maybe_quiet = SuppressStdoutStderr()

    with maybe_quiet:
        imageio.mimsave(video_path, array, fps=fps, quality=quality, codec=vcodec)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, input_format="btchw"):
    assert input_format[0] == "b", "First dimension of input_format should be batch dimension"

    def _to_tensor(video):
        if not torch.is_tensor(video):
            video = torch.tensor(video)
        return video

    if isinstance(videos, list):
        videos = [_to_tensor(v) for v in videos]
        videos = torch.stack(videos)

    input_format = " ".join(input_format)
    src = " ".join(input_format)
    videos = rearrange(videos, f"{src} -> t b c h w")

    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    outputs = np.stack(outputs)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# copied from https://gist.github.com/willprice/83d91ad1b79d78db977e25c3486504ab
def array_to_video_bytes(video: np.ndarray, fps: float = 24, format="mp4") -> bytes:
    """
    Args:
        video: Video array of shape :math:`(T, H, W, 3)` and range :math:`[0, 255]` with type ``np.uint8``.
        fps: FPS of video
    Returns:
        bytes containing the encoded video as MP4/h264.
    Examples:
        >>> from IPython.display import HTML, display
        >>> from base64 import b64encode
        >>> src = 'data:video/mp4;base64,' + b64encode(video_array_to_mp4_blob(video)).decode('utf8')
        >>> display(HTML(f'<video width="{video.shape[-2]}" height="{video.shape[-3]}" controls><source src="{src}"></video>'))
    """
    f = BytesIO()
    container = av.open(f, mode="w", format=format)

    # https://github.com/PyAV-Org/PyAV/issues/242
    stream = container.add_stream(
        "h264",
        rate=Fraction(fps).limit_denominator(65535),
    )
    stream.width = video.shape[-2]
    stream.height = video.shape[-3]
    stream.pix_fmt = "yuv420p"
    for frame in video:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return f.getvalue()
