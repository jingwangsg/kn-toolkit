import random
import numpy as np
from PIL import Image

def generate_timestamps(num_timestamps: int, duration: float):
    timestamps = []
    for _ in range(num_timestamps):
        st = random.uniform(0, duration)
        ed = st + random.uniform(0, duration - st)
        timestamps.append([st, ed])
    
    return timestamps

def generate_bboxes(
    num_bboxes: int,
    width: int,
    height: int,
):
    bboxes = []
    for _ in range(num_bboxes):
        x = random.randint(0, width)
        y = random.randint(0, height)
        w = random.randint(0, width - x)
        h = random.randint(0, height - y)
        bboxes.append([x, y, w, h])
    return bboxes


def generate_videos(
    num_videos: int,
    max_seq_length: int,
    min_seq_length: int = 1,
    random_length: bool = False,
    **image_kwargs,
):
    length_func = lambda: max_seq_length
    if random_length:
        length_func = lambda: random.randint(min_seq_length, max_seq_length)

    videos = [
        np.stack(
            generate_images(
                length_func(),
                **image_kwargs,
            )
        )
        for _ in range(num_videos)
    ]
    return videos


def generate_images(
    num_images: int,
    max_width: int,
    max_height: int,
    random_size: bool = False,
    min_width: int = 1,
    min_height: int = 1,
):
    if not random_size:
        images = [np.random.randint(0, 255, (3, max_height, max_width)) for _ in range(num_images)]
    else:
        images = [
            np.random.randint(
                0,
                255,
                (3, random.randint(min_height, max_height), random.randint(min_width, max_width)),
            )
            for _ in range(num_images)
        ]
    return images
