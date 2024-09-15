from enum import Enum

import cv2
import numpy as np


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    yellow = (255, 255, 0)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    dark_green = (1, 50, 32)


def color_val(color) -> tuple:
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value  # type: ignore
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def draw_opaque_mask(img, start_point, end_point, alpha=0.5):
    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes = np.zeros_like(img, np.uint8)

    # Draw shapes
    cv2.rectangle(shapes, start_point, end_point, (255, 255, 255), cv2.FILLED)
    out = img.copy()
    mask = shapes.astype(bool)

    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
    return out
