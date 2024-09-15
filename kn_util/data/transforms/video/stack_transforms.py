import numpy as np
import PIL



class ToStackedArray(object):
    """Converts a list of T (H x W x C) numpy.ndarrays to a numpy array of shape (T x H x W x C)"""

    def __call__(self, clip):
        """
        Args:
            clip (list of numpy.ndarray or PIL.Image.Image): clip
            (list of images) to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == 3, "got {} channels instead of 3".format(ch)
        elif isinstance(clip[0], PIL.Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = []

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, PIL.Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            np_clip.append(img)
        tensor_clip = np.stack(np_clip, axis=0)

        return tensor_clip
