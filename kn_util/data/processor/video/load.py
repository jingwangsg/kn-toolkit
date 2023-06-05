from PIL import Image

class DecordFrameLoader:

    def __init__(self,
                 stride=1,
                 from_key=None,
                 width=224,
                 height=224,
                 to_array=False,
                 to_images=False) -> None:
        # decord_args: width=224, height=224
        self.from_key = from_key
        self.width = width
        self.height = height
        self.stride = stride
        self.to_images = to_images
        self.to_array = to_array


    def __call__(self, path_or_buffer):
        """
        Args:
            path_or_buffer (str or FileIO): path to video or FileIO of video
        Returns:
            result (dict): result dict with frame array or frame images
            
        """
        from decord import VideoReader
        vr = VideoReader(path_or_buffer, width=self.width, height=self.height)

        indices = list(range(0, len(vr), self.stride))
        arr = vr.get_batch(indices).asnumpy()

        result = dict()

        if self.to_array:
            result["array"] = arr
        if self.to_images:
            result["images"] = [Image.fromarray(a) for a in arr]
        return result