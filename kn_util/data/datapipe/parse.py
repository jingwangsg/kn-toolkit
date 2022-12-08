from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import functional_datapipe
import numpy as np

@functional_datapipe("parse_tacos_tsgv")
class TACoSParser(IterDataPipe):

    def __init__(self, src_pipeline) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline

    def __iter__(self):
        for json_data in self.src_pipeline:
            for video_id, annot in json_data.items():
                video_id = video_id[:-4]
                for idx, (sentence, timestamp) in enumerate(zip(annot["sentences"], annot["timestamps"])):
                    gt = np.array(timestamp) / annot["num_frames"]
                    text_id = f"{video_id}_{idx}"

                    yield dict(video_id=video_id, text_id=text_id, gt=gt, text=sentence)

@functional_datapipe("parse_activitynet_tsgv")
class ActivityNetParser(IterDataPipe):
    def __init__(self, src_pipeline) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
    
    def __iter__(self):
        for json_data in self.src_pipeline:
            for video_id, annot in json_data.items():
                for idx, (sentence, timestamp) in enumerate(zip(annot["sentences"], annot["timestamps"])):
                    gt = np.array(timestamp) / annot["duration"]
                    text_id = f"{video_id}_idx"

                    yield dict(video_id=video_id, text_id=text_id, gt=gt, text=sentence)
    