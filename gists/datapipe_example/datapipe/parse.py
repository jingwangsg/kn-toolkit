from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torch.utils.data import functional_datapipe
import numpy as np
import os.path as osp
from operator import itemgetter


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


@functional_datapipe("parse_charades_tsgv")
class CharadesParser(IterDataPipe):

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


def build_tsgv_parser(cfg, split):
    dataset_dir = cfg.data.dataset_dir
    if cfg.data.dataset == "tacos":
        annot_path = osp.join(dataset_dir, "annot", split + ".json")
        annot_path_dp = IterableWrapper([annot_path])
        dataset_dp = annot_path_dp.open_files("r", encoding="utf-8").parse_json_files().map(
            itemgetter(1)).parse_tacos_tsgv()
    if cfg.data.dataset == "activitynet":
        annot_path = osp.join(dataset_dir, "annot", split + ".json")
        annot_path_dp = IterableWrapper([annot_path])
        dataset_dp = annot_path_dp.open_files("r", encoding="utf-8").parse_json_files().map(
            itemgetter(1)).parse_activitynet_tsgv()
    if cfg.data.dataset == "charades":
        if split == "val":
            split = "test"
        annot_path = osp.join(dataset_dir, "annot", f"charades_sta_{split}.json")
        annot_path_dp = IterableWrapper([annot_path])
        dataset_dp = annot_path_dp.open_files("r", encoding="utf-8").parse_json_files().map(
            itemgetter(1)).parse_charades_tsgv()

    return dataset_dp