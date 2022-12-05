from torchdata.dataloader2.reading_service import MultiProcessingReadingService, DistributedReadingService
from torchdata.dataloader2.dataloader2 import DataLoader2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from .default import build_datapipe_default


def build_datapipe(cfg, split):
    if cfg.data.datapipe == "default":
        return build_datapipe_default(cfg, split=split)


def build_dataloader(cfg, split="train"):
    assert split in ["train", "test", "val"]
    datapipe = build_datapipe(cfg, split=split)

    reading_service = None
    if cfg.flags.ddp:
        reading_service = DistributedReadingService()
    elif cfg.train.num_workers > 0:
        reading_service = MultiProcessingReadingService(num_workers=cfg.train.num_workers,
                                                        prefetch_factor=cfg.train.prefetch_factor)

    return DataLoader2(datapipe, reading_service=reading_service)


def build_dataloader_v1(cfg, split="train"):
    assert split in ["train", "test", "val"]
    datapipe = build_datapipe(cfg, split=split)

    num_workers = cfg.train.num_workers
    prefetch_factor = cfg.train.prefetch_factor

    if cfg.flags.ddp:
        sampler = DistributedSampler(datapipe)
    elif cfg.train.num_workers > 0:
        sampler = SequentialSampler(datapipe)

    return DataLoader(dataset=datapipe, sampler=sampler, num_workers=num_workers, prefetch_factor=prefetch_factor)