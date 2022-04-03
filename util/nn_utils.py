import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
import os


def net_stat(model):
    param_list = []
    for name, param in model.named_parameters():
        param_list += [
            {
                "name": name,
                "param_shape": param.shape,
                "#param": int(torch.tensor(param.shape).prod(0).item()),
            }
        ]
    df = pd.DataFrame(param_list)
    df.sort_values(by="#param", inplace=True, ascending=False)
    tot_param = int(df["#param"].sum())

    return df, tot_param


def set_env_seed(seed_idx=24):
    torch.manual_seed(seed_idx)
    torch.cuda.manual_seed(seed_idx)


def get_mask_from_lens(s_lens, max_len):
    batch_size = s_lens.shape[0]

    mask = torch.arange(0, max_len).cuda().unsqueeze(0).repeat(
        (batch_size, 1)
    ) < s_lens.unsqueeze(1)

    return mask.cuda()


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def save_checkpoint(conf, model: nn.Module, optimizer, scheduler, epoch, is_best=False):
    if is_best:
        to_file = os.path.join(conf.OUTPUT, f"checkpoint.epoch{epoch}.pkl")
    else:
        to_file = os.path.join(conf.OUTPUT, f"checkpoint.bst.pkl")
    
    scheduler_dict = None
    if scheduler:
        scheduler_dict = scheduler.state_dict()

    cache_dict = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler_dict,
                  "epoch": epoch}
    torch.save(cache_dict, to_file)
    print(f"=> checkpoint {to_file} saved")


def load_checkpoint(conf, model, optimizer, scheduler):
    from_file = conf.resume
    cache_dict = torch.load(from_file)
    model = model.load_state_dict(cache_dict["model"])
    optimizer.load_state_dict(cache_dict["optimizer"])
    scheduler.load_state_dict(cache_dict["scheduler"])

    conf.defrost()
    conf.TRAIN.START_EPOCH = int(cache_dict["epoch"]) + 1
    conf.freeze()

    print(f"=> checkpoint {from_file} loaded")
    