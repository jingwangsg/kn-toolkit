from yacs.config import CfgNode as CN
import os
import yaml

_C = CN()

##### data config #####
_C.DATA = CN()
_C.DATA.DATA_DIR = "./data_bin"
_C.DATA.BATCH_SIZE = 512
_C.DATA.CACHE_DIR = "./cache_bin"
_C.DATA.WIN_SIZE = 35  # follow pytorch tutorial
_C.DATA.NUM_WORKER = 1
_C.DATA.PREFETCH_FACTOR = 3

##### optimizer & schedular config #####
_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCH = 10
_C.TRAIN.LR = 3e-4
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.VAL_INTERVAL = 1

##### model config #####
_C.MODEL = CN()
_C.MODEL.SOURCE = ""
_C.MODEL.NAME = "base"
_C.MODEL.WORD_DIM = 300
_C.MODEL.D_MODEL = 512
_C.MODEL.NHEAD = 8
_C.MODEL.NLAYER = 6
_C.MODEL.DROPOUT = 0.3
_C.MODEL.D_FF = 512

##### misc #####
_C.EVAL_MODE = False
_C.USE_CKPT = False
_C.RESUME = ""
_C.OUTPUT = ""
_C.SEED = 42
_C.DEBUG_FLAG = False


def _update_config_from_file(conf: CN, cfg_file):
    conf.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(conf,
                                     os.path.join(os.path.dirname(cfg_file), cfg))
    print(f"=> merge config from {cfg_file}")
    conf.merge_from_file(cfg_file)
    conf.freeze()


def update_config(conf: CN, args):
    _update_config_from_file(conf, args.cfg)
    conf.defrost()

    # if args.data_dir:
    #     conf.DATA.DATA_DIR = args.data_dir
    # if args.eval:
    #     conf.EVAL_MODE = True
    # if args.lr:
    #     conf.TRAIN.LR = args.lr
    # if args.win_size:
    #     conf.DATA.WIN_SIZE = args.win_size
    # if args.batch_size:
    #     conf.DATA.BATCH_SIZE = args.batch_size
    # if args.resume:
    #     conf.RESUME = args.resume
    # if args.use_ckpt:
    #     conf.USE_CKPT = True
    # if args.output:
    #     conf.OUTPUT = args.output
    # if args.dryrun:
    #     conf.DEBUG_FLAG = True
    # if args.source:
    #     conf.MODEL.SOURCE = args.source

    conf.freeze()


def get_config(args):
    conf = _C.clone()
    update_config(conf, args)

    return conf
