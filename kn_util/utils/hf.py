import os, os.path as osp
from diffusers import StableDiffusionXLPipeline
from functools import wraps


def patch_pretrained_from_local(local_dir=osp.expanduser("~/HF/")):
    # ! Deprecated, not working yet
    # ! AttributeError: 'NoneType' object has no attribute 'from_pretrained'
    from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
    from diffusers import DiffusionPipeline

    def _patch(cls):
        assert hasattr(cls, "from_pretrained"), f"{cls} does not have a from_pretrained method"
        orig_fn = cls.from_pretrained

        @wraps(orig_fn)
        def patched_fn(pretrained_model_name_or_path, *args, **kwargs):
            print("Entering patched from_pretrained")
            if not pretrained_model_name_or_path.startswith("/"):
                pretrained_model_name_or_path = osp.join(local_dir, pretrained_model_name_or_path)
            if osp.exists(pretrained_model_name_or_path):
                kwargs["local_files_only"] = True
                return orig_fn(pretrained_model_name_or_path, *args, **kwargs)
            else:
                print(f"{pretrained_model_name_or_path} not found in local directory. Please download it first.")
                exit(0)

        cls.from_pretrained = patched_fn

    _patch(PreTrainedModel)
    _patch(PretrainedConfig)
    _patch(PreTrainedTokenizer)
    _patch(DiffusionPipeline)


# if __name__ == "__main__":
#     from transformers import AutoModel

#     patch_pretrained_from_local()
#     pipeline = StableDiffusionXLPipeline.from_pretrained("CompVis/stable-diffusion-v1-3")

#     # pipeline = StableDiffusionXLPipeline.from_pretrained(
#     #     "/home/aiops/wangjing/HF/CompVis/stable-diffusion-v1-3",
#     #     local_files_only=True,
#     # )

#     import ipdb

#     ipdb.set_trace()
