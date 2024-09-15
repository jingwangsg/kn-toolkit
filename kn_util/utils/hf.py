import os
import os.path as osp
from typing import Optional, Union

from diffusers import StableDiffusionXLPipeline
from huggingface_hub.utils import validate_hf_hub_args


def patch_pretrained_from_local(local_dir=osp.expanduser("~/HF/")):
    import diffusers
    import transformers
    from transformers import PretrainedConfig

    def _patch(cls, validate_wrapper=False):
        assert hasattr(cls, "from_pretrained"), f"{cls} does not have a from_pretrained method"
        orig_fn = cls.from_pretrained

        def patched_fn(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
        ):
            print("Entering patched from_pretrained")
            if not pretrained_model_name_or_path.startswith("/"):
                pretrained_model_name_or_path = osp.join(local_dir, pretrained_model_name_or_path)
            if osp.exists(pretrained_model_name_or_path):
                return orig_fn(
                    cls,
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    cache_dir=cache_dir,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    force_download=force_download,
                    local_files_only=True,
                    token=token,
                    revision=revision,
                    use_safetensors=use_safetensors,
                    **kwargs,
                )
            else:
                print(f"{pretrained_model_name_or_path} not found in local directory. Please download it first.")
                exit(0)

        # classmethod(*) is necessary here
        if validate_wrapper:
            patched_fn = validate_hf_hub_args(patched_fn)
        patched_fn = classmethod(patched_fn)
        cls.from_pretrained = patched_fn

    _patch(transformers.modeling_utils.PreTrainedModel)
    _patch(transformers.configuration_utils.PretrainedConfig)
    _patch(transformers.tokenization_utils.PreTrainedTokenizerBase)
    _patch(diffusers.pipelines.pipeline_utils.DiffusionPipeline, validate_wrapper=True)


if __name__ == "__main__":

    patch_pretrained_from_local()
    pipeline = StableDiffusionXLPipeline.from_pretrained("CompVis/stable-diffusion-v1-3")

    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     "/home/aiops/wangjing/HF/CompVis/stable-diffusion-v1-3",
    #     local_files_only=True,
    # )

    import ipdb

    ipdb.set_trace()
