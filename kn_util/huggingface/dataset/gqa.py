# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The GQA dataset."""

import json
import os

import datasets

_CITATION = """\
@inproceedings{hudson2019gqa,
  title={Gqa: A new dataset for real-world visual reasoning and compositional question answering},
  author={Hudson, Drew A and Manning, Christopher D},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={6700--6709},
  year={2019}
}
"""

_DESCRIPTION = """\
GQA is a new dataset for real-world visual reasoning and compositional question answering,
seeking to address key shortcomings of previous visual question answering (VQA) datasets.
"""

_URLS = {
    "train": "https://nlp.cs.unc.edu/data/lxmert_data/gqa/train.json",
    "valid": "https://nlp.cs.unc.edu/data/lxmert_data/gqa/valid.json",
    "testdev": "https://nlp.cs.unc.edu/data/lxmert_data/gqa/testdev.json",
    "img": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
}

_IMG_DIR = "images"


class Gqa(datasets.GeneratorBasedBuilder):
    """The GQA dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="gqa", version=datasets.Version("1.0.0"), description="GQA dataset."),
    ]

    def _info(self):
        features = datasets.Features({
            "question": datasets.Value("string"),
            "question_id": datasets.Value("int32"),
            "image_id": datasets.Value("string"),
            "label": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir["train"],
                    "img_dir": os.path.join(dl_dir["img"], _IMG_DIR)
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir["valid"],
                    "img_dir": os.path.join(dl_dir["img"], _IMG_DIR)
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir["testdev"],
                    "img_dir": os.path.join(dl_dir["img"], _IMG_DIR)
                },
            ),
        ]

    def _generate_examples(self, filepath, img_dir):
        """ Yields examples as (key, example) tuples. """
        with open(filepath, encoding="utf-8") as f:
            gqa = json.load(f)
            for id_, d in enumerate(gqa):
                img_id = os.path.join(img_dir, d["img_id"] + ".jpg")
                label = next(iter(d["label"]))
                yield id_, {
                    "question": d["sent"],
                    "question_id": d["question_id"],
                    "image_id": img_id,
                    "label": label,
                }
