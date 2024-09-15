from types import SimpleNamespace

import numpy as np
import torch
import torchtext
from kn_util.basic import global_get, global_set
from loguru import logger
from torchtext.data.utils import get_tokenizer

from kn_util.data.vocab import delete_noisy_char


def extend_vocab(pretrained_vocab, token, vector):
    pretrained_vocab.itos.extend([token])
    pretrained_vocab.stoi[token] = pretrained_vocab.vectors.shape[0]
    pretrained_vocab.vectors = torch.cat([pretrained_vocab.vectors, vector], dim=0)


class GloveTokenizeProcessor:

    def __init__(
        self,
        glove="glove.6B.300d",
        vocab_file=None,
        global_vocab_key=None,
        tokenizer="split",
        cache_dir=None,
        special_tokens=["<pad>", "<unk>"],
        to_words=False,
        to_indices=False,
        to_embeddings=False,
    ) -> None:
        """
        At least one of to_words, to_indices, to_embeddings must be True.
        Args:
            glove (str, optional): Glove name. Defaults to "glove.6B.300d".
            vocab_file (str, optional): Path to vocab file. Defaults to None.
            global_vocab_key (str, optional): Global key to upload vocab. Defaults to None.
            tokenizer (str, optional): Tokenizer name. Defaults to "split".
            cache_dir (str, optional): Path for storing downloaded glove file. Defaults to None.
            to_words (bool, optional): Whether to return words. Defaults to False.
            to_indices (bool, optional): Whether to return indices. Defaults to False.
            to_embeddings (bool, optional): Whether to return embeddings. Defaults to False.
        """
        super().__init__()
        assert to_words or to_indices or to_embeddings

        self.vocab_file = vocab_file
        self.glove = glove
        self.global_vocab_key = global_vocab_key
        self.to_words = to_words
        self.to_indices = to_indices
        self.to_embeddings = to_embeddings
        self.cache_dir = cache_dir
        self.special_tokens = special_tokens

        if tokenizer == "split":
            self.tokenizer = lambda s: delete_noisy_char(s).lower().split()
        else:
            self.tokenizer = get_tokenizer(tokenizer)

        self._load_vocab()

    def _load_vocab(self):
        global_vocab = global_get(self.global_vocab_key)
        if global_vocab:
            # vocab has been built and uploaded
            itos = global_vocab.itos
            vectors = global_vocab.vectors
        else:
            # build vocab
            pretrained_vocab = torchtext.vocab.pretrained_aliases[self.glove](cache=self.cache_dir)
            if self.vocab_file:
                # use external vocab file as vocab list
                with open(self.vocab_file, "r") as f:
                    lines = f.readlines()
                zero_vector = torch.zeros(1, pretrained_vocab.vectors.shape[-1])
                for token in self.special_tokens:
                    extend_vocab(pretrained_vocab, token, zero_vector)
                itos = ["<pad>", "<unk>"] + [w.strip() for w in lines]
                extracted_indicies = [pretrained_vocab.stoi.get(w, pretrained_vocab.stoi["<unk>"]) for w in itos]
                vectors = pretrained_vocab.vectors[extracted_indicies]
            else:
                # use original glove vocab
                zero_vector = torch.zeros(1, pretrained_vocab.vectors.shape[-1])
                for token in self.special_tokens:
                    extend_vocab(pretrained_vocab, token, zero_vector)
                itos = pretrained_vocab.itos
                vectors = pretrained_vocab.vectors

        stoi = {w: idx for idx, w in enumerate(itos)}
        self.itos = itos
        self.stoi = stoi
        self.vectors = vectors.float().numpy()

        logger.info(f"glove vocab built with {len(itos)} words")

        if global_vocab is None and self.global_vocab_key is not None:
            cache = SimpleNamespace(itos=itos, stoi=stoi, vectors=vectors)
            global_set(self.global_vocab_key, cache)

    def __call__(self, text):
        result = dict()

        text_tok = self.tokenizer(text)
        text_inds = np.array([self.stoi.get(w, self.stoi["<unk>"]) for w in text_tok])
        text_embeddings = np.stack([self.vectors[ind] for ind in text_inds], axis=0)
        if self.to_words:
            result["tokens"] = text_tok
        if self.to_indices:
            result["indices"] = text_inds
        if self.to_embeddings:
            result["embeddings"] = text_embeddings

        return SimpleNamespace(**result)