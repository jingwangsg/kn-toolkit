from kn_util.general import registry
from kn_util.data import delete_noisy_char
from torchtext.data.utils import get_tokenizer
import torchtext
import torch
import numpy as np
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("tokenize_glove")
class GloveTokenizer(IterDataPipe):

    def __init__(
        self,
        src_pipeline,
        glove="glove.6B.300d",
        vocab_file=None,
        upload_vocab_key=None,
        tokenizer="split",
        from_key=None,
        cache_dir=None,
        to_words=False,
        to_indices=False,
        to_embeddings=False,
    ) -> None:
        super().__init__()
        assert from_key is not None
        assert cache_dir is not None
        assert to_words or to_indices or to_embeddings
        self.src_pipeline = src_pipeline
        self.from_key = from_key
        self.vocab_file = vocab_file
        self.glove = glove
        self.upload_vocab_key = upload_vocab_key
        self.cache_dir = cache_dir
        self.to_words = to_words
        self.to_indices = to_indices
        self.to_embeddings = to_embeddings

        if tokenizer == "split":
            self.tokenizer = lambda s: delete_noisy_char(s).lower().split()
        else:
            self.tokenizer = get_tokenizer(tokenizer)

        self._load_vocab()

    def _load_vocab(self):
        pretrained_vocab = torchtext.vocab.pretrained_aliases[self.glove](cache=self.cache_dir)
        if self.vocab_file:
            with open(self.vocab_file, "r") as f:
                lines = f.readlines()
            itos = [w.strip() for w in lines]
            extracted_indicies = [pretrained_vocab.stoi.get(w, 1) for w in itos[1:]]
            vectors = pretrained_vocab.vectors[extracted_indicies]
            vectors = torch.concat([torch.zeros((1, vectors.shape[-1]), dtype=vectors.dtype), vectors], dim=0)
        else:
            itos = ["<unk>", "<pad>"] + pretrained_vocab.itos
            vectors = pretrained_vocab.vectors
            vectors = torch.concat(
                [
                    torch.zeros((1, vectors.shape[-1]), dtype=vectors.dtype),  # <pad>
                    pretrained_vocab["<unk>"].unsqueeze(0),  # <unk>
                    vectors,
                ],
                dim=0,
            )

        stoi = {w: idx for idx, w in enumerate(itos)}
        self.itos = itos
        self.stoi = stoi
        self.vectors = vectors.float().numpy()

        print(f"glove vocab built with {len(itos)} words")

        if self.upload_vocab_key:
            registry.register_object(self.upload_vocab_key, (itos, vectors))

        del pretrained_vocab
    
    def __iter__(self):
        for x in self.src_pipeline:
            result = dict()

            text = x[self.from_key]
            text_tok = self.tokenizer(text)
            text_inds = np.array([self.stoi.get(w, 1) for w in text_tok])
            text_embeddings = np.stack([self.vectors[ind] for ind in text_inds], axis=0)
            if self.to_words:
                result[self.from_key + ".tok"] = text_tok
            if self.to_indices:
                result[self.from_key + ".inds"] = text_inds
            if self.to_embeddings:
                result[self.from_key + ".embs"] = text_embeddings
            
            x.update(result)
            
            yield x