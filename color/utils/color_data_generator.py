import os
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

class Tokenizer:
    def __init__(self, config):
        with open(config["vocabulary_file_path"], "r", encoding="utf-8") as f:
            self.dict = ["PST", "PAD", "UNK", "PEND"] + eval(f.read())
        self.word2id = {self.dict[i]: i for i in range(len(self.dict))}
        self.id2word = {i: self.dict[i] for i in range(len(self.dict))}

        self._token_start_id = self.word2id["PST"]
        self._token_pad_id = self.word2id["PAD"]
        self._token_unknown_id = self.word2id["UNK"]
        self._token_end_id = self.word2id["PEND"]

    def encode(self, text):
        token_ids = [self._token_start_id] + [self.word2id.get(char, self._token_unknown_id) for char in text] + [self._token_end_id]
        return token_ids

    def decode(self, ids):
        return self.id2word[ids]


class Corpus:
    def __init__(self, config):
        self.config = config
        self.vocab2id, self.id2vocab = self.generate_vocabulary()
        self.data = []

    def generate_vocabulary(self):

        if os.path.exists(self.config["vocabulary_file_path"]):
            with open(self.config["vocabulary_file_path"], "r", encoding="utf-8") as f:
                vocabs = eval(f.read())
        else:
            with open(self.config["corpus_file_path"], "r", encoding="utf-8") as f:
                corpus_ = f.read()
            vocabs_with_frequency = Counter(corpus_).most_common()
            vocabs = [
                word for (word, freq) in vocabs_with_frequency if freq > self.config["character_frequency_threshold"]
            ]
            with open(self.config["vocabulary_file_path"], "w", encoding="utf-8") as f:
                f.write(str(vocabs))

        vocabs = ["PST", "PAD", "UNK", "PEND"] + vocabs
        vocab2id = dict(zip(vocabs, list(range(len(vocabs)))))
        id2vocab = dict(zip(list(range(len(vocabs))), vocabs))

        #         print('Vocabulary Size = {}'.format(len(vocab2id)))

        return vocab2id, id2vocab

    def make_and_parse_passages(self):
        with open(self.config["corpus_file_path"], "r", encoding="utf-8") as f:
            corpus_ = f.readlines()
        for line in corpus_:
            yield line.replace('"', "")

    def make_bert_data(self):
        passages = self.make_and_parse_passages()
        for passage in passages:
            sentences = passage.strip("\n").split(" ; ")
            one_sample = []
            for i in range(len(sentences)):
                one_sample.append(self.vocab2id["PST"])
                for color in sentences[i].split(" "):
                    if color == "":
                        one_sample.append(self.vocab2id["PAD"])
                    else:
                        if color in self.vocab2id:
                            one_sample.append(self.vocab2id[color])
                        else:
                            one_sample.append(self.vocab2id["UNK"])
                # add PAD when color number in a palette is less then max_palette_length
                for r in range(len(sentences[i].split(" ")), self.config["max_palette_length"][i]):
                    one_sample.append(self.vocab2id["PAD"])
                one_sample.append(self.vocab2id["PEND"])

            if len(one_sample) < self.config["max_sequence_length"]:
                one_sample += [self.vocab2id["PAD"]] * (self.config["max_sequence_length"] - len(one_sample))
            self.data.append(one_sample[: self.config["max_sequence_length"]])
            
    def token_id_to_word_list(self, token_id_list):
        """
        transfer token_id to original word list
        """
        word_list = []
        for token_id in token_id_list:
            if token_id in self.id2vocab:
                word_list.append(self.id2vocab[token_id])
            else:
                word_list.append("[UNK]")
        return word_list


class ColorGenerator(Dataset):
    def __init__(self, config):
        self.config = config
        self.corpus = Corpus(config)
        self.corpus.make_bert_data()
        self.data = self.corpus.data
        self.batch_size = self.config["batch_size"]

    def __len__(self):
        return len(self.data) // self.batch_size

    def make_padding_mask(self, batch_token_id):
        batch_padding_mask = (np.array(batch_token_id) == self.corpus.vocab2id["PAD"]).astype(int)
        return batch_padding_mask

    def __getitem__(self, idx):
        # padding_mask = self.make_padding_mask(batch_data)
        origin_x = np.array(self.data[idx * self.batch_size : (idx + 1) * self.batch_size])

        return origin_x

Config = {
    "corpus_file_path": os.path.join(f"../../training/image-palette/color/color_corpus_lab_bins_16_train.txt"),
    "vocabulary_file_path": os.path.join(f"../../training/image-palette/color/color_vocab_lab_bins_16_train.txt"),
    "character_frequency_threshold": 1,  # 3 may be better for large dataset
    "batch_size": 1,  # 2048 for training on GPU
    "max_palette_length": [5],
    "max_sequence_length": 7,
    "vocab_size": 671,  # fix vocab_size? len(color_freq)+4 (PST, PAD, UNK, PEND)
}

if __name__ == "__main__":
    dataset = DataGenerator(Config)
    for i in range(270, 280):
        origin_x = dataset[i]
        print(f"original sequence: {dataset.corpus.token_id_to_word_list(list(origin_x[0]))}")
        print(f"original id: {origin_x[0]}")