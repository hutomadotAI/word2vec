# -*- coding: utf-8 -*-

from time import time
import numpy as np
from tqdm import tqdm
import logging


def _get_logger():
    logger = logging.getLogger('svclass.word2vec')
    return logger


class Word2Vec(object):
    def __init__(self, path=None):
        self.__logger = _get_logger()
        self.path = path
        self.use = 'glove' if 'glove' in self.path else 'w2v'

    @property
    def logger(self):
        return self.__logger

    def get_new_vocab(self, docs):
        vocab = set([w for doc in docs for w in doc]) + 'UNK'
        self.logger.info("New vocab: {}".format(len(vocab)))
        return vocab

    def load_w2v(self, docs=None):
        if docs is None:
            self.logger.info("Loading word2vec...")
            w2v = self.load_embeddings()
            return w2v
        else:
            vocab = self.get_new_vocab(docs)
            w2v = self.load_embeddings(vocab)
            return w2v

    def load_embeddings(self, vocab=None):
        """
        Reads word embeddings from disk.
        """
        print("INFO: loading embeddings")
        if self.use == "w2v":
            return self._load_w2v_emb(vocab)
        elif self.use == "glove":
            return self._load_glove_emb(vocab)

    def _load_glove_emb(self, vocab=None):
        count = 0
        tStart = time()
        word_vecs = {}
        with open(self.path, "r") as f:
            for line in tqdm(f, desc='loading glove embeddings'):
                line = line.strip()
                if not line:
                    continue
                line = line.split(' ')
                if vocab is None:
                    emb = list(map(np.float32, line[1:]))
                    word_vecs[line[0]] = np.array(emb)
                    count += 1
                elif line[0] in vocab:
                    emb = map(np.float32, line[1:])
                    word_vecs[line[0]] = np.array(emb)
                    count += 1

        self.logger.info("Finished loading embeddings: {} mins".format(
            (time() - tStart) / 60.))
        if vocab is not None:
            self.logger.info("Words not found: {}".format(len(vocab) - count))
        return word_vecs

    def _load_w2v_emb(self, vocab=None):
        count = 0
        tStart = time()
        word_vecs = {}
        with open(self.path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in tqdm(range(vocab_size), desc="loading w2v embeddings"):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        """print(ch.decode('cp437'))"""
                        word.append(ch.decode('cp437'))

                if vocab is None:
                    word_vecs[word] = np.fromstring(
                        f.read(binary_len), dtype='float32')
                    count += 1
                elif word in vocab:
                    word_vecs[word] = np.fromstring(
                        f.read(binary_len), dtype='float32')
                    count += 1
                else:
                    f.read(binary_len)
        self.logger.info("Finished loading embeddings: {} mins".format(
            (time() - tStart) / 60.))
        if vocab is not None:
            self.logger.info("Words not found: {}".format(len(vocab) - count))
        return word_vecs
