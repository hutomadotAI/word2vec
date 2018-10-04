# -*- coding: utf-8 -*-

from time import time
import numpy as np
from tqdm import tqdm
import logging
import pickle
from pathlib import Path


def _get_logger():
    logger = logging.getLogger('svclass.word2vec')
    return logger


class Word2Vec(object):

    PICKLED_VECTORS_FILE_EXT = ".pkl"

    def __init__(self, path=None):
        self.__logger = _get_logger()
        self.path = path
        if 'glove' in self.path:
            self.use = 'glove'
        elif 'GoogleNews' in self.path:
            self.use = 'w2v'
        elif 'wiki' in self.path:
            self.use = 'fasttext'

    @property
    def logger(self):
        return self.__logger

    def get_new_vocab(self, docs):
        vocab = set([w for doc in docs for w in doc]) + 'UNK'
        self.logger.info("New vocab: {}".format(len(vocab)))
        return vocab

    def get_mean_norm(self, w2v):
        mean_norm = np.mean(
            np.linalg.norm(np.array(list(w2v.values())), axis=1))
        return mean_norm

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

        # First try to load a pickle file, if it exists, in the same directory as
        # the vectors file
        local_path = Path(self.path)
        pickled_vectors_file_path = local_path.with_suffix(
            self.PICKLED_VECTORS_FILE_EXT)
        if pickled_vectors_file_path.exists():
            self.logger.info("found pickled file for embeddings at {}".format(
                str(pickled_vectors_file_path)))
            tStart = time()
            with pickled_vectors_file_path.open('rb') as pkl_file:
                embeddings = pickle.load(pkl_file)
            self.logger.info(
                "Finished loading embeddings from pickle: {} mins".format(
                    (time() - tStart) / 60.))
        else:
            print("loading embeddings")
            if self.use == "w2v":
                embeddings = self._load_w2v_emb(vocab)
            elif self.use == "glove":
                embeddings = self._load_glove_emb(vocab)
            elif self.use == 'fasttext':
                embeddings = self._load_fasttext_emb(vocab)

            # Try to save the embeddings a a pickle to speedup next initialisation
            try:
                with pickled_vectors_file_path.open('wb') as pkl_file:
                    self.logger.info(
                        "saving pickled file with vectors to {}".format(
                            str(pickled_vectors_file_path)))
                    print("saving pkl")
                    pickle.dump(embeddings, pkl_file)
                    print("file saved")
            except IOError:
                self.logger.exception(
                    "Could not save the pickled file, will keep on using the original one"
                )

        return embeddings

    def _load_glove_emb(self, vocab=None):
        count = 0
        tStart = time()
        word_vecs = {}
        with open(self.path, "r", encoding='utf-8') as f:
            for line in tqdm(f, desc='loading glove embeddings'):
                line = line.rstrip()
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

    def _load_fasttext_emb(self, vocab=None):
        count = 0
        tStart = time()
        word_vecs = {}
        with open(self.path, "r", encoding='utf-8') as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            for line in tqdm(f, desc='loading fasttext embeddings', total=vocab_size):
                line = line.strip()
                if not line:
                    continue
                line = line.split(' ')
                if vocab is None:
                    emb = list(map(np.float32, line[1:]))
                    word_vecs[line[0]] = np.array(emb)
                    count += 1
                elif line[0] in vocab:
                    emb = list(map(np.float32, line[1:]))
                    word_vecs[line[0]] = np.array(emb)
                    count += 1

        self.logger.info("finished loading embeddings: {} mins".format((time() - tStart) / 60.))
        if vocab is not None:
            self.logger.info("words not found: {}".format(len(vocab) - count))
        return word_vecs
