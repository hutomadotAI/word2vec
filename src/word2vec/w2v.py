# -*- coding: utf-8 -*-

from time import time
import numpy as np
import logging
import pickle
from pathlib import Path


def _get_logger():
    logger = logging.getLogger('svclass.word2vec')
    return logger


class Word2VecError(Exception):
    pass


class Word2Vec(object):

    PICKLED_VECTORS_FILE_EXT = ".pkl"

    def __init__(self, path=None):
        self.__logger = _get_logger()
        self.path = path
        self.use = 'glove' if 'glove' in self.path else 'w2v'

    @property
    def logger(self):
        return self.__logger

    def get_mean_norm(self, w2v):
        mean_norm = np.mean(
            np.linalg.norm(np.array(list(w2v.values())), axis=1))
        return mean_norm

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
            raise Word2VecError("Couldn't find required .pkl file at {}".format(pickled_vectors_file_path))

        return embeddings




