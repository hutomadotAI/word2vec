# -*- coding: utf-8 -*-

from time import time
import numpy as np
import _pickle as pickle
from tqdm import tqdm

fPath = "/home/thomas/hutoma_research/text_sim_and_class/siamese_lstm_theano/"
w2vPath = fPath + "GoogleNews-vectors-negative300.bin"
dwordsPath = fPath + "dwords.p"

glovePath = "/home/thomas/SeqMatchSeq/data/glove/glove.840B.300d.txt"


class Word2Vec(object):
    def __init__(self, path=None, savePath=None):
        if savePath is not None:
            self.w2v, self.use, self.path = self.load(savePath)
        else:
            self.path = w2vPath if path is None else path
            self.use = 'glove' if 'glove' in self.path else 'w2v'
            self.w2v = {}

    def get_new_vocab(self, docs):
        vocab = set([w for doc in docs for w in doc if w not in self.w2v.keys()])
        print("INFO: new vocab: {}".format(len(vocab)))
        return vocab

    def add_embeddings(self, w2v_update):
        self.w2v.update(w2v_update)

    def load_w2v(self, docs=None):
        if docs is None:
            print("WARNING: This function loads all words in file; lots of RAM needed")
            print("INFO: loading word2vec...")
            w2v = self.load_embeddings()
            return w2v
        else:
            vocab = self.get_new_vocab(docs)
            w2v_new = self.load_embeddings(vocab)
            self.add_embeddings(w2v_new)
            return self.w2v

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
        with open(self.path, "rb") as f:
            for line in tqdm(f, desc='loading glove embeddings'):
                line = line.strip()
                if not line:
                    continue
                line = line.split(' ')
                if vocab is None:
                    emb = map(np.float32, line[1:])
                    word_vecs[line[0]] = np.array(emb)
                    count += 1
                elif line[0] in vocab:
                    emb = map(np.float32, line[1:])
                    word_vecs[line[0]] = np.array(emb)
                    count += 1

        print("INFO: finished loading embeddings: {} mins".format((time() - tStart) / 60.))
        if vocab is not None:
            print("INFO: words not found: {}".format(len(vocab) - count))
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
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                    count += 1
                elif word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                    count += 1
                else:
                    f.read(binary_len)
        print("INFO: finished loading embeddings: {} mins".format((time() - tStart) / 60.))
        if vocab is not None:
            print("INFO: words not found: {}".format(len(vocab) - count))
        return word_vecs

    def save(self, fpath):
        with open(fpath, 'w') as f:
            pickle.dump([self.w2v, self.use, self.path], f)

    def load(self, fpath):
        with open(fpath, 'r') as f:
            d = pickle.load(f)
        return d
