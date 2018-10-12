import argparse
import os
from time import time
from tqdm import tqdm
import pickle

import numpy as np
from pathlib import Path

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


def load_glove_emb(input_path, vocab=None):
    count = 0
    tStart = time()
    word_vecs = {}
    with input_path.open("r", encoding="utf8") as f:
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

    print("Finished loading embeddings: {} mins".format(
        (time() - tStart) / 60.))
    if vocab is not None:
        print("Words not found: {}".format(len(vocab) - count))
    return word_vecs


def load_w2v_emb(input_path, vocab=None):
    count = 0
    tStart = time()
    word_vecs = {}
    with input_path.open("rb") as f:
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
    print("Finished loading embeddings: {} mins".format(
        (time() - tStart) / 60.))
    if vocab is not None:
        print("Words not found: {}".format(len(vocab) - count))
    return word_vecs


def load_fasttext_emb(input_path, vocab=None):
    count = 0
    tStart = time()
    word_vecs = {}
    with input_path.open("r", encoding='utf-8') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        for line in tqdm(
                f, desc='loading fasttext embeddings', total=vocab_size):
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

    print("finished loading embeddings: {} mins".format(
        (time() - tStart) / 60.))
    if vocab is not None:
        print("words not found: {}".format(len(vocab) - count))
    return word_vecs


def main(args):
    print("loading embeddings")
    input_path = Path(args.input_file)
    if args.file_type == "w2v":
        embeddings = load_w2v_emb(input_path)
    elif args.file_type == "glove":
        embeddings = load_glove_emb(input_path)
    elif args.file_type == "wiki":
        embeddings = load_fasttext_emb(input_path)

    pickled_vectors_file_path = input_path.with_suffix(".pkl")
    with pickled_vectors_file_path.open('wb') as pkl_file:
        print("saving pickled file with vectors to {}".format(
            str(pickled_vectors_file_path)))
        pickle.dump(embeddings, pkl_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Preprocess word2vec data and generate a PKL file')
    PARSER.add_argument('input_file', help='Input name of word2vec data')
    PARSER.add_argument(
        'file_type', help='File type', choices=['w2v', 'glove', 'wiki'])
    BUILD_ARGS = PARSER.parse_args()
    main(BUILD_ARGS)
