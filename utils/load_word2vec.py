import numpy as np
import logging

def load_word2vec_embeddings(dictionary, vocab_embed_file, embed_dim, first_time):
    if vocab_embed_file is None:
        return None, embed_dim

    fp = open(vocab_embed_file, encoding='utf-8')

    info = fp.readline().split()
    embed_dim = int(info[1])

    if first_time == False:
        fp.close()
        return None, embed_dim

    # vocab_embed: word --> vector
    vocab_embed = {}

    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(
            list(map(float, line[1:])), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i, :] = vocab_embed[w]
            n += 1

    logging.info("{}/{} vocabs are initialized with word2vec embeddings."
                 .format(n, vocab_size))

    return W, embed_dim