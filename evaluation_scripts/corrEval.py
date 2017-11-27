import sys
import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import norm
import common

def cosine(u, v):
    return np.dot(u,v)/(norm(u)*norm(v))

def hyper_score(u,v):
    sim = np.dot(u,v)/(norm(u)*norm(v))
    direct = norm(v)/norm(u)
    return sim*direct

def load_data(embeddings_file, dataset_file, mode='cosine'):
    golds, scores = [], []
    unseen = 0
    with open(dataset_file, 'r') as fin:
        data = [line.strip().split(' ') for line in fin]
    vecs, words = common.load_vecs(embeddings_file, binary=1)
    embs = {word:vec for word,vec in zip(words,vecs)}
    for rec in data:
        if rec[0] in embs and rec[1] in embs:
            golds.append(float(rec[5]))
            if mode=='hyper':
                grade = hyper_score(embs[rec[0]], embs[rec[1]])
                scores.append(grade)
            elif mode=='cosine':
                grade = cosine(embs[rec[0]], embs[rec[1]])
                scores.append(grade)
        else:
            unseen += 1
    print 'unseen-words: %d' %unseen
    return golds, scores

if __name__=='__main__':
    embeddings_file = sys.argv[1]
    dataset_file = sys.argv[2]
    mode = sys.argv[3] # either 'cosine' or 'hyper'
    golds, scores = load_data(embeddings_file, dataset_file, mode)
    rho = spearmanr(golds, scores)[0]
    print 'Spearman correlation: %f' %rho
    
    
    
    
