import numpy as np
import sys
from sklearn.metrics import average_precision_score
from numpy.linalg import norm
import common

def cosine_sim(u, v):
    return np.dot(u,v)/(norm(u)*norm(v))

def computeAP(targets, preds):
    paired = zip(preds, targets)
    sorted_paired = sorted(paired, key=lambda x:x[0], reverse=True)
    preds, targets = zip(*sorted_paired)
    preds, targets = list(preds), list(targets)
    
    ap = 0.0
    retrievedCounter = 0.0;
    relevantCounter = 0.0;

    for i in range(len(targets)):
        retrievedCounter += 1
        if int(targets[i]) == 1:
            relevantCounter += 1
            ap += relevantCounter / retrievedCounter
    ap /= relevantCounter
    return ap

def _filter(word):
    word = word.split('-')
    if len(word) > 2:
        f_word = '-'.join(word[:-1])
    else:
        f_word = word[0]
    return f_word

def load_dataset(dataset_file):
    dataset = []
    with open(dataset_file, 'r') as fin:
        for line in fin:
            left, right, label = line.strip().split('\t')
            dataset.append((left, right, int(label)))
    return dataset

def compute_similarity(dataset, embs):
    data = []
    for (left, right, label) in dataset:
        if left in embs and right in embs:
            #direct = norm(embs[right]) / norm(embs[left])
            score = cosine_sim(embs[left], embs[right]) #* direct
            data.append((left, right, label, score))
        else:
            continue
    return data

def build_data(dataset_file, embeddings_file):
    vecs, words = common.load_vecs(embeddings_file, binary=1) #TODO: set binary=0 to read text file
    embs = {word:vecs[idx] for idx,word in enumerate(words)}
    dataset = load_dataset(dataset_file)
    data = compute_similarity(dataset, embs)
    
    return data

def ap_evaluation(data, cutoff=-1):
    
    data = sorted(data, key=lambda line:line[-1], reverse=True)
    targets, scores = [], []
    for (left, right, label, score) in data:
        targets.append(label)
        scores.append(score)
    if cutoff > 0:
        ap_score = average_precision_score(targets[:cutoff], scores[:cutoff])
        #ap_score = computeAP(targets, scores)
        print 'AP at %d cutoff: %f' %(cutoff, ap_score)
    else:
        ap_score = average_precision_score(targets, scores)
        #ap_score = computeAP(targets, scores)
        print 'AP score: %f' %ap_score     
    
    return ap_score

if __name__=='__main__':
    dataset_file = sys.argv[1]
    embeddings_file = sys.argv[2]
    data = build_data(dataset_file, embeddings_file)
    ap_evaluation(data)





