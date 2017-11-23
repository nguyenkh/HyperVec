import numpy as np
import collections
from space import Space


def prec_at(ranks, cut):
    return len([r for r in ranks if r <= cut])/float(len(ranks))

def get_rank(nn, gold):
    for idx,word in enumerate(nn):
        if word in gold:
            return idx + 1
    return idx + 1

        
def read_dict(dict_file):
    return [tuple(line.strip().split()) for line in file(dict_file)]


def apply_tm(sp, tm):
    
    print "Applying the translation matrix, size of data: %d" % sp.mat.shape[0] 
    return Space(sp.mat*tm, sp.id2row)
    
def get_valid_data(sp1, sp2, data):
    return [(el1, el2) for el1,el2 in data if 
            el1 in sp1.row2id and el2 in sp2.row2id]

def train_tm(sp1, sp2, data):

    data = get_valid_data(sp1, sp2, data)
    print "Training using: %d word pairs" % len(data)
    
    els1, els2 = zip(*data)
    m1 = sp1.mat[[sp1.row2id[el] for el in els1],:]
    m2 = sp2.mat[[sp2.row2id[el] for el in els2],:]

    tm = np.linalg.lstsq(m1, m2, -1)[0]

    return tm 
    

def score(sp1, sp2, gold, additional):
    
    sp1.normalize()

    print "Computing cosines and sorting target space elements"    
    sim_mat = -sp2.mat*sp1.mat.T
    
    if additional:
        #for each element, computes its rank in the ranked list of
        #similarites. sorting done on the opposite axis (inverse querying) 
        srtd_idx = np.argsort(np.argsort(sim_mat, axis=1), axis=1)

        #for each element, the resulting rank is combined with cosine scores. 
        #the effect will be of breaking the ties, because cosines are smaller
        #than 1. sorting done on the standard axis (regular NN querying)
        srtd_idx = np.argsort(srtd_idx + sim_mat, axis=0)
    else:
        srtd_idx = np.argsort(sim_mat, axis=0)

    ranks = []
    for i,el1 in enumerate(gold.keys()):

        sp1_idx = sp1.row2id[el1]

        #print the top 5 translations
        translations = []        
        for j in range(5):
            sp2_idx = srtd_idx[j, sp1_idx]
            word, score = sp2.id2row[sp2_idx], -sim_mat[sp2_idx, sp1_idx]        
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations) 

        #get the rank of the (highest-ranked) translation
        rnk = get_rank(srtd_idx[:,sp1_idx].A.ravel(), 
                        [sp2.row2id[el] for el in gold[el1]])
        ranks.append(rnk)

        print ("\nId: %d Source: %s \n\tTranslation:\n%s \n\tGold: %s \n\tRank: %d" %
               (len(ranks), el1, translations, gold[el1], rnk))

    print "Corrected: %s" % str(additional)
    if additional:
        print "Total extra elements, Test(%d) + Additional:%d" % (len(gold.keys()),
                                                           sp1.mat.shape[0]) 
    for k in [1,5,10]:
        print "Prec@%d: %.3f" % (k, prec_at(ranks, k))
        
