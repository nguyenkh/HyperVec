import argparse
import spacy
from spacy.lang.en import English
import gzip
from collections import Counter, defaultdict
import six.moves.cPickle as pickle
from itertools import count

def main():
    """
    TODO: extracts the feature files in the corpus
    Usage: python create_features.py -input corpus -output output-file-name -pos
    -pos: <NN/VB>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-pos', type=str)
    args = parser.parse_args()
    
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    window_size = 5
    dfeatures = defaultdict(set)
        
    output_dir = '/mount/arbeitsdaten34/projekte/slu/KimAnh/Corpora/'
    
    vocab_to_id = defaultdict(count(0).next)
    
    with gzip.open(args.input,'rb') as fin:
        para_num = 0
        # Read each paragraph in corpus
        for paragraph in fin:
            # Check empty paragraph
            paragraph = paragraph.strip()
            if len(paragraph) == 0: continue
            para_num += 1
            print 'Processing para: %d' %para_num
            # Parse each sentence
            parsed_para = nlp(unicode(paragraph))
            for sent in parsed_para.sents:
                features = process_one_sentence(sent, args.pos, window_size, vocab_to_id)
                dfeatures.update(features)

    id_to_vocab = {idx:word for word,idx in vocab_to_id.iteritems()}
    save_file(dfeatures, id_to_vocab, args.output)
    
    print 'Parsing corpus done....!'                    

def save_file(dfeatures, id_to_vocab, outfile):
    with gzip.open(outfile, 'w') as fout:
        for kk,vv in dfeatures:
            contexts = [id_to_vocab[idx] for idx in list(vv)]
            fout.write(str(id_to_vocab[kk]))
            for word in contexts:
                fout.write('\t' + str(word))
            fout.write('\n')
    print 'Saved file!'

def process_one_sentence(sent, pos, window_size, vocab_to_id):
    features = defaultdict(set)
    
    for idx,token in enumerate(sent):
        if token.tag_[:2] == pos and len(token.string.strip()) > 2:
            for idw in range(idx-window_size, idx+window_size):
                if idw != idx and idw >= 0 and idw < len(sent):
                    features[vocab_to_id[sent[idx]]].add(vocab_to_id[sent[idw]])
                    
    return features

if __name__=='__main__':
    main()
