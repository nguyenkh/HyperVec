from gensim.models import word2vec
import sys

# Script that converts word2vec txtfile into word2vec binary
print ("Script name: %s" % str(sys.argv[1]))
model = word2vec.Word2Vec.load_word2vec_format(str(sys.argv[1]),binary=False)
model.save_word2vec_format(str(sys.argv[1])+'.bin',binary=True)

