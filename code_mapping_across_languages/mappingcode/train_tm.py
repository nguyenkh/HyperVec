import sys
import getopt
import numpy as np
from space import Space
from utils import read_dict, train_tm

def usage(errno=0):
    print >>sys.stderr,\
    """
    Given train data (pairs of words and their translation), source language and 
    target language vectors, it outputs a translation matrix between source and 
    target spaces.

    Usage:
    python train_tm.py [options] train_data source_vecs target_vecs 
    \n\
    Options:
    -o --output <file>: output file prefix. Optional. Default is ./tm
    -h --help : help

    Arguments:
    train_data: <file>, train dictionary, list of word pairs (space separated words, 
            one word pair per line)
    source_vecs: <file>, vectors in source language. Space-separated, with string 
                identifier as first column (dim+1 columns, where dim is the dimensionality
                of the space)
    target_vecs: <file>, vectors in target language


    Example:
    python train_tm.py train_data.txt ENspace.pkl ITspace.pkl

    """
    sys.exit(errno)


def main(sys_argv):

    try:
        opts, argv = getopt.getopt(sys_argv[1:], "ho:",
                                   ["help", "output="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    out_file = "./tm"
    for opt, val in opts:
        if opt in ("-o", "--output"):
            out_file = val
        elif opt in ("-h", "--help"):
            usage(0)
        else:
            usage(1)

    if len(argv) == 3:
        source_file = argv[1]	
        target_file = argv[2]
	dict_file = argv[0]
    else:
	print str(err)
	usage(1)


    print "Reading the training data"
    train_data = read_dict(dict_file)

    #we only need to load the vectors for the words in the training data
    #semantic spaces contain additional words
    source_words, target_words = zip(*train_data)

    print "Reading: %s" % source_file
    source_sp = Space.build(source_file, set(source_words))
    source_sp.normalize()

    print "Reading: %s" % target_file
    target_sp = Space.build(target_file, set(target_words))
    target_sp.normalize()

    print "Learning the translation matrix"
    tm = train_tm(source_sp, target_sp, train_data)

    print "Printing the translation matrix"
    np.savetxt("%s.txt" % out_file, tm)


if __name__ == '__main__':
    main(sys.argv)

