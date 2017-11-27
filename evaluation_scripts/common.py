import numpy as np
from numpy import fromstring, dtype

def smart_open(fname, mode='rb'):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode)
    elif fname.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(fname, mode)
    else:
        return open(fname, mode)

def load_vecs(binary_file, binary=1):
    vecs = []
    vocab = []
    if binary==1:
        with smart_open(binary_file, 'rb') as f:
            header = to_unicode(f.readline())
            vocab_size, vector_size = map(int, header.split())
            binary_len = dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word.append(ch)
                word = to_unicode(b''.join(word))
                vocab.append(word)
                vec = fromstring(f.read(binary_len), dtype=np.float32)
                vecs.append(vec)
    else:
        with smart_open(binary_file, 'rb') as f:
            header = to_unicode(f.readline())
            if len(header.split()) == 2: vocab_size, vector_size = map(int, header.split())
            elif len(header.split()) > 2:
                parts = header.rstrip().split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                vocab.append(to_unicode(word))
                vecs.append(vec)
            for _, line in enumerate(f):
                parts = to_unicode(line.rstrip()).split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                vocab.append(to_unicode(word))
                vecs.append(vec)
                
    #embs_dim = len(vecs[1])   
    #UNKNOWN_WORD = np.random.uniform(-0.25,0.25,embs_dim)
    #vecs = np.vstack((UNKNOWN_WORD, vecs))               
    #vocab = ['#UNKNOWN#'] + list(vocab)
    #words = {word:idx for idx,word in enumerate(vocab)}
    
    return vecs, vocab

def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    else:
        return unicode(text, encoding, errors=errors).encode('utf8')

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding=encoding, errors=errors)