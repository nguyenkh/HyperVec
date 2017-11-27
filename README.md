## HyperVec
Hierarchical Embeddings for Hypernymy Detection and Directionality

### Prerequisite
  - [spaCy](https://spacy.io): for parsing
  - a corpus such as wikipedia corpus (plain-text)

### Preprocess
 - Create the feature files:
 
    ```python create_features.py -input corpus-file.txt -output output-file-name -pos pos_tag```
    in which: pos_tag is either NN (for the noun features) or VB (for the verb features)     

### Configuration
See config.cfg to set agruments for model.

### Training embeddings
  ```java -jar HyperVec.jar config.cfg vector-size window-size```
  
  For example, training embeddings with 100 dimensions; window-size = 5:

  ```java -jar HyperVec.jar config.cfg 100 5```

### Citation info
If you use the code or the created feature norms, please [cite our paper (Bibtex)](http://www2.ims.uni-stuttgart.de/bibliographie/entry/2811b00e1bbd503adf28648ddb737132dc67a091/), the paper can be found here: [PDF](http://www.aclweb.org/anthology/D17-1022), the poster from EMNLP can be found here: [Poster](http://www.ims.uni-stuttgart.de/institut/mitarbeiter/koepermn/publications/poster_EMNLP2017.pdf)
