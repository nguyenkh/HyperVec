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
See the config.cfg to set agruments for model.

### Training embeddings
  ```java -jar HyperVec.jar config.cfg vector-size window-size```
  
  For example, training embeddings with 100 dimensions; window-size = 5:

  ```java -jar HyperVec.jar config.cfg 100 5```
  
 ### Pretrained (hypervec) embeddings
 The embeddings used in our paper can be downloaded by using the script in `get-pretrainedHyperVecEmbeddings/download_embeddings.sh`. Note that the script downloads 9 files and concatenates them again to a single file (`hypervec.txt.gz`). The format is the default word2vec format: first line with header information, other lines word followed by whitespace seperated vector.

Information about the embeddings: creatd using the ENCOW14A corpus (14.5bn token), 100 dimensions, sym. window of 5, 15 negative samples, 0.025 learning rate, threshhold set to 0.05. The resulting vocabulary contains about 2.7m words. 
  
### Example usage: Evaluation BLESS,BIBLESS and AWBLESS
To reproduce our experiments from Table 3 use the code in the `datasets_classification/`, 
assuming your vector file is located in the same folder and named `hypervec.txt.gz`. 
  `java -jar eval-dir.jar hypervec.txt.gz` (Evaluate directionality on `BLESS.txt` using hyperscore)
  `java -jar eval-bless.jar hypervec.txt.gz 2 1000` (Evaluate classification on `BIBLESS.txt, AWBLESS.txt` using 2% of the training data and 1000 random iterations)
  

### Citation info
If you use the code or the created feature norms, please [cite our paper (Bibtex)](http://www2.ims.uni-stuttgart.de/bibliographie/entry/2811b00e1bbd503adf28648ddb737132dc67a091/), the paper can be found here: [PDF](http://www.aclweb.org/anthology/D17-1022), the poster from EMNLP can be found here: [Poster](http://www.ims.uni-stuttgart.de/institut/mitarbeiter/koepermn/publications/poster_EMNLP2017.pdf)
