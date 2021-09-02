## This repository contains the source code for the experiments presented in the paper Generating Vector Table Representations.

### Experiments and evaluation
The file experiments.py contains the code for running the experiments with each of the 5 methods discussed in the paper
and the evaluation of their performance. 



### Experiments with the TaBERT model
The code for generating vector table representations with TaBERT are in the TaBert_table_encode.py file.
To run the experiments, the pre-trained model files are required and these can be downloaded from the [TaBERT](https://github.com/facebookresearch/TaBERT) git repository.

### Experiments with TF-IDF, Word2Vec, Spacy and BERT
The setup for generating vector rable representations with the other 4 methods are in the baselines_table_encode.py file.
For table-encoding with TF-IDF the [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidfvectorizer#sklearn.feature_extraction.text.TfidfVectorizer) is used.

The pre-trained vocabulary for Word2Vec can be downloaded from https://fasttext.cc/docs/en/pretrained-vectors.html.

The pre-trained Spacy model can be downloaded with 
```bash
pip install -U spacy
python -m spacy download en_core_web_sm
```
The pre-trained BERT model can be downloaded and used with the following command:

```bash
pip install pytorch_pretrained_bert
from pytorch_pretrained_bert import import BertTokenizer, BertForMaskedLM
```

The files conf_mtx_tabert.pdf and conf_mtx_w2vec.pdf show the confusion matrices which are presented in the paper, in their original size. 





