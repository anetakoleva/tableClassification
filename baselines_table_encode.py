from utils.classification_utils import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import glob
import nltk
import torch
# nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


import spacy
import logging

BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'experiment_scaled_BERT')
DATASET_TABLES_PATH = os.path.join(BASE_DIR, 'data/datasets/tables_classification/tables_class/')
#LABELS_FILE_PATH = os.path.join(BASE_DIR, 'data/datasets/tables_classification/classes_complete.csv')

#DATASET_TABLES_PATH = os.path.join(BASE_DIR, 'data/datasets/extended_instance_goldstandard/tables_v2/') # 223 tables
LABELS_FILE_PATH = os.path.join(BASE_DIR, 'data/datasets/extended_instance_goldstandard/classes_GS.csv') # 223 tables with classes

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# configuration for logging
rootLogger = logging.getLogger()
logging_level = logging.INFO
rootLogger.setLevel(logging_level)
beautify_logging_handlers(rootLogger, logging_level, file_name=os.path.join(RESULTS_DIR, 'log_info'))
# logger = logging.getLogger("spacy")
# logger.setLevel(logging.ERROR)

#nlp = spacy.load('en_core_web_md', disable=["parser", "ner"])  # , disable=["parser", "ner"]
#nlp.max_length = 35000000


class BaselineEncoder():

    def __init__(self, model='spacy', header='concatSingle', masked_col=True, rows=3,
                 preprocess='none', dataset=DATASET_TABLES_PATH, labels=LABELS_FILE_PATH, results=RESULTS_DIR):
        super().__init__()
        if model == 'spacy':
            self.model = 'spacy'
            self.nlp = spacy.load('en_core_web_md', disable=["parser", "ner"])
            self.nlp.max_length = 35000000
        
        elif model == 'word2vec':
            self.model = 'word2vec'
            self.nlp = spacy.load(os.path.join(BASE_DIR, 'w2vec_en_vectors'))
        elif model == 'bert':
            self.model = 'bert'
        else:
            self.model = 'td-idf'

        self.name = '{}'.format(rows) + '_model_{}'.format(model) + '_header_{}'.format(header) + '_masked_{}'.format(
            masked_col)# + '_preprocess_{}'.format(preprocess)
        self.header = header
        self.masked_col = masked_col
        self.rows = rows
        self.preprocess = preprocess
        self.dataframes, self.labels_dict = self.load_data(dataset, labels)
        self.results = os.path.join(results, self.model, self.name)
        os.makedirs(self.results, exist_ok=True)

    def load_data(self, dataset_tables, labels):

        labels_dict = load_labels(labels)

        dataframes = []
        table_label = {}
        for filename in glob.glob(dataset_tables + '*.csv'):
            name = os.path.basename(os.path.normpath(filename)).strip('.csv')
            if name in labels_dict.keys():
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False)
                if self.masked_col == True:
                    df.rename(columns=lambda x: '[UNK]', inplace=True)
                else:
                    pass
                table_label[name] = labels_dict[name]
                dataframes.append(df)

        logging.info("Created {} dataframes ".format(len(dataframes)))
        logging.info("With labels len = {} ".format(len(table_label)))

        return dataframes, table_label

    def tf_idf_encode_table(self):
        """
        Encode tables with CountVectorizer and Tf-Idf.
        Option whether to use stemming or lemmatizer, specified at init of the baseline encoder
        :return:
        """

        logging.info("Tf idf encoding start")
        if self.preprocess == 'stemming':
            vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
                                         tokenizer=StemTokenizer())
        elif self.preprocess == 'lemmatize':
            vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
                                         tokenizer=LemmaTokenizer())
        else:
            vectorizer = TfidfVectorizer()

        string_tables = table_to_string(self.sampled_dfs)
        vector_tables = vectorizer.fit_transform(string_tables).toarray()

        save_embeddings(vector_tables, self.results)

        return np.asarray(vector_tables)

    def encode_tables(self):

        self.sampled_dfs = sample_rows(self.dataframes, k=self.rows)

        if self.model == 'spacy':
            logging.info(
                "Encoding tables with baseline model spacy, rows = {} and header = {}".format(self.rows, self.header))
            self.table_embeddings = self.spacy_encode_table(nlp=self.nlp)

        elif self.model == 'word2vec':
            logging.info(
                "Encoding tables with baseline model word2vec, rows = {} and header = {}".format(self.rows,
                                                                                                 self.header))
            self.table_embeddings = self.spacy_encode_table(nlp=self.nlp)

        elif self.model == 'td-idf':
            logging.info(
                "Encoding tables with baseline model tf-idf, rows = {} and preprocess {} ".format(
                    self.rows, self.preprocess))
            self.table_embeddings = self.tf_idf_encode_table()
            
        elif self.model == 'bert':
            logging.info(
                "Encoding tables with baseline model bert, rows = {} and header = {}".format(self.rows,
                                                                                                 self.header))
            self.table_embeddings = self.bert_encode_table()

        #print(self.table_embeddings.shape)

        # Transform labels to int and to array
        #print("Labels d {}".format(self.labels_dict))
        labels, class2idx = label_to_int(self.labels_dict)
        labels = np.array(labels)

        #print("Labels int {}".format(labels))

        return self.table_embeddings, labels

    def bert_encode_table(self):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logging.info("Encoding tables with BERT started")
        all_tables = []
        to_remove = []
        
        for i, df in enumerate(self.sampled_dfs):           
            
            table_encoding = []
            max_sequence_len = check_max_leng(df)
            
            if max_sequence_len > 512:            
                to_remove.append(i)
                logging.info("Skipping dataframe with index {}".format(i))            
                continue

            else:

                header_data = ' '.join(map(str, df.columns.tolist()))
                header_tokens = tokenizer.tokenize(header_data)

                if self.header == 'concatSingle':
                    # add the embedding of the column names first, then append the embedded rows
                    header_encoding = bert_sequence(header_data, len(header_tokens))
                    table_encoding.append(header_encoding)

                    for i, row in enumerate(df.values.tolist()):
                        data = ' '.join(map(str, row))
                        data_tokens = tokenizer.tokenize(data)
                        row_encoding = bert_sequence(data, len(data_tokens))
                        table_encoding.append(row_encoding)

                    table_encoding_tensor = torch.tensor(table_encoding)
                    table_embed = torch.mean(table_encoding_tensor, dim=0)
                    table_embed = table_embed.detach().numpy()

                if self.header == 'tabert':
                    # add the column names at the beggining of each row
                    max_sequence_len = max_sequence_len + len(header_tokens)
                    
                    if max_sequence_len > 512:            
                        to_remove.append(i)
                        logging.info("Skipping dataframe with index {}".format(i))            
                        continue

                    for i, row in enumerate(df.values.tolist()):
                        data = ' '.join(map(str, df.columns.tolist())) + ' '.join(map(str, row))
                        row_encoding = bert_sequence(data, max_sequence_len)
                        table_encoding.append(row_encoding)

                    table_encoding_tensor = torch.tensor(table_encoding)
                    table_embed = torch.mean(table_encoding_tensor, dim=0)
                    table_embed = table_embed.detach().numpy()

                if self.header == 'asRow' :
                    data = df.to_string(index=False).strip()
                    data_tokens = tokenizer.tokenize(data)
                    if len(data_tokens) > 512:            
                        to_remove.append(i)
                        logging.info("Skipping dataframe with index {}".format(i))            
                        continue
                    
                    table_embed = bert_sequence(data, len(data_tokens))

                all_tables.append(table_embed)
                
        if len(to_remove)>0:
            self.labels_dict = self.remove_from_labels(to_remove)
            
        logging.info("Current all_tables len {}, labels dict len {}".format(len(all_tables), len(self.labels_dict)))

        save_embeddings(all_tables, self.results)

        return np.asarray(all_tables)
    
    def remove_from_labels(self, index_to_delete):
        
        i = 0
        keys_to_delete = []
        for key in self.labels_dict.keys():
            if i in index_to_delete:
                print(key)
                keys_to_delete.append(key)
            i = i + 1

        for key in keys_to_delete:
            if key in self.labels_dict:
                logging.info("Removing table {}".format(key))
                del self.labels_dict[key]

        return self.labels_dict

    def spacy_encode_table(self, nlp):
        """
        Embed table with spacy. Input is list of dataframes and flag for column handling.
        If header = "concatN" - (df.column x len(df)).mean | df.data.mean - same as concatSingle
        if header = "concatSingle" - df.column.mean | df.rows.mean
        if header = "tabert" - concatenate the (df.column | df.row for every row in df.rows then embed).mean
        if header = "asRow" - transform the table to string and then embed
        """
        all_tables = []
        vector_concat = np.array([])
        
        logging.info("Encoding with {}".format(nlp))
        logging.info("init with {}".format(self.nlp))

        for df in self.sampled_dfs:
            df = df.fillna('')
            df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""], regex=True)

            columns = " ".join(list(df.columns.values))
            if self.header == "concatSingle":
                vector_header = nlp(columns).vector

                data = df.to_string(header=False, index=False)
                vector_data = nlp(data).vector

                # concat the vector for the header and the vector for the data
                # result vector of shape (600,)
                vector_concat = np.concatenate((vector_header, vector_data))

            if self.header == "tabert":
                table_string = ""
                for i in range(len(df)):
                    data = " ".join([str(x) for x in df.iloc[i, :]])
                    header_data = columns + ' ' + data + ' '
                    table_string += header_data

                vector_concat = nlp(table_string).vector

            if self.header == "asRow":
                table_string = df.to_string(index=False).strip()
                vector_concat = nlp(table_string).vector

            #  table_vec = np.asarray(table).mean(axis=0)

            all_tables.append(vector_concat)
        #    logging.info("Embeded {} tables so far".format(len(self.dataframes)-len(all_tables)))

        save_embeddings(all_tables, self.results)

        return np.asarray(all_tables)

    def classify_tables(self, embeddings, labels, k=20):
        """
        Run MLP classifer on the table embeddings to predict their class
        :param embeddings:  np array with table embedding
        :param labels: np array with unique labels
        :return:
        """
        # Transform labels to ints
        #  labels, class2idx = label_to_int(table_labels)
        # Cross-validation
        logging.info("Classification with cross-validation")
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, solver='adam')
        scores = cross_val_score(clf, embeddings, labels, cv=k)
        logging.info("{} accuracy with a standard deviation of {} ".format(scores.mean(), scores.std()))
        with open(os.path.join(self.results, 'classification.txt'), 'a+') as f:
            f.write("{} accuracy with a standard deviation of {} \n".format(scores.mean(), scores.std()))


if __name__ == '__main__':
    baseline_spacy = BaselineEncoder(model='word2vec', header='asRow', masked_col=False, rows=1)

    logging.info("Embed tables with model: {} using header method: {}, column names: {}"
                 .format(baseline_spacy.model, baseline_spacy.header, baseline_spacy.masked_col))

    embeddings, labels = baseline_spacy.encode_tables()

    baseline_spacy.classify_tables(embeddings, labels)
