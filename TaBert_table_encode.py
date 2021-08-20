import numpy as np
import glob
from table_bert import TableBertModel, Table, Column
from utils.classification_utils import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from statistics import mean, stdev
import logging
import random
import string

BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'experiment_scaled_MLP')
DATASET_TABLES_PATH = os.path.join(BASE_DIR, 'data/datasets/tables_classification/tables_class/') # 684 tables
#LABELS_FILE_PATH = os.path.join(BASE_DIR, 'data/datasets/tables_classification/classes_complete.csv') # 684 tables with classes

#DATASET_TABLES_PATH = os.path.join(BASE_DIR, 'data/datasets/extended_instance_goldstandard/tables_v2/') # 223 tables
LABELS_FILE_PATH = os.path.join(BASE_DIR, 'data/datasets/extended_instance_goldstandard/classes_GS.csv') # 223 tables with classes

# configuration for logging
rootLogger = logging.getLogger()
logging_level = logging.INFO
rootLogger.setLevel(logging_level)
beautify_logging_handlers(rootLogger, logging_level, file_name=os.path.join(RESULTS_DIR, 'log_info'))

class TableEncoder():

    def __init__(self, model_path, rows=1, masked_col=False, context = 'class', dataset = DATASET_TABLES_PATH, labels =LABELS_FILE_PATH, results = RESULTS_DIR):
        super().__init__()
        self.name = '{}'.format(rows) + '_masked_' + '{}'.format(masked_col)+ '_context_'+ '{}'.format(context)
        self.model = TableBertModel.from_pretrained(model_name_or_path=os.path.join(model_path, 'model.bin'))
        self.masked_col = masked_col
        self.context = context
        self.tables, self.labels_dict = self.load_tables(dataset, labels, masked_col=self.masked_col)
        self.results = os.path.join(results, 'TaBERT', self.name)
        os.makedirs(self.results, exist_ok=True)

    def create_table(self, dataframe, t_id):

        """
        For a dataframe, separate the resources and the column names in two lists to be passed as input to generate object Table.
        The csv file name is saved as the table id - it will be used for finding the class of the table.
        """
        data = []
        header = []
        col_datatype = {}
        for index, row in dataframe.iterrows():
            data.append(row.to_list())
        #for col in dataframe.columns:
        #    header.append(col)

        for c in range(dataframe.shape[1]):
            if dataframe.dtypes[c] == 'int64':
                col_datatype[c] = 'int'
            elif dataframe.dtypes[c] == 'float64':
                col_datatype[c] = 'float'
            elif dataframe.dtypes[c] == 'datetime64':
                col_datatype[c] = 'datetime'
            else:
                col_datatype[c] = 'text'

        table = Table(
            id=t_id,            
            header=[Column(dataframe.columns[col], dtype, sample_value = dataframe.iloc[:, col].values[0]) for col, dtype in col_datatype.items()],
            data=data
        ).tokenize(self.model.tokenizer)

        return table

    def load_tables(self, dataset_tables=DATASET_TABLES_PATH, labels=LABELS_FILE_PATH, masked_col=False):
        """
        Load the csv files into pandas dfs, change the column names if indicated with masked_col
        Create table where the name of the table is the name of the csv file
        :param dataset_tables: path to folder with table csv files
        :param labels: path to csv file with tables and their classes
        :param masked_col: flag to mask or not the column names
        :return: tables - list with Table objects, labels_dict - {table_name:class}
        """
        
        
        logging.info("Loading tables with masked columns {}".format(masked_col))
        labels_dict = load_labels(labels_file = labels)

        tables = []
        for filename in glob.glob(dataset_tables + '*.csv'):
            name = os.path.basename(os.path.normpath(filename)).strip('.csv')
            if name in labels_dict.keys():
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False)
                if masked_col == True:
                    df.rename(columns=lambda x: '[UNK]', inplace=True)
                else:
                    pass
                t = self.create_table(df, name)
                tables.append(t)

        if self.context == 'fake':
            logging.info("Updating labels")
            labels_dict = update_labels(labels_dict)
            
        logging.info("Created {} tables ".format(len(tables)))
        logging.info("With labels {} ".format(len(labels_dict)))       

        return tables, labels_dict

    def encode_tables(self):
        """
        Encode the tables from the list tables with the specified model
        :
        :return: dict {table_id: table_embedding}
        """
        logging.info("Encoding tables with class as context {}".format(self.context))
        tables_emb = {}
        for t in self.tables:
            if self.context == 'class':
                context = self.labels_dict[t.id]
            elif self.context == 'noise':
                context = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            elif self.context == 'Thing':
                context = 'Thing'
            elif self.context == 'fake':
                context = self.labels_dict[t.id]
            else:
                context = ''
           # logging.info("Context {}".format(context))
            context_encoding, column_encoding, info_dict = self.model.encode(
                contexts=[self.model.tokenizer.tokenize(context)],
                tables=[t])

            table_encoding = reshape_column(column_encoding)
            tables_emb[t.id] = table_encoding

        save_embeddings(tables_emb, self.results)
        # Get the labels for the encoded tables

        table_embeddings, labels = self.get_tables_labels(tables_emb)

        # Transform labels to int and to array
        labels, class2idx = label_to_int(labels)
        labels = np.array(labels)
        tables = np.concatenate(list(table_embeddings.values()))

        return tables, labels

    def encode_n_classes(self):
        """
        Encode all of the tables with each of the 90 classes as context
        :return:
        """
        classes = set(self.labels_dict.values())
        logging.info(" All the classes: {}".format(classes))
        tables_emb = {}
        for t in self.tables:
            logging.info("Start encoding of table {}".format(t.id))
            for c in classes:
                context = c
                context_encoding, column_encoding, info_dict = self.model.encode(
                    contexts=[self.model.tokenizer.tokenize(context)],
                    tables=[t])

                table_encoding = reshape_column(column_encoding)

                if t.id in tables_emb.keys():
                    tables_emb[t.id].append(table_encoding)
                else:
                    tables_emb[t.id] = [table_encoding]

        save_embeddings(tables_emb, self.results, file_name='n_classes_embed')

        return tables_emb


    def get_tables_labels(self, tables_embeddings):
        """
        Get the labels for the table embeddings
        :param tables_embeddings: dict {table_id : table_embedding}
        :return: table_embeddings - dict {table_id : table_embedding}
                 labels - dict {table_id : class of the table}
        """

        logging.info("Embeddings for {} tables".format(len(tables_embeddings)))
        labels = {}
        to_remove = []
        for key in tables_embeddings.keys():
            if key in self.labels_dict:
                labels[key] = self.labels_dict[key]
            else:
                to_remove.append(key)

        logging.info("To remove {}".format(to_remove))
        tables_embeddings = {key: tables_embeddings[key] for key in tables_embeddings if key not in to_remove}

        return tables_embeddings, labels



    def classify_tables(self, table_embeddings, table_labels, k = 5):
        """
        Run MLP classifer on the table embeddings to predict their class
        :param table_embeddings:  dict {table_id : table embedding}
        :return:
        """
        # Transform labels to ints
        labels, class2idx = label_to_int(table_labels)
        tables = np.concatenate(list(table_embeddings.values()))
        labels = np.array(labels)
        # Cross-validation
        logging.info("Classification with cross-validation k = {}".format(k))
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        scaler = StandardScaler()
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, solver='adam')  # , verbose=10)
        accuracy_stratified = []
        # enumerate the splits and summarize the distributions
        for train_ix, test_ix in kfold.split(tables, labels):

            x_train, x_test = tables[train_ix], tables[test_ix]
            y_train, y_test = labels[train_ix], labels[test_ix]

            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            clf.fit(x_train, y_train)
           # y_pred = clf.predict(x_test)

            accuracy_stratified.append(clf.score(x_test, y_test))

        logging.info('Maximum Accuracy : {} \n'.format(max(accuracy_stratified)))
        logging.info('Minimum Accuracy : {} \n'.format(min(accuracy_stratified)))
        logging.info('Overall Accuracy : {} \n'.format(mean(accuracy_stratified)))

        with open(os.path.join(self.results, 'classification.txt'), 'a+') as f:
            f.write("{} accuracy \n".format(mean(accuracy_stratified)))

        #plot_confusion_matrix(y_test, y_pred, class2idx, self.results)

        #return acc_score, train_score, test_score


if __name__ == '__main__':


    tabert_table = TableEncoder(model_path = 'tabert_base_k1', rows=1, masked_col=False, context = 'class')

    n_embeddings = tabert_table.encode_n_classes()

   # tables_embeddings, table_labels = tabert_table.encode_tables()

   # tabert_table.classify_tables(tables_embeddings, table_labels, k = 10)








