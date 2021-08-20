import torch
import pandas as pd
import numpy as np
import os
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import collections
import pickle
import random
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from pytorch_pretrained_bert import logging as hf_logging
hf_logging.set_verbosity_error()

import logging
import logging.handlers
module_log = logging.getLogger(__name__)


BASE_DIR = os.getcwd()
DATASET_TABLES_PATH = os.path.join(BASE_DIR, 'data/datasets/tables_classification/tables_class/')
LABELS_FILE_PATH = os.path.join(BASE_DIR,'data/datasets/tables_classification/classes_complete.csv')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MODEL = BertForMaskedLM.from_pretrained("bert-base-uncased")

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def evaluate_mrr(y_predicts, y_true):
  #  print('in MRR')
    mrr_list = []
    metrics_output = {'MRR': 0, 'hit@1': 0, 'hit@3': 0, 'hit@5': 0, 'hit@10': 0}

    for i, ys in enumerate(y_true):

        predicts = y_predicts[i]
        rank = list(np.argsort(-np.array(predicts))).index(ys)
        # print(rank)
        rank = rank + 1

        mrr_i = 1 / rank
        mrr_list.append(mrr_i)
        # print(mrr_list)

        if rank <= 1:
            metrics_output['hit@1'] += 1
        if rank <= 3:
            metrics_output['hit@3'] += 1
        if rank <= 5:
            metrics_output['hit@5'] += 1
        if rank <= 10:
            metrics_output['hit@10'] += 1

    mrr = np.sum(mrr_list)
    metrics_output['MRR'] = mrr

    for key, val in metrics_output.items():
        metrics_output[key] = float(val / len(y_true))

    return metrics_output


def update_labels(labels_dict):    
    logging.info("Updating labels with fake ones")
    class_map ={'Building': 'Monarch',
 'Lake': 'Country',
 'Scientist': 'Museum',
 'RadioStation': 'Saint',
 'BaseballPlayer': 'Building',
 'Airline': 'Animal',
 'City': 'Person',
 'PoliticalParty': 'AcademicJournal',
 'AcademicJournal': 'BaseballPlayer',
 'Currency': 'Hospital',
 'Mountain': 'City',
 'Bird': 'Wrestler',
 'Election': 'RadioStation',
 'Film': 'Lake',
 'Country': 'Company',
 'Plant': 'PoliticalParty',
 'Monarch': 'Film',
 'Saint': 'Election',
 'VideoGame': 'Currency',
 'Newspaper': 'Airline',
 'Airport': 'Mountain',
 'Wrestler': 'Airport',
 'Animal': 'VideoGame',
 'Museum': 'Newspaper',
 'Person': 'Scientist',
 'Hospital': 'Plant',
 'Company': 'Bird'}
    
    for k, val in labels_dict.items():
        labels_dict[k] = class_map[val]
        
    return labels_dict
   

def check_max_leng(df):
    
    df_values = df.values.tolist()
    max_len = 0
    for row in df_values:
        row = ' '.join(map(str, row))
        input_ids = TOKENIZER.encode(row, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    #  print('Max sentence length: ', max_len)
    return max_len

def bert_sequence(data, max_sequence_len):
    """ For a given string, generate vector using the BERT model.
    The max_sequence_len is the max len of the row tokens"""

   # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   # bert_model = BertModel.from_pretrained("bert-base-uncased")

    encoded_dict = TOKENIZER.encode_plus(
        data,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        truncation=True,
        max_length=max_sequence_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    encoded_dict['token_type_ids'] = torch.ones((1, max_sequence_len), dtype=int)
    #position_ids = torch.zeros((1, max_sequence_len),dtype = int)

    with torch.no_grad():
        vector = BERT_MODEL(input_ids=encoded_dict['input_ids'], token_type_ids=encoded_dict['token_type_ids'], attention_mask=encoded_dict['attention_mask'], output_hidden_states=True)
   
    hidden_states = vector[1]
    # get last four layers
    last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
    
    # cast layers to a tuple and concatenate over the second dimension
    cat_hidden_states = torch.cat(tuple(last_four_layers), dim=1)
   
    data_encoding = torch.mean(cat_hidden_states, dim=1).squeeze()

   # hidden_state = vector[1][1]   
   # data_encoding = torch.mean(hidden_state, dim=1).squeeze()
    
    #data_encoding = torch.squeeze(vector[1][0], dim=0)
    data_encoding = data_encoding.detach().numpy()   

    return data_encoding

def get_tables_labels(tables_embeddings, labels_dict):
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
        if key in labels_dict:
            labels[key] = labels_dict[key]
        else:
            to_remove.append(key)

    tables_embeddings = {key: tables_embeddings[key] for key in tables_embeddings if key not in to_remove}

    return tables_embeddings, labels


def restrict_classes(labels_dict):
    """
    Remove from the dataset classes which have only 1 representative.
    Remove these entries from the labels_dict and from the table_embeddings
    :param tables_embeddings: dict {table_id : table embedding}
    :return:
    """

    df_labels = pd.DataFrame.from_dict(labels_dict, orient='index')
    df_labels = df_labels.reset_index()
    df_labels.rename(columns={"index": "table", 0: "class"}, inplace=True)
    df_labels['count'] = df_labels.groupby(['class'])['table'].transform('count')
    df_labels = df_labels[df_labels["count"] > 1]

    labels_dict = dict(zip(df_labels['table'], df_labels['class']))

  #  tables_emb, labels = get_tables_labels(tables_embeddings, labels_dict)
    logging.info("After removing classes, num of tables {}, num of labels {}".format(len(labels_dict), len(np.unique(list(labels_dict.values())))))

    return labels_dict


def load_labels(labels_file):
    """
    Read the csv file with the labels for the tables.
    Remove all the tables which are of class 'Thing' and remove all the tables which are only one representative of a class.
    :param labels_file:
    :return:
    """
    labels = pd.read_csv(labels_file, names=['table', 'class', 'uri', 'unk'])
    labels = labels[['table', 'class']]
    labels['table'] = labels['table'].str.strip('.tar.gz')
    labels['count'] = labels.groupby(['class'])['table'].transform('count')

    labels = labels[labels["class"] != 'Thing']
    labels = labels[labels["count"] > 1]

    labels_dict = labels.set_index('table')['class'].to_dict()

    return labels_dict


def reshape_column(column_encoding):

    column_encoding = torch.mean(column_encoding, dim=1)
    column_encoding = column_encoding.detach().numpy()

    return column_encoding


def label_to_int(labels_dict):
    """
    From the unique labels in the dataset, create a mapping from a class to an integer.
    :param labels_dict: dict {table_id : class}
    :return: labels_train - list of integers which represent the table classes
             class2idx - dict {class_original : int}
    """
   # logging.info("Changing classes to integers ")
    unique = set(labels_dict.values())

   # logging.info("Number of classes {}".format(len(unique)))
    class2idx = {}
    labels_int = {}

    # dict with class to int mapping
    for i, label in enumerate(unique):
        class2idx[label] = i

    # changing the class to integers
    # labels_int is dict {table_id : class_integer}
    for key, value in labels_dict.items():
        labels_int[key] = class2idx[value]

    labels_train = list(labels_int.values())

 #   logging.info("Number of labels for training {}".format(len(labels_train)))

    return labels_train, class2idx

def labels_to_original(labels, mapping):
    """
    Change classes from integers to the original strings
    :param labels: list with int classes
    :param mapping: dict {class : int}
    :return: list with original classes
    """
    # flip dict to make the keys integers
    mapping_labels = {v: k for k, v in mapping.items()}
    original_labels = []
    for label in labels:

        original_labels.append(mapping_labels[label])

    return original_labels


def plot_confusion_matrix(y_test, y_pred, class2idx, path):
    """
    Calculate the confusion matrix for the true and predicted classes. Plot the result and save
    :param y_test: true classes
    :param y_pred: predicted classes
    :param class2idx: dict {class : int}
    :param path: where to save
    :return:
    """
    all_labels_int = list(y_test + list(y_pred))
    int_classes = [key for key, value in collections.Counter(all_labels_int).most_common()]

    # labels from int to the original strings for plot

    original_y_test = labels_to_original(y_test, class2idx)
    original_y_pred = labels_to_original(y_pred, class2idx)
    all_labels = list(original_y_test + original_y_pred)
    unique_labels = [key for key, value in
                     collections.Counter(all_labels).most_common()]  # classes in y_test and y_pred

    cm_int = confusion_matrix(y_test, y_pred, labels=int_classes, normalize='all')
    # sns.heatmap(np.log(cm_int + 1e-15))

    # plot
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_int)
    fig.colorbar(cax)
    # add labels to plot
    plt.xlabel("Predicted")
    plt.ylabel("True")

    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)

    plt.savefig(os.path.join(path, "confusion_mtx.png"))


def save_embeddings(embed_dict, folder_path, file_name = 'embeddings'):
    with open(os.path.join(folder_path, '{}.pkl'.format(file_name)), 'wb') as f:
        pickle.dump(embed_dict, f, pickle.HIGHEST_PROTOCOL)


def update_tb_config(model, rows):
    with open(os.path.join(BASE_DIR, model, 'tb_config.json'), 'r') as tb_config_file:
        data = json.load(tb_config_file)
    data['sample_row_num'] = rows
    with open(os.path.join(BASE_DIR, model, 'tb_config.json'), 'w') as tb_config_file:
        json.dump(data, tb_config_file)




def transform_tsne(self, tables_embeddings):
    """
    Transform the embeddings to 2d
    """

    all_tables = np.concatenate(list(tables_embeddings.values()))
    tables_ids = list(tables_embeddings.keys())

    labels_tables = []
    for key in tables_embeddings.keys():
        t_class = tables_embeddings[key]
        labels_tables.append(t_class)

    tsne_model = TSNE(perplexity=20, n_components=2, n_iter=1000)
    embeddings_vis = pd.DataFrame(tsne_model.fit_transform(all_tables), index=tables_ids)
    embeddings_vis['tab_id'] = embeddings_vis.index
    embeddings_vis['class'] = labels_tables
    embeddings_vis = embeddings_vis.rename(columns={0: 'x', 1: 'y'})

    return embeddings_vis


# def cluster_KMeans(self, df, K=10):
#
#     embeddings = df[['x', 'y']]
#     table_ids = list(df.index)
#
#     kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
#     labels = kmeans.labels_
#
#     cluster = pd.DataFrame(labels, index=table_ids)
#     cluster['index'] = cluster.index
#     cluster = cluster.rename(columns={0: 'cluster'})
#
#     cluster_tables = pd.merge(df, cluster, left_on='tab_id', right_on='index').drop(columns=['index'])
#     cluster_tables['cluster'] = cluster_tables['cluster'].astype(str)
#
#     return cluster_tables

# def plot_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=13):
#
#     df_cm = pd.DataFrame(
#         confusion_matrix, index=class_names, columns=class_names,
#     )
#     #print(df_cm)
#     try:
#         heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
#     except ValueError:
#         raise ValueError("Confusion matrix values must be integers.")
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#     axes.set_ylabel('True label')
#     axes.set_xlabel('Predicted label')
#     axes.set_title(class_label)

################################################  baselines utils #############################################################

class LemmaTokenizer:
    """
    Tokenizer + lemmatizer for tfidf vectorizer
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, data):
        return [self.wnl.lemmatize(word) for word in word_tokenize(data)]


class StemTokenizer():
    """
    Stemmer for preprocessing data for tfidf vectorizer
    """
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
    def __call__(self, data):
        return [self.stemmer.stem(word) for word in word_tokenize(data)]


def sample_rows(dataframes, k=3):
    """
    Sampling rows from dataframes for baseline models.
    If k < 8, then we are not sampling, instead take the whole dataframe
    :param dataframes:
    :param k: how many rows to be sampled
    :return:
    """
    logging.info("Sampling k: {} rows".format(k))
    sampled_dfs = []
    for df in dataframes:
        sampled = df[:k]
        sampled_dfs.append(sampled)

    return sampled_dfs

def table_to_string(sampled_dfs):

    logging.info("Transforming tables to strings")

    strings_table = []
    for df in sampled_dfs:
        df = df.fillna('')
        df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)
        df = df.to_string(index=False).strip()
        df = re.sub('\s+',' ',df)

        strings_table.append(df)

    return strings_table


def beautify_logging_handlers(logger, logging_level=logging.INFO, file_name=None):
    """
    Change formatters for existing handlers of the logger.
    If none exist, then add a console handler.
    :param logger: should be root logger
    :param logging_level: e.g. logging.INFO or logging.DEBUG
    :param file_name: whether you wish logs to be written to the file system;
    by default None is provided then no logs are written to file
    :return:
    """
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(module)s [%(funcName)s]: %(message)s",
                                     "%Y-%m-%d %H:%M:%S")
    if logger.handlers:
        for h in logger.handlers:
            h.setLevel(logging_level)
            h.setFormatter(logFormatter)
    else:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        ch.setFormatter(logFormatter)
        logger.addHandler(ch)

    if file_name:
        fileHandler = logging.handlers.RotatingFileHandler('{}.log'.format(file_name), mode='a',
                                                           maxBytes=0.5 * 10 ** 9,
                                                           backupCount=5)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)

    # silence some very verbose loggers like werkzeug, pika, sparql
    for log_name in ['werkzeug', 'pika', 'sparql', 'py4j']:
        slogger = logging.getLogger(log_name)
        slogger.setLevel(logging.WARN)
        if slogger.handlers:
            for h in slogger.handlers:
                h.setFormatter(logFormatter)
