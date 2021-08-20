import logging
import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve, auc, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, KFold
from statistics import mean, stdev
from sklearn import svm

from TaBert_table_encode import TableEncoder
from utils.classification_utils import *
from baselines_table_encode import BaselineEncoder

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import logging

BASE_DIR = os.getcwd()

rootLogger = logging.getLogger()
logging_level = logging.INFO
rootLogger.setLevel(logging_level)
beautify_logging_handlers(rootLogger, logging_level, file_name=os.path.join(BASE_DIR, 'log_info'))

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Experiment(object):
    def __init__(self, config):
        self.grid = ParameterGrid(config)
        
    def seed(self):
        torch.manual_seed(2809)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def prepare(self, config):

        columns = config['columns']
        rows = config['rows']

        if 'tabert' in config['model']:
            if rows > 1:
                model = 'tabert_base_k3'
                update_tb_config(model, rows)
            else:
                model = 'tabert_base_k1'
            context = config['context']

            table_encoder = TableEncoder(model_path=model, rows=rows, masked_col=columns, context=context)

        elif 'baseline' in config['model']:

            model = config['baseline']
            header = config['header']

            if model == 'tf-idf':
                preprocess = config['preprocess']
                table_encoder = BaselineEncoder(model, header, masked_col=columns, rows=rows, preprocess=preprocess)
            else:
                table_encoder = BaselineEncoder(model, header, masked_col=columns, rows=rows)

        return table_encoder

    def classification(self, table_encoder, k=20):
        """
        Run MLP classifer on the table embeddings to predict their class
        :param table_encoder:
        :return:
        """
        eval_res = {}
        tables, labels = table_encoder.encode_tables()

        # Cross-validation
        logging.info("Classification with cross-validation k = {}".format(k))
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(500), activation = 'tanh', max_iter=1000, solver='adam', random_state=42)  # , verbose=10)
      #  clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', probability =True, random_state=42)
        scaler = StandardScaler()
        
       # eval_res = {}
        accuracy_stratified = []
        f1_score_list = []
        precision_score_list = []
        recall_score_list = []
        roc_auc = []
        mrr = []
        hits1 = []
        hits3 = []
        hits5 = []
        hits10 = []

        # Binarize the labels
        y = label_binarize(labels, classes=list(set(labels)))
        n_classes = y.shape[1]
        #print(y)

        for train_ix, test_ix in kfold.split(tables, labels):
            x_train, x_test = tables[train_ix], tables[test_ix]
            y_train, y_test = labels[train_ix], labels[test_ix]
            
            scaler.fit(x_train)  
            x_train = scaler.transform(x_train)  
            x_test = scaler.transform(x_test) 

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_prob = clf.predict_proba(x_test)

            accuracy_stratified.append(accuracy_score(y_test, y_pred))
            f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
            recall_score_list.append(recall_score(y_test, y_pred, average='macro'))
            precision_score_list.append(precision_score(y_test, y_pred, average='macro'))

            # Compute ROC curve and area the curve
            y_train_b, y_test_b = y[train_ix], y[test_ix]

            fpr = dict()
            tpr = dict()
            n_roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true = y_test_b[:,i], y_score = y_prob[:,i])
                n_roc_auc[i] = auc(fpr[i], tpr[i])
            
                if np.isnan(n_roc_auc[i]) :
                    del n_roc_auc[i]
            
            roc_auc.append(mean(n_roc_auc.values()))

            result_evaluate = evaluate_mrr(y_prob, y_test)
            mrr.append(result_evaluate['MRR'])
            hits1.append(result_evaluate['hit@1'])
            hits3.append(result_evaluate['hit@3'])
            hits5.append(result_evaluate['hit@5'])
            hits10.append(result_evaluate['hit@10'])

    #    logging.info('Maximum Accuracy : {} \n'.format(max(accuracy_stratified)))
    #    logging.info('Minimum Accuracy : {} \n'.format(min(accuracy_stratified)))
    #    logging.info('Overall Accuracy : {} \n'.format(mean(accuracy_stratified)))
        
        eval_res['f1'] = mean(f1_score_list)
        eval_res['recall'] = mean(recall_score_list)
        eval_res['accuracy'] = mean(accuracy_stratified)
        eval_res['precision'] = mean(precision_score_list)
    #    eval_res['roc_auc'] = mean(roc_auc)
    #    eval_res['mrr'] = mean(mrr)
    #    eval_res['h1'] = mean(hits1)
    #    eval_res['h3'] = mean(hits3)
    #    eval_res['h5'] = mean(hits5)
    #    eval_res['h10'] = mean(hits10)

        with open(os.path.join(table_encoder.results, 'classification.txt'), 'a+') as f:
            f.write("Accuracy: {}, F1: {}, Recall: {}, Precision: {}, ROC_AUC: {}, MRR: {}, H@1: {}, H@3: {}, H@5: {} , H@10: {}  \n".format(mean(accuracy_stratified), mean(f1_score_list), mean(recall_score_list), mean(precision_score_list), mean(roc_auc), mean(mrr), mean(hits1), mean(hits3), mean(hits5), mean(hits10)))

        return eval_res

    def run(self):
        results = {}
        grid_unfold = list(self.grid)

        logging.info("Total number of configs {}".format(len(grid_unfold)))

        for i, params in enumerate(grid_unfold):

            logging.info("Run {}".format(i))

            for model_config in self.grid:
                current = {}
                table_encoder = self.prepare(model_config)

                current['model_name'] = table_encoder.name

                evaluate = self.classification(table_encoder)

                results[current['model_name']+'_{}'.format(i)] = evaluate

                logging.info("Model {} with results {}".format(current['model_name'], results))

        df_results = pd.DataFrame(results)#.sort_values(by=['accuracy'], ascending=False)

        df_results.to_csv('experiment_results.csv')


if __name__ == '__main__':
    # config = dict()

    config= [{'model': ['baseline'], 'baseline': ['bert'], 'header': ['concatSingle'], 'rows': [1,3,5,7],'columns': [True,False]}]
#    config = [{'model':['tabert'], 'context': ['_'], 'rows':[1,3,5,7], 'columns':[True, False]}]#{'model': ['baseline'], 'baseline': ['tf-idf'], 'header': ['asRow'], 'rows': [1,3, 5, 7], 'columns': [True, False], 'preprocess':['lemmatize']}, {'model': ['baseline'], 'baseline': ['spacy', 'word2vec', 'bert'], 'header': ['concatSingle'],'rows': [1,3,5,7],'columns': [True, False]}]
# Run this next:
#    config= [{'model': ['baseline'], 'baseline': ['bert'], 'header': ['concatSingle'], 'rows': [1,3,5,7],'columns': [False]}]
    test = Experiment(config)
    test.run()

#  print(len(list(ParameterGrid(config))))
