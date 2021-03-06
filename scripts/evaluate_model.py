#! ust/bin/env python
# -*- coding : utf-8 -*-


import codecs
from glob import glob
import os
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix



def import_predict_file(predict_file_path):
    pf = pd.read_csv(predict_file_path, delimiter = ',', header = 0)
    return pf


def import_gold_file(gold_file_path):
    gf = pd.read_json(gold_file_path, lines=True)
    return gf

def evaluate_model(predict_file_path, gold_file_path):
    reference =  import_gold_file(gold_file_path)
    predictions = import_predict_file(predict_file_path)
    eval_dataset = predictions.merge(reference, on='id')

    y_pred = eval_dataset.label_x
    y_gold = eval_dataset.label_y
    roc_auc = roc_auc_score(y_gold, eval_dataset.proba, average="micro", multi_class="ovr")
    f1 = f1_score(y_gold, y_pred, average="macro")
    target_names = ['Non-Hate', 'Hate']

    print('roc_auc_score: %s' %roc_auc)
    print('f1(macro): %s' % f1)
    print(classification_report(y_gold, y_pred, target_names=target_names))
    print(confusion_matrix(y_gold, y_pred))

def main():
    pred_file = sys.argv[1]
    gold_file = sys.argv[2]

    evaluate_model(pred_file, gold_file)


if __name__ == '__main__':
    main()


