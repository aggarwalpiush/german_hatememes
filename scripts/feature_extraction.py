#! usr/bin/env python
# -*- coding : utf-8 -*-


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from sentence_transformers import SentenceTransformer, util
import codecs
import json
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score, accuracy_score


vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)


def extract_image_features(img_paths, img_model):
    all_img_features = []
    for img_path in tqdm(img_paths):
        img = load_img(img_path, target_size=(224,224))
        img = np.array(img)
        reshaped_img = img.reshape(1,224,224,3)
        imgx = preprocess_input(reshaped_img)
        img_features = img_model.predict(imgx, use_multiprocessing=True)
        all_img_features.append(img_features.reshape((4096,)))
    return all_img_features

model = SentenceTransformer('bert-base-german-cased')

def extract_bert_features(sentence):
    return model.encode(sentence, convert_to_tensor=True)


def train_model(X_train, X_dev, Y_train, Y_dev):
    clf = Sequential()
    clf.add(Dense(1000, kernel_initializer="uniform", input_shape=(4396,)))
    clf.add(Dropout(0.5))
    clf.add(Dense(500))
    clf.add(Dropout(0.5))
    clf.add(Dense(100))
    clf.add(Dropout(0.5))
    clf.add(Dense(50))
    clf.add(Dropout(0.5))
    clf.add(Dense(2, activation="softmax"))

    clf.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    X_train = np.array(X_train)
    X_dev = np.array(X_dev)

    Y_train = to_categorical(Y_train)
    Y_dev = to_categrical(Y_dev)


    file_path = f"german_hate_meme_detection_model.hdf5"
    if not os.path.exists(file_path):
        clf.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=10, batch_size=32)
        clf.save_weights(file_path)
    else:
        clf.load_weights(file_path)
    return clf


def test_model(clf, X_test, Y_test):
    X_test = np.array(X_test)
    Y_test = to_categorical(Y_test)
    Y_pred = clf.predict(X_test)
    Y_pred = [np.argmax(pred) for pred in Y_pred]
    print("----- Classification report -----")
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    print("Accuracy score:", accuracy_score(Y_test, Y_pred))
    print("F1 score:", f1_score(Y_test, Y_pred))
    print("Precision score:", precision_score(Y_test, Y_pred))
    print("Recall score:", recall_score(Y_test, Y_pred))

if __name__ == '__main__':
    train_image_path = []
    train_text = []
    train_labels = []

    dev_image_path = []
    dev_text = []
    dev_labels = []

    test_image_path = []
    test_text = []
    test_labels = []

    print('Loading train data...')
    with codecs.open('../data/train.jsonl', 'r', 'utf-8') as r_json:
        for line in tqdm(r_json):
            entry = json.loads(line)
            train_image_path.append(os.path.join('../data/img', entry['img'].split('/')[-1]))
            train_text.append(entry['text'])
            train_labels.append(entry['label'])

    print('Loading dev data...')
    with codecs.open('../data/dev_unseen.jsonl', 'r', 'utf-8') as r_json:
        for line in tqdm(r_json):
            entry = json.loads(line)
            dev_image_path.append(os.path.join('../data/img', entry['img'].split('/')[-1]))
            dev_text.append(entry['text'])
            dev_labels.append(entry['label'])

    print('Loading test data...')
    with codecs.open('../data/de_test_second.tsv', 'r', 'utf-8') as r_tsv:
        for line in tqdm(r_tsv):
            tokens = line.split('\t')
            if len(tokens[0]) < 5:
                img_name = '0'+ str(tokens[0]) + '.png'
            else:
                img_name =  str(tokens[0]) + '.png'
            if tokens[2] == 'Hate':
                label = 1
            else:
                label = 0
            test_image_path.append(os.path.join('../data/img', img_name))
            test_text.append(tokens[1])
            test_labels.append(label)

    print('Extracting test features...')
    X_test = np.concatenate((extract_bert_features(test_text),
                              extract_image_features(test_image_path, vgg_model)), axis=0)
    Y_test = np.array(test_labels)

    print('Extracting dev features...')
    X_dev = np.concatenate((extract_bert_features(dev_text),
                              extract_image_features(dev_image_path, vgg_model)), axis=0)
    Y_dev = np.array(dev_labels)

    print('Extracting train features...')
    X_train = np.concatenate((extract_bert_features(train_text),
                              extract_image_features(train_image_path, vgg_model)), axis=0)
    Y_train = np.array(train_labels)





    print('Model Training...')
    clf = train_model(X_train, Y_dev, y_train, Y_dev)

    print('Model Evaluation...')
    test_model(clf, X_test, Y_test)



