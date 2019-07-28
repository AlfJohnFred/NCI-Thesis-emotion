import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def mfcc_only_svm(input_dataset):
    X = input_dataset.drop('emotion_name', axis=1)
    y = input_dataset['emotion_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_classifier_poly = SVC(kernel='poly', degree=8)

    # SVM with polynomial kernel
    svm_classifier_poly.fit(X_train, y_train)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    csv_classification = pd.DataFrame(classification_report(y_test, y_pred_poly, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_only_classification_poly.csv',  sep=',')


def feature_gender_svm(input_dataset):
    X = input_dataset.drop('emotion_name', axis=1)
    y = input_dataset['emotion_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_classifier_poly = SVC(kernel='poly', degree=8)

    # SVM with polynomial kernel
    svm_classifier_poly.fit(X_train, y_train)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    csv_classification = pd.DataFrame(classification_report(y_test, y_pred_poly, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_gender_classification_poly.csv',  sep=',')


def feature_gender_intensity_svm(input_dataset):
    X = input_dataset.drop('emotion_name', axis=1)
    y = input_dataset['emotion_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_classifier_poly = SVC(kernel='poly', degree=8)

    # SVM with polynomial kernel
    svm_classifier_poly.fit(X_train, y_train)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    csv_classification = pd.DataFrame(classification_report(y_test, y_pred_poly, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_gender_intensity_classification_poly.csv',  sep=',')


if __name__ == '__main__':
    full_data = pd.read_csv('dataset.csv')
    mfcc_only_data = full_data.drop(['filename', 'value', 'gender', 'intensity'], axis=1)
    feature_gender_data = full_data.drop(['filename', 'value', 'intensity'], axis=1)
    feature_gender_intensity_data = full_data.drop(['filename', 'value'], axis=1)
    mfcc_only_svm(mfcc_only_data)
    feature_gender_svm(feature_gender_data)
    feature_gender_intensity_svm(feature_gender_intensity_data)

