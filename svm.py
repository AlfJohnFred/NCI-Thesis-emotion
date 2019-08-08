import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def feature_only_svm(input_dataset):
    X = input_dataset.drop('emotion_name', axis=1)
    y = input_dataset['emotion_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_classifier_poly = SVC(kernel='poly', degree=8)
    # SVM with polynomial kernel
    svm_classifier_poly.fit(X_train, y_train)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    print(confusion_matrix(y_test, y_pred_poly))
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
    print(confusion_matrix(y_test, y_pred_poly))
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
    print(confusion_matrix(y_test, y_pred_poly))
    csv_classification = pd.DataFrame(classification_report(y_test, y_pred_poly, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_gender_intensity_classification_poly.csv',  sep=',')


if __name__ == '__main__':
    full_data = pd.read_csv('maxFeatures.csv')
    no_high = full_data[full_data.intensity != 1]
    feature_only_data = no_high.drop(['filename', 'value', 'gender', 'intensity'], axis=1)
    feature_gender_data = no_high.drop(['filename', 'value', 'intensity'], axis=1)
    no_low = full_data[full_data.intensity != 0]
    feature_gender_intensity_data = no_low.drop(['filename', 'value'], axis=1)
    feature_only_svm(feature_only_data)
    feature_gender_svm(feature_gender_data)
    feature_gender_intensity_svm(feature_gender_intensity_data)

