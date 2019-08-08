import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels + 2))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode = np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode


def create_model(n_hidden_units_1, n_hidden_units_2, n_hidden_units_3, n_hidden_units_4, n_classes, n_dim,
                 activation_function='relu', init_type='normal', optimiser='adamax', dropout_rate=0.2, ):
    model = Sequential()
    # layer 1
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, init=init_type, activation=activation_function))
    # layer 2
    model.add(Dense(n_hidden_units_2, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(n_hidden_units_3, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer4
    model.add(Dense(n_hidden_units_4, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # output layer
    model.add(Dense(n_classes, init=init_type, activation='softmax'))
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['categorical_accuracy'])
    return model


def ann_feature(input_data):
    labels = input_data['value']
    np.save('X', input_data.drop('value', axis=1))
    # one hot encoding labels
    labels_one = one_hot_encode(labels)
    np.save('y', labels_one)

    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    n_dim = train_x.shape[1]
    n_classes = train_y.shape[1]
    n_hidden_units_1 = n_dim
    n_hidden_units_2 = 400  # approx n_dim * 2
    n_hidden_units_3 = 200  # half of layer 2
    n_hidden_units_4 = 100

    # create the model
    model = create_model(n_hidden_units_1, n_hidden_units_2, n_hidden_units_3, n_hidden_units_4, n_classes, n_dim)
    # train the model
    history = model.fit(train_x, train_y, epochs=400, batch_size=4)
    # plot metrics
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Accuracy over number of Epochs')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('Metrics\\featureepoch.png')
    # predicting from the model
    predict = model.predict(test_x, batch_size=4)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # predicted emotions from the test set
    y_pred = np.argmax(predict, 1)
    predicted_emo = []
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_pred[i]]
        predicted_emo.append(emo)

    actual_emo = []
    y_true = np.argmax(test_y, 1)
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_true[i]]
        actual_emo.append(emo)
    csv_classification = pd.DataFrame(classification_report(actual_emo, predicted_emo, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_only_dnn.csv', sep=',')
    # generate the confusion matrix
    cm = confusion_matrix(actual_emo, predicted_emo)
    print(cm)
    index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    cm_df = pd.DataFrame(cm, index, columns)
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(cm_df, annot=True)
    fig = heatmap.get_figure()
    fig.savefig("Metrics\\featuresHeat.png")


def ann_feature_gender(input_data):
    labels = input_data['value']
    np.save('X', input_data.drop('value', axis=1))
    # one hot encoding labels
    labels_one = one_hot_encode(labels)
    np.save('y', labels_one)

    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    n_dim = train_x.shape[1]
    n_classes = train_y.shape[1]
    n_hidden_units_1 = n_dim
    n_hidden_units_2 = 400  # approx n_dim * 2
    n_hidden_units_3 = 200  # half of layer 2
    n_hidden_units_4 = 100

    # create the model
    model = create_model(n_hidden_units_1, n_hidden_units_2, n_hidden_units_3, n_hidden_units_4, n_classes, n_dim)
    # train the model
    history = model.fit(train_x, train_y, epochs=400, batch_size=4)
    # plot metrics
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Accuracy over number of Epochs')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('Metrics\\featureGenderepoch.png')
    # predicting from the model
    predict = model.predict(test_x, batch_size=4)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # predicted emotions from the test set
    y_pred = np.argmax(predict, 1)
    predicted_emo = []
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_pred[i]]
        predicted_emo.append(emo)

    actual_emo = []
    y_true = np.argmax(test_y, 1)
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_true[i]]
        actual_emo.append(emo)
    csv_classification = pd.DataFrame(classification_report(actual_emo, predicted_emo, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_gender_dnn.csv', sep=',')
    # generate the confusion matrix
    cm = confusion_matrix(actual_emo, predicted_emo)
    print(cm)
    index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    cm_df = pd.DataFrame(cm, index, columns)
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(cm_df, annot=True)
    fig = heatmap.get_figure()
    fig.savefig("Metrics\\featuresGenderHeat.png")


def ann_feature_gender_intensity(input_data):
    labels = input_data['value']
    np.save('X', input_data.drop('value', axis=1))
    # one hot encoding labels
    labels_one = one_hot_encode(labels)
    np.save('y', labels_one)

    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    n_dim = train_x.shape[1]
    n_classes = train_y.shape[1]
    n_hidden_units_1 = n_dim
    n_hidden_units_2 = 400  # approx n_dim * 2
    n_hidden_units_3 = 200  # half of layer 2
    n_hidden_units_4 = 100

    # create the model
    model = create_model(n_hidden_units_1, n_hidden_units_2, n_hidden_units_3, n_hidden_units_4, n_classes, n_dim)
    # train the model
    history = model.fit(train_x, train_y, epochs=400, batch_size=4)
    # plot metrics
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Accuracy over number of Epochs')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('Metrics\\featureGenderIntepoch.png')
    # predicting from the model
    predict = model.predict(test_x, batch_size=4)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # predicted emotions from the test set
    y_pred = np.argmax(predict, 1)
    predicted_emo = []
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_pred[i]]
        predicted_emo.append(emo)

    actual_emo = []
    y_true = np.argmax(test_y, 1)
    for i in range(0, test_y.shape[0]):
        emo = emotions[y_true[i]]
        actual_emo.append(emo)

    csv_classification = pd.DataFrame(classification_report(actual_emo, predicted_emo, output_dict=True)).transpose()
    csv_classification.to_csv('Metrics\\feature_gender_intensity_dnn.csv', sep=',')
    # generate the confusion matrix
    cm = confusion_matrix(actual_emo, predicted_emo)
    print(cm)
    index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    cm_df = pd.DataFrame(cm, index, columns)
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(cm_df, annot=True)
    fig = heatmap.get_figure()
    fig.savefig("Metrics\\featuresGenderIntHeat.png")


if __name__ == '__main__':
    full_data = pd.read_csv('maxFeatures.csv')
    no_high = full_data[full_data.intensity != 1]
    feature_only_data = no_high.drop(['filename', 'emotion_name', 'gender', 'intensity'], axis=1)
    feature_gender_data = no_high.drop(['filename', 'emotion_name', 'intensity'], axis=1)
    no_low = full_data[full_data.intensity != 0]
    feature_gender_intensity_data = no_low.drop(['filename', 'emotion_name'], axis=1)
    ann_feature(feature_only_data)
    ann_feature_gender(feature_gender_data)
    ann_feature_gender_intensity(feature_gender_intensity_data)
