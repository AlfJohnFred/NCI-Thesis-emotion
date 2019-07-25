import os
import pandas as pd
import numpy as np
import librosa
import scipy
import csv


def get_field_names():
    fieldnames = ['filename']
    for i in range(0, 20):
        fieldnames.append('mfcc_feature_' + str(i))
    fieldnames.append('emotion_name')
    fieldnames.append('value')
    fieldnames.append('gender')
    fieldnames.append('intensity')
    return fieldnames


def get_gender(gender_bit):
    if (gender_bit % 2) == 0:
        gender = 'F'
    else:
        gender = 'M'

    return gender


def get_emotion(emotion_bit):
    emotion_name = ''
    if emotion_bit == '01':
        emotion_name = 'neutral'
    elif emotion_bit == '02':
        emotion_name = 'calm'
    elif emotion_bit == '03':
        emotion_name = 'happy'
    elif emotion_bit == '04':
        emotion_name = 'sad'
    elif emotion_bit == '05':
        emotion_name = 'angry'
    elif emotion_bit == '06':
        emotion_name = 'fearful'
    elif emotion_bit == '07':
        emotion_name = 'disgust'
    elif emotion_bit == '08':
        emotion_name = 'surprise'

    return emotion_name


def get_intensity(emotional_intensity_bit):
    if emotional_intensity_bit == '01':
        intensity = 'L'
    else:
        intensity = 'H'

    return intensity


def segregate_function(file_name_without_ext, mean, fieldnames):
    file_name_without_ext = os.path.splitext(file_name_without_ext)[0]
    emotion_bit = file_name_without_ext.split("-")[2]
    gender_bit = int(file_name_without_ext.split("-")[6])
    emotional_intensity_bit = file_name_without_ext.split("-")[3]

    emotion_name = get_emotion(emotion_bit)
    gender = get_gender(gender_bit)
    intensity = get_intensity(emotional_intensity_bit)

    if not os.path.exists('C:\\users\\dvada\\Desktop\\Dissertation\\Code\\dataset.csv'):
        with open('dataset.csv', 'w', newline='') as my_csv:
            writer = csv.DictWriter(my_csv, fieldnames=fieldnames)
            writer.writeheader()
            my_csv.close()

    with open('dataset.csv', 'a', newline='') as my_csv:
        writefile = csv.writer(my_csv)
        row = [file_name_without_ext]
        for i in range(0, 20):
            row.append(mean[i])
        row.append(emotion_name)
        row.append(emotion_bit)
        row.append(gender)
        row.append(intensity)
        writefile.writerow(row)
        my_csv.close()


def extract_features(file_name, headers):
    file_name_without_ext = os.path.splitext(file_name)[0]
    sig, rate = librosa.load(path=path+file_name, sr=SAMPLE_RATE)
    mfcc_features = librosa.feature.mfcc(sig, sr=rate)
    mean = mfcc_features.mean(axis=1)
    segregate_function(file_name_without_ext, mean, headers)


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    path = "C:\\users\\dvada\\Desktop\\Dissertation\\Data\\"
    field_names = get_field_names()
    for file_name_with_ext in os.listdir(path):
        extract_features(file_name_with_ext, field_names)

