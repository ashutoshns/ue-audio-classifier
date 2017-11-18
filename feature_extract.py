# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 10:41:31 2017

@author: ashutoshns
"""

import numpy as np
import glob
import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

#matplotlib
plt.style.use('ggplot')
parent_path = 'audio/'

def load_audio(parent_path, file_paths):
    bare_sound_files = []
    for fp in file_paths:
        X,sr = librosa.load(parent_path, fp)
        bare_sound_files.append(X)
    return bare_sound_files 

def get_labels(parent_path,sub_dirs,file_ext='*.wav'):
    labels = np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_path, sub_dir, file_ext)):
            try:
                class_value = fn.split('fold')[1].split('-')[1]
                labels = np.append(labels, class_value)
            except:
                print "Error processing" + fn + " - skipping"
    return labels

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    print "Features :", len(X) ,"sampled at ", sample_rate ,"hz"
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def process_audio(parent_path,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_path, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, fn.split('fold')[1].split('-')[1])
            except:
                print "Error processing" + fn + " - skipping"
    return np.array(features), np.array(labels, dtype = np.int)


def encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    encode = np.zeros((n_labels,n_unique_labels))
    encode[np.arange(n_labels), labels] = 1
    return encode
    
def path_check(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)
   

audio_directory = 'audio/'
subsequent_fold = False
for k in range(1,11):
    fold_name = 'fold' + str(k)
    labels = get_labels(audio_directory, [fold_name])
    if subsequent_fold:
        all_labels = np.concatenate((all_labels, labels))
    else:
        all_labels = labels
        subsequent_folds = True


def save_data(data_dir):
    for k in range(1,11):
        fold_name = 'fold' + str(k)
        print "Saving" + fold_name
        features, labels = process_audio(parent_path, [fold_name])
        labels = encode(labels)
        print "Features of", fold_name , " = ", features.shape
        print "Labels of", fold_name , " = ", labels.shape
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        np.save(feature_file, features)
        print "Saved " + feature_file
        np.save(labels_file, labels)
        print "Saved " + labels_file
audio_directory = "audio/"
save_dir = "data/us8k-np-ffn/"
path_check(save_dir)
    
save_data(save_dir)
