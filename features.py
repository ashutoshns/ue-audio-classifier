"""
Feature fold generation code
"""

import numpy as np
import glob
import os
import librosa
import stft_gen as FT
import chroma_gen as CH
import mfcc_gen as MF
import spectral_contrast_gen as SP
parent_path = 'audio/'

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    print 'Features :', len(X) ,'sampled at ', sample_rate ,'hz'
    stft1 = np.abs(FT.stft(X))
    mfccs = np.mean(MF.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    print mfccs.size
    chroma = np.mean(CH.chroma_stft(S=stft1, sr=sample_rate).T,axis=0)
    print chroma.size
    mel = np.mean(MF.melspectrogram(X, sr=sample_rate).T,axis=0)
    print mel.size
    contrast = np.mean(SP.spectral_contrast(S=stft1, sr=sample_rate).T,axis=0)
    print contrast.size
#    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
#    print tonnetz.size
    return mfccs,chroma,mel,contrast

def process_audio(parent_path,sub_dirs,file_ext='*.wav'):
    features,labels = np.empty((0,187)),np.empty((0))
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_path, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, fn.split('fold')[1].split('-')[1])
            except:
                print ("Error processing" + fn + " - skipping")
    return np.array(features), np.array(labels, dtype = np.int)

def path_check(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)
   

audio_directory = 'audio/'

def encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    encode = np.zeros((n_labels,n_unique_labels))
    encode[np.arange(n_labels), labels] = 1
    return encode
    

def save_data(data_dir):
        """fold_name = 'samples'
        print ("Saving" + fold_name)
        features = process_audio(parent_path, [fold_name])
        print ("Features of", fold_name , " = ", features.shape)
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        np.save(feature_file, features)
        print ("Saved " + feature_file)
        return features"""
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
save_dir = "data/"
path_check(save_dir)
save_data(audio_directory)
