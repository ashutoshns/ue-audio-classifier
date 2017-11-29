import numpy as np
import os
import librosa
import stft_gen
import mfcc_gen
import chroma_gen
import spectral_contrast_gen
from train import model


#extract feature numpy array from eac audio file fo prediction
def extract_feature(audio_directory,file_name):
    features= np.empty((0,187))
    fn=audio_directory+'/'+file_name
    try:
        X, sample_rate = librosa.load(fn)
        print ("Features :", len(X) ,"sampled at ", sample_rate ,"hz")
        sft = np.abs(stft_gen.stft(X))
        mfccs = np.mean(mfcc_gen.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        chroma = np.mean(chroma_gen.chroma_stft(S=sft, sr=sample_rate).T,axis=0)
        mel = np.mean(mfcc_gen.melspectrogram(X, sr=sample_rate).T,axis=0)
        contrast = np.mean(spectral_contrast_gen.spectral_contrast(S=sft, sr=sample_rate).T,axis=0)
        features = np.hstack([mfccs,chroma,mel,contrast])
    except:
        print ("Error processing" + fn + " - skipping")
    
    return features

audio_directory = "samples"


#making the predictions
file_names=["aircon1.wav","aircon2.wav","carhorn1.wav","carhorn2.wav","carhorn3.wav","child1.wav","child2.wav","music1.wav","music2.wav"]
sound_names = ["air conditioner","car horn","child","street music"]
for s in range(len(file_names)):
    file_name=file_names[s]
    print("---"+file_name+"---")
    features = extract_feature(audio_directory, file_name)
    features=features.reshape((1,193))
    predictions = model.predict(features)
    ind = np.argpartition(predictions[0], -2)[-2:]
    ind[np.argsort(predictions[0][ind])]
    ind = ind[::-1]
    print ("Top guess: ", sound_names[ind[0]], " (",round(predictions[0,ind[0]],3),")")
    print ("2nd guess: ", sound_names[ind[1]], " (",round(predictions[0,ind[1]],3),")\n")
