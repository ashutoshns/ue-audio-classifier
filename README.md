# urban environment audio classification

The goal of this project is to build an audio classifier capable of recognising different sounds. The sounds we have used are ambient noises from an urban environment. Right now we have focused on 4 categories of sounds namely air conditioner, car horn, children playing and street music.

### dependencies

- This package uses os, glob, numpy, matplotlib, pandas, tensorflow, librosa(for resampling and reduction in the audio rate)
- To install these dependencies:

```
sudo apt install python-pip
sudo pip install os glob numpy matplotlib pandas tensorflow librosa
```

### step 1

- Get a good dataset, we used [UrbanSound8k](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html "UrbanSound8k Dataset"). 
- Download the dataset and place the audio files in the `audio` folder and delete the README file present there.
> please remember that all the audio files need to be in '.wav' format.

### step 2 

- Open a terminal and navigate to the project directory i.e. the directory with all the python codes.
- Ensure that you have installed all the dependencies.
- Then run the feature extraction code
```python features.py```
- This should give a several numpy array files in the folder `data`

### step 3

- Ensure that step 2 did create nimpy arrays with as many total number of rows as the number of audio files supplied and 187b columns.
- Now delete the README file in `data` folder.
- Run the training code
```python train.py```

### step 4

- Now put all your prediction test files in `samples` folder.
- Make appropriate changes to the `file_names` and `audio_names` arrays in `predict .py` and run it.
```python predict.py```

Tadaaa! You're done.
