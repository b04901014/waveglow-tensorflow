# WAVEGLOW
This is a tensorflow implementation of [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow).
Now some samples at step 375k are at */step_375k_samples*, please wait patiently for further results, the original paper is at 580k.

## Setup
First we need python3 along with [Tensorflow](https://github.com/tensorflow/tensorflow) with gpu support, the version this repository use is **r1.12**.
Other versions may also work, but I'm not sure which version will have error.

We also need:
 - [tqdm](https://github.com/tqdm/tqdm)
 - [librosa](https://github.com/librosa/librosa)

You can also setup the environment by the Dockerfile in the repository.
However, I build the tensorflow r1.12 from source to specify the cuda version to be 9.2, thus it may take much more time to setup the docker image.
> docker build -t {IMAGE\_NAME\_YOU\_LIKE} .

## Dataset Preparation
For getting the dataset prepared, first **adjust the *Input Path* section of *src/hparams.py* **to our dataset path, then run:
```
cd src
python3 dataset/procaudio.py
```
The *metadata.csv* described in *src/hparams.py* is as the following format:
```
Audio Name without extension|Text only for notation|True Text
```

Example:
```
LJ001-0008|has never been surpassed.|has never been surpassed.
LJ001-0009|Printing, then, for our purpose, may be considered as the art of making books by means of movable types.|Printing, then, for our purpose, may be considered as the art of making books by means of movable types.
```

We take this format as input since we use the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) as our training data.
It's *metadata.csv* is exactly this format and it's used for TTS.
If you are training on your own dataset, you can **pad the text part since the vocoder will not use it**:
```
audio1|deadbeef|deadbeef
audio2|deadbeef|deadbeef
audio3|deadbeef|deadbeef
```
Then there should be audio1.wav, audio2.wav and audio3.wav in the corresponding *dataset\_dir* you specified in *src/hparams.py*


All audio files should be in wav format.

## Training
To start training, run:
```
cd src
python3 main.py --use_weight_norm --truncate_sample
```

The configurations, hyperparams and descriptions are in *src/hparams.py*

## TODO
 - Inference is not tested
 - Adding loss curve
 - Multiprocess Dataset Preparation

## References
 - [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)
 - [openai/glow](https://github.com/openai/glow)
 - [tensorflow/docker](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu)
