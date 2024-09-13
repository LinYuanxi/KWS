# TorchKWS 

# Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [ConvMixer](#convmixer)


# Implementations
## About DataSet
[Speech Commands DataSet](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) is a set of one-second .wav audio files, each containing a single spoken English word.
These words are from a small set of commands, and are spoken by a variety of different speakers.
The audio files are organized into folders based on the word they contain, and this dataset is designed to help train simple machine learning models.

## Installation
We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```python
cd <ROOT>/dataset
python process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset>\
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --rebalance \
    --log
```

## ConvMixer
_ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-field Keyword Spotting_
[[Paper]](https://arxiv.org/abs/2201.05863) [[Code]](networks/convmixer.py)


# Reference
6. https://github.com/dianwen-ng/Keyword-Spotting-ConvMixer
