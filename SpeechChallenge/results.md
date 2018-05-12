# Convolutional Neural Networks for Keyword Classification
## Introduction
Convolutional neural networks (CNN) are commonly used for image classification because they are able to generalize an image into a set of features. CNNs can be used similarly for audio recognition. Spectrograms of audio clips are classified like images would be. I designed a CNN architecture to classify a small set of short spoken words.

This network completes a task based on the TensorFlow Speech Recognition Challenge from Kaggle [found here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). My goal was to build a neural network capable of classifying several keywords from the TensorFlow Speech Recognition data set. The keywords classified by the network described here are "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", and "unknown".

I initially decided to work off of the architecture described in the paper, ["Convolutional Neural Networks for Small -footprint Keyword Spotting"](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf), but instead switched to a fully convolutional network due to better performance.

## Data Set
The data set used in this network is the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). The dataset includes one second spoken commands donated by thousands of people. The training set consists of 64721 .wav files.
### Organization
All wavfiles in the training data are organized into labelled directories in the root directory ```train/audio```. As an example, one wavfile in the training set is ```train/audio/go/fd395b74_nohash_0.wav```. The first component is the label of the utterance, in this case 'go'. In the filename itself, there are three components: the speaker ID, 'nohash', and the utterance number. The speaker ID is a unique identifier of the person who donated the utterance to the dataset. One speaker may donate utterances of multiple words, and the speaker ID can be used to ensure that all utterances by one speaker are partitioned into the same set. This keeps the network from learning to identify voices instead of words. The final component is the utterance number. A speaker may contribute multiple utterances of the same word. The first utterance is numbered 0, and subsequent utterances are numbered upwards from there.

The ```_background_noise_``` directory contains a combination of wavfiles of generated noise and recorded noises. These are available to be mixed with the spoken words before they are input into the network to imporve performance.

I include in my network an additional ```_silence_``` directory that includes one files of all 0s. This is meant to be used as the silence samples input to the dataset along with utterances.

## Architecture
The network described here is designed to classify ten utterances as well as differentiate unknown words from silence.

## Results
## Future Work
## Reflection
