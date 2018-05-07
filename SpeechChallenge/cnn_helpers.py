"""Implement the CNN used for keyword spotting from the TensorFlow Simple Audio
Recognition Tutorial example. The paper can be found at
http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
"""

# Signal processing and ML libraries
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import tensorflow.contrib.signal as contrib_signal
from tensorflow.python.ops import io_ops
import random

def add_background_noise(audio, sample_rate, mode='all'):
    """Add background noise to one second audio clips. It has been shown that
    adding some additional background noise increases the accuracy of the
    network. There are four recorded noise clips, and two generated noise
    clips.

    Parameters
    ----------
    audio : the audio clip to add noise to
    sample_rate : the sample rate of the audio clip
    mode : the origin of the background noise. Either 'all', 'recorded', or 'generated'

    Returns
    -------
    mixed_audio : a one second clip of the audio combined with the background noise.
    """
    signal_noise_ratio_db = np.random.randint(-5, 11)  #dB
    signal_noise_ratio = np.power(10, signal_noise_ratio_db * 0.1)
    background_sample_rate, background_noise = load_random_background_noise(samples=sample_rate, mode=mode)
    print(audio.dtype, audio.shape)
    print(background_noise.dtype, background_noise.shape)
    mixed_audio = background_noise * signal_noise_ratio + audio
    # TODO: this adds way too much noise. figure out how to make it work correctly.
    return mixed_audio
    

# TODO: make this part of the audio processing workflow. Don't return real values!
def load_random_background_noise(samples=16000, mode='all'):
    """Loads a clip of random background noise from the set of available noises.

    Parameters
    ----------
    samples : The number of samples of background noise desired. Defaults to
        1600, or one second of audio at 1600Hz.
    mode : One of 'generated', 'recorded', or 'all'. If the mode is
        'generated', the background noise loaded will be from one of the files of
        computer generated noise. If the mode is 'recorded', the background noise
        loaded will be from one of the files of recorded noise. If the mode is
        'all', the background noise can be from any of the available files.
        Defaults to 'all'.

    Returns
    -------
    sample_rate : the sample rate of the audio, in samples per second.
    background_noise : an audio clip of one of the randomly chosen background noise files from the set of options.
    """
    base_path = '_background_noise_/'
    recorded_noise_files = ['doing_the_dishes.wav', 'dude_miaowing.wav', 'exercise_bike.wav', 'running_tap.wav']
    generated_noise_files = ['pink_noise.wav', 'white_noise.wav']
    all_noise_files = list(recorded_noise_files + generated_noise_files)

    if mode == 'all':
        noise_file = random.choice(all_noise_files)
    elif mode == 'recorded':
        noise_file = random.choice(recorded_noise_files)
    elif mode == 'generated':
        noise_file = random.choice(generated_noise_files)

    sample_rate, full_noise, _ = load_audio_clip(base_path + noise_file)
    start_index = np.random.randint(0, full_noise.shape[0] - samples)
    return sample_rate, np.squeeze(full_noise[start_index:start_index + samples])

def load_wav(filepath, desired_channels=1, desired_samples=16000):
    """Defines the piece of the computational graph that loads an audio file.

    Parameters
    ----------
    filepath : a tensor of dtype tf.string and shape (1,) representing
        the wav file being loaded.
    desired_channels : the number of channels the wav file should be decoded into.
        Defaults to 1.
    desired_samples : the number of samples to be loaded from the wav file.
        Defaults to 16000.

    Returns
    -------
    a tensor of dtype tf.float32 and shape (1, desired_samples) representing a loaded wav file
    """
    wav_loader = io_ops.read_file(filepath)
    wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=desired_channels,
            desired_samples=desired_samples)
    audio = wav_decoder.audio
    return audio

def standardize_batch(audio, mean, var):
    """Calculate the z-scores of the audio samples to normalize each batch.
    """
    standardized_audio = tf.nn.batch_normalization(audio, mean, var, None, None, 1e-6)
    return standardized_audio

def compute_logmel_spectrograms(audio, sample_rate, frame_length_seconds, frame_step_seconds):
    """Computes the log-mel spectrograms of a batch of audio clips

    Parameters
    ----------
    audio : a two dimensional tensor of audio samples of shape (num_samples, num_signals)
    sample_rate : the sample rate of the audio signals in Hz
    frame_length_seconds : the width of the STFT, in seconds
    frame_step_seconds : the number of seconds the STFTs are shifted from each other

    Returns
    -------
    A tensor of spectrograms of shape (num_signals, time_units, mel_bins) and dtype tf.float32
    """
    # Convert time parameters to samples
    frame_length_samples = int(frame_length_seconds * sample_rate)
    frame_step_samples = int(frame_step_seconds * sample_rate)

    # Create a spectrogram by taking the magnitude of the Short Time fourier Transform
    stft = contrib_signal.stft(audio, frame_length=frame_length_samples,
            frame_step=frame_step_samples, fft_length=frame_length_samples)
    
    magnitude_spectrograms = tf.abs(stft)

    # Warp the linear scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 40
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
            magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compress the mel spectrogram magnitudes.
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    return log_mel_spectrograms

def load_and_process_batch(filepaths, mean, var, desired_channels=1, desired_samples=16000, frame_length=0.025, frame_width=0.010):
    """Creates a batch of log-mel spectrograms from a list of paths to .wav files.

    Parameters
    ----------
    filepaths : a list of paths to wav files in the dataset.
    desired_channels : the number of channels of audio data to load from the
        .wav files.
    desired_samples : the number of samples to load from each .wav file.
    frame_length : the length in seconds of the STFT frame.
    frame_width : the step size in seconds between STFT frames.

    Returns
    -------
    A tensor of dtype tf.float32 and shape (batch_size, time_bins, mel_bins)
    """
    audio_signals = tf.squeeze(tf.map_fn(load_wav, filepaths, tf.float32))
    standardized_audio = standardize_batch(audio_signals, mean, var)
    spectrograms = compute_logmel_spectrograms(
            standardized_audio,
            sample_rate=desired_samples,
            frame_length_seconds=frame_length,
            frame_step_seconds=frame_width)
    return spectrograms
