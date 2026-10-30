
import json
from pathlib import Path
from math import gcd

import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from scipy.signal import resample_poly


class VoiceCommandPredictor:

    def __init__(
        self,
        model_path="voice_command_cnn.keras",
        config_path="preprocessing_config.json"
    ):

        self.model = tf.keras.models.load_model(model_path)

        self.c = json.loads(
            Path(config_path).read_text()
        )

        self.labels = self.c["class_names"]
        self.ncmd = len(self.c["target_words"])

        self.threshold = float(
            self.c["confidence_threshold"]
        )

        self.mean = np.asarray(
            self.c["mfcc_mean"], np.float32
        ).reshape(1, 1, -1)

        self.std = np.asarray(
            self.c["mfcc_std"], np.float32
        ).reshape(1, 1, -1)


    # --------------------------------------------------------
    # AUDIO PREPROCESSING
    # --------------------------------------------------------

    def _wave(self, audio, sr):

        x = np.asarray(audio)

        if x.size == 0:
            raise ValueError("Audio input is empty.")

        if x.ndim == 2:
            x = x.astype(np.float32).mean(axis=1)

        if x.ndim != 1:
            raise ValueError("Audio must be mono or stereo.")

        if np.issubdtype(x.dtype, np.integer):

            info = np.iinfo(x.dtype)

            x = x.astype(np.float32) / float(
                max(abs(info.min), info.max)
            )

        else:

            x = x.astype(np.float32)

            peak = np.max(np.abs(x))

            if peak > 1.5:
                x = x / peak

        target = self.c["sample_rate"]

        if int(sr) != target:

            g = gcd(int(sr), target)

            x = resample_poly(
                x,
                target // g,
                int(sr) // g
            ).astype(np.float32)

        n = self.c["num_samples"]

        if len(x) > n:

            st = (len(x) - n) // 2

            x = x[st:st+n]

        elif len(x) < n:

            left = (n - len(x)) // 2

            x = np.pad(
                x,
                (left, n-len(x)-left)
            )

        return np.clip(x, -1, 1).astype(np.float32)


    # --------------------------------------------------------
    # MFCC FEATURE EXTRACTION
    # --------------------------------------------------------

    def _features(self, wave):

        c = self.c

        stft = tf.signal.stft(
            tf.convert_to_tensor(
                wave[None, :], tf.float32
            ),
            frame_length=c["frame_length"],
            frame_step=c["frame_step"],
            fft_length=c["fft_length"],
            window_fn=tf.signal.hann_window
        )

        magnitude = tf.abs(stft)

        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=c["num_mel_bins"],
            num_spectrogram_bins=c["fft_length"] // 2 + 1,
            sample_rate=c["sample_rate"],
            lower_edge_hertz=20.0,
            upper_edge_hertz=7600.0
        )

        mel = tf.tensordot(
            magnitude,
            mel_matrix,
            axes=1
        )

        logmel = tf.math.log(
            tf.maximum(mel, 1e-6)
        )

        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
            logmel
        )[..., :c["num_mfcc"]].numpy()

        normalized = (
            (mfcc - self.mean) / self.std
        )

        return normalized[..., None].astype(np.float32)


    # --------------------------------------------------------
    # PREDICT COMMAND
    # --------------------------------------------------------

    def predict_audio(self, audio, sample_rate):

        wave = self._wave(audio, sample_rate)

        features = self._features(wave)

        probabilities = self.model.predict(
            features,
            verbose=0
        )[0]

        raw_id = int(np.argmax(probabilities))

        command_confidence = float(
            np.max(probabilities[:self.ncmd])
        )

        final_id = raw_id

        # Confidence-based rejection
        if (
            raw_id < self.ncmd
            and command_confidence < self.threshold
        ):

            final_id = self.labels.index("unknown")

        return {

            "label": self.labels[final_id],

            "confidence": float(
                probabilities[final_id]
            ),

            "command_confidence": command_confidence,

            "raw_prediction": self.labels[raw_id],

            "threshold": self.threshold

        }


    # --------------------------------------------------------
    # RETURN ONLY CLASS LABEL
    # --------------------------------------------------------

    def predict_command(self, audio, sample_rate):

        return self.predict_audio(
            audio,
            sample_rate
        )["label"]


    # --------------------------------------------------------
    # PREDICT FROM WAV FILE
    # --------------------------------------------------------

    def predict_file(self, wav_path):

        sr, audio = wavfile.read(
            str(wav_path)
        )

        return self.predict_audio(
            audio,
            sr
        )
