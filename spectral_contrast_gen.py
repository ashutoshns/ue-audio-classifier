import numpy as np
import scipy
import scipy.signal
import mfcc_gen


def spectral_contrast(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False):

    S, n_fft = mfcc_gen._spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)
    # Compute the center frequencies of each bin
    if freq is None:
        freq = mfcc_gen.fft_frequencies(sr=sr, n_fft=n_fft)

    freq = np.atleast_1d(freq)
    if freq.ndim != 1 or len(freq) != S.shape[0]:
        raise ParameterError('freq.shape mismatch: expected ({:d},)'.format(S.shape[0]))

    if n_bands < 1 or not isinstance(n_bands, int):
        raise ParameterError('n_bands must be a positive integer')

    if not 0.0 < quantile < 1.0:
        raise ParameterError('quantile must lie in the range (0, 1)')

    if fmin <= 0:
        raise ParameterError('fmin must be a positive number')

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0**np.arange(0, n_bands + 1))

    if np.any(octa[:-1] >= 0.5 * sr):
        raise ParameterError('Frequency band exceeds Nyquist. '
                             'Reduce either fmin or n_bands.')

    valley = np.zeros((n_bands + 1, S.shape[1]))
    peak = np.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)
        idx = np.flatnonzero(current_band)
        if k > 0:
            current_band[idx[0] - 1] = True
        if k == n_bands:
            current_band[idx[-1] + 1:] = True
        sub_band = S[current_band]
        if k < n_bands:
            sub_band = sub_band[:-1]

        # Always take at least one bin from each side
        idx = np.rint(quantile * np.sum(current_band))
        idx = int(np.maximum(idx, 1))
        sortedr = np.sort(sub_band, axis=0)
        valley[k] = np.mean(sortedr[:idx], axis=0)
        peak[k] = np.mean(sortedr[-idx:], axis=0)

    if linear:
        return peak - valley
    else:
        return mfcc_gen.power_to_db(peak) - mfcc_gen.power_to_db(valley)
