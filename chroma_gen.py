import numpy as np
import mfcc_gen
import six
import cache

@cache(level=40)

def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)
    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))
    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)
    # For max/min norms, filling with 1 works
    fill_norm = 1
    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)
    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')
        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)
        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)
    elif norm is None:
        return S
    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold
    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length
    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length
 
    return Snorm

@cache(level=10)

def chroma(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=2, norm=2, base_c=True):

    wts = np.zeros((n_chroma, n_fft))
    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]
    frqbins = n_chroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))
    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))
    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype='d')).T
    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10*n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (n_chroma, 1)))**2)

    # normalize each column
    wts = normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(np.exp(-0.5 * (((frqbins/n_chroma - ctroct)/octwidth)**2)), (n_chroma, 1))

    if base_c:
        wts = np.roll(wts, -3, axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, :int(1 + n_fft/2)])


def hz_to_octs(frequencies, A440=440.0):

    return np.log2(np.asanyarray(frequencies) / (float(A440) / 16))


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):

    frequencies = np.atleast_1d(frequencies)
    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]
    if not np.any(frequencies):
        print ('Trying to estimate tuning from empty frequency set.')
        return 0.0
    residual = np.mod(bins_per_octave * hz_to_octs(frequencies), 1.0)
    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05 from the next tone up.

    residual[residual >= 0.5] -= 1.0
    bins = np.linspace(-0.5, 0.5, int(np.ceil(1./resolution)), endpoint=False)
    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]


def localmax(x, axis=0):

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)
    x_pad = np.pad(x, paddings, mode='edge')
    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)
    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


def tiny(x):

    # Make sure we have an array view
    x = np.asarray(x)
    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, float) or np.issubdtype(x.dtype, complex):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny

@cache(level=30)

def piptrack(y=None, sr=22050, S=None, n_fft=2048, hop_length=None, fmin=150.0, fmax=4000.0, threshold=0.1):

    # Check that we received an audio time series or STFT
    if hop_length is None:
        hop_length = int(n_fft // 4)
    S, n_fft = mfcc_gen._spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)
    fft_freqs = mfcc_gen.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,then figure out where the peaks are then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])
    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < tiny(shift)))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode='constant')
    shift = np.pad(shift, ([1, 1], [0, 0]), mode='constant')
    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    idx = np.argwhere(freq_mask & localmax(S * (S > (threshold * S.max(axis=0)))))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]])*float(sr) / n_fft)
    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]] + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags


def estimate_tuning(y=None, sr=22050, S=None, n_fft=2048,resolution=0.01, bins_per_octave=12, **kwargs):

    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0
    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask], resolution=resolution, bins_per_octave=bins_per_octave)


def chroma_stft(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048, hop_length=512, tuning=None, **kwargs):
    S, n_fft = mfcc_gen._spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=2)
    n_chroma = kwargs.get('n_chroma', 12)

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = chroma(sr, n_fft, **kwargs)
    # Compute raw chroma
    raw_chroma = np.dot(chromafb, S)

    # Compute normalization factor for each frame
    return normalize(raw_chroma, norm=norm, axis=0)
