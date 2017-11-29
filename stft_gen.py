import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.signal
import scipy.interpolate
import six
import cache
from numpy.lib.stride_tricks import as_strided

MAX_MEM_BLOCK = 2**8 * 2**10

def frame(y, frame_length=2048, hop_length=512):

    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d}) for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames), strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames

@cache(level=20)

def valid_audio(y, mono=True):

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.float):
        raise ParameterError('data must be floating-point')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError('Audio must have shape (samples,) or (channels, samples). Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True

@cache(level=10)

def get_window(window, Nx, fftbins=True):

    if (isinstance(window, (six.string_types, tuple)) or np.isscalar(window)):
        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)
        raise ParameterError('Window size mismatch: {:d} != {:d}'.format(len(window), Nx))
    
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))


def pad_center(data, size, axis=-1, **kwargs):
 
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

@cache(level=20)

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'):
    
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft
        
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
        
    fft_window = get_window(window, win_length, fftbins=True)
    
    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)
    
    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)
    
    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window * y_frames[:, bl_s:bl_t], axis=0)[:stft_matrix.shape[0]]

    return stft_matrix
