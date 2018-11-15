try:
    import blosc
except:
    pass
import numpy as np
import numba
import pickle


def quantize_array(value):
    _min, _max = np.min(value), np.max(value)
    diff = _max-_min if _max > _min else 1
    return np.round((value-_min)/diff*255).astype(np.uint8), _min, _max


def dequantize_array_2(value, _min, _max):
    diff = _max-_min if _max > _min else 1
    # Use of a LUT
    lut = (np.arange(256)*(diff/255)+_min).astype(np.float32)
    return np.take(lut, value)


@numba.jit(nopython=True, nogil=True)
def dequantize_array(value, _min, _max):
    diff = _max-_min if _max > _min else 1
    # Use of a LUT
    lut = (np.arange(256)*(diff/255)+_min).astype(np.float32)
    result = np.empty(value.shape, dtype=np.float32)
    for i in range(value.size):
        result.flat[i] = lut[value.flat[i]]
    return result


def delta_encoding(f_map):
    inds = np.nonzero(f_map.flat)[0]
    d_inds = (inds - np.concatenate([[0], inds[:-1]]))
    max_ind = np.max(d_inds)
    if max_ind < 256:
        d_inds = d_inds.astype(np.uint8)
    elif max_ind < 2**16 - 1:
        d_inds = d_inds.astype(np.uint16)
    return f_map.shape, d_inds, f_map.flat[inds]


@numba.jit(nopython=True, nogil=True)
def delta_decoding(shape, d_inds, values):
    result = np.zeros(shape, dtype=values.dtype)
    ind = 0
    for i in range(values.size):
        ind += d_inds[i]
        result.flat[ind] = values[i]
    return result


def compress_sparse_data(f_map, use_blosc=True):
    assert f_map.dtype == np.float32
    # Find sparse elements
    shape, d_inds, values = delta_encoding(f_map)
    # Quantize
    values, _min, _max = quantize_array(values)
    if use_blosc:
        compress_fn = lambda arr: blosc.pack_array(arr, shuffle=blosc.SHUFFLE, clevel=9, cname='zstd')
        return pickle.dumps((f_map.shape, compress_fn(d_inds), compress_fn(values), _min, _max, True))
    else:
        return pickle.dumps((f_map.shape, d_inds, values, _min, _max, False))


def decompress_sparse_data(data):
    compressed_data = pickle.loads(data)
    if len(compressed_data) == 5:
        shape, d_inds, values, _min, _max = compressed_data
        blosc_used = False
    else:
        shape, d_inds, values, _min, _max, blosc_used = compressed_data
    if blosc_used:
        # Decompress
        decompress_fn = lambda d: blosc.unpack_array(d)
        d_inds, values, = decompress_fn(d_inds), decompress_fn(values)
    # Dequantize data
    values = dequantize_array(values, _min, _max)
    return delta_decoding(shape, d_inds, values)


def compress_array(value):
    value, _min, _max = quantize_array(value)
    # NB : seems that NOSHUFFLE and BITSHUFFLE performs better than SHUFFLE
    #      because of the many zeros in the array
    #      use zstd for slower reading but better compression
    return np.array([_min, _max], dtype=np.float64).tobytes() +\
           blosc.pack_array(value, shuffle=blosc.NOSHUFFLE, cname='blosclz')


def decompress_array(value):
    _min, _max = np.fromstring(value[:16], np.float64)
    return dequantize_array(blosc.unpack_array(value[16:]), _min, _max)