import sqlalchemy.types as types
import io
import blosc
import numpy as np
import numba
import pickle


class NpArrayType(types.TypeDecorator):

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        out = io.BytesIO()
        np.save(out, value)
        out.seek(0)
        return out.read()

    def process_result_value(self, value, dialect):
        out = io.BytesIO(value)
        out.seek(0)
        return np.load(out)


class BloscNpArrayType(types.TypeDecorator):

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        # NB : seems that NOSHUFFLE and BITSHUFFLE performs better than SHUFFLE
        #      because of the many zeros in the array
        #      using zstd now because of the better compression, use zstd for slower reading but better compression
        return blosc.pack_array(value, shuffle=blosc.NOSHUFFLE, cname='blosclz')

    def process_result_value(self, value, dialect):
        return blosc.unpack_array(value)


class QuantizedNpArrayType(types.TypeDecorator):
    """
    Quantizes the data between min value and max value with 256 evenly spaced points.
    After reading data is always converted to np.float32.
    Should allow for interesting compression, will take a bit more time to read compared to a pure blosc version
    """

    impl = types.LargeBinary

    @staticmethod
    def _quantize_array(value):
        _min, _max = np.min(value), np.max(value)
        diff = _max-_min if _max > _min else 1
        return np.round((value-_min)/diff*255).astype(np.uint8), _min, _max

    @staticmethod
    def _dequantize_array(value, _min, _max):
        diff = _max-_min if _max > _min else 1
        # Use of a LUT
        return (np.arange(256)*(diff/255)+_min).astype(np.float32)[value]

    def process_bind_param(self, value, dialect):
        value, _min, _max = self._quantize_array(value)
        # NB : seems that NOSHUFFLE and BITSHUFFLE performs better than SHUFFLE
        #      because of the many zeros in the array
        #      using zstd now because of the better compression, use zstd for slower reading but better compression
        return np.array([_min, _max], dtype=np.float64).tobytes() +\
               blosc.pack_array(value, shuffle=blosc.NOSHUFFLE, cname='zstd')

    def process_result_value(self, value, dialect):
        _min, _max = np.fromstring(value[:16], np.float64)
        return self._dequantize_array(blosc.unpack_array(value[16:]), _min, _max)
