# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeNumpyVectorArrray:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        import numpy as np
        from pymor.vectorarrays.numpy import NumpyVectorArray
        self.U = NumpyVectorArray(np.random.random((100, 100000)))

    def time_copy(self):
        self.U.copy()

    def time_gramian(self):
        self.U.gramian()
