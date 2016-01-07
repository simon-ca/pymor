# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeNumpyVectorArrray:

    params = [(10, 100), (10, 10000, 1000000)]
    param_names = ['len', 'dim']

    def setup(self, len_, dim):
        import numpy as np
        from pymor.vectorarrays.numpy import NumpyVectorArray
        self.U = NumpyVectorArray(np.random.random((len_, dim)))

    def time_copy(self, len_, dim):
        self.U.copy()

    def time_gramian(self, len_, dim):
        self.U.gramian()
