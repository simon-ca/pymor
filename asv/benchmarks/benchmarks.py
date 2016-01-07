# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from __future__ import division

import timeit


class TimeNumpyVectorArrray:

    params = [(10, 100), (10, 10000, 1000000)]
    param_names = ['len', 'dim']

    def setup(self, len_, dim):
        import numpy as np
        from pymor.vectorarrays.numpy import NumpyVectorArray
        np.random.seed(42)
        self.U = NumpyVectorArray(np.random.random((len_, dim)))
        self.W = self.U.copy()
        self.V = NumpyVectorArray(np.random.random((len_, dim)))
        U_ind = np.random.randint(0, len(self.U), len(self.U) // 4)
        seen = set()
        for i, v in enumerate(U_ind):
            while v in seen:
                v += 1
                if v == len(self.U):
                    v = 0
            seen.add(v)
            U_ind[i] = v
        self.U_ind = U_ind
        self.V_ind = np.random.randint(0, len(self.V), len(self.V) // 4)

    def time_copy(self, len_, dim):
        self.U.copy()

    def time_copy_indexed(self, len_, dim):
        self.U.copy(ind=self.U_ind)

    def time_gramian(self, len_, dim):
        self.U.gramian()

    def time_gramian_indexed(self, len_, dim):
        self.U.gramian(ind=self.U_ind)

    def time_scal(self, len_, dim):
        self.U.scal(42.)
        self.U.scal(1/42.)  # important! setup is not called in the inner loop of timeit

    def time_scal_indexed(self, len_, dim):
        self.U.scal(42., ind=self.U_ind)
        self.U.scal(1/42., ind=self.U_ind)

    def time_axpy(self, len_, dim):
        self.U.axpy(42., self.V)
        self.U.axpy(-42., self.V)

    def time_axpy_indexed(self, len_, dim):
        self.U.axpy(42., self.V, ind=self.U_ind, x_ind=self.V_ind)
        self.U.axpy(-42., self.V, ind=self.U_ind, x_ind=self.V_ind)


class TimeDemos:

    timer = timeit.default_timer  # use wall time for demos
    timeout = 60 * 60

    def teardown(self):
        from pymor.gui.qt import stop_gui_processes
        stop_gui_processes()

    def _run(self, module, args):
        import sys
        import runpy
        sys.argv = [module] + [str(a) for a in args]
        runpy.run_module(module, init_globals=None, run_name='__main__', alter_sys=True)

    def time_thermalblock_small(self):
        self._run('pymordemos.thermalblock', [2, 2, 2, 10])

    def time_thermalblock_highdim(self):
        self._run('pymordemos.thermalblock', [2, 2, 2, 10, '--grid=300'])

    def time_thermalblock_manymu(self):
        self._run('pymordemos.thermalblock', [2, 2, 16, 10, '--test=1'])
