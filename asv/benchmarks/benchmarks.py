# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from __future__ import division

import os
import timeit

os.environ['OMP_NUM_THREADS'] = '1'  # ensure we are running single threaded


class TimeVectorArray:

    params = [('numpy', 'numpylist'), (10, 100), (10, 10000, 1000000)]
    param_names = ['type', 'len', 'dim']

    def setup(self, type_, len_, dim):
        import numpy as np
        np.random.seed(42)

        U = np.random.random((len_, dim))
        V = np.random.random((len_, dim))
        if type_ == 'numpy':
            from pymor.vectorarrays.numpy import NumpyVectorArray
            self.U = NumpyVectorArray(U, copy=False)
            self.V = NumpyVectorArray(V, copy=False)
        elif type_ == 'numpylist':
            from pymor.vectorarrays.list import ListVectorArray, NumpyVector
            self.U = ListVectorArray([NumpyVector(u) for u in U], copy=False)
            self.V = ListVectorArray([NumpyVector(v) for v in V], copy=False)
        else:
            assert False

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
        self.coeffs = np.random.random(len(self.U))
        self.coeffs_ind = self.coeffs[self.U_ind]
        self.components = np.random.randint(0, self.U.dim, 7)

    def time_empty(self, type_, len_, dim):
        self.U.empty(len_)

    def time_zeros(self, type_, len_, dim):
        self.U.zeros(len_)

    def time_copy(self, type_, len_, dim):
        self.U.copy()

    def time_copy_indexed(self, type_, len_, dim):
        self.U.copy(ind=self.U_ind)

    def time_gramian(self, type_, len_, dim):
        self.U.gramian()

    def time_gramian_indexed(self, type_, len_, dim):
        self.U.gramian(ind=self.U_ind)

    def time_dot(self, type_, len_, dim):
        self.U.dot(self.V)

    def time_dot_indexed(self, type_, len_, dim):
        self.U.dot(self.V, ind=self.U_ind, o_ind=self.V_ind)

    def time_pairwise_dot(self, type_, len_, dim):
        self.U.pairwise_dot(self.V)

    def time_pairwise_dot_indexed(self, type_, len_, dim):
        self.U.pairwise_dot(self.V, ind=self.U_ind, o_ind=self.V_ind)

    def time_scal(self, type_, len_, dim):
        self.U.scal(42.)
        self.U.scal(1/42.)  # important! setup is not called in the inner loop of timeit

    def time_scal_indexed(self, type_, len_, dim):
        self.U.scal(42., ind=self.U_ind)
        self.U.scal(1/42., ind=self.U_ind)

    def time_axpy(self, type_, len_, dim):
        self.U.axpy(42., self.V)
        self.U.axpy(-42., self.V)

    def time_axpy_indexed(self, type_, len_, dim):
        self.U.axpy(42., self.V, ind=self.U_ind, x_ind=self.V_ind)
        self.U.axpy(-42., self.V, ind=self.U_ind, x_ind=self.V_ind)

    def time_lincomb(self, type_, len_, dim):
        self.U.lincomb(self.coeffs)

    def time_lincomb_indexed(self, type_, len_, dim):
        self.U.lincomb(self.coeffs_ind, ind=self.U_ind)

    def time_l2_norm(self, type_, len_, dim):
        self.U.l2_norm()

    def time_l2_norm_indexed(self, type_, len_, dim):
        self.U.l2_norm(ind=self.U_ind)

    def time_l1_norm(self, type_, len_, dim):
        self.U.l1_norm()

    def time_l1_norm_indexed(self, type_, len_, dim):
        self.U.l1_norm(ind=self.U_ind)

    def time_sup_norm(self, type_, len_, dim):
        self.U.sup_norm()

    def time_sup_norm_indexed(self, type_, len_, dim):
        self.U.sup_norm(ind=self.U_ind)

    def time_amax(self, type_, len_, dim):
        self.U.amax()

    def time_amax_indexed(self, type_, len_, dim):
        self.U.amax(ind=self.U_ind)

    def time_components(self, type_, len_, dim):
        self.U.components(self.components)

    def time_components_indexed(self, type_, len_, dim):
        self.U.components(self.components, ind=self.U_ind)


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

    def time_thermalblock_small_listva(self):
        self._run('pymordemos.thermalblock', [2, 2, 2, 10, '--list-vector-array'])

    def time_thermalblock_highdim_listva(self):
        self._run('pymordemos.thermalblock', [2, 2, 2, 10, '--grid=300', '--list-vector-array'])

    def time_thermalblock_manymu_listva(self):
        self._run('pymordemos.thermalblock', [2, 2, 16, 10, '--test=1', '--list-vector-array'])

    def time_burgersei_small(self):
        self._run('pymordemos.burgers_ei', [1, 2, 3, 40, 3, 10, '--test=0', '--cache-region=disk'])

    def time_burgersei_highdim(self):
        self._run('pymordemos.burgers_ei', [1, 2, 3, 40, 3, 10, '--test=0', '--cache-region=disk',
                                            '--grid=120', '--nt=200'])

    def time_burgersei_largecb(self):
        self._run('pymordemos.burgers_ei', [1, 2, 10, 700, 3, 10, '--test=0', '--cache-region=disk'])
