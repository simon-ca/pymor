# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.referenceelements import triangle
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.parameters.base import Parametric
from pymor.vectorarrays.numpy import NumpyVectorSpace


class RotRotOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, grid, boundary_info, coefficient=1.,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element is triangle
        assert isinstance(coefficient, Number) \
            or (isinstance(coefficient, FunctionInterface) and
                coefficient.dim_domain == 2 and
                coefficient.shape_range == tuple())
        self.source = self.range = NumpyVectorSpace(grid.size(1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.coefficient = coefficient
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if isinstance(coefficient, Parametric):
            self.build_parameter_type(inherits=(coefficient,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        self.logger.info('Calulate local matrices ...')
        M = np.empty((g.size(0), 3, 3))

        edge_lengths = g.volumes(1)[g.subentities(0, 1)]
        vol_inverse = 1. / g.volumes(0)

        M[:, 0, 0] = edge_lengths[:, 0] * edge_lengths[:, 0] * vol_inverse
        M[:, 1, 1] = edge_lengths[:, 1] * edge_lengths[:, 1] * vol_inverse
        M[:, 2, 2] = edge_lengths[:, 2] * edge_lengths[:, 2] * vol_inverse

        M[:, 2, 1] = vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 1]
        M[:, 2, 0] = vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 0]
        M[:, 1, 0] = vol_inverse * edge_lengths[:, 1] * edge_lengths[:, 0]

        M[:, 1, 2] = M[:, 2, 1]
        M[:, 0, 2] = M[:, 2, 0]
        M[:, 0, 1] = M[:, 1, 0]

        del vol_inverse, edge_lengths

        if isinstance(self.coefficient, FunctionInterface):
            M *= self.coefficient.evaluate(g.centers(0), mu=mu)[:, np.newaxis, np.newaxis]
        else:
            M *= self.coefficient

        SIGNS = np.empty((g.size(0), 3))
        SE = g.subentities(0, 2)
        SIGNS[:, 0] = np.sign(SE[:, 2] - SE[:, 1])
        SIGNS[:, 1] = np.sign(SE[:, 0] - SE[:, 2])
        SIGNS[:, 2] = np.sign(SE[:, 1] - SE[:, 0])
        M *= SIGNS[:, :, np.newaxis] * SIGNS[:, np.newaxis, :]
        del SE, SIGNS

        self.logger.info('Determine global dofs ...')
        M = M.ravel()
        SF_I0 = np.repeat(g.subentities(0, 1), 3, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 1), [1, 3]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            M = np.where(bi.dirichlet_mask(1)[SF_I0], 0, M)
            if self.dirichlet_clear_columns:
                M = np.where(bi.dirichlet_mask(1)[SF_I1], 0, M)

            if not self.dirichlet_clear_diag:
                M = np.hstack((M, np.ones(bi.dirichlet_boundaries(1).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(1)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(1)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((M, (SF_I0, SF_I1)), shape=(g.size(1), g.size(1)))
        del M, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class L2ProductOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, grid, boundary_info, coefficient=1.,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element is triangle
        assert isinstance(coefficient, Number) \
            or (isinstance(coefficient, FunctionInterface) and
                coefficient.dim_domain == 2 and
                coefficient.shape_range == tuple())
        self.source = self.range = NumpyVectorSpace(grid.size(1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.coefficient = coefficient
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if isinstance(coefficient, Parametric):
            self.build_parameter_type(inherits=(coefficient,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        self.logger.info('Calulate local matrices ...')
        M = np.empty((g.size(0), 3, 3))

        edge_lengths = g.volumes(1)[g.subentities(0, 1)]
        vol_inverse = 1. / g.volumes(0)
        CO = g.centers(2)[g.subentities(0, 2)]

        M[:, 0, 0] = (1. / 24. * edge_lengths[:, 0] * edge_lengths[:, 0] * vol_inverse *
                      (edge_lengths[:, 0] * edge_lengths[:, 0]
                       + 3. * ((CO[:, 2, 0] - CO[:, 0, 0]) * (CO[:, 1, 0] - CO[:, 0, 0]) +
                               (CO[:, 2, 1] - CO[:, 0, 1]) * (CO[:, 1, 1] - CO[:, 0, 1]))))

        M[:, 1, 1] = (1. / 24. * edge_lengths[:, 1] * edge_lengths[:, 1] * vol_inverse *
                      (edge_lengths[:, 1] * edge_lengths[:, 1]
                       + 3. * ((CO[:, 1, 0] - CO[:, 2, 0]) * (CO[:, 1, 0] - CO[:, 0, 0]) +
                               (CO[:, 1, 1] - CO[:, 2, 1]) * (CO[:, 1, 1] - CO[:, 0, 1]))))

        M[:, 2, 2] = (1. / 24. * edge_lengths[:, 2] * edge_lengths[:, 2] * vol_inverse *
                      (edge_lengths[:, 2] * edge_lengths[:, 2]
                       + 3. * ((CO[:, 1, 0] - CO[:, 2, 0]) * (CO[:, 0, 0] - CO[:, 2, 0]) +
                               (CO[:, 1, 1] - CO[:, 2, 1]) * (CO[:, 0, 1] - CO[:, 2, 1]))))

        M[:, 2, 1] = (-1. / 24. * vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 1] *
                      (edge_lengths[:, 0] * edge_lengths[:, 0]
                       - ((CO[:, 1, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 0, 0]) +
                          (CO[:, 1, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 0, 1]))))

        M[:, 2, 0] = (-1. / 24. * vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 0] *
                      (edge_lengths[:, 1] * edge_lengths[:, 1]
                       + ((CO[:, 1, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 1, 0]) +
                          (CO[:, 1, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 1, 1]))))

        M[:, 1, 0] = (-1. / 24. * vol_inverse * edge_lengths[:, 1] * edge_lengths[:, 0] *
                      (edge_lengths[:, 2] * edge_lengths[:, 2]
                       - ((CO[:, 2, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 1, 0]) +
                          (CO[:, 2, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 1, 1]))))

        M[:, 1, 2] = M[:, 2, 1]
        M[:, 0, 2] = M[:, 2, 0]
        M[:, 0, 1] = M[:, 1, 0]

        del CO, vol_inverse, edge_lengths

        if isinstance(self.coefficient, FunctionInterface):
            M *= self.coefficient.evaluate(g.centers(0), mu=mu)[:, np.newaxis, np.newaxis]
        else:
            M *= self.coefficient

        SIGNS = np.empty((g.size(0), 3))
        SE = g.subentities(0, 2)
        SIGNS[:, 0] = np.sign(SE[:, 2] - SE[:, 1])
        SIGNS[:, 1] = np.sign(SE[:, 0] - SE[:, 2])
        SIGNS[:, 2] = np.sign(SE[:, 1] - SE[:, 0])
        M *= SIGNS[:, :, np.newaxis] * SIGNS[:, np.newaxis, :]
        del SE, SIGNS

        self.logger.info('Determine global dofs ...')
        M = M.ravel()
        SF_I0 = np.repeat(g.subentities(0, 1), 3, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 1), [1, 3]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            M = np.where(bi.dirichlet_mask(1)[SF_I0], 0, M)
            if self.dirichlet_clear_columns:
                M = np.where(bi.dirichlet_mask(1)[SF_I1], 0, M)

            if not self.dirichlet_clear_diag:
                M = np.hstack((M, np.ones(bi.dirichlet_boundaries(1).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(1)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(1)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((M, (SF_I0, SF_I1)), shape=(g.size(1), g.size(1)))
        del M, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A
