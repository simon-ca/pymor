# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.analyticalproblems.maxwell import MaxwellProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.gui.qt import PatchVisualizer
from pymor.operators.nedelec import RotRotOperator, L2ProductOperator, L2ProductFunctional, CenterEvaluation
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


def discretize_maxwell(analytical_problem, diameter=None, domain_discretizer=None,
                       grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The :class:`~pymor.analyticalproblems.maxwell.MaxwellProblem` to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, MaxwellProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    rotrot_op = RotRotOperator(grid, boundary_info)
    l2_op = L2ProductOperator(grid, boundary_info)
    op = LincombOperator([rotrot_op, l2_op],
                         [ExpressionParameterFunctional('1./mu', {'mu': tuple()}),
                          ExpressionParameterFunctional('- eps * w**2', {'eps': tuple(), 'w': tuple()})])
    rhs = L2ProductFunctional(grid, p.excitation, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data)
    l2_product = L2ProductOperator(grid, EmptyBoundaryInfo(grid))
    visualizer = NedelecVisualizer(grid)

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(op, rhs, products={'l2': l2_product}, visualizer=visualizer,
                                              parameter_space=parameter_space, name=p.name)

    return discretization, {'grid': grid, 'boundary_info': boundary_info}


class NedelecVisualizer(PatchVisualizer):

    def __init__(self, grid, backend=None, block=False):
        super(NedelecVisualizer, self).__init__(grid, bounding_box=grid.bounding_box(), codim=0,
                                                backend=backend, block=block)
        self.evaluation_operator = CenterEvaluation(grid)

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2):

        def center_norm(U):
            return NumpyVectorArray(np.linalg.norm(self.evaluation_operator.apply(U).data.ravel().reshape((-1, 2)),
                                                   axis=1))

        assert isinstance(U, VectorArrayInterface) \
            or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) for u in U))
        if isinstance(U, tuple):
            U = tuple(center_norm(u) for u in U)
        else:
            U = center_norm(U)

        super(NedelecVisualizer, self).visualize(U, discretization, title=title, legend=legend,
                                                 separate_colorbars=separate_colorbars,
                                                 rescale_colorbars=rescale_colorbars,
                                                 block=block, filename=filename, columns=columns)
