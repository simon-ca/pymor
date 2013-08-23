# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.interfaces import OperatorInterface, LincombOperatorInterface
from pymor.operators.basic import (OperatorBase, MatrixBasedOperatorBase, LincombOperatorBase, LincombOperator,
                                   NumpyGenericOperator, NumpyMatrixBasedOperator, NumpyMatrixOperator)
from pymor.operators.constructions import (project_operator, rb_project_operator, ConstantOperator,
                                           FixedParameterOperator)
