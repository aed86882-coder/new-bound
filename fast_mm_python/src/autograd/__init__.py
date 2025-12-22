"""
Automatic differentiation module for gradient computation.
"""

from .GVar import GVar
from .ParamManager import ParamManager, get_param_manager, set_param_manager

__all__ = ['GVar', 'ParamManager', 'get_param_manager', 'set_param_manager']

