import warnings
from .datareps import SymmetricMatrices		
from .designspace import *
from .policies import thompson_sampling
from .utils import get_spectrum, ExampleRunnerSimulation, ExampleRunner
from .metrics import euclidean_dist
from .modelling import Emulator, EmulatorMultiShape
from .expts import TestShapeMatchBO