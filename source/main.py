from Kinetic import Kinetic
from Optimization import Optimization as Opt
import numpy as np

Start = np.random.rand(21) * 100 - 50
Sat = Kinetic(21)
Test = Opt(21)
Test.MINE(Sat.Obj_Func, Start, 45, 4)
