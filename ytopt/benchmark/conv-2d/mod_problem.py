# import required library
from plopper.plopper import Plopper
import numpy as np
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

# HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(1, os.path.dirname(HERE) + '/plopper')

# create an object of ConfigSpace 
cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=['#pragma omp target map(tofrom: O) map(to: I, K) teams distribute #P3 is_device_ptr(I, O, K)'])
p3 = CSH.CategoricalHyperparameter(name='p3', choices=['#P7 #P9'])
p7 = CSH.CategoricalHyperparameter(name='p7', choices=['num_teams(#P12)'])
p9 = CSH.CategoricalHyperparameter(name='p9', choices=['thread_limit(#P14)'])
p12 = CSH.OrdinalHyperparameter(name='p12', sequence=['2', '4', '16', '32', '64']) 
p14 = CSH.OrdinalHyperparameter(name='p14', sequence=['16', '32', '64', '128', '256', '512'])
#p2 (check if cuda is available): already exists in convolution-2d.c since it is a cuda example.
cs.add_hyperparameters([p1,p3,p7,p9,p12,p14])

cond1 = CS.EqualsCondition(p3, p1, '#pragma omp target map(tofrom: O) map(to: I, K) teams distribute #P3 is_device_ptr(I, O, K)')
cond2 = CS.EqualsCondition(p9, p3, '#P7 #P9')
cond3 = CS.EqualsCondition(p7, p3, '#P7 #P9')
cond9 = CS.EqualsCondition(p12, p7, 'num_teams(#P12)')
cond11 = CS.EqualsCondition(p14, p9, 'thread_limit(#P14)')


cs.add_conditions([cond1, cond2, cond9, cond11])

#cs.add_conditions([cond0, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond11])
#in case there is #P2 that needs to be replaced

# problem space
task_space = None
input_space = cs
output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/conv-2d.c',dir_path)

def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = list(point.values())
        print('VALUES:', point)
        params = {k.upper(): v for k, v in point.items()}
        result, power, filename = obj.findRuntime(value, params)
        return result, power, filename
    
    x = np.array(list(point.values())) #len(point) = 13 or 26
    results, power, filename = plopper_func(x)

    return results, power, filename

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )