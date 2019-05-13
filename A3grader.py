import os
import copy
import signal
import pandas as pd
import numpy as np

# Code to limit running time of specific parts of code.
#  To use, do this for example...
#
#  signal.alarm(seconds)
#  try:
#    ... run this ...
#  except TimeoutException:
#     print(' 0/8 points. Your depthFirstSearch did not terminate in', seconds/60, 'minutes.')
# Exception to signal exceeding time limit.


# class TimeoutException(Exception):
#     def __init__(self, *args, **kwargs):
#         Exception.__init__(self, *args, **kwargs)


# def timeout(signum, frame):
#     raise TimeoutException

# seconds = 60 * 5

# Comment next line for Python2
# signal.signal(signal.SIGALRM, timeout)

import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '3'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
else:
    import subprocess, glob, pathlib
    filename = next(glob.iglob('*.ipynb'.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         '*.ipynb'.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

g = 0

not_implemented = False

for func in ['run_parameters_act']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        def run_parameters_act(*args, verbose=False):
            global not_implemented
            not_implemented = True
            return pd.DataFrame([{'Activation': 'nope','RMSE Test': 0}, {'Activation': 'nope', 'RMSE Test':0, 'Epochs': 0}])


print('''\nTesting:
import neuralnetworks as nn
nnet = nn.NeuralNetwork(4, [10], 1)
acts = nnet.activation(np.array([-0.5, 1.5]))''')
      
import neuralnetworks as nn
nnet = nn.NeuralNetwork(4, [10], 1)
try:
    acts = nnet.activation(np.array([-0.5, 1.5]))
    correct_acts = np.array([-0.46211716,  0.90514825])
    if np.sum(np.abs(acts - correct_acts)) < 0.1:
        g += 10
        print('\n--- 10/10 points. nnet.activation() is correct.')
    else:
        g += 0
        print('\n---  0/10 points. nnet.activation() is {} but correct value is {}.'.format(acts, correct_acts))
except Exception as ex:
    print('\n--- 0/10 points. nnet.activation() raised exception', ex)

print('''\nTesting:
dacts = nnet.activation_derivative({})'''.format(correct_acts))

try:
    dacts = nnet.activation_derivative(correct_acts)
    correct_dacts = np.array([0.78644773, 0.18070664])
    if np.sum(np.abs(dacts - correct_dacts)) < 0.1:
        g += 10
        print('\n--- 10/10 points. nnet.activation_derivative() is correct.')
    else:
        g += 0
        print('\n---  0/10 points. nnet.activation_derivative() is {} but correct value is {}.'.format(dacts, correct_dacts))
except Exception as ex:
    print('\n--- 0/10 points. nnet.activation_derivative() raised exception', ex)


print('''\nTesting:
import neuralnetworks as nn
nnet_relu = nn.NeuralNetwork_relu(4, [10], 1)
acts = nnet_reul.activation(np.array([-0.5, 1.5]))''')
      
import neuralnetworks as nn
nnet_relu = nn.NeuralNetwork_relu(4, [10], 1)
try:
    acts = nnet_relu.activation(np.array([-0.5, 1.5]))
    correct_acts = np.array([0. , 1.5])
    if np.sum(np.abs(acts - correct_acts)) < 0.1:
        g += 10
        print('\n--- 10/10 points. nnet.activation() is correct.')
    else:
        g += 0
        print('\n---  0/10 points. nnet.activation() is {} but correct value is {}.'.format(acts, correct_acts))
except Exception as ex:
    print('\n--- 0/10 points. nnet.activation() raised exception', ex)

print('''\nTesting:
dacts = nnet_relu.activation_derivative({})'''.format(correct_acts))

try:
    dacts = nnet_relu.activation_derivative(correct_acts)
    correct_dacts = np.array([0., 1.])
    if np.sum(np.abs(dacts - correct_dacts)) < 0.1:
        g += 10
        print('\n--- 10/10 points. nnet.activation_derivative() is correct.')
    else:
        g += 0
        print('\n---  0/10 points. nnet.activation_derivative() is {} but correct value is {}.'.format(dacts, correct_dacts))
except Exception as ex:
    print('\n--- 0/10 points. nnet.activation_derivative() raised exception', ex)

print('''\nTesting:
import subprocess
subprocess.call(['curl -O www.cs.colostate.edu/~anderson/cs445/notebooks/machine.data'], shell=True)

data = np.loadtxt('machine.data', delimiter=',', usecols=range(2, 10))
X = data[:, :4]
T = data[:, -2:]
Xtrain = X[20:60, :]
Ttrain = T[20:60, :]
Xtest = X[70:100:, :]
Ttest = T[70:100:, :]

results = run_parameters_act(Xtrain, Ttrain, Xtest, Ttest, ['tanh', 'relu'],
                             [1, 1000], [[2], [100]], verbose=False)
results = results.sort_values('RMSE Test')
print(results)''')

import subprocess
subprocess.call(['curl -O www.cs.colostate.edu/~anderson/cs445/notebooks/machine.data'], shell=True)
# ! curl -O http://www.cs.colostate.edu/~anderson/cs445/notebooks/machine.data
data = np.loadtxt('machine.data', delimiter=',', usecols=range(2, 10))
X = data[:, :4]
T = data[:, -2:]
Xtrain = X[20:60, :]
Ttrain = T[20:60, :]
Xtest = X[70:100:, :]
Ttest = T[70:100:, :]

try:
    results = run_parameters_act(Xtrain, Ttrain, Xtest, Ttest, ['tanh', 'relu'],
                                 [1, 1000], [[2], [100]], verbose=False)
    results = results.sort_values('RMSE Test')
    print(results)

    if results.iloc[0]['Activation'] == 'relu':
        g += 40
        print('\n--- 40/40 points. Best activation function is correctly \'relu\'')
    else:
        g += 0
        print('\n---  0/40 points. Best activation function is incorrectly \'tanh\'')
except Exception as ex:
    print('\n--- 0/40 points. run_parameters_act  raised exception', ex)





name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 80'.format(name, g))

print('\n Remaining 20 points will be based on your text descriptions of results and plots.')

print('\n{} FINAL GRADE is   / 100'.format(name))

print('\n{} EXTRA CREDIT is   / 1'.format(name))

