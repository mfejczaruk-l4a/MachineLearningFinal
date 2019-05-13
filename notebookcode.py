#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\Yv}{\mathbf{Y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\gv}{\mathbf{g}}
# \newcommand{\Hv}{\mathbf{H}}
# \newcommand{\dv}{\mathbf{d}}
# \newcommand{\Vv}{\mathbf{V}}
# \newcommand{\vv}{\mathbf{v}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\Zv}{\mathbf{Z}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# $

# ## Scaled Conjugate Gradient Algorithm

# ### The Scaled Part

# The first derivative of an error function with respect to the
# parameters of your model tells you which direction in the parameter
# space to proceed to reduce the error function.  But how far do you go?
# So far we have just taken a small step by subtracting a small constant
# times the derivative from our current parameter values.
# 
# If we are in the vicinity of a minimum of the error function, we could
# do what Newton did...approximate the function at the current parameter
# value with a parabola and solve for the minimum of the parabola.  Use
# this as the next guess at a good parameter value.  If the error
# function is quadratic in the parameter, then we jump to the true
# minimum immediately.
# 
# How would you fit a parabola to a function at a particular value of
# $x$?  We can derive a way to do this using a truncated Taylor series
# (google that) to approximate the function about a value of $x$:
# 
# 
# $$
# f(x+\Delta x) \approx \hat{f}(x+\Delta x) = f(x) + f'(x) \Delta x + 
# \frac{1}{2} f''(x) \Delta x^2 + 
# $$
# 
# Now we want to know what value of $\Delta x$ minimizes
# $\hat{f}(x+\Delta x)$.  So take its derivative and set equal to zero.
# 
# $$
# \begin{align*}
# \frac{d \hat{f}(x+\Delta x)}{d\Delta x} &= f'(x) + \frac{1}{2} 2 f''(x)
# \Delta x\\
# & = f'(x) + f''(x) \Delta x
# \end{align*}
# $$
# 
# Setting equal to zero we get
# 
# $$
# \begin{align*}
# 0 &= f'(x) + f''(x) \Delta x\\
# \Delta x &= -\frac{f'(x)}{f''(x)}
# \end{align*}
# $$
# 
# Now we can update our guess for $x$ by adding $\Delta x$ to it.  Then,
# fit a new parabola at the new value of $x$, calculate $\Delta x$, and
# update $x$ again.  Actually, the last equation above does the parabola
# approximation and calculation of $\Delta x$.
# 
# Here is a simple example.  Say we want to find the minimum of
# 
# $$
# f(x) = 2 x^4 + 3 x^3 + 3
# $$
# To calculate
# 
# $$
# \begin{align*}
# \Delta x &= -\frac{f'(x)}{f''(x)}
# \end{align*}
# $$
# 
# we need the function's first and second derivatives.  The are
# 
# $$
# \begin{align*}
# f'(x) &= 8 x^3 + 9 x^2\\
# f''(x) &= 24 x^2 + 18 x
# \end{align*}
# $$
# 
# All together now, in python!

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import IPython.display as ipd  # for display and clear_output
import time  # for sleep


# In[2]:


def f(x):
    return 2 * x**4 + 3 * x**3 + 3

def df(x): 
    return 8 * x**3 + 9 * x**2

def ddf(x):
    return 24 * x**2 + 18*x

x = -2  # our initial guess
def taylorf(x,dx):
    return f(x) + df(x) * dx + 0.5 * ddf(x) * dx**2

x = np.random.uniform(-2, 1)  # first guess at minimum

xs = np.linspace(-2,1,num=100)

fig = plt.figure()

dxs = np.linspace(-0.5,0.5,num=100)

for rep in range(10):
    time.sleep(2) # sleep 2 seconds
    plt.clf()
    plt.plot(xs,f(xs))
    plt.grid('on')
    plt.plot(x+dxs, taylorf(x,dxs),'g-',linewidth=5,alpha=0.4)
    plt.plot(x,f(x),'ro')         
    y0,y1 = plt.ylim()
    plt.plot([x,x],[y0,y1],'r--')
    
    x = x - df(x) / float(ddf(x))
    plt.plot(x,f(x),'go')
    plt.text(x,(y0+y1)*0.5,str(x),color='r')
    plt.legend(('$f(x)$','$\hat{f}(x)$'))
    
    ipd.clear_output(wait=True)
    ipd.display(fig)
ipd.clear_output(wait=True)


# This has all been for a function $f(x)$ of a single, scalar variable
# $x$.  To minimize a squared error function for a neural network, $x$
# will consist of all the weights of the neural network.  If all of the
# weights are collected into the vector $\wv$, then the first derivative
# of the squared error function, $f$, with respect to the weight vector,
# $\wv$, is a vector of derivatives like $\frac{\partial f}{\partial
# w_{i}}$.  This is usually written as the gradient
# 
# $$
# \nabla_{\wv} f =
# (\frac{\partial f}{\partial w_{1}}, \frac{\partial f}{\partial w_{2}},
# \ldots, \frac{\partial f}{\partial w_{n}}).
# $$
# 
# The second derivative will be $n\times n$ matrix of values like
# $\frac{\partial^2 f}{\partial w_i \partial w_j}$, usually
# written as the Hessian
# 
# $$
# \nabla^2_{\wv} f =
# \begin{pmatrix}
# \frac{\partial^2 f}{\partial w_1 \partial w_1} & 
# \frac{\partial^2 f}{\partial w_1 \partial w_2} & 
# \cdots
# \frac{\partial^2 f}{\partial w_1 \partial w_n}\\
# \frac{\partial^2 f}{\partial w_2 \partial w_1} & 
# \frac{\partial^2 f}{\partial w_2 \partial w_2} & 
# \cdots
# \frac{\partial^2 f}{\partial w_2 \partial w_n}\\
# \vdots \\
# \frac{\partial^2 f}{\partial w_n \partial w_1} & 
# \frac{\partial^2 f}{\partial w_n \partial w_2} & 
# \cdots
# \frac{\partial^2 f}{\partial w_n \partial w_n}
# \end{pmatrix}
# $$
# 
# It is often impractical to
# construct and use the Hessian.  We
# will consider ways to approximate the product of the Hessian and a
# matrix as part of the Scaled Conjugate Gradient algorithm.

# ### The Conjugate Part

# Let $E(\wv)$ be the error function (mean square error over training samples) we wish to minimize by
# findig the best $\wv$. Steepest descent will find new $\wv$ by
# minimizing $E(\wv)$ in successive directions $\dv_0, \dv_1, \ldots$
# for which $\dv_i^T \dv_j = 0$ for $i \neq j$.  In other words, the
# search directions are orthogonal to each other, resulting in a zig-zag
# pattern of steps, some of which are in the same directions.  
# 
# Another problem with orthogonal directions is that forcing the second
# direction, for example, to be orthogonal to the first will not be in
# the direction of the minimum unless the error function is quadratic
# and its contours are circles.
# 
# We would rather choose a new direction based on the previous ones and
# on the curvature, or second derivative, of the error function at the
# current $\wv$.  This is the idea behind conjugate gradient methods.
# 
# The Scaled Conjugate Gradient (SCG) algorithm,
# [Efficient
# Training of Feed-Forward Neural Networks, by Moller](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.8063&rep=rep1&type=pdf), combines conjugate gradient directions with an local, quadratic approximation to the error function and solving
# for the new value of $\wv$ that would minimize the quadratic function.
# A number of additional steps are taken to improve the quadratic
# approximation.

# In[3]:


get_ipython().system('curl -O http://www.cs.colostate.edu/~anderson/cs445/notebooks/mlutilities.tar')


# In[4]:


get_ipython().system('tar xvf mlutilities.tar')


# In[5]:


import mlutilities as ml

def parabola(x,xmin,s):
    d = x - xmin
    return np.dot(np.dot(d,S),d.T)

def parabolaGrad(x,xmin,s):
    d = x - xmin
    return 2 * np.dot(s,d)

f = parabola
df = parabolaGrad
center = np.array([5,5])
S = np.array([[5,4],[4,5]])

n = 10
xs = np.linspace(0,10,n)
ys = np.linspace(0,10,n)
X,Y = np.meshgrid(xs,ys)
both = np.vstack((X.flat,Y.flat)).T
nall = n*n
Z = np.zeros(nall)
for i in range(n*n):
    Z[i] = parabola(both[i,:],center,S)
Z.resize((n,n))

fig = plt.figure(figsize=(8,8))

for reps in range(10):
    time.sleep(2)
    
    firstx = np.random.uniform(0,10,2)

    resultSCG = ml.scg(firstx, f, df, center, S, xtracep=True)
    resultSteepest =  ml.steepest(firstx, f, df, center, S, stepsize=0.05, xtracep=True)

    plt.clf()
    plt.contourf(X, Y, Z, 20, alpha=0.3)
    plt.axis('tight')
    
    xt = resultSteepest['xtrace']
    plt.plot(xt[:, 0], xt[:, 1], 'ro-')

    xt = resultSCG['xtrace']
    plt.plot(xt[:, 0], xt[:, 1], 'go-')

    plt.text(2, 3, "%s steps for Steepest Descent" % resultSteepest['xtrace'].shape[0], color='red')
    plt.text(2, 2.5, "%s steps for SCG" % resultSCG['xtrace'].shape[0], color='green')
    
    ipd.clear_output(wait=True)
    ipd.display(fig)
ipd.clear_output(wait=True) 


# Rosenbrock's function is often used to test optimization algorithms.
# It is
# 
# $$
# f(x,y) = (1-x)^2 + 100(y-x^2)^2
# $$

# In[6]:


def rosen(x):
    v = 100 * ((x[1] - x[0]**2)**2) + (1.0 - x[0])**2
    return v

def rosenGrad(x):
    g1 = -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    g2 =  200 * (x[1] - x[0]**2)
    return np.array([g1, g2])

f = rosen
df = rosenGrad


# In[7]:


n = 10
xmin, xmax = -1,2
xs = np.linspace(xmin, xmax, n)
ys = np.linspace(xmin, xmax, n)
X, Y = np.meshgrid(xs, ys)
both = np.vstack((X.flat, Y.flat)).T
nall = n * n
Z = np.zeros(nall)
for i in range(n * n):
    Z[i] = f(both[i, :])
Z.resize((n, n))

fig = plt.figure(figsize=(8, 8))

for reps in range(10):
    time.sleep(2)
    
    firstx = np.random.uniform(xmin, xmax, 2)

    # resultSCG = ml.scg(firstx, f, df, xPrecision=0.000001, xtracep=True)
    # resultSteepest =  ml.steepest(firstx, f, df, stepsize=0.001, xPrecision=0.000001, xtracep=True)
    resultSCG = ml.scg(firstx, f, df, xtracep=True)
    resultSteepest =  ml.steepest(firstx, f, df, stepsize=0.001, xtracep=True)

    plt.clf()
    plt.contourf(X, Y, Z, 20, alpha=0.3)
    plt.axis('tight')
    
    xt = resultSteepest['xtrace']
    plt.plot(xt[: ,0], xt[:, 1], 'ro-')

    xt = resultSCG['xtrace']
    plt.plot(xt[:, 0], xt[:, 1], 'go-')

    plt.text(0.7, -0.5, "%s steps for Steepest Descent" % resultSteepest['nIterations'], color='red')
    plt.text(0.7, -0.8, "%s steps for SCG" % resultSCG['nIterations'], color='green')
    
    ipd.clear_output(wait=True)
    ipd.display(fig)
ipd.clear_output(wait=True) 


# Only difficulty is that our *scg* (and *steepest*) implementation
# requires all parameters to be concatenated in a single vector.  We
# will use *pack* and *unpack* functions to concatentate and extract
# $V$ and $W$ matrices.
# 
# Here is our example from last time again, but now using *scg* (scaled conjugate gradient).

# In[8]:


# Make some training data
n = 20
X = np.linspace(0., 20.0, n).reshape((n, 1))
T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.2 * np.random.normal(size=(n, 1))

# Make some testing data
Xtest = X + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * X + 0.4 * np.sin(Xtest) + 0.2 * np.random.normal(size=(n, 1))

def addOnes(A):
    return np.insert(A, 0, 1, axis=1)


# In[9]:


# Set parameters of neural network
nInputs = X.shape[1]
nHiddens = 10
nOutputs = T.shape[1]

# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
V = np.random.uniform(-0.1, 0.1, size=(1 + nInputs, nHiddens))
W = np.random.uniform(-0.1, 0.1, size=(1 + nHiddens, nOutputs))

X1 = addOnes(X)
Xtest1 = addOnes(Xtest)

### gradientDescent functions require all parameters in a vector.
def pack(V, W):
    return np.hstack((V.flat, W.flat))

def unpack(w):
    '''Assumes V, W, nInputs, nHidden, nOuputs are defined in calling context'''
    V[:] = w[:(1 + nInputs) * nHiddens].reshape((1 + nInputs, nHiddens))
    W[:] = w[(1 + nInputs)*nHiddens:].reshape((1 + nHiddens, nOutputs))


# In[10]:


### Function f to be minimized
def objectiveF(w):
    unpack(w)
    # Forward pass on training data
    Y = addOnes(np.tanh(X1 @ V)) @ W
    return 0.5 * np.mean((T - Y)**2)


# In[11]:


### Gradient of f with respect to V,W
def gradientF(w):
    unpack(w)
    Z = np.tanh(X1 @ V)
    Z1 = addOnes(Z)
    Y = Z1 @ W
    nSamples = X1.shape[0]
    nOutputs = T.shape[1]
    error = -(T - Y) / (nSamples * nOutputs)
    dV = X1.T @ (error @ W[1:, :].T * (1 - Z**2))
    dW = Z1.T @ error
    return pack(dV, dW)


# In[12]:


# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
V = np.random.uniform(-0.1, 0.1, size=(1 + nInputs, nHiddens))
W = np.random.uniform(-0.1, 0.1, size=(1 + nHiddens, nOutputs))

result = ml.scg(pack(V, W), objectiveF, gradientF, nIterations = 2000, ftracep = True)

unpack(result['x'])  # copy best parameters into V and W
errorTrace = result['ftrace']
print('Ran for', len(errorTrace), 'iterations')


# In[13]:


errorTrace[:10]


# In[14]:


plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(errorTrace)
nEpochs = len(errorTrace)
plt.xlim(0 - 0.05 * nEpochs, nEpochs * 1.05)
plt.xlabel('Epochs')
plt.ylabel('Train RMSE')

plt.subplot(3, 1, 2)
Y = addOnes(np.tanh(X1 @ V)) @ W 
Ytest = addOnes(np.tanh(Xtest1 @ V)) @ W
plt.plot(X, T, 'o-', Xtest, Ttest, 'o-', Xtest, Ytest, 'o-')
plt.xlim(0, 20)
plt.legend(('Training', 'Testing', 'Model'), loc='upper left')
plt.xlabel('$x$')
plt.ylabel('Actual and Predicted $f(x)$')
        
plt.subplot(3, 1, 3)
Z = np.tanh(X1 @ V)
plt.plot(X, Z)
plt.xlabel('$x$')
plt.ylabel('Hidden Unit Outputs ($z$)');


# ## Neural Network Class

# Python includes the ability to define new classes.  Let's write one for neural networks.  First, let's discuss how it might be used. Make it as easy for the user as possible.
# 
#     X = ...
#     T = ...
#     nnet = NeuralNetwork(1, 5, 1)  # 1 input, 5 hidden units, 1 output
#     nnet.train(X, T, nIterations=100)
#     Y = nnet.use(X)
#     
# This implementation is for any number of hidden layers!

# In[1]:


get_ipython().run_cell_magic('writefile', 'neuralnetworks.py', "\nimport numpy as np\nimport mlutilities as ml\nimport matplotlib.pyplot as plt\nfrom copy import copy\nimport time\n\n\nclass NeuralNetwork:\n\n    def __init__(self, ni, nhs, no):\n\n        if isinstance(nhs, list) or isinstance(nhs, tuple):\n            if len(nhs) == 1 and nhs[0] == 0:\n                nihs = [ni]\n                nhs = []\n            else:\n                nihs = [ni] + list(nhs)\n        else:\n            if nhs > 0:\n                nihs = [ni, nhs]\n                nhs = [nhs]\n            else:\n                nihs = [ni]\n                nhs = []\n\n        if len(nihs) > 1:\n            self.Vs = [1/np.sqrt(nihs[i]) *\n                       np.random.uniform(-1, 1, size=(1 + nihs[i], nihs[i + 1])) for i in range(len(nihs) - 1)]\n            self.W = 1/np.sqrt(nhs[-1]) * np.random.uniform(-1, 1, size=(1 + nhs[-1], no))\n        else:\n            self.Vs = []\n            self.W = 1 / np.sqrt(ni) * np.random.uniform(-1, 1, size=(1 + ni, no))\n        self.ni, self.nhs, self.no = ni, nhs, no\n        self.Xmeans = None\n        self.Xstds = None\n        self.Tmeans = None\n        self.Tstds = None\n        self.trained = False\n        self.reason = None\n        self.errorTrace = None\n        self.numberOfIterations = None\n        self.trainingTime = None\n\n    def __repr__(self):\n        str = 'NeuralNetwork({}, {}, {})'.format(self.ni, self.nhs, self.no)\n        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'\n        if self.trained:\n            str += '\\n   Network was trained for {} iterations that took {:.4f} seconds. Final error is {}.'.format(self.numberOfIterations, self.getTrainingTime(), self.errorTrace[-1])\n        else:\n            str += '  Network is not trained.'\n        return str\n\n    def _standardizeX(self, X):\n        result = (X - self.Xmeans) / self.XstdsFixed\n        result[:, self.Xconstant] = 0.0\n        return result\n\n    def _unstandardizeX(self, Xs):\n        return self.Xstds * Xs + self.Xmeans\n\n    def _standardizeT(self, T):\n        result = (T - self.Tmeans) / self.TstdsFixed\n        result[:, self.Tconstant] = 0.0\n        return result\n\n    def _unstandardizeT(self, Ts):\n        return self.Tstds * Ts + self.Tmeans\n\n    def _pack(self, Vs, W):\n        return np.hstack([V.flat for V in Vs] + [W.flat])\n\n    def _unpack(self, w):\n        first = 0\n        numInThisLayer = self.ni\n        for i in range(len(self.Vs)):\n            self.Vs[i][:] = w[first:first + (1 + numInThisLayer) * \n                              self.nhs[i]].reshape((1 + numInThisLayer, self.nhs[i]))\n            first += (numInThisLayer+1) * self.nhs[i]\n            numInThisLayer = self.nhs[i]\n        self.W[:] = w[first:].reshape((1 + numInThisLayer, self.no))\n\n    def _objectiveF(self, w, X, T):\n        self._unpack(w)\n        # Do forward pass through all layers\n        Zprev = X\n        for i in range(len(self.nhs)):\n            V = self.Vs[i]\n            Zprev = np.tanh(Zprev @ V[1:, :] + V[0:1, :])  # handling bias weight without adding column of 1's\n        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]\n        return 0.5 * np.mean((T - Y)**2)\n\n    def _gradientF(self, w, X, T):\n        self._unpack(w)\n        # Do forward pass through all layers\n        Zprev = X\n        Z = [Zprev]\n        for i in range(len(self.nhs)):\n            V = self.Vs[i]\n            Zprev = np.tanh(Zprev @ V[1:, :] + V[0:1, :])\n            Z.append(Zprev)\n        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]\n        # Do backward pass, starting with delta in output layer\n        delta = -(T - Y) / (X.shape[0] * T.shape[1])\n        dW = np.vstack((np.ones((1, delta.shape[0])) @ delta, \n                        Z[-1].T @ delta))\n        dVs = []\n        delta = (1 - Z[-1]**2) * (delta @ self.W[1:, :].T)\n        for Zi in range(len(self.nhs), 0, -1):\n            Vi = Zi - 1  # because X is first element of Z\n            dV = np.vstack((np.ones((1, delta.shape[0])) @ delta,\n                            Z[Zi-1].T @ delta))\n            dVs.insert(0, dV)\n            delta = (delta @ self.Vs[Vi][1:, :].T) * (1 - Z[Zi-1]**2)\n        return self._pack(dVs, dW)\n\n    def train(self, X, T, nIterations=100, verbose=False,\n              weightPrecision=0, errorPrecision=0, saveWeightsHistory=False):\n        \n        if self.Xmeans is None:\n            self.Xmeans = X.mean(axis=0)\n            self.Xstds = X.std(axis=0)\n            self.Xconstant = self.Xstds == 0\n            self.XstdsFixed = copy(self.Xstds)\n            self.XstdsFixed[self.Xconstant] = 1\n        X = self._standardizeX(X)\n\n        if T.ndim == 1:\n            T = T.reshape((-1, 1))\n\n        if self.Tmeans is None:\n            self.Tmeans = T.mean(axis=0)\n            self.Tstds = T.std(axis=0)\n            self.Tconstant = self.Tstds == 0\n            self.TstdsFixed = copy(self.Tstds)\n            self.TstdsFixed[self.Tconstant] = 1\n        T = self._standardizeT(T)\n\n        startTime = time.time()\n\n        scgresult = ml.scg(self._pack(self.Vs, self.W),\n                            self._objectiveF, self._gradientF,\n                            X, T,\n                            xPrecision=weightPrecision,\n                            fPrecision=errorPrecision,\n                            nIterations=nIterations,\n                            verbose=verbose,\n                            ftracep=True,\n                            xtracep=saveWeightsHistory)\n\n        self._unpack(scgresult['x'])\n        self.reason = scgresult['reason']\n        self.errorTrace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to _unstandardize the MSEs\n        self.numberOfIterations = len(self.errorTrace)\n        self.trained = True\n        self.weightsHistory = scgresult['xtrace'] if saveWeightsHistory else None\n        self.trainingTime = time.time() - startTime\n        return self\n\n    def use(self, X, allOutputs=False):\n        Zprev = self._standardizeX(X)\n        Z = [Zprev]\n        for i in range(len(self.nhs)):\n            V = self.Vs[i]\n            Zprev = np.tanh(Zprev @ V[1:, :] + V[0:1, :])\n            Z.append(Zprev)\n        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]\n        Y = self._unstandardizeT(Y)\n        return (Y, Z[1:]) if allOutputs else Y\n\n    def getNumberOfIterations(self):\n        return self.numberOfIterations\n\n    def getErrors(self):\n        return self.errorTrace\n\n    def getTrainingTime(self):\n        return self.trainingTime\n\n    def getWeightsHistory(self):\n        return self.weightsHistory\n\n    def draw(self, inputNames=None, outputNames=None, gray=False):\n        ml.draw(self.Vs, self.W, inputNames, outputNames, gray)\n \nif __name__ == '__main__':\n\n    X = np.arange(10).reshape((-1, 1))\n    T = X + 2\n\n    net = NeuralNetwork(1, 0, 1)\n    net.train(X, T, 10)\n    print(net)\n    \n    net = NeuralNetwork(1, [5, 5], 1)\n    net.train(X, T, 10)\n    print(net)")


# If the above class definition is placed in a file named *neuralnetworks.py*, then an instance of this class can be instantiated using code like
# 
#     import neuralnetworks as nn
#     nnet = nn.NeuralNetwork(1, 4, 1)
#     
# The files *neuralnetworks.py* and *mlutilities.py*  must be in your working directory.

# In[16]:


import sys
sys.path = ['.'] + sys.path
sys.path


# In[17]:


import neuralnetworks
neuralnetworks


# In[18]:


import neuralnetworks as nn
import imp
imp.reload(nn)  # in case neuralnetworks.py has been changed

# Make some training data
nSamples = 10
Xtrain = np.linspace(0, 10, nSamples).reshape((-1, 1))
Ttrain = 1.5 + 0.6 * Xtrain + 8 * np.sin(2.2 * Xtrain)
Ttrain[np.logical_and(Xtrain > 2, Xtrain < 3)] *= 3
Ttrain[np.logical_and(Xtrain > 5, Xtrain < 7)] *= 3

nSamples = 100
Xtest = np.linspace(0, 10, nSamples).reshape((-1, 1)) + 10.0/nSamples/2
Ttest = 1.5 + 0.6 * Xtest + 8 * np.sin(2.2  *Xtest) + np.random.uniform(-2, 2, size=(nSamples, 1))
Ttest[np.logical_and(Xtest > 2, Xtest < 3)] *= 3
Ttest[np.logical_and(Xtest > 5, Xtest < 7)] *= 3

# Create the neural network, with one hidden layer of 4 units
nnet = nn.NeuralNetwork(1, [4], 1)

# Train the neural network
nnet.train(Xtrain, Ttrain, nIterations=1000)

# Print some information when done
print('SCG stopped after', nnet.getNumberOfIterations(), 'iterations:', nnet.reason)
print('Training took', nnet.getTrainingTime(), 'seconds.')

# Print the training and testing RMSE
Ytrain = nnet.use(Xtrain)
Ytest, Ztest = nnet.use(Xtest, allOutputs=True)
print("Final RMSE: train", np.sqrt(np.mean((Ytrain - Ttrain)**2)),
      "test", np.sqrt(np.mean((Ytest - Ttest)**2)))

plt.figure(figsize=(15, 15))

nHLayers = len(nnet.nhs)
nPlotRows = 3 + nHLayers

plt.subplot(nPlotRows, 1, 1)
plt.plot(nnet.getErrors())
plt.title('Regression Example')

plt.subplot(nPlotRows, 1, 2)
plt.plot(Xtrain, Ttrain, 'o-', label='Training Data')
plt.plot(Xtrain, Ytrain, 'o-', label='Train NN Output')
plt.ylabel('Training Data')
plt.legend(loc='lower right', prop={'size': 9})

plt.subplot(nPlotRows, 1, 3)
plt.plot(Xtest, Ttest, 'o-', label='Test Target')
plt.plot(Xtest, Ytest, 'o-', label='Test NN Output')
plt.ylabel('Testing Data')
plt.xlim(0, 10)
plt.legend(loc='lower right', prop={'size': 9})
for i in range(nHLayers):
    layer = nHLayers - i - 1
    plt.subplot(nPlotRows, 1, i + 4)
    plt.plot(Xtest, Ztest[layer])
    plt.xlim(0,10)
    plt.ylim(-1.1,1.1)
    plt.ylabel('Hidden Units')
    plt.text(8,0, 'Layer {}'.format(layer+1))


# In[19]:


nnet.draw(['x'],['y'])


# In[20]:


nnet


# What happens if we add another two hidden layers, for a total of three hidden layers?  Let's use 5 units in each hidden layer.

# In[21]:


nnet = nn.NeuralNetwork(1, [5, 5], 1)


# In[22]:


nnet


# The rest of the code is the same.  Even the plotting code written above works for as many hidden layers as we create.

# In[23]:


nnet.train(Xtrain, Ttrain, nIterations=1000)
print("SCG stopped after", nnet.getNumberOfIterations(), "iterations:", nnet.reason)
Ytrain = nnet.use(Xtrain)
Ytest, Ztest = nnet.use(Xtest, allOutputs=True)
print("Final RMSE: train", np.sqrt(np.mean((Ytrain - Ttrain)**2)),
      "test", np.sqrt(np.mean((Ytest - Ttest)**2)))

plt.figure(figsize=(10, 15))

nHLayers = len(nnet.nhs)
nPlotRows = 3 + nHLayers

plt.subplot(nPlotRows, 1, 1)
plt.plot(nnet.getErrors())
plt.title('Regression Example')

plt.subplot(nPlotRows, 1, 2)
plt.plot(Xtrain, Ttrain, 'o-', label='Training Data')
plt.plot(Xtrain, Ytrain, 'o-', label='Train NN Output')
plt.ylabel('Training Data')
plt.legend(loc='lower right', prop={'size': 9})

plt.subplot(nPlotRows, 1, 3)
plt.plot(Xtest, Ttest, 'o-', label='Test Target')
plt.plot(Xtest, Ytest, 'o-', label='Test NN Output')
plt.ylabel('Testing Data')
plt.xlim(0, 10)
plt.legend(loc='lower right', prop={'size': 9})
for i in range(nHLayers):
    layer = nHLayers-i-1
    plt.subplot(nPlotRows, 1, i + 4)
    plt.plot(Xtest, Ztest[layer])
    plt.xlim(0, 10)
    plt.ylim(-1.1, 1.1)
    plt.ylabel('Hidden Units')
    plt.text(8, 0, 'Layer {}'.format(layer + 1))


# For more fun, wrap the above code in a function to make it easy to try different network structures.

# In[24]:


def run(Xtrain, Ttrain, Xtest, Ttest, hiddenUnits, nIterations=100, verbose=False):
    if Xtrain.shape[1] != 1 or Ttrain.shape[1] != 1:
        print('This function written for one-dimensional input samples, X, and one-dimensional targets, T.')
        return
    
    nnet = nn.NeuralNetwork(1, hiddenUnits,1 )

    nnet.train(Xtrain, Ttrain, nIterations=nIterations, verbose=verbose)

    Ytrain = nnet.use(Xtrain)
    Ytest, Ztest = nnet.use(Xtest, allOutputs=True)
    print('Training took {:.4f} seconds.'.format(nnet.getTrainingTime()))
    print("Final RMSE: train", np.sqrt(np.mean((Ytrain - Ttrain)**2)),
          "test", np.sqrt(np.mean((Ytest - Ttest)**2)))

    plt.figure(figsize=(10, 15))
    nHLayers = len(nnet.nhs)
    nPlotRows = 3 + nHLayers

    plt.subplot(nPlotRows, 1, 1)
    plt.plot(nnet.getErrors())
    plt.title('Regression Example')

    plt.subplot(nPlotRows, 1, 2)
    plt.plot(Xtrain, Ttrain, 'o-', label='Training Data')
    plt.plot(Xtrain, Ytrain, 'o-', label='Train NN Output')
    plt.ylabel('Training Data')
    plt.legend(loc='lower right', prop={'size': 9})

    plt.subplot(nPlotRows, 1, 3)
    plt.plot(Xtest, Ttest, 'o-', label='Test Target')
    plt.plot(Xtest, Ytest, 'o-', label='Test NN Output')
    plt.ylabel('Testing Data')
    plt.xlim(0, 10)
    plt.legend(loc='lower right', prop={'size': 9})
    for i in range(nHLayers):
        layer = nHLayers-i-1
        plt.subplot(nPlotRows, 1, i+4)
        plt.plot(Xtest, Ztest[layer])
        plt.xlim(0, 10)
        plt.ylim(-1.1, 1.1)
        plt.ylabel('Hidden Units')
        plt.text(8, 0, 'Layer {}'.format(layer+1))
    return nnet


# In[25]:


run(Xtrain, Ttrain, Xtest, Ttest, (2, 2), nIterations=1000)


# In[26]:


run(Xtrain, Ttrain, Xtest, Ttest, (2, 2, 2, 2), nIterations=1000)


# In[27]:


[2]*6


# In[28]:


run(Xtrain, Ttrain, Xtest, Ttest, [2]*6, nIterations=4000)


# Can you say "deep learning"?

# This last example doesn't always work.  Depends a lot on good initial random weight values. Go back to the above cell and run it again and again, until you see it not work.

# In[29]:


nnet = run(Xtrain, Ttrain, Xtest, Ttest, [50,10,3,1,3,10,50], nIterations=1000)
nnet


# Run the above cell several times to see very different solutions as observed in the pattern of hidden layer outputs.

# In[30]:


plt.figure(figsize=(10, 10))
nnet.draw()


# Let's try saving the weights after each iteration and then make a movie showing how the outputs of each layer change as the network is trained.

# In[31]:


def run(Xtrain, Ttrain, hiddenUnits, nIterations=100, verbose=False, saveWeightsHistory=False):
    if Xtrain.shape[1] != 1 or Ttrain.shape[1] != 1:
        print('This function written for one-dimensional input samples, X, and one-dimensional targets, T.')
        return
    nnet = nn.NeuralNetwork(1, hiddenUnits,1 )
    nnet.train(Xtrain, Ttrain, nIterations=nIterations, verbose=verbose, saveWeightsHistory=saveWeightsHistory)
    return nnet


# In[32]:


nnet = run(Xtrain, Ttrain, [50,10,3,1,3,10,50], nIterations=500, verbose=False, saveWeightsHistory=True)
nnet


# In[33]:


nnet = run(Xtrain, Ttrain, [10, 5, 5], nIterations=1000, verbose=False, saveWeightsHistory=True)
nnet


# In[34]:


from matplotlib import animation, rc
# from IPython.display import HTML
rc('animation', html='jshtml')
rc('animation', embed_limit=50)


# In[35]:


weightsHistory = nnet.getWeightsHistory()
nnet._unpack(weightsHistory[0, :])
Ytrain = nnet.use(Xtrain)
Ytest, Ztest = nnet.use(Xtest, allOutputs=True)

nHLayers = len(nnet.nhs)
nPlotRows = 3 + nHLayers

fig = plt.figure(figsize=(10, 15))
plt.subplot(nPlotRows, 1, 1)
errors = nnet.getErrors()
errorsLine = plt.plot(0,errors[0])[0]
plt.xlim(0,len(errors))
plt.ylim(0,np.max(errors)*1.1)

plt.title('Regression Example')

plt.subplot(nPlotRows, 1, 2)
plt.plot(Xtrain, Ttrain, 'o-', label='Training Data')
trainLine = plt.plot(Xtrain, Ytrain, 'o-', label='Train NN Output')[0]
plt.ylabel('Training Data')
plt.legend(loc='lower right', prop={'size': 9})

plt.subplot(nPlotRows, 1, 3)
plt.plot(Xtest, Ttest, 'o-', label='Test Target')
testLine = plt.plot(Xtest, Ytest, 'o-', label='Test NN Output')[0]
plt.ylabel('Testing Data')
plt.xlim(0, 10)
plt.legend(loc='lower right', prop={'size': 9})
hiddenOutputLines = []
for i in range(nHLayers):
    layer = nHLayers-i-1
    plt.subplot(nPlotRows, 1, i+4)
    hiddenOutputLines.append( plt.plot(Xtest, Ztest[layer]) )
    plt.xlim(0, 10)
    plt.ylim(-1.1, 1.1)
    plt.ylabel('Hidden Units')
    plt.text(8, 0, 'Layer {}'.format(layer+1))
    
updatesPerFrame = 4

def animator(framei):
    step = framei * updatesPerFrame
    nnet._unpack(weightsHistory[step,:])
    Ytrain = nnet.use(Xtrain)
    Ytest, Ztest = nnet.use(Xtest, allOutputs=True)
    errorsLine.set_data(range(step),errors[:step])
    trainLine.set_ydata(Ytrain)
    testLine.set_ydata(Ytest)
    
    for iLayer in range(len(hiddenOutputLines)):
        HOLines = hiddenOutputLines[iLayer]
        Zlayer = Ztest[nHLayers - iLayer - 1]
        for iUnit in range(len(HOLines)):
            HOLines[iUnit].set_ydata(Zlayer[:,iUnit])
    return [errorsLine, trainLine, testLine] + hiddenOutputLines


# In[36]:


nFrames = len(nnet.getErrors()) // updatesPerFrame # integer divide
anim = animation.FuncAnimation(fig, animator, frames=nFrames, interval=50, blit=False)


# In[37]:


# HTML(anim.to_html5_video())
# HTML(anim.to_jshtml())
anim


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\Yv}{\mathbf{Y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\gv}{\mathbf{g}}
# \newcommand{\Hv}{\mathbf{H}}
# \newcommand{\dv}{\mathbf{d}}
# \newcommand{\Vv}{\mathbf{V}}
# \newcommand{\vv}{\mathbf{v}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\Zv}{\mathbf{Z}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# $

# # A2 Adam vs SGD

# Abigail Rictor

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import IPython.display as ipd  # for display and clear_output
import time  # for sleep


# Let's create a long vector of all of the weights in both layers.  Then we can define the weight matrix for each layer as views into this vector.  This allows us to take steps down the gradient of the error function by incrementing the whole weight vector.
# 
# Spend a little time understanding the difference between numpy views and copies.  [Here is a good tutorial](http://www.jessicayung.com/numpy-views-vs-copies-avoiding-costly-mistakes/).

# In[2]:


def make_weights(shapes):
    '''make_weights(shape): weights is list of pairs of (n_inputs, n_units) for each layer.
    n_inputs includes the constant 1 input.
    Returns weight vector w of all weights, and list of matrix views into w for each layer'''
    # Make list of number of weights in each layer
    n_weights_each_matrix = [sh[0] * sh[1] for sh in shapes]
    # Total number of weights
    n_weights = sum(n_weights_each_matrix)
    # Allocate weight vector with component for each weight
    w = np.zeros(n_weights)
    # List Ws will be list of weight matrix views into w for each layer
    Ws = make_views_on_weights(w, shapes)
    return w, Ws    


# In[3]:


def make_views_on_weights(w, shapes):    
    Ws = []
    first = 0
    for sh in shapes:
        # Create new view of w[first:last]
        last = first + sh[0] * sh[1]
        # Create new view of w[first:last] as matrix W to be matrix for a layer
        W = w[first:last].reshape(sh)
        # Initialize weight values to small uniformly-distributed values
        n_inputs = sh[0]
        scale = 1.0 / np.sqrt(n_inputs)
        W[:] = np.random.uniform(-scale, scale, size=sh)
        # Add to list of W matrices, Ws.
        Ws.append(W)
        first = last
    return Ws


# In[4]:


# Set parameters of neural network
nHiddens = 10
nOutputs = 1

# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
Vshape = (1 +1, nHiddens)
Wshape = (nHiddens + 1, nOutputs)
w, [V, W] = make_weights([Vshape, Wshape])

print('w\n', w)
print('V\n', V)
print('W\n', W)


# Now for some functions for calculating the output of our network, and for backpropagating the error to get a gradient of error with respect to all weights.

# In[5]:


def forward(Ws, X1):
    # Forward pass on training data
    V, W = Ws
    Z = np.tanh(X1 @ V)
    Z1 = np.insert(Z, 0, 1, 1)
    Y = Z1 @ W
    return Z1, Y


# In[6]:


def backward(w, Ws, X1, Z1, T, error):
    V, W = Ws
    # Backward pass. 
    # Calculate the gradient of squared error with respect to all weights in w.
    #   Order of in w is all hidden layer weights followed by all output layer weights,
    #   so gradient values are ordered this way.
    gradient =  np.hstack(((- X1.T @ ( ( error @ W[1:, :].T) * (1 - Z1[:, 1:]**2))).flat,  # for hidden layer
                          (- Z1.T @ error).flat))  # for output layer
    return gradient


# Given these functions, we can now define the stochastic gradient descent, `sgd`, procedure to update the weights.

# In[7]:


def sgd_init():
    pass

def sgd(w, Ws, X1, T, learning_rate):
    Z1, Y = forward(Ws, X1)

    # Error in output
    n_samples = X1.shape[0]
    n_outputs = T.shape[1]
    
    error = (T - Y) / (n_samples + n_outputs)

    gradient = backward(w, Ws, X1, Z1, T, error)
   
    # update values of w, in place. Don't need to return it.
    
    w -= learning_rate * gradient


# Here is another way to update the weights, the `adam` procedure.  See [this discussion](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) of Adam and other gradient descent methods.

# In[8]:


grad_trace = 0
grad_squared_trace = 0
update_step = 0

def adam_init():
    global grad_trace, grad_squared_trace, update_step

    grad_trace = 0
    grad_squared_trace = 0
    update_step = 0
    
def adam(w, Ws, X1, T, learning_rate):
    global grad_trace, grad_squared_trace, update_step

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    Z1, Y = forward(Ws, X1)

    # Error in output
    error = T - Y

    gradient = backward(w, Ws, X1, Z1, T, error)
    
    # approximate first and second moment
    grad_trace = beta1 * grad_trace + (1 - beta1) * gradient
    grad_squared_trace = beta2 * grad_squared_trace + (1 - beta2) * np.square(gradient)
    
    # bias corrected moment estimates
    grad_hat = grad_trace / (1 - beta1 ** (update_step + 1) )
    grad_squared_hat = grad_squared_trace / (1 - beta2 ** (update_step + 1) )
                
    dw = grad_hat / (np.sqrt(grad_squared_hat) + epsilon)
    
    n_samples = X1.shape[0]
    n_outputs = T.shape[1]
    
    # update values of w, in place. Don't need to return it.
    w -= learning_rate / (n_samples + n_outputs) * dw
    
    update_step += 1


# Now, wrap these all together into a function to train a neural network given the training and testing data, the gradient descent function names, the batch size, the number of epochs, the learning rate, and the graphics update rate.

# In[9]:


def train(Xtrain, Ttrain, Xtest, Ttest,
          n_hiddens, 
          gradient_descent_method_init, gradient_descent_method, 
          batch_size, n_epochs, learning_rate, graphics_rate=0):

    if graphics_rate > 0 and Xtrain.shape[1] > 1:
        print('Graphics only works when X has one column (data has one input variable)')
        print('Setting graphics_rate to 0')
        graphics_rate = 0
    
    # Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
    n_inputs = Xtrain.shape[1]
    n_outputs = Ttrain.shape[1]
    Vshape = (1 + n_inputs, n_hiddens)
    Wshape = (1 + n_hiddens, n_outputs)
    w, [V, W] = make_weights([Vshape, Wshape])

    error_trace = np.zeros((n_epochs, 2))

    if graphics_rate > 0:
        fig = plt.figure(figsize=(12, 10))
        
    Xtrain1 = np.insert(Xtrain, 0, 1, 1)
    Xtest1 = np.insert(Xtest, 0, 1, 1)
    n_samples = Xtrain1.shape[0]
        
    gradient_descent_method_init()
    
    for epoch in range(n_epochs):

        # gradient_descent_method_init()
        
        # Reorder samples
        rows = np.arange(n_samples)
        np.random.shuffle(rows)
        
        for first_n in range(0, n_samples, batch_size):
            last_n = first_n + batch_size
            rows_batch = rows[first_n:last_n]
            Xtrain1_batch = Xtrain1[rows_batch, :]
            Ttrain_batch = Ttrain[rows_batch, :]
            # gradient_descent method changes values of w
            gradient_descent_method(w, [V, W], Xtrain1_batch, Ttrain_batch, learning_rate)
    
        # error traces for plotting
        Z1train, Ytrain = forward([V, W], Xtrain1)
        error_trace[epoch, 0] = np.sqrt(np.mean(((Ttrain - Ytrain)**2)))
    
        Z1test, Ytest = forward([V, W], Xtest1)
        error_trace[epoch, 1] = np.sqrt(np.mean((Ytest - Ttest)**2))

        if graphics_rate > 0 and (epoch % graphics_rate == 0 or epoch == n_epochs - 1):
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.plot(error_trace[:epoch, :])
            plt.ylim(0, 0.4)
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.legend(('Train','Test'), loc='upper left')
        
            plt.subplot(3, 1, 2)
            plt.plot(Xtrain, Ttrain, 'o-', Xtest, Ttest, 'o-', Xtrain, Ytrain, 'o-')
            plt.xlim(-1, 1)
            plt.ylim(-0.2, 1.6)
            plt.legend(('Training', 'Testing', 'Model'), loc='upper left')
            plt.xlabel('$x$')
            plt.ylabel('Actual and Predicted $f(x)$')
        
            plt.subplot(3, 1, 3)
            plt.plot(Xtrain, Z1train[:, 1:])  # Don't plot the constant 1 column
            plt.ylim(-1.1, 1.1)
            plt.xlabel('$x$')
            plt.ylabel('Hidden Unit Outputs ($z$)');
        
            ipd.clear_output(wait=True)
            ipd.display(fig)

    ipd.clear_output(wait=True)

    return Ytrain, Ytest, error_trace


# Here are some demonstrations.

# In[10]:


# Make some training data
n = 20
Xtrain = np.linspace(0.,20.0,n).reshape((n,1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.sin(Xtrain * 3) + 0.01 * np.random.normal(size=(n, 1))
Xtrain = Xtrain / 10

# Make some testing data
n = n // 3
Xtest = np.linspace(0, 20, n).reshape((-1, 1)) - 10
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.2 * np.sin(Xtest + 10) +  0.1 * np.sin(Xtest * 3) + 0.01 * np.random.normal(size=(n, 1))
Xtest = Xtest / 10


# In[79]:


Ytrain, Ytest, error_trace = train(Xtrain, Ttrain, Xtest, Ttest, n_hiddens=20, 
                       gradient_descent_method_init=sgd_init, gradient_descent_method=sgd,
                       batch_size=Xtrain.shape[0], n_epochs=200000, learning_rate=0.2, graphics_rate=5000)
print('Final RMSE', np.sqrt(np.mean((Ttest - Ytest)**2)))


# In[ ]:


Ytrain, Ytest, error_trace = train(Xtrain, Ttrain, Xtest, Ttest, n_hiddens=20, 
                       gradient_descent_method_init=adam_init, gradient_descent_method=adam,
                       batch_size=Xtrain.shape[0], n_epochs=100000, learning_rate=0.05, graphics_rate=5000)
print('Final RMSE', np.sqrt(np.mean((Ttest - Ytest)**2)))


# In[12]:


n_samples = Xtrain.shape[0]

_, _, error_trace_adam = train(Xtrain, Ttrain, Xtest, Ttest, n_hiddens=20, 
                       gradient_descent_method_init=adam_init, gradient_descent_method=adam,
                       batch_size=n_samples, n_epochs=20000, learning_rate=0.01, graphics_rate=0)

_, _, error_trace_sgd = train(Xtrain, Ttrain, Xtest, Ttest, n_hiddens=20, 
                       gradient_descent_method_init=sgd_init, gradient_descent_method=sgd,
                       batch_size=n_samples, n_epochs=20000, learning_rate=0.01, graphics_rate=0)

plt.plot(np.hstack((error_trace_sgd[:, 1:], error_trace_adam[:, 1:])))
plt.legend(('SGD', 'Adam'))

print('SGD', error_trace_sgd[-1, 1], 'Adam', error_trace_adam[-1, 1])


# # Search for Good Parameter Values on a New Data Set

# Now your work begins.  First, download this [Real Estate Valuation Data](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set) from the UCI machine learning repository. Read it in to python and form an input matrix `X` that contains six columns, and target matrix `T` of one column containing the house price of unit area.  This is easiest to do with the [pandas](https://pandas.pydata.org/) package.  Check out the [Getting Started](http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) material at the pandas site.  Near the bottom of that page are some simple examples of how to use the `read_excel` pandas function.  Pretty handy since the Real Estate data is an `.xlsx` file.  Other helpful functions are the `drop` function that can be used to remove a column, such as the one labeled `No` in the data file, which is just an index that we should ignore, and `data.columns.tolist()` where `data` is a Dataframe.  Also note that a Dataframe can be converted to a `numpy` array by `npdata = np.array(data)` where `data` again is a Dataframe.
# 
# We want to try to predict the target value from the six input values.
# 
# Randomly partition the data into an 80% partition for training, making `Xtrain` and `Ttrain`, and a 20% partition for testing, makng `Xtest` and `Ttest`.
# 
# Standardize the input `Xtrain` and `Xtest` matrices by subtracting by the column means and dividing by the column standard deviations, with the means and standard deviations determined only from the `Xtrain` matrix.

# In[13]:


import pandas as pd
import numpy as np


# In[47]:


def partition(data):
    data_copy = np.random.permutation(data)
    division_index = int(.8*len(data_copy))
    training = data_copy[:division_index]
    testing = data_copy[division_index:]               
    return training, testing

def separate_targets(data):
    X = data[:, :data.shape[1]-1]
    T = data[:,data.shape[1]-1:data.shape[1]]
    return X, T


# In[48]:


data = pd.read_excel('data.xlsx', 'Sheet1', index_col=None, na_values=['NA']).drop("No", axis=1)
npdata = np.array(data)
training, testing = partition(npdata)
Xtrain, Ttrain = separate_targets(training)
Xtest, Ttest = separate_targets(testing)


# Using the one-hidden layer neural network implemented here, train neural networks in two ways: one with SGD and one with Adam.  For each, try at least three values for each of the following parameters:
# 
#   * number of hidden units, from 1 to 50,
#   * batch size,
#   * number of epochs
#   * learning_rate
# 
# Create a table of results containing the algorithm name ('sgd' or 'adam'), the values of the above four parameters, and the RMSE on the training and testing data.  Since this is a mixed-type table, use the `pandas` package.  Sort your table by the test RMSE. 
# 
# Here are some clues on how to do this. To initialize a `pandas` Dataframe, called `results` do
# 
#       import pandas as pd
#       
#       results = pd.DataFrame(columns=['Algorithm', 'Epochs', 'Learning Rate', 'Hidden Units', 'Batch Size', 
#                                 'RMSE Train', 'RMSE Test'])
# To add a row to this, do
# 
#       results.loc[len(results)] = [algo, n_epochs, lr, nh, bs, rmse(Ytrain, Ttrain), rmse(Ytest, Ttest)]
#       
# assuming those variables have appropriate values.  Then, to sort `results` by `RMSE Test` and just see the top 50 entries, do
# 
#       results.sort_values('RMSE Test').head(50)

# In[16]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))


# In[50]:


def run_parameters(Xtrain, Ttrain, Xtest, Ttest, epoch_list, learning_rate_list, hidden_unit_list, batch_size_list, verbose=False):
    results = pd.DataFrame(columns=['Algorithm', 'Epochs', 'Learning Rate', 'Hidden Units', 'Batch Size', 
                            'RMSE Train', 'RMSE Test'])
    for algorithm in ['adam', 'sgd']:
        init = adam_init if algorithm is 'adam' else sgd_init
        method = adam if algorithm is 'adam' else sgd
        for epoch in epoch_list:
            for learning_rate in learning_rate_list:
                for hidden_units in hidden_unit_list:
                    for batch_size in batch_size_list:
                        Ytrain, Ytest, error_trace = train(Xtrain, Ttrain, Xtest, Ttest, n_hiddens=20, 
                                               gradient_descent_method_init=init, gradient_descent_method=method,
                                               batch_size=batch_size, n_epochs=epoch, learning_rate=learning_rate, graphics_rate=0)
                        results.loc[len(results)] = [algorithm, epoch, learning_rate, hidden_units, batch_size, rmse(Ytrain, Ttrain), rmse(Ytest, Ttest)]
                    
    return results


# In[51]:


data = np.loadtxt('machine.data', delimiter=',', usecols=range(2, 10))
X = data[:, :-2]
T = data[:, -2:-1]
Xtrain = X[:160, :]
Ttrain = T[:160, :]
Xtest = X[160:, :]
Ttest = T[160:, :]

means = Xtrain.mean(0)
stds = Xtrain.std(0)
Xtrains = (Xtrain - means) / stds
Xtests = (Xtest - means) / stds


# Put the above steps into a new function named `run_parameters` that accepts the arguments
# 
#     * Xtrain, standardized
#     * Ttrain
#     * Xtest, standardized
#     * Ttest
#     * list of numbers of epochs
#     * list of learning rates
#     * list of numbers of hidden units
#     * list of batch sizes
#     * verbose, if True then print results of each parameter value combination
# 
# and returns a pandas DataFrame containing the results of runs for all combinations of the above parameter values. The DataFrame must have columns titled `[Algorithm', 'Epochs', 'Learning Rate', 'Hidden Units', 'Batch Size', 'RMSE Train', 'RMSE Test']`.  So, if eac of the above lists contains two values, the resulting DataFrame must have 16 rows.

# Describe your experiments, including how you decided on what values to test, and what the results tell you.  
# 
# Extract the `RMSE test` values for all values of `Hidden Units` using the best values for the other parameters.  Here is an example of how to do this.
# 
#     nh =results.loc[(results['Algorithm'] == 'adam') & (results['Epochs'] == 100) &
#                     (results['Learning Rate'] == 0.002) & (results['Batch Size'] == 1)]
#                     
# Now you can plot the training and testing RMSE versus the hidden units.  Describe what you see.

# In[84]:


results = run_parameters(Xtrains, Ttrain, Xtests, Ttest, [100, 1000, 5000], [0.01, 0.05, .1], [1, 25, 50], [25, 50, 75], verbose=False)
nh = results.loc[(results['Algorithm'] == 'adam') & (results['Epochs'] == 100) &
                (results['Learning Rate'] == 0.001) & (results['Batch Size'] == 1)]


# I used the following values in my test run.
# 
# epochs: 100, 500, 1000
# 
# learning rates: .01, .05, .1 
# 
# hidden units: 1, 25, 50
# 
# batch sizes: 25, 50, 75
# 
# My intention has been to gather data for a great range of potential values for each input, though my epoch counts are admittedly low (this is from a practical standpoint).
# 
# Below are the ten best RMSE Test values from this run, showing 'sgd' as the prevailing algorithm in this context. There are also plots showing the RMSE Train (orange) and Test (green) values in comparison to the number of hidden units (blue) in a test with 1000 epochs, a batch size of 25, and a learning rate controlled values. I've included one for the sgd algorithm (top) and one for the adam algorithm (bottom).

# In[86]:


sgd_nh = results.loc[(results['Algorithm'] == 'sgd') & (results['Epochs'] == 100) &
                (results['Learning Rate'] == 0.05) & (results['Batch Size'] == 25)].drop("Epochs", axis=1).drop("Batch Size", axis=1).drop("Learning Rate", axis=1)

adam_nh = results.loc[(results['Algorithm'] == 'adam') & (results['Epochs'] == 100) &
                (results['Learning Rate'] == 0.05) & (results['Batch Size'] == 25)].drop("Epochs", axis=1).drop("Batch Size", axis=1).drop("Learning Rate", axis=1)

sgd_nh.plot(title="SGD")

adam_nh.plot(title="ADAM")

results.sort_values('RMSE Test').head(10)


# The data above shows sgd to have more accurate results, and that those results are more accurate at the midpoint of possible hidden units values (25). This reflects what I know about the adam optimization algorithm, showing that using fewer epochs, sgd can be as effective or moreso (as seen here) than the optimization, which allows more effective training relative to the number of epochs, unlike sgd which flattens out at an earlier point.

# ## Grading

# Download [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  You should see a perfect execution score of 90 out of 90 points if your functions are defined correctly. The remaining 10 points will be based on the results you obtain from the energy data and on your discussions.
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A2.ipynb' with 'Lastname' being your last name, and then save this notebook.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook.  It will include additional tests.  You need not include code to test that the values passed in to your functions are the correct form.

# In[81]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Repeat the evaluation of parameter values for another data set of your choice.
#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# * [Assignment 3: Neural Network Regression with tanh and relu](#Assignment-3:-Neural-Network-Regression-with-tanh-and-relu)
# 	* [Overview](#Overview)
# 	* [Neural Network Code](#Neural-Network-Code)
# 	* [Neural Network Performance with Different Hidden Layer Structures and Numbers of Training Iterations](#Neural-Network-Performance-with-Different-Hidden-Layer-Structures-and-Numbers-of-Training-Iterations)
# 		* [Example with Toy Data](#Example-with-Toy-Data)
# 		* [Experiments with Automobile Data](#Experiments-with-Automobile-Data)
# 	* [Experiments with relu activation function](#Experiments-with-relu-activation-function)
# 	* [Text descriptions](#Text-descriptions)
# 	* [Grading and Check-in](#Grading-and-Check-in)
# 	* [Extra Credit](#Extra-Credit)
# 

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 3: Neural Network Regression with tanh and relu

# Abigail Rictor
# *Type your name here and rewrite all of the following sections.  Add more sections to present your code, results, and discussions.*

# ## Overview

# The goal of this assignment is to 
#    * gain some experience in comparing different sized neural networks when applied to a data set, 
#    * implement a different activation function, relu, and compare with Tanh, and
#    * learn about object-oriented programming in python and to gain some experience in comparing different sized neural networks when applied to a data set.
# 
# Starting with the ```NeuralNetwork``` class from the lecture notes, you will create a new version of that class, apply it to a data set, and discuss the results. You will then create a second version, named ```NeuralNetwork_relu```, that uses the relu activation function instead of the tanh function.

# In[91]:


#!curl -O http://www.cs.colostate.edu/~anderson/cs445/notebooks/mlutilities.tar
#!tar xvf mlutilities.tar
#!del mlutilities.tar
# !curl -O http://www.cs.colostate.edu/~anderson/cs445/notebooks/machine.data


# ## Neural Network Code

# Start with the ```NeuralNetwork``` class defined in lecture notes 09. Put that class definition as written into *neuralnetworks.py* into your current directory.  Also place *mlutilities.py* from lecture notes 09 in your current directory. If this is done correctly, then the following code should run and produce results similar to what is shown here.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import neuralnetworks as nn

X = np.arange(10).reshape((-1,1))
T = np.sin(X)

nnet = nn.NeuralNetwork(1, [10], 1)
nnet.train(X, T, 100, verbose=True)
nnet


# In[6]:


plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(nnet.getErrors())

plt.subplot(3, 1, 2)
plt.plot(X, T, 'o-', label='Actual')
plt.plot(X, nnet.use(X), 'o-', label='Predicted')

plt.subplot(3, 1, 3)
nnet.draw()


# Now extract the parts of the neural network code that refer to the activation function and its derivative into two new methods.  Modify the code in *neuralnetworks.py* by adding these two methods to the ```NeuralNetwork``` class:
# 
#     def activation(self, weighted_sum):
#         return np.tanh(weighted_sum)
#         
#     def activation_derivative(self, activation_value):
#         return 1 - activation_value * activation_value
#         
# Now replace the code in the appropriate places in the ```NeuralNetwork``` class so that ```np.tanh``` is replaced with a call to the ```self.activation``` method and its derivative is replaced by calls to ```self.activation_derivative```. Tell jupyter to reload your changed code using the following lines.

# In[5]:


import imp
imp.reload(nn)

nnet = nn.NeuralNetwork(1, [10], 1)


# In[8]:


[nnet.activation(s) for s in [-2, -0.5, 0, 0.5, 2]]


# In[9]:


[nnet.activation_derivative(nnet.activation(s)) for s in [-2, -0.5, 0, 0.5, 2]]


# In[10]:


nnet.train(X, T, 100, verbose=True)
nnet


# In[11]:


plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(nnet.getErrors())

plt.subplot(3, 1, 2)
plt.plot(X, T, 'o-', label='Actual')
plt.plot(X, nnet.use(X), 'o-', label='Predicted')

plt.subplot(3, 1, 3)
nnet.draw()


# ## Neural Network Performance with Different Hidden Layer Structures and Numbers of Training Iterations

# ### Example with Toy Data

# Using your new ```NeuralNetwork``` class, you can compare the error obtained on a given data set by looping over various hidden layer structures.  Here is an example using the simple toy data from above.

# In[12]:


nRows = X.shape[0]
rows = np.arange(nRows)
np.random.shuffle(rows)
nTrain = int(nRows * 0.8)
trainRows = rows[:nTrain]
testRows = rows[nTrain:]
Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
Xtest, Ttest = X[testRows, :], T[testRows, :]


# In[13]:


Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[14]:


def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))


# In[15]:


import pandas as pd


# In[16]:


def run_parameters(Xtrain, Ttrain, Xtest, Ttest, epochs_list, nh_list, verbose=True):

    n_inputs = Xtrain.shape[1]  # number of columns in X
    n_outputs = Ttrain.shape[1]  # number of columns in T
    
    results = pd.DataFrame(columns=['Epochs', 'Hidden Units', 'RMSE Train', 'RMSE Test'])
    for n_epochs in epochs_list:        
        for nh in nh_list:
            nnet = nn.NeuralNetwork(Xtrain.shape[1], nh, n_outputs)
            nnet.train(Xtrain, Ttrain, n_epochs)
            Ytrain = nnet.use(Xtrain)
            Ytest = nnet.use(Xtest)
            results.loc[len(results)] = [n_epochs, nh, rmse(Ytrain, Ttrain), 
                                         rmse(Ytest, Ttest)]
            if verbose:
                display(results.tail(1))  # not print
    return results


# In[17]:


results = run_parameters(Xtrain, Ttrain, Xtest, Ttest, [10, 100], [[0], [10], [10, 10]])


# In[18]:


results


# In[19]:


hiddens = [[0]] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
hiddens


# In[20]:


results = run_parameters(Xtrain, Ttrain, Xtest, Ttest, [500], hiddens, verbose=False)
results


# In[21]:


errors = np.array(results[['RMSE Train', 'RMSE Test']])
plt.figure(figsize=(10, 10))
plt.plot(errors, 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.xlabel('Hidden Layers Structure')
plt.grid(True)


# For this data (and the random shuffling of the data used when this notebook was run and random assignment of initial weight values), `[20, 20, 20, 20, 20]` produced the lowest test error.  
# 
# Now, using the best hidden layer structure found, write the code that varies the number of training epochs and plot 'RMSE Train' and 'RMSE Test' versus the number of epochs.

# **DO SOMETHING HERE**

# ### Experiments with Automobile Data

# Now, repeat the above steps with the automobile mpg data we have used before.  Set it up to use 
# 
#   * cylinders,
#   * displacement,
#   * weight,
#   * acceleration,
#   * year, and
#   * origin
#   
# as input variables, and
# 
#   * mpg
#   * horsepower
#   
# as output variables.

# In[75]:


# !curl -O http://mlr.cs.umass.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
# !curl -O http://mlr.cs.umass.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names


# In[60]:


def makeMPGData(filename='auto-mpg.data'):
    def missingIsNan(s):
        return np.nan if s == b'?' else float(s)
    data = np.loadtxt(filename, usecols=range(8), converters={3: missingIsNan})
    print("Read",data.shape[0],"rows and",data.shape[1],"columns from",filename)
    goodRowsMask = np.isnan(data).sum(axis=1) == 0
    data = data[goodRowsMask,:]
    print("After removing rows containing question marks, data has",data.shape[0],"rows and",data.shape[1],"columns.")
    T = data[:, [0,3]]
    X = data[:, [1,2,4,5,6,7]]
    Xnames =  ['bias', 'cylinders','displacement','weight','acceleration','year','origin']
    Tnames = ['mpg', 'horsepower']
    return X,T,Xnames,Tnames


# In[63]:


X, T, Xnames, Tnames = makeMPGData()


# In[64]:


nRows = X.shape[0]
rows = np.arange(nRows)
np.random.shuffle(rows)
nTrain = int(nRows * 0.8)
trainRows = rows[:nTrain]
testRows = rows[nTrain:]
Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
Xtest, Ttest = X[testRows, :], T[testRows, :]


# In[65]:


Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[70]:


results = run_parameters(Xtrain, Ttrain, Xtest, Ttest, [10, 100], [[0], [10], [10, 10]])


# In[71]:


results


# In[72]:


hiddens = [[0]] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
hiddens


# In[73]:


results = run_parameters(Xtrain, Ttrain, Xtest, Ttest, [500], hiddens, verbose=False)
results


# In[74]:


errors = np.array(results[['RMSE Train', 'RMSE Test']])
plt.figure(figsize=(10, 10))
plt.plot(errors, 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.xlabel('Hidden Layers Structure')
plt.grid(True)


# ## Experiments with relu activation function

# Now define the ```NeuralNetwork_relu``` class by extending ```NeuralNetwork``` and simply redefine the ```activation``` and ```activation_derivative``` methods.  Feel free to search the net for examples of how to define these functions.  Try keywords like *relu derivative python*.  Acknowledge the sites that you find helpful.
# 
# Your `NeuralNetwork_relu` class must be defined in your `neuralnetworks.py` file.
# 
# Write and run the code that repeats the above experiments with the auto-mpg data using `NeuralNetwork_relu` to evaluate different hidden layer structures and different numbers of epochs. To accomplish this, define a new function named `run_parameters_act` that has a new fifth argument to `run_parameters` called `activation_functions` that is passed a list with value `['tanh']`, `['relu']`, or `['tanh', 'relu']` to try both activation functions. In the body of `run_parameters_act` you must create and train the appropriate neural network based on the value of that parameter. The pandas DataFrame returned by `run_parameters_act` must include a column named 'Activation'.
# 
# Sort the results by the 'RMSE Test'.  Pick one value of number of epochs that tends to produce the lowest 'RMSE Test' and select all rows of results for that number of epochs.  Make one plot that shows the 'RMSE Test' versus hidden layer structure with one curve for 'tanh' and one curve for 'relu'.  Use the `plt.legend` function to add a legend to the plot.

# In[14]:


def run_parameters_act(Xtrain, Ttrain, Xtest, Ttest, activation_functions, epochs_list, nh_list, verbose=True):

    n_inputs = Xtrain.shape[1]  # number of columns in X
    n_outputs = Ttrain.shape[1]  # number of columns in T
    
    results = pd.DataFrame(columns=['Activation', 'Epochs', 'Hidden Units', 'RMSE Train', 'RMSE Test'])
    for act in activation_functions:
        for n_epochs in epochs_list:        
            for nh in nh_list:
                nnet = nn.NeuralNetwork(Xtrain.shape[1], nh, n_outputs) if act is 'tanh' else nn.NeuralNetwork_relu(Xtrain.shape[1], nh, n_outputs)
                nnet.train(Xtrain, Ttrain, n_epochs)
                Ytrain = nnet.use(Xtrain)
                Ytest = nnet.use(Xtest)
                entry = [act, n_epochs, nh, rmse(Ytrain, Ttrain), 
                                             rmse(Ytest, Ttest)]
                results.loc[len(results)] = entry
                if verbose:
                    display(results.tail(1))  # not print
    return results


# In[15]:


run_parameters_act(Xtrain, Ttrain, Xtest, Ttest, ['tanh', 'relu'], [1, 1000], [[2], [100]], verbose=False)


# ## Text descriptions

# As always, discuss your results on the auto-mpg data.  Discuss which hidden layer structures, numbers of iterations, and activation functions seem to do best.  Your results will vary with different random partitions of the data. Investigate and discuss how much the best hidden layer structure and number of training iterations vary when you repeat the runs.

# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A3grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A3grader.tar) and extract `A3grader.py` from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect execution score of  80 / 80 if your functions and class are defined correctly. The remaining 20 points will be based on the results you obtain from the comparisons of hidden layer structures and numbers of training iterations on the automobile data.
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A3.ipynb` with `Lastname` being your last name, and then save this notebook. Define both neural network classes, `NeuralNetwork` and `NeuralNetwork_relu` in the file named `neuralnetworks.py`.
# 
# Combine your notebook and `neuralnetwork.py` into one zip file or tar file.  Name your tar file `Lastname-A3.tar` or your zip file `Lastname-A3.zip`.  Check in your tar or zip file using the `Assignment 3` link in Canvas.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include other tests.

# In[17]:


imp.reload(nn)
get_ipython().run_line_magic('run', '-i A3grader.py')


# ## Extra Credit

# Create yet another version of the neural network class, called ```NeuralNetwork_logistic```, that uses the `logistic` activation function and its derivative, and repeat the above comparisons.
