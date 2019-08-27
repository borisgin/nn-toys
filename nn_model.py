# http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import math
import random
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

#from scipy.optimize import minimize
#from collections import defaultdict
#from functools import partial

def f(x,y):
    z = (x*y-1)*(x*y-1)
    return z

def grad(x,y):
    dx = 2*y*(x*y-1)
    dy = 2*x*(x*y-1)
    return (dx,dy)

def init_weights(init_range):
    x_0= random.uniform(-init_range, init_range)
    y_0= random.uniform(-init_range, init_range)
    #x_t = -10.0; y_t=0.005
    return (x_0, y_0)

def learning_rate(lr0, lr_policy, t, steps, lr_min=0, rampup=0):
    if (rampup > 0) and (t < rampup):
        lr = lr0  * t / rampup
    else:
        if (lr_policy == 'fixed'):
            lr = lr0
        if (lr_policy == 'poly'):
            r = 1. - (t - rampup) / (steps - rampup)
            lr = lr0 * r * r
        elif (lr_policy == 'cosine'):
            r = 1. - (t - rampup) / (steps - rampup)
            lr = lr0 *  math.cos(3.1415 * r + 1) / 2.
        else:
            print("lr policy {} not supported".format(lr_policy))
            lr=lr0

    lr= max(abs(lr), lr_min)
    return lr

def add_grad_noise(dx, dy, grad_noise):
    dx += grad_noise * abs(dx) * random.uniform(-1, 1)
    dy += grad_noise * abs(dy) * random.uniform(-1, 1)
    return (dx, dy)

def plot_loss(loss):
    T=loss.size
    max_loss=np.nanmax(loss)
    t=np.arange(0,T ,1)
    # Now we are using the Qt4Agg backend open an interactive plot window
    plt.plot(t, loss)
    plt.axis([0., T+1, 0., max_loss+0.1])
    plt.show()

class SGD(object):
    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "Adam"

    def get_update(self, dx, dy):
        if self.m_x ==0 and self.m_y ==0:
            self.m_x = dx
            self.m_y = dy
        else:
            self.m_x = self.momentum * self.m_x + dx
            self.m_y = self.momentum * self.m_y + dy
        return  (self.m_x, self.m_y)


class Adam(object):
    def __init__(self, beta1=0.95, beta2=0.99):
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_x = 0
        self.v_y = 0
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "Adam"

    def get_update(self, dx, dy):
        if self.m_x==0 and self.m_y==0:
            self.v_x = dx * dx
            self.v_y = dy * dy
            self.m_x = dx
            self.m_y = dy
        else:
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * dx * dx
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * dy * dy

            self.m_x = self.beta1 * self.m_x + (1- self.beta1) * dx
            self.m_y = self.beta1 * self.m_y + (1- self.beta1) * dy
        ux = self.m_x / math.sqrt(self.v_x )
        uy = self.m_y / math.sqrt(self.v_y)
        return (ux, uy)

class Novograd(object):
    def __init__(self, beta1=0.95, beta2=0.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_x = 0
        self.v_y = 0
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "Novograd"

    def get_update(self, dx, dy):
        if self.m_x==0 and self.m_y==0:
            self.v_x = dx * dx
            self.v_y = dy * dy
            self.m_x = dx
            self.m_y = dy
        else:
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * dx * dx
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * dy * dy
            self.m_x = self.beta1 * self.m_x + (1- self.beta1) * dx / math.sqrt(self.v_x)
            self.m_y = self.beta1 * self.m_y + (1- self.beta1) * dy / math.sqrt(self.v_y)
            # self.m_x = self.beta1 * self.m_x + (1- self.beta1) * dx / abs(dx)
            # self.m_y = self.beta1 * self.m_y + (1- self.beta1) * dy / abs(dy)
        ux = self.m_x
        uy = self.m_y
        return (ux, uy)


def show_fun(f):
    A = 10
    xmin, xmax, xstep = -A, A, .1
    ymin, ymax, ystep = -A, A, .1

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax = plt.axes(projection='3d', elev=50, azim=-50)
    # ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)

    ax.contour(x, y, z, levels=np.arange(0, A, 0.1), norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    dz_dx, dz_dy = grad(x, y)
    # ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(x, y, -dz_dx, -dz_dy)  # alpha=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    plt.show()

'''
fig = plt.figure(figsize=(14,6))

# `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
p = ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
#p = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=1)
# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=1, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)
plt.show()
'''

#--------------------------------------------------


def plot_trajectory(xt, yt):
    A = 10
    xmin, xmax, xstep = -A, A, .1
    ymin, ymax, ystep = -A, A, .1

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    ax = plt.axes()
    ax.contour(x, y, z, levels=np.arange(0, 2, 0.2),  cmap=plt.cm.jet) # norm=LogNorm())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    s=np.arange(xmin, - xstep, xstep)
    t=1/s
    plt.plot(s, t, color='black', linestyle='dashed')
    s = np.arange(xstep, xmax+ xstep, xstep)
    t = 1 / s
    plt.plot(s, t, color='black', linestyle='dashed')

    # plt.plot(xt, yt, color='red',) #marker='*', linestyle='dashed')
    plt.quiver(xt[:-1], yt[:-1], xt[1:]- xt[:-1], yt[1:]-yt[:-1], color='red', angles='xy', scale_units='xy',  scale=1)
    plt.show()


N=200
p=np.zeros((N,3), dtype=float)

lr_min=0.00001
lr_policy = 'poly'
lr_policy = 'cosine'

lr0 = 0.5;
rampup = 0;
wd=0.2
grad_noise = 0.1 #.0
init_range= 0.9
(xt,yt) = init_weights(init_range)
# xt = 0.01 ; yt = -4

optimzer=SGD(momentum=0.95);   decoupled_wd = False; lr0 = 0.01
optimzer=Adam(beta1=0.95, beta2=0.99);  decoupled_wd = False; lr0 = 0.3
optimzer=Adam(beta1=0.95, beta2=0.99);decoupled_wd = True;    lr0 = 0.3; wd=0.1
# optimzer=Novograd(beta1=0.95, beta2=0.5); decoupled_wd = True; lr0 = 0.05 ; wd=0.1

for t in range(0, N-1):
    if abs(xt) >  100 or abs (yt)>100:
        break

    p[t, :] = [xt, yt, f(xt, yt)]
    (dx,dy)= grad(xt,yt)
    if grad_noise > 0.0 :
        (dx, dy) = add_grad_noise(dx, dy, grad_noise)

    if wd > 0.0 and not decoupled_wd:
        dx += wd * xt
        dy += wd * yt

    ux, uy = optimzer.get_update(dx,dy)

    if wd > 0.0 and decoupled_wd:
        ux += wd * xt
        uy += wd * yt

    lr = learning_rate(lr0=lr0, lr_policy=lr_policy, t=t, steps=N, rampup=rampup, lr_min=0)
    xt = xt - lr * ux
    yt = yt - lr * uy
    p[t+1,:] = [xt, yt, f(xt,yt)]

print(p)
x=p[:,0]
y=p[:,1]
plot_trajectory(x, y)
loss=p[:,2]
#print loss
plot_loss(loss)