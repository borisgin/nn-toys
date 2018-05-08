# http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import math
import random
import matplotlib.pyplot as plt
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
    x_t= random.uniform(-init_range, init_range)
    y_t= random.uniform(-init_range, init_range)
    #x_t = -10.0; y_t=0.005
    return (x_t, y_t)

def learning_rate(lr0, lr_min, lr_policy, t, xt, yt, rampup=0):
    lr=lr0
    if (rampup > 0) and (t < rampup):
        lr = lr0  * t / rampup
    else:
        if (lr_policy == 'fixed'):
            lr=lr0
        if (lr_policy == 'decay'):
            lr = lr0 / math.sqrt(1. + (t-rampup))
        elif (lr_policy == 'opt'):
            n=xt*xt + yt*yt
            d=(xt*yt - 1) * xt*yt
            epsilon= 0.000001
            if abs(d) < epsilon :
                 d=sign(d)*epsilon
            else:
                if n*n > 8*d:
                    lr=(n - math.sqrt(n*n - 8*d))/(8*d)
                else:
                    lr=n/(8*d)

    lr= lr0*max(abs(lr), lr_min)

    return lr

def plot_loss(loss):
    T=loss.size
    max_loss=np.nanmax(loss)
    t=np.arange(0,T ,1)
    # Now we are using the Qt4Agg backend open an interactive plot window
    plt.plot(t, loss)
    plt.axis([0., T+1, 0., max_loss+0.1])
    plt.show()

#=============================================================
'''
xmin, xmax, xstep = -1.5, 1.5, .1
ymin, ymax, ystep = -1.5, 1.5, .1
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
#z = (x*y-1)*(x*y-1)
z = f(x,y)

fig, ax = plt.subplots(figsize=(10, 10))
#ax = plt.axes(projection='3d', elev=50, azim=-50)
#ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.contour(x, y, z, levels=np.arange(0, 10, 0.1), norm=LogNorm(), cmap=plt.cm.jet)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
#ax.set_zlabel('$z$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
dz_dx, dz_dy=grad(x,y)
#ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.quiver(x, y, -dz_dx, -dz_dy) # alpha=1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.show()
'''

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
N=100
p=np.zeros((N,3), dtype=float)

lr_min=0.00001
lr0 = 0.2 #0.1
lr_policy = 'opt' #'decay'#'opt'

grad_noise= 3.0

rampup = 0
larc = False #
eta  = 0.1
epsilon=0.1

momentum = 0.9 #
wd=0.000 #5 #0.001

init_range= 0.5
xt,yt = init_weights(init_range) #1.005001 #0.05
#xt=0.1 ; yt = -0.2

#-------------------------------
lr=lr0
p[0,:]=[xt, yt, f(xt,yt)]
m_x=0
m_y=0
for t in range(0, N-1):
    lr=learning_rate(lr0, lr_min, lr_policy, t, xt, yt, rampup)

    dx,dy= grad(xt,yt)
    if grad_noise > 0.0 :
        dx += grad_noise * abs(dx) * random.uniform(-1, 1)
        dy += grad_noise * abs(dy) * random.uniform(-1, 1)

    if wd > 0.0 :
        dx += wd*xt
        dy += wd*yt

    #print (dx,dy)
    if larc:
        # if (xt*xt + yt*yt< epsilon): # and (abs(yt) < epsilon):
        #     dx = (dx / abs(dx)) * (1. + eta) * abs(xt) / lr
        #     dy = (dy / abs(dy)) * (2. + eta) * abs(yt) / lr
        #if (abs(xt) + abs(yt)> epsilon):
        if (abs(xt) > epsilon):
            if abs(dx) > abs(xt)*eta:
                dx = dx * max(epsilon, abs(xt)* eta / abs(dx))
                #dx = (dx/abs(dx)) * eta * abs(xt)
#        else:
#             dx = (dx / abs(dx)) * (1 + eta) * abs(xt) / lr
        if (abs(yt) > epsilon):
            if abs(dy) > abs(yt)*eta :
                dy = dy * max(epsilon, abs(yt) * eta / abs(dy))
             #dy = dy / abs(dy) * abs(yt)  * eta
                #dy = (dy/abs(dy)) * 2* eta * abs(yt)

    #print (dx,dy)
    if (momentum>0):
        m_x = momentum*m_x + (1-momentum)*dx
        m_y = momentum*m_y + (1-momentum)*dy
        dx=m_x
        dy=m_y

    #lr = max(lr_min,  lr0* abs((xt * yt - 1)))

    xt = xt - lr*dx
    yt = yt - lr*dy

    p[t+1,:] = [xt, yt, f(xt,yt)]

print p
loss=p[:,2]
#print loss
plot_loss(loss)