# http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML


def f(x, y):
    z = (x*y-1)*(x*y-1)
    return z


def grad(x, y):
    dx = 2 * y * (x * y - 1)
    dy = 2 * x * (x * y - 1)
    return (dx, dy)


def polar_weights(r, phi=None):
    if phi is None:
        phi = random.uniform(0, 2*3.1415)
    x_0 = r * math.cos(phi)
    y_0 = r * math.sin(phi)
    return (x_0, y_0)


def init_weights(init_range):
    x_0 = random.uniform(-init_range, init_range)
    y_0 = random.uniform(-init_range, init_range)
    return (x_0, y_0)


def learning_rate(lr0, lr_policy, t, steps, lr_min=0, rampup=0):
    if (rampup > 0) and (t < rampup):
        lr = lr0 * t / rampup
    else:
        r = 1. - (t - rampup) / (steps - rampup)
        if (lr_policy == 'poly'):
            lr = lr0 * r * r
        elif (lr_policy == 'cosine'):
            lr = lr0 * math.cos(3.1415 * r + 1) / 2.
        elif (lr_policy == 'fixed'):
            lr = lr0
        else:
            print("lr policy {} not supported".format(lr_policy))
            lr = lr0
    lr = max(abs(lr), lr_min)
    return lr


def add_grad_noise(dx, dy, grad_noise):
    dx += grad_noise * abs(dx) * random.uniform(-1, 1)
    dy += grad_noise * abs(dy) * random.uniform(-1, 1)
    return (dx, dy)


class SGD(object):
    def __init__(self, beta1=0.95, grad_averaging=True):
        self.beta1 = beta1
        self.grad_averaging = grad_averaging
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "SGD"

    def get_update(self, dx, dy):
        if self.m_x == 0 and self.m_y == 0:
            self.m_x = dx
            self.m_y = dy
        else:
            if self.grad_averaging:
                self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * dx
                self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * dy
            else:
                self.m_x = self.beta1 * self.m_x + dx
                self.m_y = self.beta1 * self.m_y + dy
        return (self.m_x, self.m_y)


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
        if self.m_x == 0 and self.m_y == 0:
            self.v_x = dx * dx
            self.v_y = dy * dy
            self.m_x = dx
            self.m_y = dy
        else:
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * dx * dx
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * dy * dy
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * dx
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * dy
        ux = self.m_x / math.sqrt(self.v_x)
        uy = self.m_y / math.sqrt(self.v_y)
        return (ux, uy)


class Novograd(object):
    def __init__(self, beta1=0.95, beta2=0.0, grad_averaging=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_averaging = grad_averaging
        self.v_x = 0
        self.v_y = 0
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "Novograd"

    def get_update(self, dx, dy):
        if self.m_x == 0 and self.m_y == 0:
            self.v_x = dx * dx
            self.v_y = dy * dy
            self.m_x = dx
            self.m_y = dy
        else:
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * dx * dx
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * dy * dy
            if self.grad_averaging:
                self.m_x = self.beta1 * self.m_x + \
                    (1 - self.beta1) * dx / math.sqrt(self.v_x)
                self.m_y = self.beta1 * self.m_y + \
                    (1 - self.beta1) * dy / math.sqrt(self.v_y)
            else:
                self.m_x = self.beta1 * self.m_x + dx / math.sqrt(self.v_x)
                self.m_y = self.beta1 * self.m_y + dy / math.sqrt(self.v_y)

            # self.m_x = self.beta1 * self.m_x + (1- self.beta1) * dx / abs(dx)
            # self.m_y = self.beta1 * self.m_y + (1- self.beta1) * dy / abs(dy)
        ux = self.m_x
        uy = self.m_y
        return (ux, uy)


class Novograd_v1(object):
    def __init__(self, beta1=0.95, beta2=0.0, grad_averaging=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_averaging = grad_averaging
        self.v_x = 0
        self.v_y = 0
        self.m_x = 0
        self.m_y = 0

    def name(self):
        return "Novograd, $L_\inf$ norm"

    def get_update(self, dx, dy):
        if self.m_x == 0 and self.m_y == 0:
            self.v_x = abs(dx)
            self.v_y = abs(dy)
            self.m_x = dx
            self.m_y = dy
        else:
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * abs(dx)
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * abs(dy)
            if self.grad_averaging:
                self.m_x = self.beta1 * self.m_x +\
                    (1 - self.beta1) * dx / self.v_x
                self.m_y = self.beta1 * self.m_y +\
                    (1 - self.beta1) * dy / self.v_y
            else:
                self.m_x = self.beta1 * self.m_x + dx / self.v_x
                self.m_y = self.beta1 * self.m_y + dy / self.v_y
            # self.m_x = self.beta1 * self.m_x + (1- self.beta1) * dx / abs(dx)
            # self.m_y = self.beta1 * self.m_y + (1- self.beta1) * dy / abs(dy)
        ux = self.m_x
        uy = self.m_y
        return (ux, uy)


def plot_loss(loss):
    T = loss.size
    max_loss = np.nanmax(loss)
    t = np.arange(0, T, 1)
    plt.plot(t, loss)
    plt.axis([0., T + 1, 0., max_loss + 0.1])
    plt.show()


def show_heatmap(f, xt, yt):
    A = 4
    xmin, xmax, xstep = -A, A, .1
    ymin, ymax, ystep = -A, A, .1
    X = np.arange(xmin, xmax + xstep, xstep)
    Y = np.arange(ymin, ymax + ystep, ystep)
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots(figsize=(10, 10))
    s = np.arange(xmin, - xstep, xstep)
    t = 1 / s
    t = np.clip(t, -A, A)
    plt.plot(s, t, color='yellow', linestyle='dashed')
    s = np.arange(xstep, xmax + xstep, xstep)
    t = 1 / s
    t = np.clip(t, -A, A)
    plt.plot(s, t, color='yellow', linestyle='dashed')

    im = plt.imshow(z, cmap=plt.cm.jet, extent=(-A, A, -A, A),
                    origin='lower', interpolation='bilinear')
    fig.colorbar(im)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    plt.grid(True)
    plt.title("$y=(w_1*w_2-1)^2$")

    plt.quiver(xt[:-1], yt[:-1], xt[1:] - xt[:-1], yt[1:] - yt[:-1],
               color='red', angles='xy', scale_units='xy', scale=1)
    plt.show()


def show_fun(f):
    A = 4
    xmin, xmax, xstep = -A, A, .1
    ymin, ymax, ystep = -A, A, .1

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contour(x, y, z, levels=np.arange(0, A, 0.1), norm=LogNorm(),
               cmap=plt.cm.jet)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    s = np.arange(xmin, -xstep, xstep)
    t = 1 / s
    t = np.clip(t, -A, A)
    plt.plot(s, t, color='black', linestyle='dashed')
    s = np.arange(xstep, xmax + xstep, xstep)
    t = 1 / s
    t = np.clip(t, -A, A)
    plt.plot(s, t, color='black', linestyle='dashed')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    dz_dx, dz_dy = grad(x, y)

    ax.quiver(x, y, -dz_dx, -dz_dy)  # alpha=1)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    s = np.arange(xmin, - xstep, xstep)
    t = 1 / s
    plt.plot(s, t, color='black', linestyle='dashed')
    s = np.arange(xstep, xmax + xstep, xstep)
    t = 1 / s
    plt.plot(s, t, color='black', linestyle='dashed')

    plt.show()

# fig = plt.figure(figsize=(14,6))
#
# # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# p = ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
# #p = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=1)
# # surface_plot with color grading and color bar
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# p = ax.plot_surface(x, y, z, rstride=1, cstride=1,
#   cmap=plt.cm.coolwarm, linewidth=1, antialiased=False)
# cb = fig.colorbar(p, shrink=0.5)
# plt.show()


def plot_trajectory(xt, yt, opt_name):
    A = 4
    xmin, xmax, xstep = -A, A, .1
    ymin, ymax, ystep = -A, A, .1

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    ax = plt.axes()
    ax.contour(x, y, z, levels=np.arange(0, 2, 0.2), cmap=plt.cm.jet)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    s = np.arange(xmin, - xstep, xstep)
    t = 1 / s
    plt.plot(s, t, color='black', linestyle='dashed')
    s = np.arange(xstep, xmax + xstep, xstep)
    t = 1 / s
    plt.plot(s, t, color='black', linestyle='dashed')

    plt.quiver(xt[:-1], yt[:-1], xt[1:] - xt[:-1], yt[1:] - yt[:-1],
               color='red', angles='xy', scale_units='xy', scale=1)
    plt.grid(True)
    plt.title("$y=(w_1*w_2-1)^2$ , {}".format(opt_name))
    plt.show()


def minimize(
        f, grad,
        xt, yt,
        optimizer,
        lr_policy,
        lr0, lr_min, rampup,
        wd, decoupled_wd,
        grad_noise=0):

    N = 1000
    p = np.zeros((N, 3), dtype=float)

    for t in range(0, N - 1):
        if abs(xt) > 100 or abs(yt) > 100:
            break
        p[t, :] = [xt, yt, f(xt, yt)]
        (dx, dy) = grad(xt, yt)
        if grad_noise > 0.0:
            (dx, dy) = add_grad_noise(dx, dy, grad_noise)
        if wd > 0.0 and not decoupled_wd:
            dx += wd * xt
            dy += wd * yt

        ux, uy = optimizer.get_update(dx, dy)

        if wd > 0.0 and decoupled_wd:
            ux += wd * xt
            uy += wd * yt

        lr = learning_rate(lr0=lr0, lr_policy=lr_policy, t=t, steps=N,
                           rampup=rampup, lr_min=0)
        xt = xt - lr * ux
        yt = yt - lr * uy
        p[t + 1, :] = [xt, yt, f(xt, yt)]

    return p


def main():

    lr_policy = 'fixed'
    lr0 = 0.05
    rampup = 0
    lr_min = 0.0
    wd = 0.1
    decoupled_wd = False
    beta1 = 0.95
    beta2 = 0.95

    optimizer = SGD(beta1=beta1)
    optimizer = Adam(beta1=beta1, beta2=beta2)
    # optimizer = Adam(beta1=beta1, beta2=beta2); decoupled_wd = True
    # optimizer = Novograd_v1(beta1=beta1, beta2=beta2); decoupled_wd = True
    # optimizer = Novograd(beta1=beta1, beta2=beta2); decoupled_wd = True
    opt_name = optimizer.name()
    if opt_name == "Adam" and decoupled_wd:
        opt_name = "AdamW"

    init_range = 0.5
    (x0, y0) = polar_weights(r=init_range, phi=-0.1)
    # (xt, yt) = init_weights(init_range)  #; xt = 0.01 ; yt = -4

    grad_noise = 0.0

    p = minimize(f=f, grad=grad, xt=x0, yt=y0, optimizer=optimizer,
                 lr_policy=lr_policy, lr0=lr0, lr_min=lr_min, rampup=rampup,
                 wd=wd, decoupled_wd=decoupled_wd, grad_noise=grad_noise)
    x = p[:, 0]
    y = p[:, 1]
    loss = p[:, 2]

    print("Last point: ({}, {}), L={}".format(x[-1], y[-1], loss[-1]))
    # show_heatmap(f, x, y)
    plot_trajectory(x, y, opt_name)
    plot_loss(loss)


if __name__ == "__main__":
    main()
