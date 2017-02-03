#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import ipdb

def main():
    nsamples = 200
    nfeatures = 2
    kernels = [dot, dot2, RBF]
    nkernels = len(kernels)
    design_matrix = np.random.normal(size=(nsamples, nfeatures))

    fig, axes = plt.subplots(1, nkernels, figsize=(5 * nkernels, 5))
    scatters = []
    for ax, kernel in zip(axes, kernels):
        scatters.append(ax.scatter(design_matrix[:, 0], design_matrix[:, 1],
                                   s=20, c='black', edgecolor='none'))
        ax.set_title(kernel.__name__)
    evhandler = PickEvent(axes, scatters, kernels)
    fig.canvas.mpl_connect('button_press_event', evhandler)

    fig.tight_layout(pad=0.5)

    plt.show()


def dot(x, y):
    return np.sum(x * y, axis=1)

def dot2(x, y):
    return np.sum((x * y)**2, axis=1)

def RBF(x, y):
    return np.exp(-np.sum((x - y)**2, axis=1))


class PickEvent(object):
    def __init__(self, axes, scatters, kernels):
        self.axes = axes
        self.scatters = scatters
        self.points = None
        self.fig = axes[0].get_figure()
        self.kernels = kernels
        self.training_positions = self.scatters[0].get_offsets()

    def __call__(self, event):
        if event.button == 1:
            print("button, x, y: ", event.button, event.xdata, event.ydata)
            # first remove all points
            if self.points is not None:
                for point in self.points:
                    point[0].remove()

            # now update the scatter plots
            self.points = []
            for iax, ax in enumerate(self.axes):
                self.points.append(ax.plot(event.xdata, event.ydata, 'x',
                                           ms=10, c='black'))
                prediction_position = np.array([event.xdata, event.ydata])
                colors = self.kernels[iax](prediction_position,
                                           self.training_positions)
                norm = plt.Normalize(colors.min(), colors.max())
                self.scatters[iax].set_array(colors)
                self.scatters[iax].set_norm(norm)
        self.fig.canvas.draw()

if __name__ == "__main__":
    main()
