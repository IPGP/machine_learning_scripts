#!/usr/bin/env python

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = np.load('dataset.npz')
    design_matrix = dataset['source_time_functions']
    labels = dataset['labels']

    nsamples, nfeatures = design_matrix.shape

    # label indices:
    # 0: year, 1: month, 2:day, 3:hour, 4:min, 5:sec, 6:lat, 7:lon
    # 8: depth, 9: M0, 10: Mw, 11:strike1, 12: dip1, 13: rake1, 14:strike2,
    # 15:dip2, 16:rake2
    fig, (col1, col2) = plt.subplots(1, 2, sharey=True)
    design_matrix_normed = (design_matrix /
                            np.max(design_matrix, axis=1)[:, None])
    col1.imshow(design_matrix_normed, origin='lower', cmap='viridis',
                aspect='auto')
    col2.plot(labels[:, 8] / labels[:, 8].max(), range(nsamples))
    col2.plot(labels[:, 10] / labels[:, 10].max(), range(nsamples))
    plt.show()


if __name__ == "__main__":
    main()
