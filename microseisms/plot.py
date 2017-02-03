#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = np.load('dataset_all.npz')
    design_matrix = dataset['design_matrix']

    depths = dataset['depths']

    design_matrix /= design_matrix.max(axis=1)[:, None]

    fig, (col1, col2) = plt.subplots(1, 2)
    col1.imshow(design_matrix, aspect='auto', origin='lower', cmap='viridis')
    col2.plot(depths, range(len(depths)))
    plt.show()


if __name__ == "__main__":
    main()
