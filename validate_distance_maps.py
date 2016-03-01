# Some helper functions for validaint distance maps

import numpy as np


def prepare_input(target, prediction):
    if isinstance(target, list):
        target = np.array(target)
    if isinstance(prediction, list):
        prediction = np.array(prediction)
    assert isinstance(target, np.ndarray), 'not a valid input dtype'
    assert isinstance(prediction, np.ndarray), 'not a valid input dtype'
    assert target.shape == prediction.shape, 'input and prediction have different shape'
    target = target.flatten()
    prediction = prediction.flatten()
    return target, prediction


def calculate_L1(target, prediction):
    target, prediction = prepare_input(target, prediction)
    L1 = np.mean(np.abs(target - prediction))
    return L1


def calculate_L2(target, prediction):
    target, prediction = prepare_input(target, prediction)
    L2 = np.mean((target - prediction)**2)
    return L2


def calculate_perc_of_correct(target, prediction):
    # Calculate #correct_voxels/#total_voxels (how many were correct?)
    target, prediction = prepare_input(target, prediction)
    assert prediction.dtype == np.int
    assert target.dtype == np.int
    perc = np.sum(target == prediction)/float(len(target))
    return perc



