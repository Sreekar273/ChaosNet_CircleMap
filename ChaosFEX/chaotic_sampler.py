#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for computing trajectory along the skew-tent map.

compute_trajectory() is the main function that wraps around smaller modular
functions composed specifically for performance optimizations by Numba's JIT

Dependencies: numpy, numba

@author: Dr. Pranay S. Yadav
"""


# Import calls
import numpy as np
from numba import vectorize, float64, njit
from ChaosFEX.input_validator import _check_trajectory_inputs

# Compute single step of iteration through the standard circle map
@vectorize([float64(float64, float64, float64)])  # theta_i, Omega, K
def _circle_map_onestep(theta, Omega, K):
    """
    Computes a single step of iteration through the standard circle map given an
    input (previous) angle, natural frequency, and coupling strength.

    Parameters
    ----------
    theta : scalar, float64
        Current angle on the circle.
    Omega : scalar, float64
        Natural frequency of the oscillator.
    K : scalar, float64
        Coupling strength.

    Returns
    -------
    Output value as float64 from the standard circle map.
    """
    return (theta + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)) % 1.0

# Multiple iterations along standard circle map
@njit
def _iterate_circle_map(Omega, K, traj_vec):
    """
    Computes multiple steps of iteration through the standard circle map given a
    starting condition and parameters Omega and K. This function calls
    _circle_map_onestep for running a single step.

    Parameters
    ----------
    Omega : scalar, float64
        Natural frequency of the oscillator.
    K : scalar, float64
        Coupling strength.
    traj_vec : array, 1D, float64
        Pre-allocated array of zeroes with the first element containing an
        initial angle.

    Returns
    -------
    traj_vec : array, 1D, float64
        Array populated with values corresponding to the trajectory taken by
        recursive iteration through the standard circle map.
    """
    for idx in range(1, len(traj_vec)):
        traj_vec[idx] = _circle_map_onestep(traj_vec[idx - 1], Omega, K)

    return traj_vec

# Compute trajectory given initial conditions, Omega, K, and size
@njit
def _compute_trajectory(init_cond, Omega, K, length):
    """
    Computes the trajectory along the standard circle map with given Omega and K.

    Parameters
    ----------
    init_cond : scalar, float64
        Initial angle for iterating through the circle map.
    Omega : scalar, float64
        Natural frequency of the oscillator.
    K : scalar, float64
        Coupling strength.
    length : scalar, integer
        Size of the trajectory to compute through iteration.

    Returns
    -------
    array, 1D, float64
        Array of demanded size filled with values corresponding to the
        trajectory.
    """
    traj_vec = np.zeros(length, dtype=np.float64)
    traj_vec[0] = init_cond
    return _iterate_circle_map(Omega, K, traj_vec)

# Warmup for Numba cache initialization
def warmup():
    """
    Runs all the Numba-optimized functions to initialize Numba's JIT.
    Returns nothing and only prints to stdout.
    """
    if _compute_trajectory(0.1, 1/3, 2.0, 3)[-1] == np.array([0.550]) % 1.0:  # Example value for testing
        print("> Numba JIT warmup successful for chaotic_sampler ...")
    else:
        print("> Numba JIT warmup failed for chaotic_sampler ...")


def compute_trajectory(init_cond, Omega, K, length, validate=False):
    """
    Computes the trajectory along the standard circle map with given parameters.

    Parameters
    ----------
    init_cond : scalar, float64
        Initial angle for iterating through the circle map.
    Omega : scalar, float64
        Natural frequency of the oscillator.
    K : scalar, float64
        Coupling strength.
    length : scalar, integer
        Size of the trajectory to compute through iteration.

    Returns
    -------
    array, 1D, float64
        Array of demanded size filled with values corresponding to the trajectory.
    """
    if validate:
        if _check_trajectory_inputs(init_cond, Omega, K, length):
            return _compute_trajectory(init_cond, Omega, K, length)
        else:
            return None
    return _compute_trajectory(init_cond, Omega, K, length)

