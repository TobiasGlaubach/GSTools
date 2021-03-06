#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""

import numpy as np

cimport cython
from cython.parallel import prange, parallel
from libcpp.vector cimport vector
from libc.math cimport fabs, sqrt, atan2, acos, asin, sin, cos, M_PI
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef inline double _distance_1d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]))

cdef inline double _distance_2d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]))

cdef inline double _distance_3d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) +
                (y[i] - y[j]) * (y[i] - y[j]) +
                (z[i] - z[j]) * (z[i] - z[j]))

cdef inline bint _angle_test_1d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const double[:] angles,
    const double angles_tol,
    const int i,
    const int j
) nogil:
    return True

cdef inline bint _angle_test_2d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const double[:] angles,
    const double angles_tol,
    const int i,
    const int j
) nogil:
    cdef double dx = x[i] - x[j]
    cdef double dy = y[i] - y[j]
    # azimuth
    cdef double phi1 = atan2(dy,dx) % (2.0 * M_PI)
    cdef double phi2 = atan2(-dy,-dx) % (2.0 * M_PI)
    # check both directions (+/-)
    cdef bint dir1 = fabs(phi1 - angles[0]) <= angles_tol
    cdef bint dir2 = fabs(phi2 - angles[0]) <= angles_tol
    return dir1 or dir2

cdef inline bint _angle_test_3d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const double[:] angles,
    const double angles_tol,
    const int i,
    const int j
) nogil:
    cdef double dx = x[i] - x[j]
    cdef double dy = y[i] - y[j]
    cdef double dz = z[i] - z[j]
    cdef double dr = sqrt(dx**2 + dy**2 + dz**2)
    # azimuth
    cdef double phi1 = atan2(dy, dx) % (2.0 * M_PI)
    cdef double phi2 = atan2(-dy, -dx) % (2.0 * M_PI)
    # elevation
    cdef double theta1 = acos(dz / dr)
    cdef double theta2 = acos(-dz / dr)
    # definitions for great-circle distance (for tolerance check)
    cdef double dx1 = sin(theta1) * cos(phi1) - sin(angles[1]) * cos(angles[0])
    cdef double dy1 = sin(theta1) * sin(phi1) - sin(angles[1]) * sin(angles[0])
    cdef double dz1 = cos(theta1) - cos(angles[1])
    cdef double dx2 = sin(theta2) * cos(phi2) - sin(angles[1]) * cos(angles[0])
    cdef double dy2 = sin(theta2) * sin(phi2) - sin(angles[1]) * sin(angles[0])
    cdef double dz2 = cos(theta2) - cos(angles[1])
    cdef double dist1 = 2.0 * asin(sqrt(dx1**2 + dy1**2 + dz1**2) * 0.5)
    cdef double dist2 = 2.0 * asin(sqrt(dx2**2 + dy2**2 + dz2**2) * 0.5)
    # check both directions (+/-)
    cdef bint dir1 = dist1 <= angles_tol
    cdef bint dir2 = dist2 <= angles_tol
    return dir1 or dir2

cdef inline double estimator_matheron(const double f_diff) nogil:
    return f_diff * f_diff

cdef inline double estimator_cressie(const double f_diff) nogil:
    return sqrt(fabs(f_diff))

ctypedef double (*_estimator_func)(const double) nogil

cdef inline void normalization_matheron(
    vector[double]& variogram,
    vector[long]& counts
):
    cdef int i
    for i in range(variogram.size()):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])

cdef inline void normalization_cressie(
    vector[double]& variogram,
    vector[long]& counts
):
    cdef int i
    for i in range(variogram.size()):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] = (
            (1./counts[i] * variogram[i])**4 /
            (0.457 + 0.494 / counts[i] + 0.045 / counts[i]**2)
        )

ctypedef void (*_normalization_func)(
    vector[double]&,
    vector[long]&
)

cdef _estimator_func choose_estimator_func(str estimator_type):
    cdef _estimator_func estimator_func
    if estimator_type == 'm':
        estimator_func = estimator_matheron
    elif estimator_type == 'c':
        estimator_func = estimator_cressie
    return estimator_func

cdef _normalization_func choose_estimator_normalization(str estimator_type):
    cdef _normalization_func normalization_func
    if estimator_type == 'm':
        normalization_func = normalization_matheron
    elif estimator_type == 'c':
        normalization_func = normalization_cressie
    return normalization_func

ctypedef double (*_dist_func)(
    const double[:],
    const double[:],
    const double[:],
    const int,
    const int
) nogil

ctypedef bint (*_angle_test_func)(
    const double[:],
    const double[:],
    const double[:],
    const double[:],
    const double,
    const int,
    const int
) nogil


def unstructured(
    const double[:] f,
    const double[:] bin_edges,
    const double[:] x,
    const double[:] y=None,
    const double[:] z=None,
    const double[:] angles=None,
    const double angles_tol=0.436332,
    str estimator_type='m'
):
    if x.shape[0] != f.shape[0]:
        raise ValueError('len(x) = {0} != len(f) = {1} '.
                         format(x.shape[0], f.shape[0]))
    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    cdef _dist_func distance
    cdef _angle_test_func angle_test

    # 3d
    if z is not None:
        if z.shape[0] != f.shape[0]:
            raise ValueError('len(z) = {0} != len(f) = {1} '.
                             format(z.shape[0], f.shape[0]))
        distance = _distance_3d
        angle_test = _angle_test_3d
    # 2d
    elif y is not None:
        if y.shape[0] != f.shape[0]:
            raise ValueError('len(y) = {0} != len(f) = {1} '.
                             format(y.shape[0], f.shape[0]))
        distance = _distance_2d
        angle_test = _angle_test_2d
    # 1d
    else:
        distance = _distance_1d
        angle_test = _angle_test_1d

    if angles is not None:
        if z is not None and angles.size < 2:
            raise ValueError('3d requested but only one angle given')
        if y is not None and angles.size < 1:
            raise ValueError('2d with angle requested but no angle given')

    if angles_tol <= 0:
        raise ValueError('tolerance for angle search masks must be > 0')

    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = x.shape[0] - 1
    cdef int k_max = x.shape[0]

    cdef vector[double] variogram = vector[double](len(bin_edges)-1, 0.0)
    cdef vector[long] counts = vector[long](len(bin_edges)-1, 0)
    cdef int i, j, k
    cdef DTYPE_t dist
    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(x, y, z, k, j)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    if angles is None or angle_test(x, y, z, angles, angles_tol, k, j):
                        counts[i] += 1
                        variogram[i] += estimator_func(f[k] - f[j])

    normalization_func(variogram, counts)
    return np.asarray(variogram), np.asarray(counts)


def structured(const double[:,:,:] f, str estimator_type='m'):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef vector[double] variogram = vector[double](l_max, 0.0)
    cdef vector[long] counts = vector[long](l_max, 0)
    cdef int i, j, k, l

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for l in prange(1, l_max-i):
                        counts[l] += 1
                        variogram[l] += estimator_func(f[i,j,k] - f[i+l,j,k])

    normalization_func(variogram, counts)
    return np.asarray(variogram)

def ma_structured(
    const double[:,:,:] f,
    const bint[:,:,:] mask,
    str estimator_type='m'
):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef vector[double] variogram = vector[double](l_max, 0.0)
    cdef vector[long] counts = vector[long](l_max, 0)
    cdef int i, j, k, l

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for l in prange(1, l_max-i):
                        if not mask[i,j,k] and not mask[i+l,j,k]:
                            counts[l] += 1
                            variogram[l] += estimator_func(f[i,j,k] - f[i+l,j,k])

    normalization_func(variogram, counts)
    return np.asarray(variogram)
