"""
Introduction onto the various options for variogram estimation
---------------------------

This example is going to give an introduction into the various options
given for estimation variograms on unstructured data 




"""
import os
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
import timeit

###############################################################################
# The Data
# ^^^^^^^^

# We are going generate some simple data ourselfs similar to the code from 
# the examples on random fields.

seed = gs.random.MasterRNG(19970221)
rng = np.random.RandomState(seed())
x = rng.randint(0, 100, size=3000)
y = rng.randint(0, 100, size=3000)

model = gs.Exponential(dim=2, var=1, len_scale=[12, 3], angles=np.pi / 8)
srf = gs.SRF(model, seed=20170519)
field = srf((x, y))



###############################################################################
# Let's have a look at the generated field

plt.scatter(x, y, c=field)
plt.show()


###############################################################################
# Variogram estimation with rotated major and minor axis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The unstructured function allows for variogram estimation with rotated axis. 
# Here only points within the given angle +/- tolerance are taken into account, 
# when calculating the variogram values. This process is implemented
# using search masks. Usage example: 

bins = np.arange(0, 50, 5)
angle_mask = 22.5
angle_tol = 15

bin_centers, gamma = gs.vario_estimate_unstructured(
        (x, y),
        field,
        bins,
        angles=[np.deg2rad(angle_mask)],
        angles_tol=np.deg2rad(angle_tol)
    )

plt.scatter(bin_centers, gamma)
plt.xlabel('bin center positions')
plt.ylabel('estimated semi-variance')
plt.title('Semi-Variogram estimation with azimzuth={:3.2f}° +/- {:3.2f}°'.format(angle_mask, angle_tol))
plt.show()

# The internal calculation uses search masks equal to the way shown below:

# helper functions:

def check_in_mask(p0, p1, angle_mask, angle_tol, bin_low, bin_high):
    """checks if p1 is within a given searach mask of p0"""
    dx, dy  = np.array(p1) - np.array(p0)
    dist_01 = np.sqrt(dx**2 + dy**2)
    angle_01 = np.arctan2(dy, dx)

    is_valid_angle = np.abs(angle_mask - angle_01) <= angle_tol 
    is_valid_dist = bin_low < dist_01 <= bin_high
    return is_valid_angle and is_valid_dist

def get_search_mask_for_plot(p0, angle_mask, angle_tol, bin_low, bin_high):
    """ for a point x0, y1 gets two arrays of x and y points marking the search mask border to use with plt.fill(...)"""
    x0, y0 = p0
    x = np.zeros((0,0))
    x = np.append(x, bin_low * np.cos(np.linspace(angle_mask - 0.5 * angle_tol, angle_mask + 0.5 * angle_tol, 10)))
    x = np.append(x, bin_high * np.cos(np.linspace(angle_mask + 0.5 * angle_tol, angle_mask - 0.5 * angle_tol, 10)))
    x += x0

    y = np.empty((0,0))
    y = np.append(y, bin_low * np.sin(np.linspace(angle_mask - 0.5 * angle_tol, angle_mask + 0.5 * angle_tol, 10)))
    y = np.append(y, bin_high * np.sin(np.linspace(angle_mask + 0.5 * angle_tol, angle_mask - 0.5 * angle_tol, 10)))
    y += y0

    return x, y

# say there are three points given
points = [(0,0), (2,1), (1,0)]

# plot the points on plane
f, ax = plt.subplots(1,1, figsize=(12,12))
for i, (xp, yp) in enumerate(points):
  ax.scatter(xp, yp, label='p_{}: {}, {}'.format(i, xp, yp))

# for p0 plot search masks for bins=0...4
for bin_i in range(4):
  bin_low = bin_i
  bin_high = bin_i+1
  xf, yf = get_search_mask_for_plot(points[0], np.deg2rad(angle_mask), np.deg2rad(angle_tol), bin_low, bin_high)
  ax.fill(xf, yf, alpha=0.2, label='search mask {} distance=[{}, {}) angle={:3.2f}° +/- {:3.2f}°'.format(bin_i, bin_low, bin_high, angle_mask, angle_tol))

ax.legend()
ax.set_xlim((-1, 3))
ax.set_ylim((-1, 3))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal', 'box')
ax.grid()


# this code mimics what the calculation then does internally to check whether or not a point is within a mask:
for bin_i in range(4):
  bin_low, bin_high = bin_i, bin_i+1
  for i, p0 in enumerate(points):
    for p1 in points[i+1:]:
      is_valid = check_in_mask(p0, p1, np.deg2rad(angle_mask), np.deg2rad(angle_tol), bin_low, bin_high)
      print('Points: {} --> {} | at search mask distance=[{}, {}) angle={:3.2f}° +/- {:3.2f}° ==> {}'.format(p0, p1, bin_low, bin_high, angle_mask, angle_tol, 'VALID' if is_valid else 'NOT VALID'))
  print('')
plt.show()

###############################################################################
# Returning the point count in each bin
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The unstructured function has a parameter which will change the return value
# of the function to also return the number of valid points founds within each
# bin of the given search mask. This can be helpful e.G. in a scenario where
# a variogram should be fitted to data and the variogram points need to be 
# weighted. 

bins = np.arange(0, 50, 5)
angle_mask = 22.5
angle_tol = 15

bin_centers, gamma, counts = gs.vario_estimate_unstructured(
        (x, y),
        field,
        bins,
        angles=[np.deg2rad(angle_mask)],
        angles_tol=np.deg2rad(angle_tol),
        return_counts=True
    )

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,12))
ax1.bar(bin_centers, counts)
ax2.plot(bin_centers, gamma)
ax2.set_xlabel('bin center positions')
ax1.set_ylabel('number of valid points')
ax2.set_ylabel('estimated semi-variance')
plt.show()

###############################################################################
# Caching calculation results for speed improvement
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# instead of using parallel computation with openMD it is also possible to 
# speed the calculation up when only single threaded calculation is available/
# needed. This can be done by instructing the unstructured function to cache 
# and reuse calculated results internally. Especially for high len(bins) this 
# can result in a huge speed improvement. This can be done by setting the
# use_caching argument to True. Here is an example on the speed improvements.
# (please note, that speeds can differ significantly based on whether or not 
# GSTools was compiled with OpenMD or not)

angle_mask = 22.5
angle_tol = 15

# make simple functions for comparing
fun = lambda bins: gs.vario_estimate_unstructured(
        (x, y),
        field,
        bins,
        angles=[np.deg2rad(angle_mask)],
        angles_tol=np.deg2rad(angle_tol)
    )

fun_cached = lambda bins: gs.vario_estimate_unstructured(
        (x, y),
        field,
        bins,
        angles=[np.deg2rad(angle_mask)],
        angles_tol=np.deg2rad(angle_tol),
        use_caching=True
    )

# compare execution speed for different dense bin vectors
for bin_steps in [10, 5, 2, 1]:
    bins = np.arange(0, 20, bin_steps)

    # timeit needs an argumentless function to measure execution time
    def fn():
        fun(bins)
    def fnc():
        fun_cached(bins)
    t_no_cache = timeit.timeit(fn, number=30) / 30.
    t_cached = timeit.timeit(fnc, number=30) / 30.
    print(f"len(bins) {bins.size}: time per execution: no_caching {t_no_cache} | with_caching {t_cached}")


