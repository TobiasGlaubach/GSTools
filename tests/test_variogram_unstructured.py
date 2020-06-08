# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
from gstools import vario_estimate_unstructured, Exponential, SRF
import gstools as gs

class TestVariogramUnstructured(unittest.TestCase):
    def setUp(self):

        # this code generates the testdata for the 2D rotation test cases
        x = np.random.RandomState(19970221).rand(20) * 10.0 - 5
        y = np.zeros_like(x)
        model = gs.Exponential(dim=2, var=2, len_scale=8)
        srf = gs.SRF(model, mean=0, seed=19970221)
        field = srf((x, y))

        bins = np.arange(10)
        bin_center, gamma = gs.vario_estimate_unstructured((x, ), field, bins)
        idx = np.argsort(x)
        self.test_data_rotation_1 = {'gamma': gamma, 'x': x[idx], 'field': field[idx], 'bins': bins, 'bin_center': bin_center}

        # CODE ABOVE SHOULD GENERATE THIS DATA
        # x = np.array([
        # -4.86210059, -4.1984934 , -3.9246953 , -3.28490663, -2.16332379,
        # -1.87553275, -1.74125124, -1.27224687, -1.20931578, -0.2413368 ,
        #  0.03200921,  1.17099773,  1.53863105,  1.64478688,  2.75252136,
        #  3.3556915 ,  3.89828775,  4.21485964,  4.5364357 ,  4.79236969]),
        # field = np.array([
        # -1.10318365, -0.53566629, -0.41789049, -1.06167529,  0.38449961,
        # -0.36550477, -0.98905552, -0.19352766,  0.16264266,  0.26920833,
        #  0.05379665,  0.71275006,  0.36651935,  0.17366865,  1.20022343,
        #  0.79385446,  0.69456069,  1.0733393 ,  0.71191592,  0.71969766])
        # gamma_exp = np.array([
        # 0.14260989, 0.18301197, 0.25855841, 0.29990083, 0.67914526,
        # 0.60136535, 0.92875492, 1.46910435, 1.10165104])

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_list(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_uncorrelated_2d(self):
        x_c = np.linspace(0.0, 100.0, 60)
        y_c = np.linspace(0.0, 100.0, 60)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured((x, y), field, bins)

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_uncorrelated_3d(self):
        x_c = np.linspace(0.0, 100.0, 15)
        y_c = np.linspace(0.0, 100.0, 15)
        z_c = np.linspace(0.0, 100.0, 15)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y, z), field, bins
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_1d(self):
        x = np.linspace(0.0, 100.0, 21000)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            [x], field, bins, sampling_size=5000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_2d(self):
        x_c = np.linspace(0.0, 100.0, 600)
        y_c = np.linspace(0.0, 100.0, 600)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y), field, bins, sampling_size=2000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_3d(self):
        x_c = np.linspace(0.0, 100.0, 100)
        y_c = np.linspace(0.0, 100.0, 100)
        z_c = np.linspace(0.0, 100.0, 100)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y, z),
            field,
            bins,
            sampling_size=2000,
            sampling_seed=1479373475,
        )
        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_assertions(self):
        x = np.arange(0, 10)
        x_e = np.arange(0, 11)
        y = np.arange(0, 11)
        y_e = np.arange(0, 12)
        z = np.arange(0, 12)
        z_e = np.arange(0, 15)
        bins = np.arange(0, 3)
        #        bins_e = np.arange(0, 1)
        field = np.arange(0, 10)
        field_e = np.arange(0, 9)

        self.assertRaises(
            ValueError, vario_estimate_unstructured, [x_e], field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x_e, y, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z), field_e, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, [x], field_e, bins
        )

    def test_angles_2D_x2x(self):
        """
        test case 1.)
           all along x axis on x axis
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y),
            field,
            bins,
            angles=[0]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    def test_angles_2D_y2x(self):
        """
        test case 2.)
        all along y axis on y axis but calculation for x axis
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

            
        bin_centres, gamma = vario_estimate_unstructured(
            (y, x),
            field,
            bins,
            angles=[0]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(0, gamma[i], places=3)
            
    def test_angles_2D_y2y(self):
        """
        test case 3.)
           all along y axis on y axis and calculation for y axis
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)


        bin_centres, gamma = vario_estimate_unstructured(
            (y, x),
            field,
            bins,
            angles=[np.pi/2.]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    def test_angles_2D_xy2x(self):
        """
        test case 4.)
           data along 45deg axis but calculation for x axis
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

                    
        ccos, csin = np.cos(np.pi/4.), np.sin(np.pi/4.)
        
        xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
        yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]
        
        bin_centres, gamma = vario_estimate_unstructured(
            (xr, yr),
            field,
            bins,
            angles=[0]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(0, gamma[i], places=3)
            
    def test_angles_2D_xy2xy(self):
        """
        test case 5.)
           data along 45deg axis and calculation for 45deg
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

        ccos, csin = np.cos(np.pi/4.), np.sin(np.pi/4.)
        
        xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
        yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]
        
        bin_centres, gamma = vario_estimate_unstructured(
            (xr, yr),
            field,
            bins,
            angles=[np.pi/4.]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    def test_angles_count_2D_case_1(self):
        """
            test case counts 1.) 
            test if the data pairs count corresponds to include/exclude behaviour
            data at +45deg and setting mask to +45deg +/- 10deg --> should find points
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

        angle_rad = np.deg2rad(45)
        ccos, csin = np.cos(angle_rad), np.sin(angle_rad)
        
        xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
        yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]
        
        bin_centres, gamma, counts = vario_estimate_unstructured(
            (xr, yr),
            field,
            bins,
            angles_tol=np.deg2rad(10.0),
            angles=[angle_rad],
            return_counts=True
        )

        self.assertNotEqual(0, np.sum(counts), "expected to find some points pairs in angle direction, but none were found")


    def test_angles_count_2D_case_2(self):
        """
            test case counts 2.) 
            test if the data pairs count corresponds to include/exclude behaviour
            data at +45deg and setting mask to +54.9deg +/- 10deg --> should find points
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

        angle_rad = np.deg2rad(45)
        ccos, csin = np.cos(angle_rad), np.sin(angle_rad)
        
        xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
        yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]
        
        bin_centres, gamma, counts = vario_estimate_unstructured(
            (xr, yr),
            field,
            bins,
            angles_tol=np.deg2rad(10.0),
            angles=[angle_rad + np.deg2rad(9.9)],
            return_counts=True
        )
        
        self.assertNotEqual(0, np.sum(counts), "expected to find some points pairs in angle direction, but none were found")


    def test_angles_count_2D_case_3(self):
        """
            test case counts 3.) 
            test if the data pairs count corresponds to include/exclude behaviour
            data at +45deg and setting mask to +55.1deg +/- 10deg --> should NOT find points
        """

        x = self.test_data_rotation_1['x']
        field = self.test_data_rotation_1['field']
        gamma_exp = self.test_data_rotation_1['gamma']
        bins = self.test_data_rotation_1['bins']
        y = np.zeros_like(x)

        angle_rad = np.deg2rad(45)
        ccos, csin = np.cos(angle_rad), np.sin(angle_rad)
        
        xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
        yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]
        
        bin_centres, gamma, counts = vario_estimate_unstructured(
            (xr, yr),
            field,
            bins,
            angles_tol=np.deg2rad(10.0),
            angles=[angle_rad + np.deg2rad(10.1)],
            return_counts=True
        )
        
        self.assertEqual(0, np.sum(counts), "expected to find NO points pairs in angle direction, but some were found")


    def test_angles_2D_caching(self):
        """
        generate a dataset with rotated anisotrope variogram and try to 
        reconstruct it using unstructured
        """
        do_timeit = False
        if do_timeit:
            import timeit

        x = y = np.arange(0, 50)
        model = gs.Exponential(dim=2, var=1, len_scale=[12.0, 3.0], angles=np.pi / 8)
        srf = gs.SRF(model, seed=20170519)
        srf.structured([x, y])

        (gridx, gridy) = srf.pos
        val = srf.field.copy()

        X, Y = np.meshgrid(gridx, gridy)

        np.random.seed(20170519)
        idx = np.random.choice(X.size, 1000)
        #idx = range(0, X.size, 3)
        x = X.flatten()[idx]
        y = Y.flatten()[idx]
        field = val.flatten()[idx]

        fun = lambda bins: vario_estimate_unstructured(
                (x, y),
                field,
                bins,
                angles=[np.pi / 8.]
            )
        fun_cached = lambda bins: vario_estimate_unstructured(
                (x, y),
                field,
                bins,
                angles=[np.pi / 8.], 
                use_caching=True
            )

        for bin_steps in [10, 5, 2, 1]:
            bins = np.arange(0, 20, bin_steps)

            if do_timeit:
                def fn():
                    fun(bins)
                def fnc():
                    fun_cached(bins)
                t_no_cache = timeit.timeit(fn, number=30) / 30.
                t_cached = timeit.timeit(fnc, number=30) / 30.
                print(f"len(bins) {bins.size}: t_no_cache {t_no_cache} | t_cached {t_cached}")
                #self.assertTrue(t_no_cache > t_cached)

            bin_centres, gamma = fun(bins)
            bin_centres_c, gamma_c = fun_cached(bins)
        


        for i in range(gamma.size):
            self.assertAlmostEqual(bin_centres[i], bin_centres_c[i], places=3)
            self.assertAlmostEqual(gamma[i], gamma_c[i], places=3)


    def test_angles_2D_vario_based(self):
        """
        generate a dataset with rotated anisotrope variogram and try to 
        reconstruct it using unstructured
        """

        x = y = np.arange(0, 50)
        model = gs.Exponential(dim=2, var=1, len_scale=[12.0, 3.0], angles=np.pi / 8)
        srf = gs.SRF(model, seed=20170519)
        srf.structured([x, y])

        (gridx, gridy) = srf.pos
        val = srf.field.copy()

        X, Y = np.meshgrid(gridx, gridy)

        np.random.seed(20170519)
        idx = np.random.choice(X.size, 1000)
        #idx = range(0, X.size, 3)
        x = X.flatten()[idx]
        y = Y.flatten()[idx]
        field = val.flatten()[idx]

        bins = np.arange(20)
        bin_centres, gamma = vario_estimate_unstructured(
            (x, y),
            field,
            bins,
            angles=[np.pi / 8.]
        )
        
        for i in range(gamma.size):
            self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

if __name__ == "__main__":
    unittest.main()
