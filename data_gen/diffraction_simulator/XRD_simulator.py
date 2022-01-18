# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "5/22/14"

"""
This module implements an XRD pattern calculator.
Modified for deepKNet point cloud model
"""

import os
import json
import math
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator

with open(os.path.join(os.path.dirname(__file__), "atomic_scattering_params.json")) as f:
    ATOMIC_SCATTERING_PARAMS = json.load(f)

class XRDSimulator(AbstractDiffractionPatternCalculator):
    def __init__(self, max_Miller):
        self.max_Miller = max_Miller

    def get_pattern(self, structure):
        latt = structure.lattice
        volume = structure.volume
        recip_latt = latt.reciprocal_lattice_crystallographic
        Miller_range = np.arange(-self.max_Miller, self.max_Miller+1, 1)
        hs, ks, ls = np.meshgrid(Miller_range, Miller_range, Miller_range)
        recip_pts = np.array(list(zip(hs.flatten(), ks.flatten(), ls.flatten())))
        recip_pos = recip_pts @ recip_latt.matrix
        norms = np.linalg.norm(recip_pos, axis=1)

        # Create a flattened array of zs, coeffs, fcoords and occus. This is
        # used to perform vectorized computation of atomic scattering factors
        # later. Note that these are not necessarily the same size as the
        # structure as each partially occupied specie occupies its own
        # position in the flattened array.
        zs = []
        coeffs = []
        fcoords = []
        occus = []

        for site in structure:
            # do not consider mixed species at the same site
            try:
                assert(len(site.species.items()) == 1)
            except:
                print('mixed species at the same site detected, abort..')
            for sp, occu in site.species.items():
                zs.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError("Unable to calculate XRD pattern as "
                                     "there is no scattering coefficients for"
                                     " %s." % sp.symbol)
                coeffs.append(c)
                fcoords.append(site.frac_coords)
                occus.append(occu)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)
        try:
            assert np.max(occus) == np.min(occus) == 1 # all sites should be singly occupied
        except:
            print('check occupancy values...')

        total_electrons = sum(zs)
        features = [[0, 0, 0, 0, 0, 0, (float(total_electrons)/volume)**2]]
        for idx in range(recip_pts.shape[0]):
            hkl = recip_pts[idx]
            xyz = recip_pos[idx]
            g_hkl = norms[idx]

            # skip origin and points on the limiting sphere to avoid precision problems
            if (g_hkl < 1e-6): continue

            # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 
            s = g_hkl / 2

            # Store s^2 since we are using it a few times.
            s2 = s ** 2

            # Vectorized computation of g.r for all fractional coords and
            # hkl. Output size is N_atom
            g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

            # Highly vectorized computation of atomic scattering factors.
            # Equivalent non-vectorized code is::
            #
            #   for site in structure:
            #      el = site.specie
            #      coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
            #      fs = el.Z - 41.78214 * s2 * sum(
            #          [d[0] * exp(-d[1] * s2) for d in coeff])
            fs = zs - 41.78214 * s2 * np.sum(
                coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)
            
            # Structure factor = sum of atomic scattering factors (with
            # position factor exp(2j * pi * g.r and occupancies).
            # Vectorized computation.
            f_hkl = np.sum(fs * occus * np.exp(2j * math.pi * g_dot_r))

            # Intensity for hkl is modulus square of structure factor.
            i_hkl = (f_hkl * f_hkl.conjugate()).real
            i_hkl_out = i_hkl / volume**2
            
            # add to features 
            features.append([hkl[0], hkl[1], hkl[2], xyz[0], xyz[1], xyz[2], i_hkl_out])

        assert len(features) == (2*self.max_Miller+1)**3
        return np.array(features)


