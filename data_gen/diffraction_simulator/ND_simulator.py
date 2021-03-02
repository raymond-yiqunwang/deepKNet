# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Yuta Suzuki"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Yuta Suzuki"
__email__ = "resnant@outlook.jp"
__date__ = "4/19/18"

"""
This module implements a neutron diffraction (ND) pattern calculator.
"""

import os
import json
import math
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator,\
                                               DiffractionPattern, get_unique_families

with open(os.path.join(os.path.dirname(__file__),
                       "neutron_scattering_length.json")) as f:
    ATOMIC_SCATTERING_LEN = json.load(f)


class NDSimulator(AbstractDiffractionPatternCalculator):
    """
    Computes the powder neutron diffraction pattern of a crystal structure.
    This code is a slight modification of XRDCalculator in
    pymatgen.analysis.diffraction.xrd. See it for details of the algorithm.
    Main changes by using neutron instead of X-ray are as follows:

    1. Atomic scattering length is a constant.
    2. Polarization correction term of Lorentz factor is unnecessary.

    Reference:
    Marc De Graef and Michael E. McHenry, Structure of Materials 2nd ed,
    Chapter13, Cambridge University Press 2003.

    """

    def __init__(self, wavelength=1.54184, symprec=0, debye_waller_factors=None):
        """
        Initializes the ND calculator with a given radiation.

        Args:
            wavelength (float): The wavelength of neutron in angstroms.
                Defaults to 1.54, corresponds to Cu K_alpha x-ray radiation.
            symprec (float): Symmetry precision for structure refinement. If
                set to 0, no refinement is done. Otherwise, refinement is
                performed using spglib with provided precision.
            debye_waller_factors ({element symbol: float}): Allows the
                specification of Debye-Waller factors. Note that these
                factors are temperature dependent.
        """
        self.wavelength = wavelength
        self.symprec = symprec
        self.debye_waller_factors = debye_waller_factors or {}

    def get_pattern(self, structure, scaled=True, two_theta_range=None):
        """
        Calculates the powder neutron diffraction pattern for a structure.

        Args:
            structure (Structure): Input structure
            scaled (bool): Whether to return scaled intensities. The maximum
                peak is set to a value of 100. Defaults to True. Use False if
                you need the absolute values to combine ND plots.
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.

        Returns:
            (NDPattern)
        """
        try:
            assert(self.symprec == 0)
        except:
            print('symprec is not zero, terminate the process and check your input..')
        if self.symprec:
            finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = finder.get_refined_structure()

        skip = False # in case element does not have neutron scattering length
        wavelength = self.wavelength
        latt = structure.lattice
        volume = structure.volume
        is_hex = latt.is_hexagonal()

        # Obtained from Bragg condition. Note that reciprocal lattice
        # vector length is 1 / d_hkl.
        try:
            assert(two_theta_range == None)
        except:
            print('two theta range is not None, terminate the process and check your input..')
        min_r, max_r = (
            (0, 2 / wavelength)
            if two_theta_range is None
            else [2 * math.sin(radians(t / 2)) / wavelength for t in two_theta_range]
        )

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
        if min_r:
            recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

        # Create a flattened array of coeffs, fcoords and occus. This is
        # used to perform vectorized computation of atomic scattering factors
        # later. Note that these are not necessarily the same size as the
        # structure as each partially occupied specie occupies its own
        # position in the flattened array.
        coeffs = []
        fcoords = []
        occus = []
        dwfactors = []

        for site in structure:
            # do not consider mixed species at the same site
            try:
                assert(len(site.species.items()) == 1)
            except:
                print('mixed species at the same site detected, abort..')
            for sp, occu in site.species.items():
                try:
                    c = ATOMIC_SCATTERING_LEN[sp.symbol]
                except KeyError:
                    print("Unable to calculate ND pattern as "
                          "there is no scattering coefficients for"
                          " {}.".format(sp.symbol))
                    # quick return
                    return None, None, None

                coeffs.append(c)
                fcoords.append(site.frac_coords)
                occus.append(occu)
                dwfactors.append(self.debye_waller_factors.get(sp.symbol, 0))

        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)
        try:
            assert(np.max(occus) == np.min(occus) == 1) # all sites should be singly occupied
        except:
            print('check occupancy values...')
        dwfactors = np.array(dwfactors)
        peaks = {}
        two_thetas = []

        total_coeffs = sum(coeffs)
        features = [[0, 0, 0, float(total_coeffs/volume)**2]]
        for hkl, g_hkl, ind, _ in sorted(recip_pts,
                                         key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])):
            # skip origin and points on the limiting sphere to avoid precision problems
            if (g_hkl < 1e-4) or (g_hkl > 2./self.wavelength): continue

            # Force miller indices to be integers.
            hkl = [int(round(i)) for i in hkl]

            d_hkl = 1 / g_hkl

            # Bragg condition
            theta = math.asin(wavelength * g_hkl / 2)

            # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
            # 1/|ghkl|)
            s = g_hkl / 2

            # Calculate Debye-Waller factor
            dw_correction = np.exp(-dwfactors * (s ** 2))

            # Vectorized computation of g.r for all fractional coords and
            # hkl.
            g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

            # Structure factor = sum of atomic scattering factors (with
            # position factor exp(2j * pi * g.r and occupancies).
            # Vectorized computation.
            f_hkl = np.sum(
                #coeffs * occus * np.exp(2j * math.pi * g_dot_r) * dw_correction
                coeffs * occus * np.exp(2j * math.pi * g_dot_r)
            )

            # Lorentz polarization correction for hkl
            lorentz_factor = 1 / (math.sin(theta) ** 2 * math.cos(theta))

            # Intensity for hkl is modulus square of structure factor.
            i_hkl = (f_hkl * f_hkl.conjugate()).real
            i_hkl_out = i_hkl / volume**2

            # add to features
            features.append([hkl[0], hkl[1], hkl[2], i_hkl_out])

            two_theta = math.degrees(2 * theta)

            if is_hex:
                # Use Miller-Bravais indices for hexagonal lattices.
                hkl = (hkl[0], hkl[1], -hkl[0] - hkl[1], hkl[2])
            # Deal with floating point precision issues.
            ind = np.where(
                np.abs(np.subtract(two_thetas, two_theta)) < self.TWO_THETA_TOL
            )
            if len(ind[0]) > 0:
                peaks[two_thetas[ind[0][0]]][0] += i_hkl * lorentz_factor
                peaks[two_thetas[ind[0][0]]][1].append(tuple(hkl))
            else:
                peaks[two_theta] = [i_hkl * lorentz_factor, [tuple(hkl)], d_hkl]
                two_thetas.append(two_theta)

        # Scale intensities so that the max intensity is 100.
        max_intensity = max([v[0] for v in peaks.values()])
        x = []
        y = []
        hkls = []
        d_hkls = []
        for k in sorted(peaks.keys()):
            v = peaks[k]
            fam = get_unique_families(v[1])
            if v[0] / max_intensity * 100 > self.SCALED_INTENSITY_TOL:
                x.append(k)
                y.append(v[0])
                hkls.append(
                    [{"hkl": hkl, "multiplicity": mult} for hkl, mult in fam.items()]
                )
                d_hkls.append(v[2])
        nd = DiffractionPattern(x, y, hkls, d_hkls)
        if scaled:
            nd.normalize(mode="max", value=100)
        return nd, recip_latt.matrix, features


"""
  implemented for debugging purpose
"""
import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.neutron import NDCalculator
if __name__ == "__main__":
    # obtain material cif file
    filenames = pd.read_csv('../MPdata_all/MPdata_all.csv', sep=';', header=0, index_col=None)['material_id']
    for ifile in filenames.sample(n=500):
        cif_file = '../MPdata_all/' + ifile + '.cif'
        assert(os.path.isfile(cif_file))
        struct = Structure.from_file(cif_file)
        if (Element('Ac') in struct.species) or (Element('Pu') in struct.species):
            continue
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        conventional_struct = sga.get_conventional_standard_structure()
            
        # compute neutron diffraction pattern and compare outputs
        pattern, _, _ = NDSimulator().get_pattern(conventional_struct, two_theta_range=None)
        pattern_pymatgen = NDCalculator().get_pattern(conventional_struct, two_theta_range=None)
        abs_error = max(abs(np.array(pattern.x) - np.array(pattern_pymatgen.x)))
        if abs_error > 1E-12:
            print('{} did not pass the test, error: {:.6f}'.format(ifile, abs_error))
    print('finished checking the implementation..')


