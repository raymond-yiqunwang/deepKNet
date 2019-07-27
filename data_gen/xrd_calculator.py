### Copied and modified from pymatgen
import os
import json
from math import sin, cos, asin, pi, degrees, radians
import numpy as np
from xrd_core import AbstractDiffractionPatternCalculator


# XRD wavelengths in angstroms
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}

with open(os.path.join(os.path.dirname(__file__),
                       "atomic_scattering_params.json")) as f:
    ATOMIC_SCATTERING_PARAMS = json.load(f)


class XRDCalculator(AbstractDiffractionPatternCalculator):
    def __init__(self, wavelength="CuKa"):
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            self.wavelength = WAVELENGTHS[wavelength]

    def get_pattern(self, structure):
        # returns an (N x 4) array where N is the number of valid hkl points within the limiting sphere
        out = []

        wavelength = self.wavelength
        latt = structure.lattice

        # Obtained from Bragg condition. Note that reciprocal lattice
        # vector length is 1 / d_hkl.
        max_r = 2. / wavelength

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_basis = recip_latt.matrix
        recip_pts = recip_latt.get_points_in_sphere(
            [[0, 0, 0]], [0, 0, 0], max_r)

        zs = []
        coeffs = []
        fcoords = []
        occus = []

        for site in structure:
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
        peaks = {}
        two_thetas = []

        for hkl, g_hkl, ind, _ in sorted(
                recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])):
            # Force miller indices to be integers.
            if g_hkl != 0:
                # normalized coordinate within limiting sphere
                
                d_hkl = 1 / g_hkl

                # Bragg condition
                theta = asin(wavelength * g_hkl / 2)

                # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
                # 1/|ghkl|)
                s = g_hkl / 2

                # Store s^2 since we are using it a few times.
                s2 = s ** 2

                # Vectorized computation of g.r for all fractional coords and
                # hkl.
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
                f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r))

                # Intensity for hkl is modulus square of structure factor.
                i_hkl = (f_hkl * f_hkl.conjugate()).real
                
                # each point in the reciprocal space is represented by a 4D vector -- [x, y, z, intensity]
                hkl_data = list(np.dot(recip_basis.T, hkl) / max_r)
                assert(np.linalg.norm(hkl_data) < 1.)
                hkl_data.append(i_hkl)
                
                out.append(hkl_data)

        return out
