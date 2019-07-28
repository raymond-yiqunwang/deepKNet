### Copied and modified from pymatgen
import os
import json
import math
import numpy as np
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.core.structure import Structure
from pymatgen import MPRester


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


def cart2sphere(cart_coord):
    x = cart_coord[0]
    y = cart_coord[1]
    z = cart_coord[2]
    r = math.sqrt(x**2 + y**2 + z**2) 
    theta = math.acos(z/r) / math.pi
    phi = (math.atan2(y, x) + math.pi) / (2 * math.pi)
    return [r, theta, phi]


class XRDCalculator(AbstractDiffractionPatternCalculator):
    def __init__(self, wavelength="CuKa"):
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            self.wavelength = WAVELENGTHS[wavelength]

    def get_pattern(self, structure):
        # deprecated function
        pass

    def get_atomic_form_factor(self, structure):
        # returns an (N x 123) array where N is the number of valid hkl points within the limiting sphere
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
        # Convert from Cartesian to spherical coordinate
        recip_pts_spherical = []
        for rpoint in recip_pts:
            hkl = rpoint[0]
            g_hkl = rpoint[1]
            if (g_hkl < 1e-8): continue # skip [0 0 0] point
            spherical = cart2sphere(np.dot(recip_basis.T, hkl))
            recip_pts_spherical.append([hkl, g_hkl, spherical])

        zs = []
        coeffs = []
        fcoords = []
        occus = []

        for site in structure:
            for sp, occu in site.species.items():
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError("Unable to calculate XRD pattern as "
                                     "there is no scattering coefficients for"
                                     " %s." % sp.symbol)
                zs.append(sp.Z)
                coeffs.append(c)
                fcoords.append(site.frac_coords)
                occus.append(occu)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)

        for hkl, g_hkl, spherical in sorted(
                recip_pts_spherical, key=lambda i: (i[2][0], i[2][1], i[2][2])):
            if g_hkl != 0:
                # Bragg condition
                theta = math.asin(wavelength * g_hkl / 2)

                s = g_hkl / 2
                s2 = s ** 2

                g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

                fs = zs - 41.78214 * s2 * np.sum(
                    coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)

                atomic_form_factor = [0.] * 120
                f_hkl = fs * occus * np.exp(2j * math.pi * g_dot_r)
                i_hkl = (f_hkl * f_hkl.conjugate()).real
                for idx in range(zs.size):
                    atomic_intensity = i_hkl[idx]
                    z = zs[idx]
                    atomic_form_factor[z] += atomic_intensity
                # normalize atomic form factor vector
                atomic_form_factor /= np.linalg.norm(atomic_form_factor)
                
                # each point in the reciprocal space is represented by a 123D vector -- [x, y, z, atomic_form_factors]
                hkl_data = spherical.copy()
                hkl_data.extend(atomic_form_factor)
                
                out.append(hkl_data)

        return out


    def get_intensity(self, structure):
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
        # Convert from Cartesian to spherical coordinate
        recip_pts_spherical = []
        for rpoint in recip_pts:
            hkl = rpoint[0]
            g_hkl = rpoint[1]
            if (g_hkl < 1e-8): continue # skip [0 0 0] point
            spherical = cart2sphere(np.dot(recip_basis.T, hkl))
            recip_pts_spherical.append([hkl, g_hkl, spherical])

        zs = []
        coeffs = []
        fcoords = []
        occus = []

        for site in structure:
            for sp, occu in site.species.items():
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError("Unable to calculate XRD pattern as "
                                     "there is no scattering coefficients for"
                                     " %s." % sp.symbol)
                zs.append(sp.Z)
                coeffs.append(c)
                fcoords.append(site.frac_coords)
                occus.append(occu)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)

        for hkl, g_hkl, spherical in sorted(
                recip_pts_spherical, key=lambda i: (i[2][0], i[2][1], i[2][2])):
            if g_hkl != 0:
                # Bragg condition
                theta = math.asin(wavelength * g_hkl / 2)

                s = g_hkl / 2
                s2 = s ** 2

                g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

                fs = zs - 41.78214 * s2 * np.sum(
                    coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)

                f_hkl = np.sum(fs * occus * np.exp(2j * math.pi * g_dot_r))
                i_hkl = (f_hkl * f_hkl.conjugate()).real
                
                # each point in the reciprocal space is represented by a 4D vector -- [x, y, z, intensity]
                hkl_data = spherical.copy()
                hkl_data.append(i_hkl)
                
                out.append(hkl_data)

        return out


if __name__ == "__main__":
    #xrd = XRDCalculator("CuKa")
    xrd = XRDCalculator(5.0)

    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    mp_data = m.query(criteria={"material_id": { "$eq" : "mp-2885" }}, properties=['cif', 'band_gap'])
    
    struct = Structure.from_str(mp_data[0]['cif'], fmt='cif')
    #pattern = xrd.get_intensity(struct)
    pattern = xrd.get_atomic_form_factor(struct)
