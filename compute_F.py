import pymatgen.analysis.diffraction.xrd_mod as xrd
from pymatgen.core.structure import Structure

pattern = xrd.XRDCalculator(wavelength="AgKa")

struct = Structure.from_file("./NaCl.cif", primitive=False, sort=False, merge_tol=0.0)
pattern = pattern.get_pattern(struct)
#print(pattern)
