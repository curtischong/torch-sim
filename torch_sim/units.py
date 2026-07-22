# ruff: noqa: N815
"""Physical constants and unit conversion factors used by TorchSim."""

from enum import Enum
from math import pi, sqrt
from typing import Self


class BaseConstant(float, Enum):
    """CODATA Recommended Values of the Fundamental Physical Constants: 2014.

    References:
        http://arxiv.org/pdf/1507.07956.pdf
        https://wiki.fysik.dtu.dk/ase/_modules/ase/units.html#create_units
    """

    def __new__(cls, value: float) -> Self:
        """Create new BaseConstant enum value."""
        return float.__new__(cls, value)

    c = 299792458.0  # speed of light, m/s
    mu0 = 4.0e-7 * pi  # permeability of vacuum
    grav = 6.67408e-11  # gravitational constant
    h_planck = 6.626070040e-34  # Planck constant, J s
    e = 1.6021766208e-19  # elementary charge
    m_e = 9.10938356e-31  # electron mass
    m_p = 1.672621898e-27  # proton mass
    n_av = 6.022140857e23  # Avogadro number
    k_B = 1.38064852e-23  # Boltzmann constant, J/K
    amu = 1.660539040e-27  # atomic mass unit, kg


bc = BaseConstant


class UnitConversion(float, Enum):
    """Unit conversion factors between common unit systems."""

    def __new__(cls, value: float) -> Self:
        """Create new UnitConversion enum value."""
        return float.__new__(cls, value)

    Ang_to_met = 1e-10
    Ang2_to_met2 = 1e-10**2
    Ang3_to_met3 = 1e-10**3
    ps_to_s = 1e-12
    fs_to_s = 1e-15
    bar_to_pa = 1e5
    atm_to_pa = 101325
    pa_to_GPa = 1e-9
    eV_per_Ang3_to_GPa = bc.e * 1e21
    cal_to_J = 4.184
    kcal_to_cal = 1e3
    eV_to_J = bc.e
    Bohr_to_Ang = 0.529177210903
    Ang_to_Bohr = 1.0 / Bohr_to_Ang
    Hartree_to_eV = 27.211386245988
    eV_to_Hartree = 1.0 / Hartree_to_eV
    e2_per_Ang_to_eV = 14.399645478425668


uc = UnitConversion

# TorchSim state uses Angstrom, eV, and atomic mass units. These are the only
# non-trivial factors needed to convert the public MD API (K, ps, bar) to the
# coherent internal values required by the low-level integrators.
BOLTZMANN_CONSTANT_EV_PER_K = bc.k_B / bc.e
PS_TO_INTERNAL_TIME = sqrt(bc.e / (bc.amu * uc.Ang2_to_met2)) * uc.ps_to_s
BAR_TO_EV_PER_ANGSTROM3 = uc.bar_to_pa * uc.Ang3_to_met3 / bc.e
