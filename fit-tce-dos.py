from pathlib import Path

from ase import io, Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from tce.calculator import TCECalculator
from tce.constants import CUTOFFS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLassoCV
import matplotlib.pyplot as plt


ELEMENTS: list[str] = ["W", "Ta", "Cr", "V"]
FREQUENCY_CUTOFF_INV_CM: float = 400.0
FREQUENCY_RESOLUTION_INV_CM: float = 10.0


def rescale(atoms: Atoms, old_a: float, new_a: float) -> Atoms:

    assert old_a > 0

    atoms_copy = atoms.copy()
    atoms_copy.cell *= new_a / old_a
    atoms_copy.positions *= new_a / old_a

    return atoms_copy


def sort(atoms, decimals=4):

    atoms = atoms.copy()

    scaled = atoms.get_scaled_positions()
    scaled = np.round(scaled, decimals)
    scaled = np.mod(scaled, 1.0)

    atoms.set_scaled_positions(scaled)

    order = np.lexsort((
        scaled[:, 0],
        scaled[:, 1],
        scaled[:, 2],
    ))

    return atoms[order]


def main():

    lattice_parameters = {}
    for element in ELEMENTS:

        unit_cell = bulk(element, cubic=True)

        lattice_parameters[element] = np.cbrt(unit_cell.get_volume())

    configurations = []
    target = []
    for p in Path("configurations").rglob("*/*/"):
        
        atoms = io.read(p / "configuration.xyz", format="extxyz")
        # rescale to a common lattice parameter
        # just make it tungsten
        lattice_parameter = np.cbrt(2.0 * atoms.get_volume() / len(atoms))
        atoms = rescale(atoms, new_a=lattice_parameters["W"], old_a=lattice_parameter)
        atoms = sort(atoms)
        configurations.append(atoms)

        vibrational_data = np.loadtxt(p / "dynmat.dat")
        
        length = int(3.0 * np.sqrt(len(vibrational_data) / 3))
        dynamical_matrix = vibrational_data.reshape(length, length)

        relative_asymmetry = np.linalg.norm(dynamical_matrix.T - dynamical_matrix) / np.linalg.norm(dynamical_matrix)
        #print(f"Relative asymmetry: {relative_asymmetry:.2%}")

        eigenvalues = np.linalg.eigvalsh(0.5 * (dynamical_matrix + dynamical_matrix.T))
        #print(f"Smallest 5 eigenvalues: {eigenvalues[:5]}")

        frequencies_THz = np.sqrt(eigenvalues[3:]) / (2.0 * np.pi)
        frequencies_inv_cm = frequencies_THz * 33.3564
        
        bin_heights, _ = np.histogram(
            frequencies_inv_cm,
            bins=np.arange(0, FREQUENCY_CUTOFF_INV_CM, FREQUENCY_RESOLUTION_INV_CM)
        )
        target.append(bin_heights)

    target = np.array(target)

    calc = TCECalculator.load("fitting/tce-energy.pkl")

    X = np.array([
        calc.get_feature_vector(config) for config in configurations
    ])
    
    model = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("fit", MultiTaskLassoCV(fit_intercept=False))
    ]).fit(X, target)

    print(model)


if __name__ == "__main__":

    main()
