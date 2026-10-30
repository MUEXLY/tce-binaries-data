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
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


ELEMENTS: list[str] = ["W", "Ta", "Cr", "V"]


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
    for p in Path("configurations").rglob("*/*/"):
        
        atoms = io.read(p / "configuration.xyz", format="extxyz")

        # rescale to a common lattice parameter
        # just make it tungsten
        lattice_parameter = np.cbrt(2.0 * atoms.get_volume() / len(atoms))
        atoms = rescale(atoms, new_a=lattice_parameters["W"], old_a=lattice_parameter)
        atoms = sort(atoms)

        energy = np.loadtxt(p / "relaxed_energy.dat")
        atoms.calc = SinglePointCalculator(atoms, energy=energy)

        configurations.append(atoms)

    configurations_train, configurations_test = train_test_split(
        configurations,
        test_size=0.2,
        random_state=0
    )

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["bcc"][:5] * lattice_parameters["W"],
        species=ELEMENTS,
        many_body_features=[
            (0, 0, 1),
            (0, 0, 2),
            (1, 1, 2),
            (2, 2, 2)
        ],
        models={
            "energy": Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("fit", XGBRegressor(random_state=0))
            ])
        }
    )
    calc.train(configurations_train)

    actual = []
    predicted = []
    for t in configurations_test:

        actual.append(t.get_potential_energy() / len(t))
        predicted.append(calc.get_potential_energy(t) / len(t))

    plt.scatter(actual, predicted, label="test", zorder=8, edgecolor="black", marker="s")

    for t in configurations_train:

        actual.append(t.get_potential_energy() / len(t))
        predicted.append(calc.get_potential_energy(t) / len(t))

    plt.scatter(actual, predicted, label="train", zorder=7, edgecolor="black")

    ax = plt.gca()
    plt.plot(ax.get_xlim(), ax.get_xlim(), linestyle="--", color="black", zorder=6)
    plt.grid()
    plt.xlabel("actual energy (eV / atom)")
    plt.ylabel("TCE energy (eV / atom)")
    plt.legend()
    plt.savefig("fitting/parity-energy.pdf")

    calc.train(configurations)
    calc.save("fitting/tce-energy.pkl")


if __name__ == "__main__":

    main()
