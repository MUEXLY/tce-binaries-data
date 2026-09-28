from pathlib import Path

from ase.build import bulk
from ase import Atoms, io
import numpy as np
from tce.datasets import Dataset, PresetDataset


SYSTEMS: list[tuple[str, str]] = [
    ("Ta", "W"),
    ("Ta", "V"),
    ("Cr", "V")
]
CONFIGS_DIR: Path = Path("configurations")


def rescale(atoms: Atoms, old_a: float, new_a: float) -> Atoms:

    assert old_a > 0

    atoms_copy = atoms.copy()
    atoms_copy.cell *= new_a / old_a
    atoms_copy.positions *= new_a / old_a

    return atoms_copy


def main():

    # guess equiatomic lattice parameters
    lattice_parameters = {}
    for first_element, second_element in SYSTEMS:

        first_unit_cell = bulk(first_element, cubic=True)
        second_unit_cell = bulk(second_element, cubic=True)

        first_lattice_parameter = np.cbrt(first_unit_cell.get_volume())
        second_lattice_parameter = np.cbrt(second_unit_cell.get_volume())

        lattice_parameters[first_element, second_element] = 0.5 * (first_lattice_parameter + second_lattice_parameter)

    # we only need to generate diverse configurations for one binary
    # we have Ta-W configurations directly from tce-lib

    dataset = Dataset.from_preset(PresetDataset.TUNGSTEN_TANTALUM_GENETIC)

    new_a = lattice_parameters["Ta", "W"]
    ta_w_binary_configurations = [
        rescale(config, new_a=new_a, old_a=dataset.lattice_parameter)
        for config in dataset.configurations
    ]

    for i, configuration in enumerate(ta_w_binary_configurations, start=1):
        config_dir = CONFIGS_DIR / "TaW" / f"{i:0d}"
        config_dir.mkdir(exist_ok=True, parents=True)
        io.write(config_dir / "configuration.xyz", configuration)

    for system in SYSTEMS[1:]:

        for i, configuration in enumerate(ta_w_binary_configurations, start=1):

            transmutated_config = configuration.copy()
            first_mask = transmutated_config.symbols == "Ta"
            second_mask = transmutated_config.symbols == "W"

            transmutated_config.symbols[np.where(first_mask)] = system[0]
            transmutated_config.symbols[np.where(second_mask)] = system[1]

            transmutated_config = rescale(
                transmutated_config, 
                new_a=lattice_parameters[system], 
                old_a=lattice_parameters["Ta", "W"]
            )

            config_dir = CONFIGS_DIR / "".join(system) / f"{i:0d}"
            config_dir.mkdir(exist_ok=True, parents=True)
            io.write(config_dir / "configuration.xyz", transmutated_config)


if __name__ == "__main__":
    main()
