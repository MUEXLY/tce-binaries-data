from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.calculators.lammpslib import LAMMPSlib
from ase.vibrations import Vibrations


CONFIGS_DIR: Path = Path("configurations")
SYSTEM_NAMES: list[str] = ["TaW", "TaV", "CrV"]

# establishing a list of array of atoms used in the repo (same as systems: list)
ATOM_TYPES = {
    "W": 1,
    "Ta": 2,
    "Cr": 3,
    "V": 4
}

# Configuration mapping for lamps commands
LAMMPS_CMDS = [
    "pair_style eam/alloy",
    "pair_coeff * * WTaCrV.eam.alloy.zbl W Ta Cr V"
]


def main():

    # Iterate through each alloy system directory
    for sys_name in SYSTEM_NAMES:
        sys_dir = CONFIGS_DIR / sys_name
        if not sys_dir.exists():
            continue
            
        print(f"Processing system: {sys_name}")
        
        # Loop through each individual configuration folder sorted numerically
        for config_dir in sorted(sys_dir.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else d.name):
            if config_dir.is_dir():
                xyz_file = config_dir / "configuration.xyz"
                if not xyz_file.exists():
                    continue
                    
                print(f"  -> Running configuration: {config_dir.name}")
                
                # Read the atomic structure from the configuration file
                atoms = read(xyz_file)
                
                # Attach the LAMMPS calculator with the consistent 4-element global atom types
                atoms.calc = LAMMPSlib(
                    lmpcmds=LAMMPS_CMDS,
                    atom_types=ATOM_TYPES,
                    log_file=str(config_dir / "lammps.log")
                )
                
                # Run vibrational analysis using ASE
                vib = Vibrations(atoms, name=str(config_dir / "vib"))
                try:
                    vib.clean(empty_files=True, combined=False)
                    vib.run()
                    freqs = vib.get_frequencies()
                    
                    if freqs is not None:
                        # Convert to numpy array and filter for purely positive frequencies
                        freqs_arr = np.asarray(freqs)
                        real_mask = (freqs_arr.imag == 0) & (freqs_arr.real > 0)
                        real_freqs = freqs_arr[real_mask].real.astype(float)
                        
                        if len(real_freqs) > 0:
                            
                            # create histogram per configuration
                            plt.figure(figsize=(8, 5))
                            
                            # Calculate histogram data as well as axis offsets
                            hist, bin_edges = np.histogram(real_freqs, bins=50, density=True)
                            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                            
                            plt.plot(bin_centers, hist, color="tab:blue", linewidth=2)
                            
                            plt.xlabel(r"$\nu$ (cm^-1)", fontsize=12)
                            plt.ylabel("Density of States (DOS)", fontsize=12)
                            plt.title(f"Vibrational DOS - {sys_name} Config {config_dir.name}", fontsize=14)
                            plt.grid(True, linestyle="--", alpha=0.6)
                            
                            # Save the plot directly inside the individual configuration folder
                            output_path = config_dir / "vibrational_histogram.png"
                            plt.savefig(output_path, dpi=300)
                            plt.close()  # Close the plot figure to free up memory
                        
                    vib.clean()
                except Exception as e:
                    print(f"[{sys_name}] Error in {config_dir.name}: {e}")

    print("Individual histograms generated")


if __name__ == "__main__":
    main()