from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
                # NEW: Look for dynmat.dat instead of configuration.xyz to avoid rerunning LAMMPS
                dynmat_file = config_dir / "dynmat.dat"
                if not dynmat_file.exists():
                    continue
                    
                print(f"  -> Reading dynmat.dat for configuration: {config_dir.name}")
                
                try:
                    # NEW: Load and diagonalize the dynamical matrix directly from dynmat.dat
                    dynmat_raw = np.loadtxt(dynmat_file)
                    dynlen = int(3 * np.sqrt(len(dynmat_raw) / 3))
                    dynmat = dynmat_raw.reshape((dynlen, dynlen))
                    
                    eigvals, _ = np.linalg.eigh(dynmat)
                    
                    # Convert eigenvalues to real positive frequencies
                    real_freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                    real_freqs = real_freqs[real_freqs > 0].astype(float)
                    
                    if len(real_freqs) > 0:
                        
                        # NEW: Convert frequency values if needed
                        
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
                        
                        # Save the plot directly inside the individual configuration folder as a PDF
                        output_path = config_dir / "vibrational_histogram.pdf"
                        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
                        plt.close()  # Close the plot figure to free up memory
                        
                except Exception as e:
                    print(f"[{sys_name}] Error in {config_dir.name}: {e}")

    print("Individual PDF histograms generated from dynmat.dat")


if __name__ == "__main__":
    main()