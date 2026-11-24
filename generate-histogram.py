from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIGS_DIR: Path = Path("configurations")
SYSTEM_NAMES: list[str] = ["TaW", "TaV", "CrV"]

# Conversion factor from THz to cm^-1
THZ_TO_CM_INV = 33.35641

def main():

    # Iterate through each alloy system directory
    for sys_name in SYSTEM_NAMES:
        sys_dir = CONFIGS_DIR / sys_name
        if not sys_dir.exists():
            continue
            
        print(f"Processing system: {sys_name}")
        
        for config_dir in sys_dir.iterdir():
            if not config_dir.is_dir():
                continue
            dynmat_file = config_dir / "dynmat.dat"
            if not dynmat_file.exists():
                continue

            print(f"Reading dynmat.dat for configuration: {config_dir.name}")
            dynmat_raw = np.loadtxt(dynmat_file)
            dynlen = int(3 * np.sqrt(len(dynmat_raw) / 3))
            dynmat = dynmat_raw.reshape((dynlen, dynlen))

            # Check and enforce symmetry required for np.linalg.eigh
            asymmetry = np.linalg.norm(dynmat - dynmat.T) / np.linalg.norm(dynmat)
            if asymmetry > 1.0e-2:
                raise ValueError(f"Dynamical matrix for {sys_name}/{config_dir.name} is not strictly symmetric, with asymmetry {asymmetry}")
            dynmat = 0.5 * (dynmat + dynmat.T)

            eigvals = np.linalg.eigvalsh(dynmat)

            # Check for acoustic modes (should be 3 near-zero frequencies due to translation)
            num_near_zero = np.sum(np.abs(eigvals) < 1e-3 * eigvals.mean())
            if num_near_zero != 3:
                raise ValueError(f"Found {num_near_zero} near-zero eigenvalues (expected 3 acoustic modes).")

            # Convert eigenvalues to real frequencies in THz, then to cm^-1
            freqs_thz = np.sqrt(eigvals[3:]) / (2.0 * np.pi)
            freqs_cm_inv = freqs_thz * THZ_TO_CM_INV

            # Filter out true negative (unstable) frequencies
            real_freqs = freqs_cm_inv[freqs_cm_inv > 0]

            # create histogram per configuration
            plt.figure()
            plt.hist(real_freqs, bins=25, density=True, linewidth=2, edgecolor="black", zorder=7)

            plt.xlabel(r"$\nu$ (cm$^{-1}$)")
            plt.ylabel("Density of States (DOS)")
            plt.title(f"Vibrational DOS - {sys_name} Config {config_dir.name}")
            plt.grid(True, linestyle="--")

            # Save the plot directly inside the individual configuration folder as a PDF
            output_path = config_dir / "vibrational_histogram.pdf"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()  # Close the plot figure to free up memory

            np.savetxt(config_dir / "frequencies.txt", freqs_cm_inv, header="vibrational frequencies in inverse cm")

    print("Individual PDF histograms generated from dynmat.dat")


if __name__ == "__main__":
    main()