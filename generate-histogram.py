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
            if config_dir.is_dir():
                dynmat_file = config_dir / "dynmat.dat"
                if not dynmat_file.exists():
                    continue
                    
                print(f"  -> Reading dynmat.dat for configuration: {config_dir.name}")
                        dynmat_raw = np.loadtxt(dynmat_file)
                        dynlen = int(3 * np.sqrt(len(dynmat_raw) / 3))
                        dynmat = dynmat_raw.reshape((dynlen, dynlen))
            
                        # Check and enforce symmetry required for np.linalg.eigh
                        if not np.allclose(dynmat, dynmat.T, atol=1e-5):
                            print(f"Warning: Dynamical matrix for {sys_name}/{config_dir.name} is not strictly symmetric. Symmetrizing.")
                        dynmat = 0.5 * (dynmat + dynmat.T)
            
                        eigvals, _ = np.linalg.eigh(dynmat)
            
                        # Check for acoustic modes (should be 3 near-zero frequencies due to translation)
                        num_near_zero = np.sum(np.abs(eigvals) < 1e-3)
                        if num_near_zero != 3:
                            print(f"Note: Found {num_near_zero} near-zero eigenvalues (expected 3 acoustic modes).")
             
                    # Convert eigenvalues to real frequencies in THz, then to cm^-1
                    # Using sign to handle any tiny numerical instabilities resulting in negative values
                    freqs_thz = np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                    freqs_cm_inv = freqs_thz * THZ_TO_CM_INV
                    
                    # Filter out true negative (unstable) frequencies
                    real_freqs = freqs_cm_inv[freqs_cm_inv > 0]  

                    if len(real_freqs) > 0:
                        
                        # create histogram per configuration
                        plt.figure(figsize=(8, 5);
                        plt.hist(real_freqs, bins=50, density=True, color="tab:blue", histtype="step", linewidth=2)
                        
                        plt.xlabel(r"$\nu$ (cm$^{-1}$)", fontsize=12)
                        plt.ylabel("Density of States (DOS)", fontsize=12)
                        plt.title(f"Vibrational DOS - {sys_name} Config {config_dir.name}", fontsize=14)
                        plt.grid(True, linestyle="--", alpha=0.6)
                        
                        # Save the plot directly inside the individual configuration folder as a PDF
                        output_path = config_dir / "vibrational_histogram.pdf"
                        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
                        plt.close()  # Close the plot figure to free up memory

    print("Individual PDF histograms generated from dynmat.dat")


if __name__ == "__main__":
    main()