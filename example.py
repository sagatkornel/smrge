import numpy as np
import matplotlib.pyplot as plt

# Load your modules
from rge import RGERunner
import defitions as defs


def main():
    # Create runner: 2-loop β functions, include yb & yτ
    runner = RGERunner(loops=2, include_btau=True)
    
    # Solve with a dense output grid (optional: adjust number of points)
    t_eval = np.linspace(defs.t_min, defs.t_max, 1000)
    sol = runner.run(t_eval=t_eval)
    
    # Convert log‐scale t back to μ in GeV
    mu = defs.M_Z * np.exp(sol.t)
    # λ is the last entry in the state vector when include_btau=True
    lam = sol.y[-1]
    
    # Plot
    plt.figure()
    plt.plot(mu, lam, label=r'$\lambda(\mu)$')
    plt.xscale('log')
    plt.xlabel(r'Energy scale $\mu$ [GeV]')
    plt.ylabel(r'Higgs self‐coupling $\lambda$')
    plt.title('2-Loop Running of Higgs Self-Coupling (with $y_b,y_\\tau$)')
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.axvspan(1.22 * 1e19, mu[-1], color='black', alpha=0.2) # Planck scale
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()