import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
cap = pd.read_csv("capacity_results.csv")
err = pd.read_csv("error_results.csv")

# --- Identify columns ---
# (Adjust names below if your actual CSV column headers differ)
N_cap = cap["N"]
beta_cap = cap["beta_star"]

N_err = err["N"]
beta_err = err["beta_star"]

# --- Plot side by side ---
plt.figure(figsize=(8, 5))
plt.plot(N_cap, beta_cap, label="β* (Capacity Maximization)", lw=2.2, color="C0")
plt.plot(N_err, beta_err, label="β* (Error Minimization)", lw=2.2, color="C3", linestyle="--")

plt.xlabel(r"Photon number $N = |\alpha|^2$")
plt.ylabel(r"Optimal displacement $\beta^*$")
plt.title("Comparison of β*(N): Capacity Maximization vs Error Minimization")
plt.grid(True, alpha=0.3)
plt.legend()

plt.xlim(left = 0)
plt.tight_layout()
plt.savefig("beta_star_comparison.png", dpi=160)
plt.show()
