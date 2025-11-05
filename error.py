# kennedy_error_min.py
# Reproduce/extend "Programming the Kennedy Receiver for Capacity Maximization"
# Part 2: Minimizing one-shot error probability for the Kennedy receiver (equal priors)

import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

# ---------------------------
# Helstrom bound (Dolinar)
# ---------------------------

def helstrom_error(N, p=0.5):
    """Helstrom minimum error probability for BPSK coherent states."""
    return 0.5 * (1 - np.sqrt(1 - 4 * p * (1 - p) * np.exp(-4 * N)))

# ---------------------------
# Kennedy average error probability
# ---------------------------

def P_error_np(N, beta, p=0.5):
    """Average error probability for Kennedy receiver (NumPy)."""
    alpha = np.sqrt(N)
    q0 = np.exp(-beta**2)
    q1 = np.exp(-(2 * alpha + beta)**2)
    return (1 - p) * (1 - q0) + p * q1

def P_error_torch(N_t, beta_t, p_t):
    """Average error probability for Kennedy receiver (Torch, autograd)."""
    alpha_t = torch.sqrt(N_t)
    q0 = torch.exp(-beta_t**2)
    q1 = torch.exp(-(2 * alpha_t + beta_t)**2)
    return (1 - p_t) * (1 - q0) + p_t * q1

# ---------------------------
# Minimize Pe over β (p fixed)
# ---------------------------

def minimize_Pe_beta(N, p=0.5, restarts=4, steps=1000, lr=0.001, seed=0,
                     device="cuda:0", init=None, patience=300, verbose=True):
    """
    For a single N, minimize one-shot error probability P_e over real β using Torch/Adam.
    Keeps p fixed (default 0.5).
    Returns best_Pe, best_beta
    """
    torch.manual_seed(seed)
    N_t = torch.tensor(float(N), dtype=torch.double, device=device)
    p_t = torch.tensor(float(p), dtype=torch.double, device=device)

    best_Pe = 1.0
    best_beta = 0.0

    for r in range(restarts):
        # ----- Initialization -----
        if init is not None and r == 0:
            beta = torch.tensor(init["beta"], dtype=torch.double, device=device)
            if verbose:
                print(f"Warm start: β={init['beta']:.4f}")
        else:
            beta = torch.randn((), dtype=torch.double, device=device) * 0.5

        beta.requires_grad_(True)

        opt = torch.optim.Adam([beta], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

        best_Pe_r = 1.0
        best_beta_r = float(beta.item())
        no_improve = 0

        # ----- Optimization Loop -----
        for t in range(steps):
            Pe = P_error_torch(N_t, beta, p_t)
            opt.zero_grad()
            Pe.backward()
            opt.step()

            # LR decay
            if (t + 1) % 50 == 0:
                scheduler.step()

            Pe_val = Pe.item()
            if Pe_val < best_Pe_r:
                best_Pe_r = Pe_val
                best_beta_r = float(beta.item())
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (t + 1) % 300 == 0:
                print(f"[N={N:.3f}] Step {t+1}/{steps} | β={beta.item():.5f}, Pe={Pe_val:.7e}")

            if no_improve > patience:
                if verbose:
                    print(f"  → Early stop at step {t+1}")
                break

        # End restart
        if best_Pe_r < best_Pe:
            best_Pe = best_Pe_r
            best_beta = best_beta_r

        if verbose:
            print(f"Restart {r+1}/{restarts}: Pe={best_Pe_r:.6e}, β={best_beta_r:.5f}")

    if verbose:
        print(f"N={N:.4f} | best_Pe={best_Pe:.7e}, β*={best_beta:.5f}")

    return best_Pe, best_beta


# ---------------------------
# Main sweep over N, plotting, CSV
# ---------------------------

def main():
    # Photon number range (log spacing helps visualize small-N behavior)
    N_vals = np.linspace(1e-6, 2.0, 10000)
    device = "cuda:0"  # or "cpu"

    Pe_helstrom = helstrom_error(N_vals)
    Pe_kennedy0 = P_error_np(N_vals, beta=0.0, p=0.5)  # classical Kennedy
    Pe_kennedy_opt = []
    beta_star = []
    init = None

    for idx, N in enumerate(N_vals):
        Pe_best, beta_best = minimize_Pe_beta(
            N,
            p=0.5,
            restarts=4,
            steps=1000,
            lr=0.001,
            seed=1234,
            device=device,
            init=init,
            patience=300,
            verbose=False
        )
        Pe_kennedy_opt.append(Pe_best)
        beta_star.append(beta_best)
        init = {"beta": beta_best}

        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(N_vals)} (N={N:.4f})")

    Pe_kennedy_opt = np.array(Pe_kennedy_opt)
    beta_star = np.array(beta_star)

    # Save CSV
    with open("kennedy_error_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "Helstrom_Pe", "Kennedy_beta0", "Kennedy_opt", "beta_star"])
        for i in range(len(N_vals)):
            w.writerow([float(N_vals[i]), float(Pe_helstrom[i]),
                        float(Pe_kennedy0[i]), float(Pe_kennedy_opt[i]),
                        float(beta_star[i])])

    # ----- Plot Pe vs N -----
    plt.figure(figsize=(7,5))
    plt.semilogy(N_vals, Pe_helstrom, label="Helstrom (Dolinar)", lw=2.5, color="C2")
    plt.semilogy(N_vals, Pe_kennedy0, "--", label="Kennedy β=0", lw=2.5, color="C1")
    plt.semilogy(N_vals, Pe_kennedy_opt, "-.", label="Kennedy β*", lw=2.5, color="C0")
    plt.xlabel(r"Photon number $N = |\alpha|^2$")
    plt.ylabel(r"Average error probability $P_e$")
    plt.title("One-shot BPSK State Discrimination Error")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("kennedy_error_curves.png", dpi=160)

    # ----- Plot β*(N) -----
    plt.figure(figsize=(7,4))
    plt.plot(N_vals, beta_star, lw=2.0)
    plt.xlabel("N = |α|²")
    plt.ylabel("β* minimizing $P_e$")
    plt.title("Optimal Kennedy Displacement β* vs Photon Number (Error Minimization)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kennedy_error_beta_star.png", dpi=160)

    print("Saved:")
    print("  kennedy_error_results.csv")
    print("  kennedy_error_curves.png")
    print("  kennedy_error_beta_star.png")


if __name__ == "__main__":
    main()
