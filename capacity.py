# kennedy_capacity_ml.py
# Reproduce/extend "Programming the Kennedy Receiver for Capacity Maximization"
# using ML-style optimization (PyTorch autograd) for β and p.

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from math import sqrt, log

# ---------------------------
# Utilities (numerically stable)
# ---------------------------

def h2_np(q):
    """Binary entropy in bits (NumPy), vectorized, stable at 0/1."""
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

def h2_t(q):
    """Binary entropy in bits (Torch), elementwise, stable at 0/1."""
    eps = 1e-12
    q = torch.clamp(q, eps, 1 - eps)
    return -(q * torch.log2(q) + (1 - q) * torch.log2(1 - q))

def holevo_C_infty(N):
    """Holevo capacity for coherent-state BPSK: C∞(N) = h2((1 - e^{-2N})/2)."""
    x = (1 - np.exp(-2 * N)) / 2.0
    return h2_np(x)

def dolinar_C1(N):
    """Symbol-by-symbol optimal capacity via Helstrom error (equal priors)."""
    # Helstrom error for two pure coherent states ±sqrt(N)
    Pe = 0.5 * (1 - np.sqrt(1 - np.exp(-4 * N)))
    return 1.0 - h2_np(Pe)

# ---------------------------
# Kennedy mutual information
# ---------------------------

def I_kennedy_np(N, beta, p):
    """
    Mutual information I(X;Y) for Kennedy/GK receiver (NumPy, scalar inputs).
    Phase aligned: alpha = sqrt(N) (real), beta real.
    """
    alpha = np.sqrt(N)
    q0 = np.exp(-beta**2)                # no-click given X = |−α>
    q1 = np.exp(-(2*alpha + beta)**2)    # no-click given X = |+α>
    qbar = p * q0 + (1 - p) * q1
    return h2_np(qbar) - p*h2_np(q0) - (1-p)*h2_np(q1)

def I_kennedy_t(N_t, beta_t, p_t):
    """
    Torch version (enables autograd). Inputs can be scalars (0-D tensors).
    """
    alpha_t = torch.sqrt(N_t)
    q0 = torch.exp(-beta_t**2)
    q1 = torch.exp(-(2*alpha_t + beta_t)**2)
    qbar = p_t * q0 + (1 - p_t) * q1
    return h2_t(qbar) - p_t * h2_t(q0) - (1 - p_t) * h2_t(q1)

# ---------------------------
# Optimize p for fixed beta (Kennedy β=0 baseline)
# ---------------------------

def optimize_p_for_fixed_beta(N, beta, tol=1e-12, maxit=200):
    """
    Optimize p in [0,1] for fixed beta using Newton's method (concave in p).
    """
    alpha = np.sqrt(N)
    q0 = np.exp(-beta**2)
    q1 = np.exp(-(2*alpha + beta)**2)
    # Start near symmetric
    p = 0.5
    for _ in range(maxit):
        qbar = p*q0 + (1 - p)*q1
        # Derivatives w.r.t p
        # dI/dp = h2'(qbar)*(q0 - q1) - h2(q0) + h2(q1)
        # h2'(x) = log2((1-x)/x)
        # h2''(x) = -(1 / ln2) * 1/(x(1-x)) < 0
        eps = 1e-15
        qbar = np.clip(qbar, eps, 1 - eps)
        d1 = np.log2((1 - qbar)/qbar) * (q0 - q1) - h2_np(q0) + h2_np(q1)
        d2 = -(q0 - q1)**2 / (np.log(2) * qbar * (1 - qbar))
        step = d1 / (d2 + 1e-30)
        p_new = np.clip(p - step, 0.0, 1.0)
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return float(p)

# ---------------------------
# ML-style optimization of (beta, p) with PyTorch + Adam
# ---------------------------

def maximize_I_ml(N, restarts=5, steps=1500, lr=0.0001, seed=0, device="cuda:0", init = None, patience = 300, verbose = True):
    """
    For a single N, maximize I(X;Y) over real beta and p∈(0,1) using Torch/Adam.
    • p parameterized via sigmoid: p = sigmoid(theta_p)
    • beta is unconstrained real
    Returns best_I, best_beta, best_p
    """
    torch.manual_seed(seed)
    N_t = torch.tensor(float(N), dtype=torch.double, device=device)

    best_I = -1.0
    best_beta = 0.0
    best_p = 0.5

    for r in range(restarts):
        # ----- Initialization -----
        if init is not None and r == 0:
            beta = torch.tensor(init["beta"], dtype=torch.double, device=device)
            theta_p = torch.logit(
                torch.tensor(init["p"], dtype=torch.double, device=device)
            )
            if verbose:
                print(f"Warm start: β={init['beta']:.4f}, p={init['p']:.4f}")
        else:
            beta = torch.randn((), dtype=torch.double, device=device) * 0.5
            theta_p = torch.randn((), dtype=torch.double, device=device) * 0.5


        beta.requires_grad_(True)
        theta_p.requires_grad_(True)

        opt = torch.optim.Adam([beta, theta_p], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

        best_I_r = -1.0
        best_beta_r = float(beta.item())
        best_p_r = float(torch.sigmoid(theta_p).item())

        no_improve = 0
        # ----- Optimization Loop -----
        for t in range(steps):
            p = torch.sigmoid(theta_p)
            p = torch.clamp(p, 1e-12, 1 - 1e-12)   # stability clamp

            I = I_kennedy_t(N_t, beta, p)
            loss = -I  # maximize I

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Learning rate decay
            if (t + 1) % 50 == 0:
                scheduler.step()

            # Track best iterate
            I_val = I.item()
            if I_val > best_I_r:
                best_I_r = I_val
                best_beta_r = float(beta.item())
                best_p_r = float(p.item())
                no_improve = 0
            else:
                no_improve += 1

            # Print occasionally
            if verbose and (t + 1) % 300 == 0:
                print(f"[N={N:.3f} | Restart {r+1}] Step {t+1}/{steps} | "
                      f"β={beta.item():.5f}, p={p.item():.5f}, I={I_val:.7f}")

            # Early stopping
            if no_improve > patience:
                if verbose:
                    print(f"  → Early stop at step {t+1} (no improvement for {patience} steps)")
                break

        # ----- Evaluate restart best -----
        if verbose:
            print(f"Restart {r+1}/{restarts} done: I={best_I_r:.6f}, β={best_beta_r:.4f}, p={best_p_r:.4f}")

        if best_I_r > best_I:
            best_I = best_I_r
            best_beta = best_beta_r
            best_p = best_p_r

    if verbose:
        print(f"N={N:.4f} | best_I={best_I:.7f}, β*={best_beta:.5f}, p*={best_p:.5f}")

    return best_I, best_beta, best_p

# ---------------------------
# Main sweep over N, plotting, CSV
# ---------------------------

def main():
    # Range of photon numbers (paper shows emphasis on small N)
    N_vals = np.linspace(1e-6, 2.0, 10000)  # from very photon-starved to moderate
    device = "cuda:0"  # set "cuda" if you want GPU and have it

    C_inf = holevo_C_infty(N_vals)
    C1 = dolinar_C1(N_vals)

    # Kennedy β=0 with p optimized
    C_kennedy_beta0 = []
    p_beta0 = []
    for N in N_vals:
        p_opt = optimize_p_for_fixed_beta(N, beta=0.0)
        I_val = I_kennedy_np(N, beta=0.0, p=p_opt)
        C_kennedy_beta0.append(I_val)
        p_beta0.append(p_opt)
    C_kennedy_beta0 = np.array(C_kennedy_beta0)
    p_beta0 = np.array(p_beta0)

    # Kennedy with β and p jointly optimized (ML optimization)
    C_kennedy_ml = []
    beta_star = []
    p_star = []
    init = None
    for idx, N in enumerate(N_vals):
        best_I, best_beta, best_p = maximize_I_ml(
            float(N), restarts=6, steps=1800, lr=0.0001, seed=1234, device=device, init=init, patience=400, verbose= False
        )
        C_kennedy_ml.append(best_I)
        beta_star.append(best_beta)
        p_star.append(best_p)
        
        # Warm start next iteration
        init = {"beta": best_beta, "p": best_p}

        # optional small progress indicator
        if idx % 25 == 0:
            print(f"Progress: {idx}/{len(N_vals)} (N={N:.4f})")

    C_kennedy_ml = np.array(C_kennedy_ml)
    beta_star = np.array(beta_star)
    p_star = np.array(p_star)
    # Save CSV
    with open("kennedy_capacity_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "exp(-2N)", "Holevo_Cinf", "Dolinar_C1",
                    "Kennedy_beta0", "Kennedy_optML", "p_beta0", "beta_star", "p_star"])
        for i in range(len(N_vals)):
            w.writerow([float(N_vals[i]), float(np.exp(-2*N_vals[i])), float(C_inf[i]), float(C1[i]),
                        float(C_kennedy_beta0[i]), float(C_kennedy_ml[i]), float(p_beta0[i]),
                        float(beta_star[i]), float(p_star[i])])

    # Plot Capacity vs e^{-2N} (to mirror the paper's x-axis)
    x = np.exp(-2 * N_vals)
    plt.figure(figsize=(8.0, 5.6))
    # Sort by x decreasing if you want the same left-to-right feel as the paper
    order = np.argsort(x)
    x_plot = x[order]
    plt.plot(x_plot, C_inf[order], label="Holevo $C_\\infty(N)$", lw=2.5, color="C2")
    plt.plot(x_plot, C1[order], "--", label="Dolinar $C_1(N)$", lw=2.5, color="C3")
    plt.plot(x_plot, C_kennedy_beta0[order], "-.", label="Kennedy (β=0), p opt", lw=2.5, color="C1")
    plt.plot(x_plot, C_kennedy_ml[order], ":", label="Kennedy (β,p) opt (ML)", lw=3.0, color="C0")

    plt.xlabel(r"$e^{-2N}$")
    plt.ylabel("Capacity (bits / symbol)")
    plt.title("BPSK Capacity: Holevo vs Dolinar vs Kennedy (β=0) vs Kennedy (β,p ML-opt)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("kennedy_capacity_curves.png", dpi=160)

    # Auxiliary plots: optimal β and p vs N
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(N_vals, beta_star, lw=2.0)
    ax[0].set_xlabel("N = |α|^2")
    ax[0].set_ylabel("β* (ML)")
    ax[0].set_title("Optimal Kennedy displacement β* vs N")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(N_vals, p_star, lw=2.0)
    ax[1].set_xlabel("N = |α|^2")
    ax[1].set_ylabel("p* (ML)")
    ax[1].set_title("Optimal prior p* vs N")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kennedy_opt_params.png", dpi=160)

    print("Saved:")
    print("  kennedy_capacity_results.csv")
    print("  kennedy_capacity_curves.png")
    print("  kennedy_opt_params.png")



if __name__ == "__main__":
    main()
