# minimal 2D Lennard–Jones MD in reduced units (ε=σ=m=kB=1) potential's energy parameter (ε)), length parameter (σ), and a particle's mass ((m)

import math, random
import numpy as np

def minimum_image(dr: np.ndarray, box: float) -> np.ndarray:
    """Apply minimum image convention to a displacement vector."""
    dr -= box * np.rint(dr / box)
    return dr
    
def lj_pair_energy_force(dr: np.ndarray, box: float, rc: float = 2.5):
    """
    Compute shifted LJ pair energy and force vector for a displacement dr (2D).
    Returns (U_ij, F_ij_vector). Uses reduced units with sigma=epsilon=1.
    """
    dr = minimum_image(dr, box)
    r2 = np.dot(dr, dr)
    rc2 = rc * rc
    if r2 >= rc2:
        return 0.0, np.zeros(2)
    
    # precompute r ^ -6
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    
    # Unshifted potential and force
    U = 4.0 * (inv_r6 * inv_r6 - inv_r6)
    # Force vector: F = dU/dr * (-r_hat) => using inv_r2 form:
    # scalar factor = 24 * inv_r2 * (2*inv_r6^2 - inv_r6)
    F_scalar = 24.0 * inv_r2 * (2.0 * inv_r6 * inv_r6 - inv_r6)
    F_vec = F_scalar * dr

    # Shift potential so U(rc)=0 (force left unshifted; continuous at rc)
    inv_rc2 = 1.0 / rc2
    inv_rc6 = inv_rc2 ** 3
    U_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6)
    U -= U_shift
    return U, F_vec

# --- -----------------------quick unit checks ---------------------------------
# if __name__ == "__main__":
    
#     # potential minimum at r = 2^(1/6) with value -1 (before shifting). It can be derived by differentiating the LJ potential w.r.t. r and setting the derivative to zero.
    
#     rmin = 2.0 ** (1.0 / 6.0)
#     dr = np.array([rmin, 0.0])
#     U_unshift, F = lj_pair_energy_force(dr, box=1000.0, rc=2.5)
    
#     # Undo shift to check 'true' minimum ~ -1
#     inv_rc2 = 1.0 / (2.5 * 2.5)
#     inv_rc6 = inv_rc2 ** 3
#     U_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6)
#     U_true = U_unshift + U_shift
    
#     print("U_true(rmin) ~ -1 :", U_true)
#     print("Force at rmin ~ 0  :", F)
#-------------------------------------------------------------------------------------
    
# -------------------Initialization of positions & velocities-------------------------
def init_positions(N: int, box: float) -> np.ndarray:
    """Plcing N particles on a square lattice inside a box of given size."""
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = box / n_side
    
    # Generate co-ordinates for the atoms positions
    xs = np.arange(n_side) * spacing + 0.5 * spacing
    ys = np.arange(n_side) * spacing + 0.5 * spacing
    
    # grid of positions
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    return grid[:N].copy()

def init_velocities(N: int, T0: float, rng: np.random.Generator) -> np.ndarray:
    """Maxwell–Boltzmann in 2D, remove COM (center-of-mass) drift (i.e., total momentum = 0), rescale to exact T0 (m=kB=1)."""
    v = rng.normal(loc=0.0, scale=np.sqrt(T0), size=(N, 2))
    v -= v.mean(axis=0, keepdims=True)  # zero total momentum
    dof = 2 * N - 2                     # subtract 2 for removed COM in 2D
    K = 0.5 * np.sum(v * v)             # kinetic energy
    T_current = (2.0 * K) / dof         # current temperature, Using equipartition theorem: K = (dof/2) * T
    v *= np.sqrt(T0 / T_current)        # rescale to desired T0
    return v

# --- -----------------------Force and Energy Computation -----------------------
def compute_forces(positions: np.ndarray, box: float, rc: float = 2.5):
    """Compute total forces and potential energy via pairwise LJ with cutoff."""
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    U_total = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dr = positions[i] - positions[j]
            U_ij, F_ij = lj_pair_energy_force(dr, box, rc)
            U_total += U_ij
            forces[i] += F_ij
            forces[j] -= F_ij
    return forces, U_total

def kinetic_energy(vel: np.ndarray) -> float:
    return 0.5 * float(np.sum(vel * vel))

def temperature(vel: np.ndarray, removed_com: bool = True) -> float:
    N = vel.shape[0]
    dof = 2 * N - (2 if removed_com else 0)
    return (2.0 * kinetic_energy(vel)) / dof

# ------------------------- Velocity–Verlet + periodic boundaries -------------------
def apply_pbc(positions: np.ndarray, box: float) -> None:
    """Wrap positions into [0, L). In-place."""
    positions[:] = positions % box # the modulus operator (%) gives the remainder after division — but when used with NumPy arrays, it works element-wise.

# Velocity-verlet algorithm: to compute Newtonian dynamics numerically, i.e., conservation of energy
def velocity_verlet(positions, velocities, forces, box, dt, rc=2.5):
    """
    One Velocity–Verlet step. mass=1. Returns (positions, velocities, forces, U).
    """
    # half kick
    velocities += 0.5 * dt * forces
    # drift
    positions += dt * velocities
    apply_pbc(positions, box)
    # new forces
    new_forces, U = compute_forces(positions, box, rc)
    # half kick
    velocities += 0.5 * dt * new_forces
    return positions, velocities, new_forces, U
# -------------------------------------------------------------------------------

#------------------------- xyz trazectory output ---------------------
def write_xyz(positions, box, step, filename="traj.xyz", clear=False): # If clear=True, truncate file first (use at step==0)
    """Appending positions to an XYZ file"""
    if clear:
        open(filename, "w").close()
    N = positions.shape[0]
    with open(filename, "a") as f:
        f.write(f"{N}\nstep {step} box {box: .6f}\n")
        for (x, y) in positions:
            f.write(f"Ar{x: .6f} {y: .6f} 0.0\n")
# -------------------------------------------------------------------------------

# ------------------------- Main MD loop -------------------
# a tiny driver to run and inspect energy/temperature. Simulate N atoms in a square box, moving and colliding according to LJ forces, for nsteps time steps
def run_md(
    N=36, rho=0.8, T0=1.0, dt=0.005, nsteps=2000, rc=2.5, seed=1234,
    log_interval=100, xyz_interval=10, xyz_file="traj.xyz"
):
    """
    Minimal MD loop in reduced units (m=sigma=epsilon=kB=1).
    Prints energy/T and writes an XYZ trajectory (if xyz_interval>0).
    """
    # Setup & Initialization
    box = np.sqrt(N / rho)
    rng = np.random.default_rng(seed)
    pos = init_positions(N, box)
    vel = init_velocities(N, T0, rng)
    forces, U = compute_forces(pos, box, rc)
    
    # Simulation parameters printout
    print(f"# N={N} rho={rho:.3f} box={box:.5f} T0={T0} dt={dt} rc={rc}")
    print("# step      E_tot         E_pot         E_kin         T_inst")
    
    # Initial XYZ write
    if xyz_interval and xyz_interval > 0:
        write_xyz(pos % box, box, step=0, filename=xyz_file, clear=True) # clear=True means overwrite the file (start fresh).
        
    # Main MD loop
    for step in range(1, nsteps + 1):
        # Compute instantaneous quantities
        K = kinetic_energy(vel)
        T = temperature(vel, removed_com=True)
        E_tot = U + K
        if step % log_interval == 0:
            print(f"{step:6d}  {E_tot:12.6f}  {U:12.6f}  {K:12.6f}  {T:10.6f}")
        
        # Integrate one step of motion (Moves the system forward by dt using the Velocity–Verlet algorithm. half-kick, drift, compute new forces, half-kick again)
        pos, vel, forces, U = velocity_verlet(pos, vel, forces, box, dt, rc)
        
        # Write XYZ frames periodically
        if xyz_interval and (step % xyz_interval == 0):
            # positions are wrapped inside velocity_verlet -> safe to write
            write_xyz(pos, box, step=step, filename=xyz_file, clear=False) # clear=False means append the new frame, so you build a trajectory.

    return pos, vel, box

if __name__ == "__main__":
    run_md()