# minimal 2D Lennard–Jones MD in reduced units (ε=σ=m=kB=1)

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

# --- quick unit checks ---
if __name__ == "__main__":
    
    # potential minimum at r = 2^(1/6) with value -1 (before shifting). It can be derived by differentiating the LJ potential w.r.t. r and setting the derivative to zero.
    
    rmin = 2.0 ** (1.0 / 6.0)
    dr = np.array([rmin, 0.0])
    U_unshift, F = lj_pair_energy_force(dr, box=1000.0, rc=2.5)
    
    # Undo shift to check 'true' minimum ~ -1
    inv_rc2 = 1.0 / (2.5 * 2.5)
    inv_rc6 = inv_rc2 ** 3
    U_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6)
    U_true = U_unshift + U_shift
    
    print("U_true(rmin) ~ -1 :", U_true)
    print("Force at rmin ~ 0  :", F)