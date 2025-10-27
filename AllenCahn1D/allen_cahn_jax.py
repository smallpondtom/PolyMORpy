"""
Allen-Cahn PDE Solver using JAX

Solves the Allen-Cahn equation:
    ∂u/∂t = μ∂²u/∂x² - ε(u³ - u)

where:
    - u: state variable
    - μ: diffusion coefficient  
    - ε: nonlinear coefficient
"""

#%% 
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from typing import Tuple, Optional


def _make_cubic_mapping_jax(max_r=200):
    """
    Factory function to create a cubic mapping that completely avoids boolean indexing.
    
    This implementation uses the unique Kronecker product with redundancies eliminated.
    For a vector u of length r, this produces all products u[i]*u[j]*u[k] where i≤j≤k,
    resulting in S = r(r+1)(r+2)/6 unique terms.
    
    Args:
        max_r: Maximum dimension supported (pre-computes indices for efficiency)
    
    Returns:
        cubic_mapping: Function that computes cubic Kronecker product
        diagonal_positions: Dict mapping dimension r to positions of diagonal terms (i,i,i)
    """
    # Pre-compute ALL valid index combinations for the maximum dimension
    indices = []
    for i in range(max_r):
        for j in range(i, max_r):
            for k in range(j, max_r):
                indices.append((i, j, k))
    
    i_indices = jnp.array([idx[0] for idx in indices])
    j_indices = jnp.array([idx[1] for idx in indices])
    k_indices = jnp.array([idx[2] for idx in indices])
    
    # Pre-compute POSITIONS of valid indices for each possible r
    # (these are NOT contiguous when generated from max_r indices!)
    valid_positions_for_r = {}
    diagonal_positions_for_r = {}
    
    for r in range(1, max_r + 1):
        valid_positions = []
        diag_positions = []
        
        for idx, (i, j, k) in enumerate(indices):
            if i < r and j < r and k < r:
                # This index is valid for dimension r
                valid_positions.append(idx)
                # Track diagonal positions within the valid set
                if i == j == k:
                    diag_positions.append(len(valid_positions) - 1)
        
        valid_positions_for_r[r] = jnp.array(valid_positions, dtype=jnp.int32)
        diagonal_positions_for_r[r] = jnp.array(diag_positions, dtype=jnp.int32)
    
    @jit
    def cubic_mapping(x):
        """
        Compute unique cubic Kronecker product.
        
        Args:
            x: Input vector of shape (r,) or (r, n) for batch processing
        
        Returns:
            Cubic products of shape (S,) or (S, n) where S = r(r+1)(r+2)/6
        """
        # Handle both 1D and 2D inputs
        if x.ndim == 1:
            r = x.shape[0]
            x = x.reshape(-1, 1)
            squeeze_output = True
        else:
            r, n = x.shape
            squeeze_output = False
        
        if r > max_r:
            raise ValueError(f"Input dimension {r} exceeds max_r={max_r}")
        
        # Pad x to max_r
        x_padded = jnp.zeros((max_r, x.shape[1]))
        x_padded = x_padded.at[:r, :].set(x)
        
        # Compute all products (including invalid ones)
        all_products = x_padded[i_indices, :] * x_padded[j_indices, :] * x_padded[k_indices, :]
        
        # Select only valid products for dimension r using pre-computed positions
        valid_positions = valid_positions_for_r[r]
        result = all_products[valid_positions, :]
        
        if squeeze_output:
            result = result.squeeze()
        
        return result
    
    return cubic_mapping, diagonal_positions_for_r

#!!! MAKE SURE TO UPDATE MAX_R IF YOUR SPATIAL DIMENSION EXCEEDS THIS !!!
# Create the cubic mapping function (supports up to dimension 200)
_cubic_map, _diagonal_positions = _make_cubic_mapping_jax(max_r=200)

#%%
class AllenCahnSolver:
    """
    Allen-Cahn PDE solver with periodic boundary conditions.
    Uses Semi-Implicit Crank-Nicolson (SICN) or Crank-Nicolson Adam-Bashforth (CNAB) time integration.
    """
    
    def __init__(
        self,
        spatial_domain: Tuple[float, float] = (0.0, 1.0),
        time_domain: Tuple[float, float] = (0.0, 1.0),
        dx: float = 0.01,
        dt: float = 0.001,
        mu: float = 0.01,  # diffusion coefficient
        epsilon: float = 1.0,  # nonlinear coefficient
        boundary_condition: str = "periodic"
    ):
        """
        Initialize the Allen-Cahn solver.
        
        Args:
            spatial_domain: (x_min, x_max) spatial domain bounds
            time_domain: (t_min, t_max) temporal domain bounds
            dx: spatial grid spacing
            dt: time step size
            mu: diffusion coefficient
            epsilon: nonlinear coefficient
            boundary_condition: type of boundary condition (only 'periodic' implemented)
        """
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.dx = dx
        self.dt = dt
        self.mu = mu
        self.epsilon = epsilon
        self.bc = boundary_condition
        
        # Create spatial and temporal grids
        if boundary_condition == "periodic":
            self.x = jnp.arange(spatial_domain[0], spatial_domain[1], dx)
        else:
            self.x = jnp.arange(spatial_domain[0], spatial_domain[1] + dx, dx)
        
        self.t = jnp.arange(time_domain[0], time_domain[1] + dt, dt)
        self.nx = len(self.x)
        self.nt = len(self.t)
        
        # Create finite difference operators
        self.A, self.E = self._create_operators()
        
    def _create_operators(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create linear operator A and cubic operator E for periodic BC.
        
        Returns:
            A: Linear operator matrix (nx, nx)
            E: Cubic operator matrix (nx, S) where S = nx(nx+1)(nx+2)/6
        """
        nx = self.nx
        dx = self.dx
        mu = self.mu
        eps = self.epsilon
        
        # Create linear operator A (tridiagonal with periodic BC)
        # A = ε*I - μ*D²/dx² where D² is the second derivative operator
        diagonal = (eps - 2*mu/dx**2) * jnp.ones(nx)
        off_diagonal = (mu/dx**2) * jnp.ones(nx-1)
        
        A = jnp.diag(diagonal) + jnp.diag(off_diagonal, k=1) + jnp.diag(off_diagonal, k=-1)
        
        # Add periodic boundary conditions
        A = A.at[0, -1].set(mu/dx**2)
        A = A.at[-1, 0].set(mu/dx**2)
        
        # Create cubic operator E for sparse cubic term
        # E maps the cubic Kronecker structure to the nonlinear term -ε*u³
        S = int(nx * (nx + 1) * (nx + 2) / 6)
        
        # Get pre-computed diagonal positions for this dimension
        diagonal_positions = _diagonal_positions[nx]
        
        # Create sparse E matrix: E[i, diagonal_positions[i]] = -epsilon
        # This extracts the diagonal cubic terms (u[i]³) from the full cubic Kronecker product
        E = jnp.zeros((nx, S))
        E = E.at[jnp.arange(nx), diagonal_positions].set(-eps)
        
        return A, E
    
    def update_parameters(self, mu: Optional[float] = None, epsilon: Optional[float] = None):
        """
        Update diffusion coefficient (mu) and/or nonlinear coefficient (epsilon).
        Automatically recreates the operators A and E.
        
        Args:
            mu: New diffusion coefficient (if None, keeps current value)
            epsilon: New nonlinear coefficient (if None, keeps current value)
        """
        if mu is not None:
            self.mu = mu
        if epsilon is not None:
            self.epsilon = epsilon
        
        # Recreate operators with new parameters
        self.A, self.E = self._create_operators()
    
    @staticmethod
    @jit
    def _compute_cubic_term(u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute unique cubic Kronecker product: u ⊗ u ⊗ u.
        
        For a vector u of length nx, this computes all products u[i]*u[j]*u[k] 
        where i≤j≤k, resulting in S = nx(nx+1)(nx+2)/6 unique terms.
        
        Args:
            u: State vector of shape (nx,)
        
        Returns:
            Cubic Kronecker product of shape (S,)
        """
        # Compute unique cubic Kronecker product directly
        return _cubic_map(u)
    
    @staticmethod
    @jit
    def _sicn_step(u_prev: jnp.ndarray, A: jnp.ndarray, E: jnp.ndarray, 
                   dt: float) -> jnp.ndarray:
        """
        Single time step using Semi-Implicit Crank-Nicolson (SICN) method.
        Linear terms: Crank-Nicolson (implicit)
        Nonlinear terms: Explicit
        
        Args:
            u_prev: Solution at previous time step
            A: Linear operator matrix
            E: Cubic operator matrix
            dt: Time step size
            
        Returns:
            u_next: Solution at next time step
        """

        nx = A.shape[0]
        I = jnp.eye(nx)
        
        u3 = _cubic_map(u_prev)
        
        lhs = I - dt/2 * A
        rhs = (I + dt/2 * A) @ u_prev + E @ u3 * dt
        
        u_next = jnp.linalg.solve(lhs, rhs)
        return u_next
    
    @staticmethod
    @jit
    def _cnab_step(u_prev: jnp.ndarray, u3_prev: jnp.ndarray, A: jnp.ndarray, 
                   E: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single time step using Crank-Nicolson Adam-Bashforth (CNAB) method.
        Linear terms: Crank-Nicolson (implicit)
        Nonlinear terms: Adams-Bashforth (explicit, 2nd order)
        
        For first step (when u3_prev is zeros), uses SICN method.
        
        Args:
            u_prev: Solution at previous time step
            u3_prev: Cubic term at previous time step (zeros for first step)
            A: Linear operator matrix
            E: Cubic operator matrix
            dt: Time step size
            
        Returns:
            u_next: Solution at next time step
            u3_curr: Cubic term at current time step
        """
        nx = A.shape[0]
        I = jnp.eye(nx)
        
        u3_curr = _cubic_map(u_prev)
        
        lhs = I - dt/2 * A
        rhs = (I + dt/2 * A) @ u_prev + E @ u3_curr * (3*dt/2) - E @ u3_prev * (dt/2)
        
        u_next = jnp.linalg.solve(lhs, rhs)
        return u_next, u3_curr
    
    def solve(
        self,
        u0: jnp.ndarray,
        method: str = "cnab",
        return_all_steps: bool = True
    ) -> jnp.ndarray:
        """
        Solve the Allen-Cahn equation.
        
        Args:
            u0: Initial condition (shape: nx)
            method: Integration method - 'sicn' or 'cnab' (default: 'cnab')
            return_all_steps: If True, return solution at all time steps; 
                            if False, return only final state
        
        Returns:
            u: Solution array of shape (nx, nt) if return_all_steps=True,
               or (nx,) if return_all_steps=False
        """
        assert u0.shape[0] == self.nx, f"Initial condition must have length {self.nx}"
        method = method.lower()
        assert method in ["sicn", "cnab"], "Method must be 'sicn' or 'cnab'"
        
        if return_all_steps:
            u_all = jnp.zeros((self.nx, self.nt))
            u_all = u_all.at[:, 0].set(u0)
        
        u_curr = u0
        
        S = int(self.nx * (self.nx + 1) * (self.nx + 2) / 6)
        u3_prev = jnp.zeros(S)
        
        for j in range(1, self.nt):
            if method == "sicn":
                u_curr = self._sicn_step(u_curr, self.A, self.E, self.dt)
            else:  # cnab
                u_curr, u3_prev = self._cnab_step(u_curr, u3_prev, self.A, self.E, self.dt)
            
            if return_all_steps:
                u_all = u_all.at[:, j].set(u_curr)
        
        return u_all if return_all_steps else u_curr
    
#%% Example usage and helper functions
def example_initial_condition(x: jnp.ndarray, mode: str = "random") -> jnp.ndarray:
    """
    Generate example initial conditions.
    
    Args:
        x: Spatial grid points
        mode: Type of initial condition
            - 'random': Random perturbation around 0
            - 'sine': Sine wave
            - 'tanh': Hyperbolic tangent (interface)
            - 'multiscale': Multiple sine waves
    
    Returns:
        u0: Initial condition
    """
    if mode == "random":
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, x.shape)
    
    elif mode == "sine":
        return 0.5 * jnp.sin(2 * jnp.pi * x)
    
    elif mode == "tanh":
        return jnp.tanh((x - 0.5) / 0.1)
    
    elif mode == "multiscale":
        return 0.3 * jnp.sin(2 * jnp.pi * x) + 0.2 * jnp.sin(8 * jnp.pi * x)
    
    elif mode == "bump":
        return 0.47 * jnp.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2)) - 0.23
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

#%%
if __name__ == "__main__":
    # Example: Solve Allen-Cahn equation
    print("Setting up Allen-Cahn solver...")
    
    solver = AllenCahnSolver(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 1.0),
        dx=0.01,
        dt=0.001,
        mu=0.01,
        epsilon=1.0,
        boundary_condition="periodic"
    )
    
    print(f"Spatial grid: {solver.nx} points")
    print(f"Time steps: {solver.nt}")
    print(f"dx = {solver.dx}, dt = {solver.dt}")

#%% NOTE: You can update parameters if needed
    # solver.update_parameters(mu=0.1, epsilon=1.0)

#%% # Generate initial condition
    u0 = example_initial_condition(solver.x, mode="tanh")
    
    print("\nSolving with CNAB method...")
    u_solution = solver.solve(u0, method="cnab", return_all_steps=True)
    
    print(f"Solution shape: {u_solution.shape}")
    print(f"Final solution range: [{u_solution[:, -1].min():.4f}, {u_solution[:, -1].max():.4f}]")
    print("\nDone! Solution stored in u_solution array.")
    print("You can visualize with: plt.imshow(u_solution.T, aspect='auto', origin='lower')")

#%% # Plot the 3D plot of Allen-Cahn solution
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Solve with full time history for visualization
    print("\n[Visualization]")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(221, projection='3d')
    T, X = jnp.meshgrid(solver.t, solver.x)
    surf = ax1.plot_surface(X, T, u_solution, cmap='viridis', 
                           linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel('Space (x)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title(f'Kawahara Solution: 3D Surface\n(μ={solver.mu}, ε={solver.epsilon})')
    ax1.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. Contour Plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(T, X, u_solution, levels=50, cmap='viridis')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('Kawahara Solution: Contour Plot')
    fig.colorbar(contour, ax=ax2)
    
    # 3. Time evolution at selected spatial points
    ax3 = fig.add_subplot(223)
    spatial_indices = [0, solver.nx//4, solver.nx//2, 3*solver.nx//4]
    for idx in spatial_indices:
        ax3.plot(solver.t, u_solution[idx, :], label=f'x={solver.x[idx]:.2f}')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('u(x,t)')
    ax3.set_title('Time Evolution at Selected Spatial Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Spatial profiles at selected times
    ax4 = fig.add_subplot(224)
    time_indices = [0, solver.nt//4, solver.nt//2, 3*solver.nt//4, -1]
    for idx in time_indices:
        ax4.plot(solver.x, u_solution[:, idx], label=f't={solver.t[idx]:.1f}')
    ax4.set_xlabel('Space (x)')
    ax4.set_ylabel('u(x,t)')
    ax4.set_title('Spatial Profiles at Selected Times')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/allen_cahn_solution.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'figures/allen_cahn_solution.png'")
    plt.show()
    
# %%
