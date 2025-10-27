"""
Kawahara (Dispersively-Modified Kuramoto-Sivashinsky) Equation Solver using JAX

Solves the Kawahara equation with periodic boundary conditions:
    ∂u/∂t = -μ∂⁴u/∂x⁴ - ∂²u/∂x² - u∂u/∂x - δ∂ⁿu/∂xⁿ (- ν∂⁵u/∂x⁵ for 5th order)

where:
    - u(x,t): state variable
    - μ: viscosity/diffusion coefficient (4th order)
    - δ: dispersion coefficient
    - ν: 5th order dispersion coefficient
    - n: dispersion order (1, 3, 5, or 3+5=8 (both 3rd and 5th order terms))

Three conservation formulations are supported:
    - NC: Non-conservative (standard form)
    - C: Conservative form
    - EP: Energy-preserving form

Note: 
    - Set δ=0 and ν=0 to recover the standard Kuramoto-Sivashinsky equation.
    - If n=8, both 3rd and 5th order dispersion terms are included, which is 
      the Benney-Lin equation.
    - Dispersion of order 1 is less common but included for completeness.
"""

#%% Modules
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional, Literal

#!!! MAKE SURE TO ENABLE 64-BIT FLOATS IN JAX BY DEFAULT !!!
# jax.config.update('jax_enable_x64', True)

# Quadratic mapping function (unique Kronecker product)
# def _make_quadratic_mapping_jax(max_r=300):
#     """
#     Factory function to create quadratic mapping for unique Kronecker product.
    
#     Computes the unique quadratic Kronecker product: u ⊗ u with redundancies eliminated.
#     For a vector u of length r, this produces all products u[i]*u[j] where i≤j,
#     resulting in S = r(r+1)/2 unique terms.
    
#     Args:
#         max_r: Maximum dimension supported
    
#     Returns:
#         quadratic_mapping: Function that computes quadratic Kronecker product
#     """
#     # Pre-compute ALL valid index combinations for the maximum dimension
#     indices = []
#     for i in range(max_r):
#         for j in range(i, max_r):
#             indices.append((i, j))
    
#     i_indices = jnp.array([idx[0] for idx in indices])
#     j_indices = jnp.array([idx[1] for idx in indices])
    
#     # Pre-compute positions of valid indices for each possible r
#     valid_positions_for_r = {}
#     for r in range(1, max_r + 1):
#         valid_positions = []
#         for idx, (i, j) in enumerate(indices):
#             if i < r and j < r:
#                 valid_positions.append(idx)
#         valid_positions_for_r[r] = jnp.array(valid_positions, dtype=jnp.int32)
    
#     @jit
#     def quadratic_mapping(x):
#         """
#         Compute unique quadratic Kronecker product.
        
#         Args:
#             x: Input vector of shape (r,) or (r, n) for batch processing
        
#         Returns:
#             Quadratic products of shape (S,) or (S, n) where S = r(r+1)/2
#         """
#         # Handle both 1D and 2D inputs
#         if x.ndim == 1:
#             r = x.shape[0]
#             x = x.reshape(-1, 1)
#             squeeze_output = True
#         else:
#             r, n = x.shape
#             squeeze_output = False
        
#         if r > max_r:
#             raise ValueError(f"Input dimension {r} exceeds max_r={max_r}")
        
#         # Pad x to max_r
#         x_padded = jnp.zeros((max_r, x.shape[1]))
#         x_padded = x_padded.at[:r, :].set(x)
        
#         # Compute all products (including invalid ones)
#         all_products = x_padded[i_indices, :] * x_padded[j_indices, :]
        
#         # Select only valid products for dimension r
#         valid_positions = valid_positions_for_r[r]
#         result = all_products[valid_positions, :]
        
#         if squeeze_output:
#             result = result.squeeze()
        
#         return result
    
#     return quadratic_mapping

#!!! CHANGE MAX_R IF YOUR DIMENSION EXCEEDS 600 !!!
# Create the quadratic mapping function (supports up to dimension 300)
# _quadratic_map = _make_quadratic_mapping_jax(max_r=600)

def default_feature_map(reduced_data_points):
    r = reduced_data_points.shape[0]
    return jnp.concatenate(
        [reduced_data_points[i] * reduced_data_points[i:] for i in range(r)],
        axis=0)

_quadratic_map = default_feature_map

#%%
class KawaharaSolver:
    """
    Kawahara equation solver with periodic boundary conditions.
    
    Features:
    - Three conservation types: NC, C, EP
    - Dispersion orders: 1, 3, 5, or both (3+5=8)
    - Dynamic parameter updates
    - Crank-Nicolson Adams-Bashforth time integration
    """
    
    def __init__(
        self,
        spatial_domain: Tuple[float, float] = (0.0, 50.0),
        time_domain: Tuple[float, float] = (0.0, 300.0),
        dx: float = None,
        dt: float = 0.01,
        mu: float = 1.0,
        delta: float = 0.15,
        nu: float = 0.01,
        nx: int = 512,
        conservation_type: Literal['NC', 'C', 'EP'] = 'NC',
        dispersion_order: Literal[1, 3, 5, 8] = 3,
        boundary_condition: str = "periodic"
    ):
        """
        Initialize the Kawahara equation solver.
        
        Args:
            spatial_domain: (x_min, x_max) spatial domain bounds
            time_domain: (t_min, t_max) temporal domain bounds
            dx: spatial grid spacing (if None, computed from nx)
            dt: time step size
            mu: viscosity coefficient (4th order diffusion)
            delta: dispersion coefficient
            nx: number of spatial grid points (used if dx is None)
            conservation_type: 'NC' (non-conservative), 'C' (conservative), or 'EP' (energy-preserving)
            dispersion_order: 1, 3, 5, or 3+5=`8`(order of dispersion term)
            boundary_condition: type of boundary condition (only 'periodic' supported)
        """
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.dt = dt
        self.bc = boundary_condition
        
        # Parameters (mutable for dynamic updates)
        self.mu = mu
        self.delta = delta
        self.nu = nu
        self.conservation_type = conservation_type
        self.dispersion_order = dispersion_order
        
        # Validation
        assert boundary_condition == "periodic", "Only periodic BC currently supported"
        assert conservation_type in ['NC', 'C', 'EP'], "conservation_type must be 'NC', 'C', or 'EP'"
        assert dispersion_order in [1, 3, 5, 8], "dispersion_order must be 1, 3, 5, or 3+5=8"
        
        # Create spatial grid
        if dx is None:
            dx = (spatial_domain[1] - spatial_domain[0]) / nx
        self.dx = dx
        
        # Periodic BC: exclude the endpoint
        self.x = jnp.arange(spatial_domain[0], spatial_domain[1], dx)
        self.t = jnp.arange(time_domain[0], time_domain[1] + dt, dt)
        self.nx = len(self.x)
        self.nt = len(self.t)
        
        print(f"Kawahara Solver initialized:")
        print(f"  Grid: {self.nx} spatial points, {self.nt} time steps")
        print(f"  dx = {self.dx:.6f}, dt = {self.dt:.6f}")
        print(f"  Conservation type: {self.conservation_type}")
        print(f"  Dispersion order: {self.dispersion_order}")
        print(f"  Parameters: μ = {self.mu}, δ = {self.delta}, ν = {self.nu}")
    
    def update_parameters(self, 
                          mu: Optional[float] = None, 
                          delta: Optional[float] = None,
                          nu: Optional[float] = None,
                          conservation_type: Optional[str] = None,
                          dispersion_order: Optional[int] = None):
        """
        Update solver parameters. Changes will be reflected in subsequent simulations.
        
        Args:
            mu: viscosity coefficient (if provided)
            delta: 1st or 3rd order dispersion coefficient (if provided)
            nu: 5th order dispersion coefficient (if provided)
            conservation_type: conservation type (if provided)
            dispersion_order: dispersion order (if provided)
        """
        if mu is not None:
            self.mu = mu
            print(f"Updated μ = {self.mu}")
        
        if delta is not None:
            self.delta = delta
            print(f"Updated δ = {self.delta}")

        if nu is not None:
            self.nu = nu
            print(f"Updated ν = {self.nu}")
        
        if conservation_type is not None:
            assert conservation_type in ['NC', 'C', 'EP']
            self.conservation_type = conservation_type
            print(f"Updated conservation type = {self.conservation_type}")
        
        if dispersion_order is not None:
            assert dispersion_order in [1, 3, 5, 8]
            self.dispersion_order = dispersion_order
            print(f"Updated dispersion order = {self.dispersion_order}")
    
    def _create_linear_operator(self) -> jnp.ndarray:
        """
        Create the linear operator A for the Kawahara equation.
        
        The linear part includes:
        - 4th order diffusion: -μ∂⁴u/∂x⁴
        - 2nd order term: -∂²u/∂x²
        - Dispersion: -δ∂ⁿu/∂xⁿ and -ν∂⁵u/∂x⁵ depending on dispersion_order
          (n=1, n=3, n=5, or both 3 and 5 => n=8)
        
        Returns:
            A: Linear operator matrix (nx, nx)
        """
        nx = self.nx
        dx = self.dx
        mu = self.mu
        delta = self.delta
        nu = self.nu
        ord = self.dispersion_order
        
        # Compute stencil coefficients based on dispersion order
        if ord == 1:
            # Dispersion order 1: -δ∂u/∂x
            np2 = -mu/dx**4
            np1 = 4*mu/dx**4 - 1/dx**2 - delta/(2*dx)
            n   = -6*mu/dx**4 + 2/dx**2
            nm1 = 4*mu/dx**4 - 1/dx**2 + delta/(2*dx)
            nm2 = -mu/dx**4
        elif ord == 3:
            # Dispersion order 3: -δ∂³u/∂x³
            np2 = -mu/dx**4 - delta/(2*dx**3)
            np1 = 4*mu/dx**4 + delta/(dx**3) - 1/dx**2
            n   = -6*mu/dx**4 + 2/dx**2
            nm1 = 4*mu/dx**4 - delta/(dx**3) - 1/dx**2
            nm2 = -mu/dx**4 + delta/(2*dx**3)
        elif ord == 5:
            # Dispersion order 5: -ν∂⁵u/∂x⁵
            np3 = -nu/(2*dx**5)
            np2 = 2*nu/dx**5 - mu/dx**4 
            np1 = -5*nu/(2*dx**5) + 4*mu/dx**4 - 1/dx**2
            n   = -6*mu/dx**4 + 2/dx**2
            nm1 = 5*nu/(2*dx**5) + 4*mu/dx**4 - 1/dx**2
            nm2 = -2*nu/dx**5 - mu/dx**4
            nm3 = nu/(2*dx**5)
        else:  # ord == 8 (both 3rd and 5th order)
            np3 = -nu/(2*dx**5)
            np2 = 2*nu/dx**5 - mu/dx**4 - delta/(2*dx**3)
            np1 = -5*nu/(2*dx**5) + 4*mu/dx**4 + delta/(dx**3) - 1/dx**2
            n   = -6*mu/dx**4 + 2/dx**2
            nm1 = 5*nu/(2*dx**5) + 4*mu/dx**4 - delta/(dx**3) - 1/dx**2
            nm2 = -2*nu/dx**5 - mu/dx**4 + delta/(2*dx**3)
            nm3 = nu/(2*dx**5)

        if ord < 5: 
            # Create pentadiagonal matrix
            A = (jnp.diag(n * jnp.ones(nx)) +
                jnp.diag(np1 * jnp.ones(nx - 1), k=1) +
                jnp.diag(nm1 * jnp.ones(nx - 1), k=-1) +
                jnp.diag(np2 * jnp.ones(nx - 2), k=2) +
                jnp.diag(nm2 * jnp.ones(nx - 2), k=-2))
            
            # Apply periodic boundary conditions
            A = A.at[0, -2:].set(jnp.array([nm2, nm1]))
            A = A.at[1, -1].set(nm2)
            A = A.at[-2, 0].set(np2)
            A = A.at[-1, :2].set(jnp.array([np1, np2]))
        else:
            # Create heptadiagonal matrix
            A = (jnp.diag(n * jnp.ones(nx)) +
                jnp.diag(np1 * jnp.ones(nx - 1), k=1) +
                jnp.diag(nm1 * jnp.ones(nx - 1), k=-1) +
                jnp.diag(np2 * jnp.ones(nx - 2), k=2) +
                jnp.diag(nm2 * jnp.ones(nx - 2), k=-2) +
                jnp.diag(np3 * jnp.ones(nx - 3), k=3) +
                jnp.diag(nm3 * jnp.ones(nx - 3), k=-3))
            
            # Apply periodic boundary conditions
            A = A.at[0, -3:].set(jnp.array([nm3, nm2, nm1]))
            A = A.at[1, -2:].set(jnp.array([nm3, nm2]))
            A = A.at[2, -1].set(nm3)
            A = A.at[-3, 0].set(np3)
            A = A.at[-2, :2].set(jnp.array([np2, np3]))
            A = A.at[-1, :3].set(jnp.array([np1, np2, np3]))
        
        return A
    
    def _create_nonlinear_operator(self) -> jnp.ndarray:
        """
        Create the nonlinear operator F for the quadratic term -u∂u/∂x.
        
        Three formulations:
        - NC: Non-conservative (standard centered difference)
        - C: Conservative (preserves mass)
        - EP: Energy-preserving (preserves energy)
        
        Returns:
            F: Quadratic operator matrix (nx, S) where S = nx(nx+1)/2
        """
        nx = self.nx
        dx = self.dx
        S = nx * (nx + 1) // 2
        
        if nx < 3:
            return jnp.zeros((nx, S))
        
        if self.conservation_type == 'NC':
            return self._create_nonconservative_F(nx, dx, S)
        elif self.conservation_type == 'C':
            return self._create_conservative_F(nx, dx, S)
        else:  # EP
            return self._create_energy_preserving_F(nx, dx, S)
    
    def _create_nonconservative_F(self, nx: int, dx: float, S: int) -> jnp.ndarray:
        """Non-conservative formulation: -u∂u/∂x using centered differences."""
        F = jnp.zeros((nx, S))
        
        # Interior points: central difference
        for i in range(1, nx - 1):
            # # Index for u[i]*u[i+1] in the quadratic Kronecker product
            # idx_ip = i * (nx + 1) - i * (i + 1) // 2 + 1
            # # Index for u[i]*u[i-1]
            # idx_im = (i - 1) * (nx + 1) - (i - 1) * i // 2
            idx_i2 = i * (nx + 1) - i * (i + 1) // 2  # u[i]²
            idx_im2 = (i - 1) * (nx + 1) - (i - 1) * i // 2  # u[i-1]²
            idx_ip = idx_i2 + 1  # u[i]*u[i+1]
            idx_im = idx_im2 + 1  # u[i-1]*u[i]
            
            F = F.at[i, idx_ip].set(-1 / (2 * dx))
            F = F.at[i, idx_im].set(1 / (2 * dx))
        
        # Periodic boundary conditions
        # First point
        F = F.at[0, 1].set(-1 / (2 * dx))  # u[0]*u[1]
        F = F.at[0, nx - 1].set(1 / (2 * dx))  # u[0]*u[nx-1] (periodic)
        
        # Last point
        F = F.at[-1, nx - 1].set(-1 / (2 * dx))  # u[nx-1]² diagonal
        F = F.at[-1, S - 2].set(1 / (2 * dx))  # u[nx-1]*u[nx-2]
        
        return F
    
    def _create_conservative_F(self, nx: int, dx: float, S: int) -> jnp.ndarray:
        """Conservative formulation: preserves mass."""
        F = jnp.zeros((nx, S))
        
        # Interior points
        for i in range(1, nx - 1):
            # Index for u[i+1]²
            idx_ip2 = (i + 1) * (nx + 1) - (i + 1) * (i + 2) // 2
            # Index for u[i-1]²
            idx_im2 = (i - 1) * (nx + 1) - (i - 1) * i // 2
            
            F = F.at[i, idx_ip2].set(-1 / (4 * dx))
            F = F.at[i, idx_im2].set(1 / (4 * dx))
        
        # Periodic boundary conditions
        F = F.at[0, nx].set(-1 / (4 * dx))  # u[1]²
        F = F.at[0, S - 1].set(1 / (4 * dx))  # u[nx-1]²
        F = F.at[-1, S - 3].set(1 / (4 * dx))  # u[nx-2]²
        F = F.at[-1, 0].set(-1 / (4 * dx))  # u[0]²
        
        return F
    
    def _create_energy_preserving_F(self, nx: int, dx: float, S: int) -> jnp.ndarray:
        """Energy-preserving formulation: preserves energy."""
        F = jnp.zeros((nx, S))
        
        # Interior points
        for i in range(1, nx - 1):
            # Indices for various products
            idx_i2 = i * (nx + 1) - i * (i + 1) // 2  # u[i]²
            idx_ip2 = (i + 1) * (nx + 1) - (i + 1) * (i + 2) // 2  # u[i+1]²
            idx_im2 = (i - 1) * (nx + 1) - (i - 1) * i // 2  # u[i-1]²
            idx_ip = idx_i2 + 1  # u[i]*u[i+1]
            idx_im = idx_im2 + 1  # u[i-1]*u[i]
            
            F = F.at[i, idx_ip2].set(-1 / (6 * dx))
            F = F.at[i, idx_im2].set(1 / (6 * dx))
            F = F.at[i, idx_ip].set(-1 / (6 * dx))
            F = F.at[i, idx_im].set(1 / (6 * dx))
        
        # Periodic boundary conditions
        F = F.at[0, 1].set(-1 / (6 * dx))
        F = F.at[0, nx].set(-1 / (6 * dx))
        F = F.at[0, nx - 1].set(1 / (6 * dx))
        F = F.at[0, S - 1].set(1 / (6 * dx))
        
        F = F.at[-1, S - 2].set(1 / (6 * dx))
        F = F.at[-1, S - 3].set(1 / (6 * dx))
        F = F.at[-1, 0].set(-1 / (6 * dx))
        F = F.at[-1, nx - 1].set(-1 / (6 * dx))
        
        return F
    
    @staticmethod
    @jit
    def _compute_quadratic_term(u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute unique quadratic Kronecker product: u ⊗ u.
        
        Args:
            u: State vector of shape (nx,)
        
        Returns:
            Quadratic Kronecker product of shape (S,) where S = nx(nx+1)/2
        """
        return _quadratic_map(u)
    
    def solve(
        self,
        u0: jnp.ndarray,
        return_all_steps: bool = True,
        verbose: bool = False
    ) -> jnp.ndarray:
        """
        Solve the Kawahara equation using Crank-Nicolson Adams-Bashforth method.
        
        Args:
            u0: Initial condition (shape: nx)
            return_all_steps: If True, return solution at all time steps;
                            if False, return only final state
            verbose: Print progress information
        
        Returns:
            u: Solution array of shape (nx, nt) if return_all_steps=True,
               or (nx,) if return_all_steps=False
        """
        assert u0.shape[0] == self.nx, f"Initial condition must have length {self.nx}"
        
        if verbose:
            print(f"Solving Kawahara equation with current parameters:")
            print(f"  μ = {self.mu}, δ = {self.delta}, ν = {self.nu}")
            print(f"  Conservation: {self.conservation_type}, Dispersion order: {self.dispersion_order}")
        
        # Create operators with current parameters
        A = self._create_linear_operator()
        F = self._create_nonlinear_operator()
        
        if return_all_steps:
            u_all = jnp.zeros((self.nx, self.nt))
            u_all = u_all.at[:, 0].set(u0)
        
        # Time integration using CNAB
        u_curr = u0
        S = self.nx * (self.nx + 1) // 2
        u2_prev = jnp.zeros(S)  # Previous quadratic term
        
        # Pre-compute matrices for efficiency
        I = jnp.eye(self.nx)
        dt = self.dt
        ImdtA_inv = jnp.linalg.inv(I - dt/2 * A)
        IpdtA = I + dt/2 * A
        
        for j in range(1, self.nt):
            u2_curr = self._compute_quadratic_term(u_curr)
            
            if j == 1:
                # First step: use implicit Euler for nonlinear term
                rhs = IpdtA @ u_curr + F @ u2_curr * dt
            else:
                # Adams-Bashforth for nonlinear term: 3/2*f(t_n) - 1/2*f(t_{n-1})
                rhs = IpdtA @ u_curr + F @ u2_curr * (3*dt/2) - F @ u2_prev * (dt/2)
            
            u_curr = ImdtA_inv @ rhs
            u2_prev = u2_curr
            
            if return_all_steps:
                u_all = u_all.at[:, j].set(u_curr)
            
            if verbose and (j % 100 == 0 or j == self.nt - 1):
                print(f"  Step {j}/{self.nt-1}, max|u| = {jnp.abs(u_curr).max():.6f}")
        
        return u_all if return_all_steps else u_curr


def example_initial_condition(x: jnp.ndarray, mode: str = "cosines", L: float = 50.0) -> jnp.ndarray:
    """
    Generate example initial conditions for the Kawahara equation.
    
    Args:
        x: Spatial grid points
        mode: Type of initial condition
        L: Domain length
    
    Returns:
        u0: Initial condition
    """
    if mode == "cosines":
        # Classic two-mode initial condition
        a = 1.0
        b = 0.1
        return a * jnp.cos(2 * jnp.pi * x / L) + b * jnp.cos(4 * jnp.pi * x / L)
    
    elif mode == "single_mode":
        return jnp.cos(2 * jnp.pi * x / L)
    
    elif mode == "gaussian":
        x0 = (x[-1] - x[0]) / 2
        return jnp.exp(-((x - x0) / 5)**2)
    
    elif mode == "random":
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, x.shape) * 0.1
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

#%% Test and visualization
if __name__ == "__main__":
    # Example: Solve Kawahara equation with parameter updates
    print("=" * 70)
    print("Kawahara Equation Solver - Basic Example")
    print("=" * 70)
    
    # Create solver
    solver = KawaharaSolver(
        spatial_domain=(0.0, 50.0),
        time_domain=(0.0, 300.0),
        nx=512,
        dt=0.01,
        mu=1.0,  # you want to keep mu near 1.0 (it'll go very chaotic if small)
        delta=0.05,  # you want to keep delta small (e.g., 0.05)
        nu=0.001,  # you also want to keep nu small (e.g., 0.001)
        conservation_type='NC', 
        dispersion_order=8
    )
    
#%% # Initial condition
    u0 = example_initial_condition(solver.x, mode="cosines", L=solver.spatial_domain[1])
    
#%% # First run with initial parameters 
    # (THIS WILL TAKE A WHILE. JULIA IS SOOOOOOOOO MUCH FASTER...)
    print("\n[Run 1] Solving with initial parameters...")
    u_solution = solver.solve(u0, return_all_steps=True, verbose=False)
    u1 = u_solution[:, -1]
    print(f"Final state range: [{u1.min():.4f}, {u1.max():.4f}]")
    
#%% # Update parameters and run again 
    # (SKIP THIS IF YOU DON'T WANNA WAIT AGAIN...)
    print("\n[Parameter Update]")
    solver.update_parameters(mu=0.8, delta=0.3, nu=0.02)
    
    print("\n[Run 2] Solving with updated parameters...")
    u2 = solver.solve(u0, return_all_steps=False, verbose=False)
    print(f"Final state range: [{u2.min():.4f}, {u2.max():.4f}]")
    
    print("\nParameter update mechanism working correctly! ✓")

#%% # Visualization of the final solution
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
    ax1.set_title(f'Kawahara Solution: 3D Surface\n(μ={solver.mu}, δ={solver.delta}, {solver.conservation_type})')
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
    plt.savefig('figures/kawahara_solution.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'figures/kawahara_solution.png'")
    plt.show()
    
    print(f"\nSolution statistics:")
    print(f"  Shape: {u_solution.shape}")
    print(f"  Min: {u_solution.min():.6f}")
    print(f"  Max: {u_solution.max():.6f}")
    print(f"  Mean: {u_solution.mean():.6f}")
    print(f"  Std: {u_solution.std():.6f}")
# %%
