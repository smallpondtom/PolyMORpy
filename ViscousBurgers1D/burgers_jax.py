"""
Viscous Burgers' equation model in JAX

Solves: ∂u/∂t = μ ∂²u/∂x² - u ∂u/∂x

Supports:
- Multiple conservation types: Non-Conservative (NC), Conservative (C), Energy Preserving (EP)
- Boundary conditions: periodic, dirichlet
- Semi-implicit Euler integration
"""

#%%
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional, Literal, Callable

#%%
def default_feature_map(reduced_data_points):
    """Default quadratic feature map (unique Kronecker product)"""
    r = reduced_data_points.shape[0]
    return jnp.concatenate(
        [reduced_data_points[i] * reduced_data_points[i:] for i in range(r)],
        axis=0)


_quadratic_map = default_feature_map

#%%
class BurgersModel:
    """
    Viscous Burgers' equation solver using JAX
    
    Parameters
    ----------
    spatial_domain : Tuple[float, float]
        Spatial domain (x_min, x_max)
    time_domain : Tuple[float, float]
        Time domain (t_min, t_max)
    dx : float
        Spatial grid size
    dt : float
        Time step size
    diffusion_coeff : float or array-like
        Viscosity parameter μ (or array of parameters)
    BC : str, optional
        Boundary condition: 'periodic' or 'dirichlet' (default: 'dirichlet')
    conservation_type : str, optional
        Conservation type: 'NC', 'C', or 'EP' (default: 'NC')
        NC = Non-Conservative, C = Conservative, EP = Energy Preserving
    """
    
    def __init__(
        self,
        spatial_domain: Tuple[float, float],
        time_domain: Tuple[float, float],
        dx: float,
        dt: float,
        diffusion_coeff: float,
        BC: Literal['periodic', 'dirichlet'] = 'dirichlet',
        conservation_type: Literal['NC', 'C', 'EP'] = 'NC'
    ):
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.dx = dx
        self.dt = dt
        self.BC = BC
        self.conservation_type = conservation_type
        
        # Validate inputs
        assert BC in ['periodic', 'dirichlet'], f"Invalid BC: {BC}"
        assert conservation_type in ['NC', 'C', 'EP'], f"Invalid conservation_type: {conservation_type}"
        
        # Create spatial and temporal grids
        if BC == 'periodic':
            self.xspan = jnp.arange(spatial_domain[0], spatial_domain[1], dx)
        else:  # dirichlet
            self.xspan = jnp.arange(spatial_domain[0], spatial_domain[1] + dx/2, dx)
        
        self.tspan = jnp.arange(time_domain[0], time_domain[1] + dt/2, dt)
        
        self.N = len(self.xspan)  # spatial dimension
        self.T = len(self.tspan)  # temporal dimension
        
        # Set diffusion coefficient
        self.mu = float(diffusion_coeff) if jnp.isscalar(diffusion_coeff) else diffusion_coeff
        
        # Build operators
        self._build_operators()
        
    def _build_operators(self):
        """Build the linear (A), quadratic (F), and control (B) operators"""
        if self.BC == 'periodic':
            if self.conservation_type == 'NC':
                self.A, self.F = self._finite_diff_periodic_nonconservative()
                self.B = None
            elif self.conservation_type == 'C':
                self.A, self.F = self._finite_diff_periodic_conservative()
                self.B = None
            elif self.conservation_type == 'EP':
                self.A, self.F = self._finite_diff_periodic_energy_preserving()
                self.B = None
        elif self.BC == 'dirichlet':
            self.A, self.F, self.B = self._finite_diff_dirichlet()
        else:
            raise NotImplementedError(f"BC {self.BC} not implemented")
    
    def _finite_diff_dirichlet(
        self,
        same_on_both_ends: bool = False,
        opposite_sign_on_ends: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate A, F, B matrices for Dirichlet boundary condition (non-conservative)"""
        N = self.N
        dx = self.dx
        dt = self.dt
        mu = self.mu
        
        # Linear operator A (diffusion)
        A = jnp.zeros((N, N))
        diag = -2 * mu / dx**2 * jnp.ones(N)
        off_diag = mu / dx**2 * jnp.ones(N - 1)
        
        A = A.at[jnp.arange(N), jnp.arange(N)].set(diag)
        A = A.at[jnp.arange(N-1), jnp.arange(1, N)].set(off_diag)
        A = A.at[jnp.arange(1, N), jnp.arange(N-1)].set(off_diag)
        
        # Boundary rows
        A = A.at[0, 0].set(-1/dt)
        A = A.at[0, 1].set(0)
        A = A.at[-1, -2].set(0)
        A = A.at[-1, -1].set(-1/dt)
        
        # Quadratic operator F (nonlinear advection)
        S = N * (N + 1) // 2
        F = jnp.zeros((N, S))
        
        if N >= 3:
            for i in range(1, N - 1):
                # Index for u_i * u_{i+1}
                idx_ip = self._get_kron_index(i, i + 1, N)
                # Index for u_i * u_{i-1}
                idx_im = self._get_kron_index(i - 1, i, N)
                
                F = F.at[i, idx_ip].set(-1 / (2 * dx))
                F = F.at[i, idx_im].set(1 / (2 * dx))
        
        # Control operator B
        if same_on_both_ends:
            B = jnp.zeros((N, 1))
            B = B.at[0, 0].set(1 / dt)
            B = B.at[-1, 0].set(1 / dt)
        elif opposite_sign_on_ends:
            B = jnp.zeros((N, 1))
            B = B.at[0, 0].set(1 / dt)
            B = B.at[-1, 0].set(-1 / dt)
        else:
            B = jnp.zeros((N, 2))
            B = B.at[0, 0].set(1 / dt)
            B = B.at[-1, 1].set(1 / dt)
        
        return A, F, B
    
    def _finite_diff_periodic_nonconservative(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate A, F matrices for periodic BC (non-conservative)"""
        N = self.N
        dx = self.dx
        mu = self.mu
        
        # Linear operator A
        A = jnp.zeros((N, N))
        diag = -2 * mu / dx**2 * jnp.ones(N)
        off_diag = mu / dx**2 * jnp.ones(N - 1)
        
        A = A.at[jnp.arange(N), jnp.arange(N)].set(diag)
        A = A.at[jnp.arange(N-1), jnp.arange(1, N)].set(off_diag)
        A = A.at[jnp.arange(1, N), jnp.arange(N-1)].set(off_diag)
        
        # Periodic boundary
        A = A.at[0, -1].set(mu / dx**2)
        A = A.at[-1, 0].set(mu / dx**2)
        
        # Quadratic operator F
        S = N * (N + 1) // 2
        F = jnp.zeros((N, S))
        
        if N >= 3:
            for i in range(1, N - 1):
                idx_ip = self._get_kron_index(i, i + 1, N)
                idx_im = self._get_kron_index(i - 1, i, N)
                F = F.at[i, idx_ip].set(-1 / (2 * dx))
                F = F.at[i, idx_im].set(1 / (2 * dx))
            
            # Periodic boundary conditions
            idx_0N = self._get_kron_index(0, N - 1, N)
            idx_01 = self._get_kron_index(0, 1, N)
            F = F.at[0, idx_01].set(-1 / (2 * dx))
            F = F.at[0, idx_0N].set(1 / (2 * dx))
            
            idx_N0 = self._get_kron_index(0, N - 1, N)
            idx_NN1 = self._get_kron_index(N - 2, N - 1, N)
            F = F.at[-1, idx_N0].set(-1 / (2 * dx))
            F = F.at[-1, idx_NN1].set(1 / (2 * dx))
        
        return A, F
    
    def _finite_diff_periodic_conservative(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate A, F matrices for periodic BC (conservative form)"""
        N = self.N
        dx = self.dx
        mu = self.mu
        
        # Linear operator A (same as non-conservative)
        A = jnp.zeros((N, N))
        diag = -2 * mu / dx**2 * jnp.ones(N)
        off_diag = mu / dx**2 * jnp.ones(N - 1)
        
        A = A.at[jnp.arange(N), jnp.arange(N)].set(diag)
        A = A.at[jnp.arange(N-1), jnp.arange(1, N)].set(off_diag)
        A = A.at[jnp.arange(1, N), jnp.arange(N-1)].set(off_diag)
        A = A.at[0, -1].set(mu / dx**2)
        A = A.at[-1, 0].set(mu / dx**2)
        
        # Quadratic operator F (conservative form uses u_{i+1}^2 - u_{i-1}^2)
        S = N * (N + 1) // 2
        F = jnp.zeros((N, S))
        
        if N >= 3:
            for i in range(1, N - 1):
                # Index for u_{i+1}^2
                idx_ip2 = self._get_kron_index(i + 1, i + 1, N)
                # Index for u_{i-1}^2
                idx_im2 = self._get_kron_index(i - 1, i - 1, N)
                
                F = F.at[i, idx_ip2].set(-1 / (4 * dx))
                F = F.at[i, idx_im2].set(1 / (4 * dx))
            
            # Periodic boundary
            idx_12 = self._get_kron_index(1, 1, N)
            idx_N2 = self._get_kron_index(N - 1, N - 1, N)
            F = F.at[0, idx_12].set(-1 / (4 * dx))
            F = F.at[0, idx_N2].set(1 / (4 * dx))
            
            idx_N12 = self._get_kron_index(N - 2, N - 2, N)
            idx_02 = self._get_kron_index(0, 0, N)
            F = F.at[-1, idx_N12].set(1 / (4 * dx))
            F = F.at[-1, idx_02].set(-1 / (4 * dx))
        
        return A, F
    
    def _finite_diff_periodic_energy_preserving(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate A, F matrices for periodic BC (energy preserving form)"""
        N = self.N
        dx = self.dx
        mu = self.mu
        
        # Linear operator A (same as others)
        A = jnp.zeros((N, N))
        diag = -2 * mu / dx**2 * jnp.ones(N)
        off_diag = mu / dx**2 * jnp.ones(N - 1)
        
        A = A.at[jnp.arange(N), jnp.arange(N)].set(diag)
        A = A.at[jnp.arange(N-1), jnp.arange(1, N)].set(off_diag)
        A = A.at[jnp.arange(1, N), jnp.arange(N-1)].set(off_diag)
        A = A.at[0, -1].set(mu / dx**2)
        A = A.at[-1, 0].set(mu / dx**2)
        
        # Quadratic operator F (energy preserving uses 4 terms)
        S = N * (N + 1) // 2
        F = jnp.zeros((N, S))
        
        if N >= 3:
            for i in range(1, N - 1):
                # u_{i+1}^2
                idx_ip2 = self._get_kron_index(i + 1, i + 1, N)
                # u_{i-1}^2
                idx_im2 = self._get_kron_index(i - 1, i - 1, N)
                # u_i * u_{i+1}
                idx_iip = self._get_kron_index(i, i + 1, N)
                # u_{i-1} * u_i
                idx_imi = self._get_kron_index(i - 1, i, N)
                
                F = F.at[i, idx_ip2].set(-1 / (6 * dx))
                F = F.at[i, idx_im2].set(1 / (6 * dx))
                F = F.at[i, idx_iip].set(-1 / (6 * dx))
                F = F.at[i, idx_imi].set(1 / (6 * dx))
            
            # Periodic boundary for first point
            idx_01 = self._get_kron_index(0, 1, N)
            idx_12 = self._get_kron_index(1, 1, N)
            idx_0N = self._get_kron_index(0, N - 1, N)
            idx_N2 = self._get_kron_index(N - 1, N - 1, N)
            
            F = F.at[0, idx_01].set(-1 / (6 * dx))
            F = F.at[0, idx_12].set(-1 / (6 * dx))
            F = F.at[0, idx_0N].set(1 / (6 * dx))
            F = F.at[0, idx_N2].set(1 / (6 * dx))
            
            # Periodic boundary for last point
            idx_N1N = self._get_kron_index(N - 2, N - 1, N)
            idx_N12 = self._get_kron_index(N - 2, N - 2, N)
            idx_N0 = self._get_kron_index(0, N - 1, N)
            idx_02 = self._get_kron_index(0, 0, N)
            
            F = F.at[-1, idx_N1N].set(1 / (6 * dx))
            F = F.at[-1, idx_N12].set(1 / (6 * dx))
            F = F.at[-1, idx_N0].set(-1 / (6 * dx))
            F = F.at[-1, idx_02].set(-1 / (6 * dx))
        
        return A, F
    
    @staticmethod
    def _get_kron_index(i: int, j: int, N: int) -> int:
        """Get index in unique Kronecker product for u_i * u_j where i <= j"""
        if i > j:
            i, j = j, i
        # Index = N*(N+1)/2 - (N-i)*(N-i+1)/2 + (j-i)
        return N * (N + 1) // 2 - (N - i) * (N - i + 1) // 2 + (j - i)
    
    def update_parameters(self, mu: float):
        """Update diffusion coefficient and rebuild operators"""
        self.mu = float(mu)
        self._build_operators()
    
    def _step(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Single time step using semi-implicit Euler"""
        u2 = _quadratic_map(u)
        # (I - dt*A) u_{n+1} = u_n + dt*F*u2
        lhs = jnp.eye(self.N) - dt * self.A
        rhs = u + dt * (self.F @ u2)
        return jnp.linalg.solve(lhs, rhs)
    
    def _step_with_control(
        self,
        u: jnp.ndarray,
        control: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """Single time step with control input"""
        u2 = _quadratic_map(u)
        lhs = jnp.eye(self.N) - dt * self.A
        rhs = u + dt * (self.F @ u2) + dt * (self.B @ control)
        return jnp.linalg.solve(lhs, rhs)
    
    def solve(
        self,
        u0: jnp.ndarray,
        tspan: Optional[jnp.ndarray] = None,
        control: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Solve the Burgers' equation
        
        Parameters
        ----------
        u0 : array-like, shape (N,)
            Initial condition
        tspan : array-like, optional
            Time points to solve at (default: self.tspan)
        control : array-like, optional
            Control input, shape (num_inputs, T) for Dirichlet BC
            
        Returns
        -------
        u : array, shape (N, T)
            Solution at each time point
        """
        u0 = jnp.asarray(u0)
        assert u0.shape[0] == self.N, f"Initial condition must have shape ({self.N},)"
        
        if tspan is None:
            tspan = self.tspan
        else:
            tspan = jnp.asarray(tspan)
        
        T = len(tspan)
        u = jnp.zeros((self.N, T))
        u = u.at[:, 0].set(u0)
        
        # Check if we have control
        has_control = control is not None and self.B is not None
        if has_control:
            control = jnp.asarray(control)
            assert control.shape[1] == T, f"Control must have {T} time steps"
        
        # Integrate
        for i in range(1, T):
            dt = tspan[i] - tspan[i - 1]
            if has_control:
                u = u.at[:, i].set(self._step_with_control(u[:, i - 1], control[:, i], dt))
            else:
                u = u.at[:, i].set(self._step(u[:, i - 1], dt))
        
        return u
    
    def __repr__(self):
        return (f"BurgersModel(N={self.N}, T={self.T}, dx={self.dx:.4f}, dt={self.dt:.4f}, "
                f"mu={self.mu}, BC='{self.BC}', type='{self.conservation_type}')")


# Convenience functions
def solve_burgers(
    spatial_domain: Tuple[float, float],
    time_domain: Tuple[float, float],
    dx: float,
    dt: float,
    mu: float,
    u0: jnp.ndarray,
    BC: str = 'dirichlet',
    conservation_type: str = 'NC',
    control: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience function to solve Burgers' equation
    
    Returns
    -------
    xspan : array
        Spatial grid points
    tspan : array
        Time points
    u : array
        Solution
    """
    model = BurgersModel(
        spatial_domain=spatial_domain,
        time_domain=time_domain,
        dx=dx,
        dt=dt,
        diffusion_coeff=mu,
        BC=BC,
        conservation_type=conservation_type
    )
    
    u = model.solve(u0, control=control)
    
    return model.xspan, model.tspan, u

#%%
if __name__ == "__main__":
    # Example usage
    print("Viscous Burgers' Equation Solver\n")
    
    # Problem setup
    L = 1.0  # Domain length
    T = 0.5  # Final time
    mu = 0.01  # Viscosity
    
    # Discretization
    N = 100
    dx = L / N
    dt = 0.001
    
    # Initial condition: smooth Gaussian bump
    model = BurgersModel(
        spatial_domain=(0.0, L),
        time_domain=(0.0, T),
        dx=dx,
        dt=dt,
        diffusion_coeff=mu,
        BC='periodic',
        conservation_type='NC'
    )
    
    # Initial condition
    x = model.xspan
    u0 = jnp.exp(-100 * (x - 0.5)**2)
    
    print(f"Model: {model}")
    print(f"Solving with initial condition u0 = exp(-100*(x-0.5)^2)...\n")
    
    # Solve
    u = model.solve(u0)
    u_solution = u
    
    print(f"Solution shape: {u.shape}")
    print(f"Initial energy: {jnp.sum(u[:, 0]**2) * dx:.6f}")
    print(f"Final energy: {jnp.sum(u[:, -1]**2) * dx:.6f}")
    
    #%% Test parameter update
    print("\nTesting parameter update...")
    model.update_parameters(mu=0.02)
    u2 = model.solve(u0)
    print(f"Solution with mu=0.02 computed successfully")

    #%% 
    print("\nComparison of conservation types:")
    for ctype in ['NC', 'C', 'EP']:
        model_test = BurgersModel(
            spatial_domain=(0.0, L),
            time_domain=(0.0, T),
            dx=dx,
            dt=dt,
            diffusion_coeff=mu,
            BC='periodic',
            conservation_type=ctype
        )
        u_test = model_test.solve(u0)
        energy_ratio = jnp.sum(u_test[:, -1]**2) / jnp.sum(u_test[:, 0]**2)
        print(f"  {ctype}: Energy ratio = {energy_ratio:.6f}")
    
    #%% Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot at different time steps
    time_indices = [0, model.T // 4, model.T // 2, -1]
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, color in zip(time_indices, colors):
        time_val = model.tspan[idx]
        axes[0].plot(model.xspan, u_solution[:, idx], label=f't={time_val:.3f}', color=color)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_title('Solution at Different Times')
    axes[0].legend()
    axes[0].grid(True)
    
    # Contour plot
    X, T = jnp.meshgrid(model.xspan, model.tspan)
    contour = axes[1].contourf(X, T, u_solution.T, levels=20, cmap='hot')
    plt.colorbar(contour, ax=axes[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Heat Distribution Over Time')
    
    # Energy decay
    energy = jnp.sum(u_solution**2, axis=0) * model.dx
    axes[2].plot(model.tspan, energy)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Energy Decay')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/burgers_solution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Model setup complete!")
    print(f"Spatial points: {model.N}")
    print(f"Time steps: {model.T}")
    print(f"Final energy: {energy[-1]:.6f}")
# %%
