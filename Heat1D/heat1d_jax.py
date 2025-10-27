"""
1D Heat Equation Model using JAX

The 1D heat equation:
∂u/∂t = μ ∂²u/∂x²

with various boundary conditions and integration schemes.
"""

#%%
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Tuple, Union, Callable, Optional, Dict, Any
import numpy as np

#%%
class Heat1DModel:
    """
    1D Heat Equation Model
    
    Solves: ∂u/∂t = μ ∂²u/∂x²
    
    Parameters
    ----------
    spatial_domain : tuple
        Spatial domain (x_min, x_max)
    time_domain : tuple
        Temporal domain (t_min, t_max)
    dx : float
        Spatial grid size
    dt : float
        Temporal step size
    diffusion_coeffs : float or array
        Diffusion coefficient(s) μ
    BC : str
        Boundary condition type: 'periodic', 'dirichlet', 'neumann', 'mixed', 'robin'
    """
    
    def __init__(
        self,
        spatial_domain: Tuple[float, float],
        time_domain: Tuple[float, float],
        dx: float,
        dt: float,
        diffusion_coeffs: Union[float, np.ndarray],
        BC: str = 'dirichlet'
    ):
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.dx = dx
        self.dt = dt
        self.BC = BC
        
        # Validate boundary condition
        valid_BCs = ['periodic', 'dirichlet', 'neumann', 'mixed', 'robin']
        if BC not in valid_BCs:
            raise ValueError(f"Invalid boundary condition. Choose from {valid_BCs}")
        
        # Create spatial grid
        if BC == 'periodic':
            self.xspan = jnp.arange(spatial_domain[0], spatial_domain[1], dx)
        else:
            self.xspan = jnp.arange(spatial_domain[0], spatial_domain[1] + dx, dx)
        
        # Create temporal grid
        self.tspan = jnp.arange(time_domain[0], time_domain[1] + dt, dt)
        
        # Dimensions
        self.spatial_dim = len(self.xspan)
        self.time_dim = len(self.tspan)
        
        # Diffusion coefficients
        if isinstance(diffusion_coeffs, (int, float)):
            self.diffusion_coeffs = jnp.array([diffusion_coeffs])
            self.current_mu = float(diffusion_coeffs)
        else:
            self.diffusion_coeffs = jnp.array(diffusion_coeffs)
            self.current_mu = float(diffusion_coeffs[0])
        
        self.param_dim = len(self.diffusion_coeffs)
        self.param_domain = (float(self.diffusion_coeffs.min()), 
                            float(self.diffusion_coeffs.max()))
        
        # Initial condition (zeros by default)
        self.IC = jnp.zeros(self.spatial_dim)
        
        # Cache for operators and settings
        self._cached_operators = None
        self._cached_mu = None
        self._bc_kwargs = {}
    
    def set_initial_condition(self, IC: Union[np.ndarray, jnp.ndarray]):
        """Set the initial condition."""
        self.IC = jnp.array(IC)
        if len(self.IC) != self.spatial_dim:
            raise ValueError(f"IC dimension {len(self.IC)} doesn't match spatial dimension {self.spatial_dim}")
    
    def update_parameter(self, mu: float):
        """
        Update the diffusion coefficient μ.
        
        Parameters
        ----------
        mu : float
            New diffusion coefficient value
        """
        self.current_mu = float(mu)
        # Clear cached operators if mu has changed
        if self._cached_mu != mu:
            self._cached_operators = None
    
    def set_bc_params(self, **kwargs):
        """
        Set boundary condition parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Boundary condition specific parameters:
            - For Dirichlet: same_on_both_ends (bool)
            - For Mixed: order (list)
            - For Robin: alpha (float), beta (float)
        """
        # Check if BC params have changed
        if kwargs != self._bc_kwargs:
            self._cached_operators = None  # Invalidate cache
        self._bc_kwargs = kwargs
    
    def finite_diff_model(self, mu: Optional[float] = None, **kwargs) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get finite difference operators for the specified boundary condition.
        
        Parameters
        ----------
        mu : float, optional
            Diffusion coefficient. If None, uses current_mu
        **kwargs : dict
            Additional parameters for specific boundary conditions
        
        Returns
        -------
        A : jnp.ndarray
            System matrix
        B : jnp.ndarray (optional)
            Input matrix for non-periodic boundary conditions
        """
        if mu is None:
            mu = self.current_mu
        
        # Merge provided kwargs with stored BC parameters
        bc_kwargs = {**self._bc_kwargs, **kwargs}
        
        # Check if we can use cached operators
        # Cache is valid only if mu and BC parameters haven't changed
        if (self._cached_mu == mu and 
            self._cached_operators is not None and 
            not kwargs):  # No new kwargs provided
            return self._cached_operators
        
        # Generate new operators
        if self.BC == 'periodic':
            operators = self._finite_diff_periodic(mu)
        elif self.BC == 'dirichlet':
            operators = self._finite_diff_dirichlet(mu, **bc_kwargs)
        elif self.BC == 'neumann':
            operators = self._finite_diff_neumann(mu)
        elif self.BC == 'mixed':
            operators = self._finite_diff_mixed(mu, **bc_kwargs)
        elif self.BC == 'robin':
            operators = self._finite_diff_robin(mu, **bc_kwargs)
        
        # Cache the operators only if no temporary kwargs were provided
        if not kwargs:
            self._cached_mu = mu
            self._cached_operators = operators
        
        return operators
    
    def _finite_diff_periodic(self, mu: float) -> jnp.ndarray:
        """Finite difference for periodic boundary conditions."""
        N = self.spatial_dim
        dx2 = self.dx ** 2
        
        # Create tridiagonal matrix
        diag = -2 * jnp.ones(N) * mu / dx2
        off_diag = jnp.ones(N - 1) * mu / dx2
        
        A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        
        # Periodic boundary conditions
        A = A.at[0, N-1].set(mu / dx2)
        A = A.at[N-1, 0].set(mu / dx2)
        
        return A
    
    def _finite_diff_dirichlet(self, mu: float, same_on_both_ends: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finite difference for Dirichlet boundary conditions."""
        N = self.spatial_dim
        dx2 = self.dx ** 2
        
        # Create tridiagonal matrix
        diag = -2 * jnp.ones(N) * mu / dx2
        off_diag = jnp.ones(N - 1) * mu / dx2
        
        A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        
        if same_on_both_ends:
            # Same Dirichlet BC at both ends
            B = jnp.zeros(N)
            B = B.at[0].set(mu / dx2)
            B = B.at[-1].set(mu / dx2)
            B = B.reshape(-1, 1)
        else:
            # Different Dirichlet BCs at each end
            B = jnp.zeros((N, 2))
            B = B.at[0, 0].set(mu / dx2)
            B = B.at[-1, 1].set(mu / dx2)
        
        return A, B
    
    def _finite_diff_neumann(self, mu: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finite difference for Neumann boundary conditions."""
        N = self.spatial_dim
        dx2 = self.dx ** 2
        
        # Create tridiagonal matrix
        diag = -2 * jnp.ones(N) * mu / dx2
        off_diag = jnp.ones(N - 1) * mu / dx2
        
        A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        
        # Neumann boundary conditions
        A = A.at[0, 0].set(-mu / dx2)
        A = A.at[-1, -1].set(-mu / dx2)
        
        # Input matrix for boundary conditions
        B = jnp.zeros((N, 2))
        B = B.at[0, 0].set(-mu / self.dx)
        B = B.at[-1, 1].set(mu / self.dx)
        
        return A, B
    
    def _finite_diff_mixed(self, mu: float, order: list = ["neumann", "dirichlet"]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finite difference for mixed boundary conditions."""
        N = self.spatial_dim
        dx2 = self.dx ** 2
        
        # Create tridiagonal matrix
        diag = -2 * jnp.ones(N) * mu / dx2
        off_diag = jnp.ones(N - 1) * mu / dx2
        
        A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        B = jnp.zeros((N, 2))
        
        if order[0] == "neumann":
            # Neumann (left) and Dirichlet (right)
            A = A.at[0, 0].set(-mu / dx2)
            B = B.at[0, 0].set(-mu / self.dx)
            B = B.at[-1, 1].set(mu / dx2)
        else:
            # Dirichlet (left) and Neumann (right)
            A = A.at[-1, -1].set(-mu / dx2)
            B = B.at[0, 0].set(mu / dx2)
            B = B.at[-1, 1].set(mu / self.dx)
        
        return A, B
    
    def _finite_diff_robin(self, mu: float, alpha: float = 0.5, beta: float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Finite difference for Robin boundary conditions.
        Robin BC: α u(0) + (1 - α) u'(0) = g(t)
        """
        N = self.spatial_dim
        dx = self.dx
        dx2 = dx ** 2
        
        # Create tridiagonal matrix
        diag = -2 * jnp.ones(N) * mu / dx2
        off_diag = jnp.ones(N - 1) * mu / dx2
        
        A = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        
        # Robin boundary conditions
        A = A.at[0, 0].add(alpha / (alpha + alpha*dx - dx) * mu / dx2)
        A = A.at[-1, -1].add(beta / (beta - beta*dx + dx) * mu / dx2)
        
        # Input matrix
        B = jnp.zeros((N, 2))
        B = B.at[0, 0].set(-dx / (alpha + alpha*dx - dx) * mu / dx2)
        B = B.at[-1, 1].set(dx / (beta - beta*dx + dx) * mu / dx2)
        
        return A, B
    
    def integrate_model(
        self,
        tdata: jnp.ndarray,
        u0: jnp.ndarray,
        input_data: Optional[jnp.ndarray] = None,
        linear_matrix: Optional[jnp.ndarray] = None,
        control_matrix: Optional[jnp.ndarray] = None,
        system_input: bool = False,
        integrator_type: str = 'ForwardEuler',
        **kwargs
    ) -> jnp.ndarray:
        """
        Integrate the 1D Heat Equation using different schemes.
        
        Parameters
        ----------
        tdata : jnp.ndarray
            Time data points
        u0 : jnp.ndarray
            Initial condition
        input_data : jnp.ndarray, optional
            Input/boundary data
        linear_matrix : jnp.ndarray
            System matrix A
        control_matrix : jnp.ndarray, optional
            Control matrix B
        system_input : bool
            Whether system has input
        integrator_type : str
            Integration scheme: 'ForwardEuler', 'BackwardEuler', 'CrankNicolson'
        
        Returns
        -------
        u : jnp.ndarray
            Solution array of shape (spatial_dim, time_dim)
        """
        if linear_matrix is None:
            raise ValueError("linear_matrix must be provided")
        
        A = linear_matrix
        xdim = len(u0)
        tdim = len(tdata)
        
        # Adjust input dimensions if needed
        if system_input:
            if control_matrix is None:
                raise ValueError("control_matrix must be provided when system_input=True")
            B = control_matrix
            
            if input_data is None:
                # Default to zero input
                input_data = jnp.zeros((B.shape[1], tdim))
            else:
                # Ensure input_data has correct shape
                if input_data.ndim == 1:
                    input_data = input_data.reshape(-1, 1)
                if input_data.shape[0] != B.shape[1]:
                    input_data = input_data.T
                if input_data.shape[1] != tdim:
                    # Pad or truncate as needed
                    if input_data.shape[1] < tdim:
                        input_data = jnp.pad(input_data, ((0, 0), (0, tdim - input_data.shape[1])))
                    else:
                        input_data = input_data[:, :tdim]
        
        # Choose integration method
        if integrator_type == 'ForwardEuler':
            return self._forward_euler(tdata, u0, A, B if system_input else None, 
                                      input_data if system_input else None)
        elif integrator_type == 'BackwardEuler':
            return self._backward_euler(tdata, u0, A, B if system_input else None,
                                       input_data if system_input else None)
        elif integrator_type == 'CrankNicolson':
            return self._crank_nicolson(tdata, u0, A, B if system_input else None,
                                       input_data if system_input else None)
        else:
            raise ValueError(f"Unknown integrator type: {integrator_type}")
    
    @partial(jit, static_argnums=(0,))
    def _forward_euler(
        self,
        tdata: jnp.ndarray,
        u0: jnp.ndarray,
        A: jnp.ndarray,
        B: Optional[jnp.ndarray] = None,
        input_data: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Forward Euler integration scheme."""
        xdim = len(u0)
        tdim = len(tdata)
        I = jnp.eye(xdim)
        
        def step_no_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            u_next = (I + dt * A) @ u_prev
            return u_next, u_next
        
        def step_with_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            u_next = (I + dt * A) @ u_prev + dt * B @ input_data[:, t_idx-1]
            return u_next, u_next
        
        if B is None:
            _, u = jax.lax.scan(step_no_input, u0, jnp.arange(1, tdim))
        else:
            _, u = jax.lax.scan(step_with_input, u0, jnp.arange(1, tdim))
        
        # Combine initial condition with solution
        return jnp.concatenate([u0.reshape(-1, 1), u.T], axis=1)
    
    @partial(jit, static_argnums=(0,))
    def _backward_euler(
        self,
        tdata: jnp.ndarray,
        u0: jnp.ndarray,
        A: jnp.ndarray,
        B: Optional[jnp.ndarray] = None,
        input_data: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Backward Euler integration scheme."""
        xdim = len(u0)
        tdim = len(tdata)
        I = jnp.eye(xdim)
        
        def step_no_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            u_next = jnp.linalg.solve(I - dt * A, u_prev)
            return u_next, u_next
        
        def step_with_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            rhs = u_prev + dt * B @ input_data[:, t_idx-1]
            u_next = jnp.linalg.solve(I - dt * A, rhs)
            return u_next, u_next
        
        if B is None:
            _, u = jax.lax.scan(step_no_input, u0, jnp.arange(1, tdim))
        else:
            _, u = jax.lax.scan(step_with_input, u0, jnp.arange(1, tdim))
        
        # Combine initial condition with solution
        return jnp.concatenate([u0.reshape(-1, 1), u.T], axis=1)
    
    @partial(jit, static_argnums=(0,))
    def _crank_nicolson(
        self,
        tdata: jnp.ndarray,
        u0: jnp.ndarray,
        A: jnp.ndarray,
        B: Optional[jnp.ndarray] = None,
        input_data: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Crank-Nicolson integration scheme."""
        xdim = len(u0)
        tdim = len(tdata)
        I = jnp.eye(xdim)
        
        def step_no_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            rhs = (I + 0.5 * dt * A) @ u_prev
            u_next = jnp.linalg.solve(I - 0.5 * dt * A, rhs)
            return u_next, u_next
        
        def step_with_input(u_prev, t_idx):
            dt = tdata[t_idx] - tdata[t_idx-1]
            rhs = ((I + 0.5 * dt * A) @ u_prev + 
                   0.5 * dt * B @ (input_data[:, t_idx-1] + input_data[:, t_idx]))
            u_next = jnp.linalg.solve(I - 0.5 * dt * A, rhs)
            return u_next, u_next
        
        if B is None:
            _, u = jax.lax.scan(step_no_input, u0, jnp.arange(1, tdim))
        else:
            # Ensure we have input data for all time steps
            if input_data.shape[1] < tdim:
                input_data = jnp.pad(input_data, ((0, 0), (0, tdim - input_data.shape[1])))
            _, u = jax.lax.scan(step_with_input, u0, jnp.arange(1, tdim))
        
        # Combine initial condition with solution
        return jnp.concatenate([u0.reshape(-1, 1), u.T], axis=1)
    
    def solve(
        self,
        mu: Optional[float] = None,
        initial_condition: Optional[jnp.ndarray] = None,
        boundary_data: Optional[jnp.ndarray] = None,
        integrator: str = 'CrankNicolson',
        **bc_kwargs
    ) -> jnp.ndarray:
        """
        Convenient method to solve the heat equation.
        
        Parameters
        ----------
        mu : float, optional
            Diffusion coefficient. If None, uses current_mu
        initial_condition : jnp.ndarray, optional
            Initial condition. If None, uses self.IC
        boundary_data : jnp.ndarray, optional
            Boundary condition data. If None, uses zeros
        integrator : str
            Integration scheme: 'ForwardEuler', 'BackwardEuler', 'CrankNicolson'
        **bc_kwargs : dict
            Boundary condition specific parameters
        
        Returns
        -------
        solution : jnp.ndarray
            Solution array of shape (spatial_dim, time_dim)
        
        Examples
        --------
        >>> model = Heat1DModel(...)
        >>> solution1 = model.solve(mu=0.01)
        >>> model.update_parameter(0.02)
        >>> solution2 = model.solve()  # Uses new mu=0.02
        """
        # Update parameters if provided
        if mu is not None:
            self.update_parameter(mu)
        
        # Use provided IC or default
        if initial_condition is not None:
            self.set_initial_condition(initial_condition)
        u0 = self.IC
        
        # Merge BC kwargs with any stored ones
        if bc_kwargs:
            self.set_bc_params(**bc_kwargs)
        
        # Get operators (will use cached if available)
        operators = self.finite_diff_model()
        
        # Determine if we have system input
        if self.BC == 'periodic':
            A = operators
            B = None
            system_input = False
            input_data = None
        else:
            A, B = operators
            system_input = True
            
            # Use provided boundary data or generate default
            if boundary_data is None:
                # Default boundary data based on BC type
                if self.BC == 'dirichlet':
                    if self._bc_kwargs.get('same_on_both_ends', True):
                        input_data = jnp.zeros((1, self.time_dim))
                    else:
                        input_data = jnp.zeros((2, self.time_dim))
                else:
                    # Neumann, mixed, robin all use 2 inputs
                    input_data = jnp.zeros((2, self.time_dim))
            else:
                input_data = boundary_data
        
        # Solve the system
        return self.integrate_model(
            tdata=self.tspan,
            u0=u0,
            input_data=input_data,
            linear_matrix=A,
            control_matrix=B,
            system_input=system_input,
            integrator_type=integrator
        )
    
    def solve_parameter_sweep(
        self,
        mu_values: Union[list, np.ndarray, jnp.ndarray],
        initial_condition: Optional[jnp.ndarray] = None,
        boundary_data: Optional[jnp.ndarray] = None,
        integrator: str = 'CrankNicolson',
        **bc_kwargs
    ) -> Dict[float, jnp.ndarray]:
        """
        Solve the heat equation for multiple parameter values.
        
        Parameters
        ----------
        mu_values : array-like
            List of diffusion coefficient values to solve for
        initial_condition : jnp.ndarray, optional
            Initial condition. If None, uses self.IC
        boundary_data : jnp.ndarray, optional
            Boundary condition data. If None, uses zeros
        integrator : str
            Integration scheme
        **bc_kwargs : dict
            Boundary condition specific parameters
        
        Returns
        -------
        solutions : dict
            Dictionary mapping mu values to solutions
        
        Examples
        --------
        >>> model = Heat1DModel(...)
        >>> mu_values = [0.005, 0.01, 0.02]
        >>> solutions = model.solve_parameter_sweep(mu_values)
        >>> for mu, solution in solutions.items():
        ...     print(f"Solution for mu={mu}: max={jnp.max(solution):.4f}")
        """
        solutions = {}
        
        for mu in mu_values:
            self.update_parameter(float(mu))
            solution = self.solve(
                initial_condition=initial_condition,
                boundary_data=boundary_data,
                integrator=integrator,
                **bc_kwargs
            )
            solutions[float(mu)] = solution
        
        return solutions


def create_gaussian_ic(x: jnp.ndarray, center: float = 0.5, width: float = 0.1) -> jnp.ndarray:
    """Create a Gaussian initial condition."""
    return jnp.exp(-((x - center) ** 2) / (2 * width ** 2))


def create_step_ic(x: jnp.ndarray, left: float = 0.4, right: float = 0.6) -> jnp.ndarray:
    """Create a step function initial condition."""
    return jnp.where((x >= left) & (x <= right), 1.0, 0.0)


def create_sinusoidal_ic(x: jnp.ndarray, n_modes: int = 1) -> jnp.ndarray:
    """Create a sinusoidal initial condition."""
    L = x[-1] - x[0]
    return jnp.sin(2 * jnp.pi * n_modes * (x - x[0]) / L)

#%%
if __name__ == "__main__":
    # Example usage demonstrating parameter updates
    import matplotlib.pyplot as plt
    
    # Set up the model
    model = Heat1DModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 1.0),
        dx=0.01,
        dt=0.0001,
        diffusion_coeffs=0.01,  # Initial diffusion coefficient
        BC='dirichlet'
    )
    
    # Set initial condition
    ic = create_gaussian_ic(model.xspan, center=0.5, width=0.05)
    model.set_initial_condition(ic)
    
    # Method 1: Using the convenient solve method
    print("Method 1: Using solve() method")
    solution1 = model.solve(mu=0.01)
    print(f"  Solution with μ=0.01: max value = {jnp.max(solution1[:, -1]):.6f}")
    
    # Update parameter and solve again
    model.update_parameter(0.02)
    solution2 = model.solve()  # Uses the new μ=0.02
    print(f"  Solution with μ=0.02: max value = {jnp.max(solution2[:, -1]):.6f}")
    
    # Method 2: Solve for multiple parameters at once
    print("\nMethod 2: Parameter sweep")
    mu_values = [0.005, 0.01, 0.02, 0.04]
    solutions = model.solve_parameter_sweep(mu_values)
    
    for mu, sol in solutions.items():
        print(f"  μ={mu:.3f}: final max = {jnp.max(sol[:, -1]):.6f}")

    #%% Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot at different time steps
    time_indices = [0, model.time_dim // 4, model.time_dim // 2, -1]
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, color in zip(time_indices, colors):
        time_val = model.tspan[idx]
        axes[0].plot(model.xspan, solution1[:, idx], label=f't={time_val:.3f}', color=color)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_title('Solution at Different Times')
    axes[0].legend()
    axes[0].grid(True)
    
    # Contour plot
    X, T = jnp.meshgrid(model.xspan, model.tspan)
    contour = axes[1].contourf(X, T, solution1.T, levels=20, cmap='hot')
    plt.colorbar(contour, ax=axes[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Heat Distribution Over Time')
    
    # Energy decay
    energy = jnp.sum(solution1**2, axis=0) * model.dx
    axes[2].plot(model.tspan, energy)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Energy Decay')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_solution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Model setup complete!")
    print(f"Spatial points: {model.spatial_dim}")
    print(f"Time steps: {model.time_dim}")
    print(f"Final energy: {energy[-1]:.6f}")
    
    #%% Visualize the effect of different μ values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Solutions at final time for different μ
    for mu, sol in solutions.items():
        axes[0].plot(model.xspan, sol[:, -1], label=f'μ={mu:.3f}', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t_final)')
    axes[0].set_title(f'Solutions at t={model.time_domain[1]} for Different μ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Maximum value decay over time
    for mu, sol in solutions.items():
        max_vals = jnp.max(sol, axis=0)
        axes[1].plot(model.tspan, max_vals, label=f'μ={mu:.3f}', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Max Temperature')
    axes[1].set_title('Maximum Value Decay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Energy decay
    for mu, sol in solutions.items():
        energy = jnp.sum(sol**2, axis=0) * model.dx
        axes[2].plot(model.tspan, energy, label=f'μ={mu:.3f}', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Energy Decay')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_parameter_update.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nModel setup complete!")
    print(f"Spatial points: {model.spatial_dim}")
    print(f"Time steps: {model.time_dim}")
    print(f"Parameter update functionality: ✓") 
# %%
