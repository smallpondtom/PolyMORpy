"""
2D Heat Equation Model using JAX

The 2D heat equation:
∂u/∂t = μ(∂²u/∂x² + ∂²u/∂y²)

with various boundary conditions and integration schemes.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Union, Callable, Optional, Dict, Any
import numpy as np
from scipy.sparse import kron, eye, spdiags, csr_matrix
import scipy.sparse as sp


class Heat2DModel:
    """
    2D Heat Equation Model
    
    Solves: ∂u/∂t = μ(∂²u/∂x² + ∂²u/∂y²)
    
    Parameters
    ----------
    spatial_domain : tuple of tuples
        Spatial domains ((x_min, x_max), (y_min, y_max))
    time_domain : tuple
        Temporal domain (t_min, t_max)
    dx : float
        Spatial grid size in x-direction
    dy : float
        Spatial grid size in y-direction
    dt : float
        Temporal step size
    diffusion_coeffs : float or array
        Diffusion coefficient(s) μ
    BC : tuple of str
        Boundary condition types for (x, y): 'periodic', 'dirichlet'
    """
    
    def __init__(
        self,
        spatial_domain: Tuple[Tuple[float, float], Tuple[float, float]],
        time_domain: Tuple[float, float],
        dx: float,
        dy: float,
        dt: float,
        diffusion_coeffs: Union[float, np.ndarray],
        BC: Tuple[str, str] = ('dirichlet', 'dirichlet')
    ):
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.BC = BC
        
        # Validate boundary conditions
        valid_BCs = ['periodic', 'dirichlet', 'neumann', 'mixed', 'robin']
        for bc in BC:
            if bc not in valid_BCs:
                raise ValueError(f"Invalid boundary condition '{bc}'. Choose from {valid_BCs}")
        
        # Create spatial grids
        if BC[0] == 'periodic':
            self.xspan = jnp.arange(spatial_domain[0][0], spatial_domain[0][1], dx)
        else:
            self.xspan = jnp.arange(spatial_domain[0][0], spatial_domain[0][1] + dx, dx)
        
        if BC[1] == 'periodic':
            self.yspan = jnp.arange(spatial_domain[1][0], spatial_domain[1][1], dy)
        else:
            self.yspan = jnp.arange(spatial_domain[1][0], spatial_domain[1][1] + dy, dy)
        
        # Create temporal grid
        self.tspan = jnp.arange(time_domain[0], time_domain[1] + dt, dt)
        
        # Dimensions
        self.spatial_dim = (len(self.xspan), len(self.yspan))
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
        # Store as 1D vector for compatibility with linear algebra
        self.IC = jnp.zeros(self.spatial_dim[0] * self.spatial_dim[1])
        
        # Cache for operators
        self._cached_operators = None
        self._cached_mu = None
    
    def set_initial_condition(self, IC: Union[np.ndarray, jnp.ndarray]):
        """
        Set the initial condition.
        
        Parameters
        ----------
        IC : array
            Initial condition. Can be 1D (flattened) or 2D array.
            If 2D with shape (Nx, Ny), it will be flattened.
        """
        IC = jnp.array(IC)
        
        # Handle 2D input
        if IC.ndim == 2:
            if IC.shape != self.spatial_dim:
                raise ValueError(f"IC shape {IC.shape} doesn't match spatial dimensions {self.spatial_dim}")
            IC = IC.flatten()
        
        # Check 1D dimensions
        expected_size = self.spatial_dim[0] * self.spatial_dim[1]
        if len(IC) != expected_size:
            raise ValueError(f"IC size {len(IC)} doesn't match expected size {expected_size}")
        
        self.IC = IC
    
    def update_parameter(self, mu: float):
        """
        Update the diffusion coefficient μ.
        
        Parameters
        ----------
        mu : float
            New diffusion coefficient value
        """
        self.current_mu = float(mu)
        if self._cached_mu != mu:
            self._cached_operators = None
    
    def finite_diff_model(
        self, 
        mu: Optional[float] = None
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get finite difference operators for the specified boundary condition.
        
        Parameters
        ----------
        mu : float, optional
            Diffusion coefficient. If None, uses current_mu
        
        Returns
        -------
        A : jnp.ndarray
            System matrix (sparse converted to dense)
        B : jnp.ndarray (optional)
            Input matrix for non-periodic boundary conditions
        """
        if mu is None:
            mu = self.current_mu
        
        # Check cache
        if self._cached_mu == mu and self._cached_operators is not None:
            return self._cached_operators
        
        # Generate operators based on BC
        if all(bc == 'periodic' for bc in self.BC):
            operators = self._finite_diff_periodic(mu)
        elif all(bc == 'dirichlet' for bc in self.BC):
            operators = self._finite_diff_dirichlet(mu)
        else:
            raise NotImplementedError(f"Boundary condition combination {self.BC} not yet implemented")
        
        # Cache operators
        self._cached_mu = mu
        self._cached_operators = operators
        
        return operators
    
    def _finite_diff_periodic(self, mu: float) -> jnp.ndarray:
        """
        Finite difference for periodic boundary conditions in both directions.
        
        Uses Kronecker product to construct 2D Laplacian from 1D operators.
        """
        Nx, Ny = self.spatial_dim
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        
        # 1D periodic second-derivative operators
        # X-direction
        diag_x = -2 * np.ones(Nx)
        off_diag_x = np.ones(Nx - 1)
        Ax = sp.diags([diag_x, off_diag_x, off_diag_x], [0, 1, -1], shape=(Nx, Nx))
        Ax = Ax.tolil()
        Ax[0, Nx-1] = 1.0
        Ax[Nx-1, 0] = 1.0
        Ax = Ax.tocsr() * (mu / dx2)
        
        # Y-direction
        diag_y = -2 * np.ones(Ny)
        off_diag_y = np.ones(Ny - 1)
        Ay = sp.diags([diag_y, off_diag_y, off_diag_y], [0, 1, -1], shape=(Ny, Ny))
        Ay = Ay.tolil()
        Ay[0, Ny-1] = 1.0
        Ay[Ny-1, 0] = 1.0
        Ay = Ay.tocsr() * (mu / dy2)
        
        # 2D Laplacian using Kronecker product
        # A = (Ay ⊗ Ix) + (Iy ⊗ Ax)
        Ix = sp.eye(Nx)
        Iy = sp.eye(Ny)
        
        A = kron(Ay, Ix) + kron(Iy, Ax)
        
        # Convert to dense JAX array
        A = jnp.array(A.toarray())
        
        return A
    
    def _finite_diff_dirichlet(self, mu: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Finite difference for Dirichlet boundary conditions in both directions.
        
        Returns both the system matrix A and control matrix B for boundary inputs.
        """
        Nx, Ny = self.spatial_dim
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        
        # 1D second-derivative operators
        # X-direction
        diag_x = -2 * np.ones(Nx)
        off_diag_x = np.ones(Nx - 1)
        Ax = sp.diags([diag_x, off_diag_x, off_diag_x], [0, 1, -1], shape=(Nx, Nx))
        Ax = Ax.tocsr() * (mu / dx2)
        
        # Y-direction
        diag_y = -2 * np.ones(Ny)
        off_diag_y = np.ones(Ny - 1)
        Ay = sp.diags([diag_y, off_diag_y, off_diag_y], [0, 1, -1], shape=(Ny, Ny))
        Ay = Ay.tocsr() * (mu / dy2)
        
        # 2D Laplacian using Kronecker product
        Ix = sp.eye(Nx)
        Iy = sp.eye(Ny)
        
        A = kron(Ay, Ix) + kron(Iy, Ax)
        
        # B matrix for boundary inputs
        # We have 4 boundaries: left, right, bottom, top
        total_points = Nx * Ny
        Bx = sp.lil_matrix((total_points, 2))  # left and right boundaries
        By = sp.lil_matrix((total_points, 2))  # bottom and top boundaries
        
        # Left boundary (x = 0) for all y
        left_indices = [j * Nx for j in range(Ny)]
        for idx in left_indices:
            Bx[idx, 0] = mu / dx2
        
        # Right boundary (x = Lx) for all y
        right_indices = [j * Nx + (Nx - 1) for j in range(Ny)]
        for idx in right_indices:
            Bx[idx, 1] = mu / dx2
        
        # Bottom boundary (y = 0) for all x
        bottom_indices = list(range(Nx))
        for idx in bottom_indices:
            By[idx, 0] = mu / dy2
        
        # Top boundary (y = Ly) for all x
        top_indices = [(Ny - 1) * Nx + i for i in range(Nx)]
        for idx in top_indices:
            By[idx, 1] = mu / dy2
        
        # Combine B matrices
        B = sp.hstack([Bx, By])
        
        # Convert to dense JAX arrays
        A = jnp.array(A.toarray())
        B = jnp.array(B.toarray())
        
        return A, B
    
    @staticmethod
    @jit
    def _forward_euler_step_no_input(u_prev, dt, A):
        """Single step of Forward Euler integration without input."""
        u_next = u_prev + dt * (A @ u_prev)
        return u_next
    
    @staticmethod
    @jit
    def _forward_euler_step_with_input(u_prev, dt, A, B, input_val):
        """Single step of Forward Euler integration with input."""
        u_next = u_prev + dt * (A @ u_prev + B @ input_val)
        return u_next
    
    @staticmethod
    @jit
    def _backward_euler_step_no_input(u_prev, dt, A):
        """Single step of Backward Euler integration without input."""
        N = len(u_prev)
        I = jnp.eye(N)
        rhs = u_prev
        u_next = jnp.linalg.solve(I - dt * A, rhs)
        return u_next
    
    @staticmethod
    @jit
    def _backward_euler_step_with_input(u_prev, dt, A, B, input_val):
        """Single step of Backward Euler integration with input."""
        N = len(u_prev)
        I = jnp.eye(N)
        rhs = u_prev + dt * B @ input_val
        u_next = jnp.linalg.solve(I - dt * A, rhs)
        return u_next
    
    @staticmethod
    @jit
    def _crank_nicolson_step_no_input(u_prev, dt, A):
        """Single step of Crank-Nicolson integration without input."""
        N = len(u_prev)
        I = jnp.eye(N)
        rhs = (I + 0.5 * dt * A) @ u_prev
        u_next = jnp.linalg.solve(I - 0.5 * dt * A, rhs)
        return u_next
    
    @staticmethod
    @jit
    def _crank_nicolson_step_with_input(u_prev, dt, A, B, input_prev, input_next):
        """Single step of Crank-Nicolson integration with input."""
        N = len(u_prev)
        I = jnp.eye(N)
        rhs = (I + 0.5 * dt * A) @ u_prev + 0.5 * dt * B @ (input_prev + input_next)
        u_next = jnp.linalg.solve(I - 0.5 * dt * A, rhs)
        return u_next
    
    def integrate_model(
        self,
        tdata: jnp.ndarray,
        u0: jnp.ndarray,
        input_data: Optional[jnp.ndarray] = None,
        linear_matrix: Optional[jnp.ndarray] = None,
        control_matrix: Optional[jnp.ndarray] = None,
        system_input: bool = False,
        integrator_type: str = 'BackwardEuler'
    ) -> jnp.ndarray:
        """
        Integrate the 2D heat equation model.
        
        Parameters
        ----------
        tdata : jnp.ndarray
            Time points
        u0 : jnp.ndarray
            Initial condition (1D flattened array)
        input_data : jnp.ndarray, optional
            Boundary input data (input_dim, time_dim)
        linear_matrix : jnp.ndarray, optional
            System matrix A
        control_matrix : jnp.ndarray, optional
            Control matrix B
        system_input : bool
            Whether system has control inputs
        integrator_type : str
            Integration scheme: 'ForwardEuler', 'BackwardEuler', 'CrankNicolson'
        
        Returns
        -------
        u : jnp.ndarray
            Solution array (spatial_dim, time_dim)
        """
        xdim = len(u0)
        tdim = len(tdata)
        
        # Initialize solution array
        u = jnp.zeros((xdim, tdim))
        u = u.at[:, 0].set(u0)
        
        A = linear_matrix
        B = control_matrix if system_input else None
        
        # Validate input dimensions
        if system_input and input_data is not None:
            expected_input_dim = B.shape[1]
            if input_data.shape[0] != expected_input_dim:
                raise ValueError(
                    f"Input dimension {input_data.shape[0]} doesn't match "
                    f"expected {expected_input_dim}"
                )
            if input_data.shape[1] != tdim:
                raise ValueError(
                    f"Input time dimension {input_data.shape[1]} doesn't match "
                    f"time points {tdim}"
                )
        
        # Integration loop
        if integrator_type == 'ForwardEuler':
            for i in range(1, tdim):
                dt = tdata[i] - tdata[i-1]
                if system_input:
                    input_val = input_data[:, i-1]
                    u = u.at[:, i].set(
                        self._forward_euler_step_with_input(u[:, i-1], dt, A, B, input_val)
                    )
                else:
                    u = u.at[:, i].set(
                        self._forward_euler_step_no_input(u[:, i-1], dt, A)
                    )
        
        elif integrator_type == 'BackwardEuler':
            for i in range(1, tdim):
                dt = tdata[i] - tdata[i-1]
                if system_input:
                    input_val = input_data[:, i-1]
                    u = u.at[:, i].set(
                        self._backward_euler_step_with_input(u[:, i-1], dt, A, B, input_val)
                    )
                else:
                    u = u.at[:, i].set(
                        self._backward_euler_step_no_input(u[:, i-1], dt, A)
                    )
        
        elif integrator_type == 'CrankNicolson':
            for i in range(1, tdim):
                dt = tdata[i] - tdata[i-1]
                if system_input:
                    input_prev = input_data[:, i-1]
                    input_next = input_data[:, i]
                    u = u.at[:, i].set(
                        self._crank_nicolson_step_with_input(u[:, i-1], dt, A, B, input_prev, input_next)
                    )
                else:
                    u = u.at[:, i].set(
                        self._crank_nicolson_step_no_input(u[:, i-1], dt, A)
                    )
        
        else:
            raise ValueError(
                f"Unknown integrator type '{integrator_type}'. "
                "Choose from 'ForwardEuler', 'BackwardEuler', 'CrankNicolson'"
            )
        
        return u
    
    def solve(
        self,
        mu: Optional[float] = None,
        initial_condition: Optional[jnp.ndarray] = None,
        boundary_data: Optional[jnp.ndarray] = None,
        integrator: str = 'CrankNicolson'
    ) -> jnp.ndarray:
        """
        Solve the 2D heat equation with current settings.
        
        Parameters
        ----------
        mu : float, optional
            Diffusion coefficient. If None, uses current_mu
        initial_condition : jnp.ndarray, optional
            Initial condition. If None, uses self.IC
        boundary_data : jnp.ndarray, optional
            Boundary condition data. If None, uses zeros
        integrator : str
            Integration scheme
        
        Returns
        -------
        solution : jnp.ndarray
            Solution array (spatial_dim, time_dim)
        """
        # Update parameter if provided
        if mu is not None:
            self.update_parameter(mu)
        
        # Set initial condition
        if initial_condition is not None:
            self.set_initial_condition(initial_condition)
        u0 = self.IC
        
        # Get operators
        operators = self.finite_diff_model()
        
        # Configure based on BC type
        if all(bc == 'periodic' for bc in self.BC):
            A = operators
            B = None
            system_input = False
            input_data = None
        else:
            A, B = operators
            system_input = True
            
            if boundary_data is None:
                # Default: zero boundary conditions
                # 4 boundaries for Dirichlet: left, right, bottom, top
                input_data = jnp.zeros((4, self.time_dim))
            else:
                input_data = boundary_data
        
        # Solve
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
        integrator: str = 'CrankNicolson'
    ) -> Dict[float, jnp.ndarray]:
        """
        Solve the heat equation for multiple parameter values.
        
        Parameters
        ----------
        mu_values : array-like
            List of diffusion coefficient values
        initial_condition : jnp.ndarray, optional
            Initial condition
        boundary_data : jnp.ndarray, optional
            Boundary condition data
        integrator : str
            Integration scheme
        
        Returns
        -------
        solutions : dict
            Dictionary mapping mu values to solutions
        """
        solutions = {}
        
        for mu in mu_values:
            self.update_parameter(float(mu))
            solution = self.solve(
                initial_condition=initial_condition,
                boundary_data=boundary_data,
                integrator=integrator
            )
            solutions[float(mu)] = solution
        
        return solutions
    
    def reshape_solution(self, solution: jnp.ndarray) -> jnp.ndarray:
        """
        Reshape flattened solution back to 2D spatial grid.
        
        Parameters
        ----------
        solution : jnp.ndarray
            Solution array (spatial_dim_flat, time_dim)
        
        Returns
        -------
        reshaped : jnp.ndarray
            Solution array (Nx, Ny, time_dim)
        """
        Nx, Ny = self.spatial_dim
        tdim = solution.shape[1]
        
        # Reshape to (Nx, Ny, time_dim)
        return solution.reshape(Nx, Ny, tdim, order='C')


# Initial condition generators
def create_gaussian_ic_2d(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    center: Tuple[float, float] = (0.5, 0.5), 
    width: float = 0.1
) -> jnp.ndarray:
    """
    Create a 2D Gaussian initial condition.
    
    Parameters
    ----------
    X, Y : jnp.ndarray
        Meshgrid arrays
    center : tuple
        Center of Gaussian (x0, y0)
    width : float
        Width parameter (standard deviation)
    
    Returns
    -------
    ic : jnp.ndarray
        2D initial condition
    """
    x0, y0 = center
    return jnp.exp(-(((X - x0)**2 + (Y - y0)**2) / (2 * width**2)))


def create_sinusoidal_ic_2d(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    n_modes: Tuple[int, int] = (1, 1),
    x_domain: Tuple[float, float] = (0.0, 1.0),
    y_domain: Tuple[float, float] = (0.0, 1.0)
) -> jnp.ndarray:
    """
    Create a 2D sinusoidal initial condition.
    
    Parameters
    ----------
    X, Y : jnp.ndarray
        Meshgrid arrays
    n_modes : tuple
        Number of modes in (x, y) directions
    x_domain, y_domain : tuple
        Domain extents
    
    Returns
    -------
    ic : jnp.ndarray
        2D initial condition
    """
    nx, ny = n_modes
    Lx = x_domain[1] - x_domain[0]
    Ly = y_domain[1] - y_domain[0]
    
    return (jnp.sin(2 * jnp.pi * nx * (X - x_domain[0]) / Lx) * 
            jnp.sin(2 * jnp.pi * ny * (Y - y_domain[0]) / Ly))


def create_step_ic_2d(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    x_range: Tuple[float, float] = (0.3, 0.7),
    y_range: Tuple[float, float] = (0.3, 0.7)
) -> jnp.ndarray:
    """
    Create a 2D step function initial condition.
    
    Parameters
    ----------
    X, Y : jnp.ndarray
        Meshgrid arrays
    x_range, y_range : tuple
        Ranges where the function is 1
    
    Returns
    -------
    ic : jnp.ndarray
        2D initial condition
    """
    in_x_range = (X >= x_range[0]) & (X <= x_range[1])
    in_y_range = (Y >= y_range[0]) & (Y <= y_range[1])
    return jnp.where(in_x_range & in_y_range, 1.0, 0.0)


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    print("Creating 2D Heat Equation Model...")
    model = Heat2DModel(
        spatial_domain=((0.0, 1.0), (0.0, 1.0)),
        time_domain=(0.0, 0.1),
        dx=0.02,
        dy=0.02,
        dt=0.0001,
        diffusion_coeffs=0.01,
        BC=('dirichlet', 'dirichlet')
    )
    
    # Create meshgrid for initial condition
    X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
    
    # Set Gaussian initial condition
    ic_2d = create_gaussian_ic_2d(X, Y, center=(0.5, 0.5), width=0.1)
    model.set_initial_condition(ic_2d)
    
    print(f"Spatial dimensions: {model.spatial_dim}")
    print(f"Time steps: {model.time_dim}")
    print(f"Total grid points: {model.spatial_dim[0] * model.spatial_dim[1]}")
    
    # Solve
    print("\nSolving...")
    solution = model.solve(integrator='CrankNicolson')
    
    # Reshape for visualization
    sol_3d = model.reshape_solution(solution)
    
    # Plot results
    fig = plt.figure(figsize=(15, 4))
    
    time_indices = [0, model.time_dim // 4, model.time_dim // 2, -1]
    for i, idx in enumerate(time_indices):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.plot_surface(X, Y, sol_3d[:, :, idx], cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f't = {model.tspan[idx]:.4f}')
    
    plt.tight_layout()
    
    from pathlib import Path
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'heat2d_test.png', dpi=150, bbox_inches='tight')
    print(f"\nTest plot saved to {output_dir / 'heat2d_test.png'}!")
