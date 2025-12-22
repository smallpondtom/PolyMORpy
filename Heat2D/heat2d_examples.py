"""
Extended examples and validation tests for Heat2D JAX implementation.
Demonstrates boundary conditions, integration schemes, and parameter sweeps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from heat2d_jax import (
    Heat2DModel, 
    create_gaussian_ic_2d, 
    create_sinusoidal_ic_2d, 
    create_step_ic_2d
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def analytical_solution_2d_dirichlet(X, Y, t, mu, n_terms=10):
    """
    Analytical solution for 2D heat equation with zero Dirichlet BCs.
    
    Initial condition: u(x,y,0) = sin(πx)sin(πy) on [0,1]×[0,1]
    
    Parameters
    ----------
    X, Y : jnp.ndarray
        Meshgrid arrays
    t : float
        Time
    mu : float
        Diffusion coefficient
    n_terms : int
        Number of terms in Fourier series
    
    Returns
    -------
    u : jnp.ndarray
        Analytical solution
    """
    Lx = Ly = 1.0
    solution = jnp.zeros_like(X)
    
    for n in range(1, n_terms + 1):
        for m in range(1, n_terms + 1):
            # Coefficient for sin(nπx)sin(mπy) initial condition
            # Only (1,1) mode is non-zero for our IC
            if n == 1 and m == 1:
                Anm = 1.0
            else:
                Anm = 0.0
            
            lambda_nm = mu * jnp.pi**2 * ((n/Lx)**2 + (m/Ly)**2)
            solution += (Anm * 
                        jnp.sin(n * jnp.pi * X / Lx) * 
                        jnp.sin(m * jnp.pi * Y / Ly) * 
                        jnp.exp(-lambda_nm * t))
    
    return solution


def example_1_boundary_conditions():
    """
    Example 1: Demonstrate different boundary conditions.
    """
    print("=" * 70)
    print("Example 1: Different Boundary Conditions")
    print("=" * 70)
    
    # Common parameters
    spatial_domain = ((0.0, 1.0), (0.0, 1.0))
    time_domain = (0.0, 0.1)
    dx = dy = 0.02
    dt = 0.0001
    mu = 0.01
    
    # Test both periodic and Dirichlet BCs
    bc_types = [
        (('periodic', 'periodic'), 'Periodic (both)'),
        (('dirichlet', 'dirichlet'), 'Dirichlet (both)')
    ]
    
    fig = plt.figure(figsize=(15, 6))
    
    for idx, (bc, bc_name) in enumerate(bc_types):
        print(f"\n{bc_name}:")
        
        # Create model
        model = Heat2DModel(
            spatial_domain=spatial_domain,
            time_domain=time_domain,
            dx=dx,
            dy=dy,
            dt=dt,
            diffusion_coeffs=mu,
            BC=bc
        )
        
        # Create meshgrid
        X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
        
        # Set initial condition
        if all(b == 'periodic' for b in bc):
            # Sinusoidal IC for periodic BC
            ic_2d = create_sinusoidal_ic_2d(
                X, Y, 
                n_modes=(2, 2),
                x_domain=spatial_domain[0],
                y_domain=spatial_domain[1]
            )
        else:
            # Gaussian IC for Dirichlet BC
            ic_2d = create_gaussian_ic_2d(X, Y, center=(0.5, 0.5), width=0.15)
        
        model.set_initial_condition(ic_2d)
        
        print(f"  Grid: {model.spatial_dim[0]} × {model.spatial_dim[1]}")
        print(f"  Time steps: {model.time_dim}")
        
        # Solve
        solution = model.solve(integrator='CrankNicolson')
        sol_3d = model.reshape_solution(solution)
        
        # Plot initial and final states
        for j, (time_idx, time_label) in enumerate([(0, 'Initial'), (-1, 'Final')]):
            ax = fig.add_subplot(2, 4, idx*4 + j*2 + 1, projection='3d')
            surf = ax.plot_surface(X, Y, sol_3d[:, :, time_idx], cmap='hot', 
                                  vmin=0, vmax=1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f'{bc_name}\n{time_label} (t={model.tspan[time_idx]:.4f})')
            ax.set_zlim(0, 1)
            
            # Contour plot
            ax2 = fig.add_subplot(2, 4, idx*4 + j*2 + 2)
            contour = ax2.contourf(X, Y, sol_3d[:, :, time_idx], levels=20, cmap='hot',
                                   vmin=0, vmax=1)
            plt.colorbar(contour, ax=ax2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title(f'{time_label} (contour)')
            ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_boundary_conditions.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved: heat2d_boundary_conditions.png")


def example_2_integration_schemes():
    """
    Example 2: Compare different integration schemes.
    """
    print("\n" + "=" * 70)
    print("Example 2: Integration Scheme Comparison")
    print("=" * 70)
    
    # Setup model
    model = Heat2DModel(
        spatial_domain=((0.0, 1.0), (0.0, 1.0)),
        time_domain=(0.0, 0.05),
        dx=0.02,
        dy=0.02,
        dt=0.0001,
        diffusion_coeffs=0.01,
        BC=('dirichlet', 'dirichlet')
    )
    
    X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
    ic_2d = create_gaussian_ic_2d(X, Y, center=(0.5, 0.5), width=0.1)
    model.set_initial_condition(ic_2d)
    
    integrators = ['ForwardEuler', 'BackwardEuler', 'CrankNicolson']
    solutions = {}
    
    fig = plt.figure(figsize=(18, 10))
    
    for idx, integrator in enumerate(integrators):
        print(f"\n{integrator}:")
        print(f"  Solving...")
        
        solution = model.solve(integrator=integrator)
        solutions[integrator] = solution
        sol_3d = model.reshape_solution(solution)
        
        # Plot at multiple time points
        time_indices = [0, model.time_dim // 3, 2 * model.time_dim // 3, -1]
        
        for j, time_idx in enumerate(time_indices):
            ax = fig.add_subplot(len(integrators), 4, idx*4 + j + 1, projection='3d')
            surf = ax.plot_surface(X, Y, sol_3d[:, :, time_idx], cmap='hot')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f'{integrator}\nt={model.tspan[time_idx]:.4f}')
            ax.view_init(elev=30, azim=45)
        
        # Compute and print metrics
        final_max = jnp.max(sol_3d[:, :, -1])
        final_energy = jnp.sum(sol_3d[:, :, -1]**2) * model.dx * model.dy
        print(f"  Final max value: {final_max:.6f}")
        print(f"  Final energy: {final_energy:.6f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_integrators.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved: heat2d_integrators.png")
    
    # Energy decay comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    for integrator, solution in solutions.items():
        sol_3d = model.reshape_solution(solution)
        energy = jnp.array([jnp.sum(sol_3d[:, :, i]**2) * model.dx * model.dy 
                            for i in range(model.time_dim)])
        ax.plot(model.tspan, energy, label=integrator, linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Decay Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_energy_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: heat2d_energy_comparison.png")


def example_3_parameter_sweep():
    """
    Example 3: Parameter sweep with different diffusion coefficients.
    """
    print("\n" + "=" * 70)
    print("Example 3: Parameter Sweep")
    print("=" * 70)
    
    # Setup model
    model = Heat2DModel(
        spatial_domain=((0.0, 1.0), (0.0, 1.0)),
        time_domain=(0.0, 0.1),
        dx=0.02,
        dy=0.02,
        dt=0.0001,
        diffusion_coeffs=0.01,
        BC=('dirichlet', 'dirichlet')
    )
    
    X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
    ic_2d = create_gaussian_ic_2d(X, Y, center=(0.5, 0.5), width=0.1)
    model.set_initial_condition(ic_2d)
    
    # Parameter sweep
    mu_values = [0.01, 0.05, 0.1, 0.5]
    print(f"\nTesting μ values: {mu_values}")
    
    solutions = model.solve_parameter_sweep(mu_values, integrator='CrankNicolson')
    
    # Plot results
    fig = plt.figure(figsize=(16, 10))
    
    for idx, (mu, solution) in enumerate(solutions.items()):
        sol_3d = model.reshape_solution(solution)
        
        # 3D surface plot
        ax1 = fig.add_subplot(3, 4, idx + 1, projection='3d')
        surf = ax1.plot_surface(X, Y, sol_3d[:, :, -1], cmap='hot')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title(f'μ = {mu:.3f} (Final)')
        ax1.view_init(elev=30, azim=45)
        
        # Contour plot
        ax2 = fig.add_subplot(3, 4, idx + 5)
        contour = ax2.contourf(X, Y, sol_3d[:, :, -1], levels=20, cmap='hot')
        plt.colorbar(contour, ax=ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'μ = {mu:.3f} (Contour)')
        ax2.set_aspect('equal')
        
        # Center line profile
        ax3 = fig.add_subplot(3, 4, idx + 9)
        center_y = model.spatial_dim[1] // 2
        ax3.plot(model.xspan, sol_3d[:, center_y, -1], linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x, y=0.5)')
        ax3.set_title(f'μ = {mu:.3f} (Center Line)')
        ax3.grid(True, alpha=0.3)
        
        # Print statistics
        final_max = jnp.max(sol_3d[:, :, -1])
        final_energy = jnp.sum(sol_3d[:, :, -1]**2) * model.dx * model.dy
        print(f"μ = {mu:.3f}: max = {final_max:.6f}, energy = {final_energy:.6f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_parameter_sweep.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved: heat2d_parameter_sweep.png")
    
    # Energy decay for all parameters
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    for mu, solution in solutions.items():
        sol_3d = model.reshape_solution(solution)
        energy = jnp.array([jnp.sum(sol_3d[:, :, i]**2) * model.dx * model.dy 
                            for i in range(model.time_dim)])
        ax.plot(model.tspan, energy, label=f'μ = {mu:.3f}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Decay for Different Diffusion Coefficients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_parameter_energy.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: heat2d_parameter_energy.png")


def example_4_analytical_validation():
    """
    Example 4: Validate against analytical solution.
    """
    print("\n" + "=" * 70)
    print("Example 4: Analytical Validation")
    print("=" * 70)
    
    # Setup model
    model = Heat2DModel(
        spatial_domain=((0.0, 1.0), (0.0, 1.0)),
        time_domain=(0.0, 0.05),
        dx=0.02,
        dy=0.02,
        dt=0.0001,
        diffusion_coeffs=0.1,
        BC=('dirichlet', 'dirichlet')
    )
    
    X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
    
    # Use sinusoidal IC that matches analytical solution
    ic_2d = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    model.set_initial_condition(ic_2d)
    
    print(f"\nInitial condition: sin(πx)sin(πy)")
    print(f"Grid: {model.spatial_dim[0]} × {model.spatial_dim[1]}")
    
    # Solve numerically
    print("Solving numerically...")
    solution = model.solve(integrator='CrankNicolson')
    sol_3d = model.reshape_solution(solution)
    
    # Compute analytical solutions at select times
    time_indices = [0, model.time_dim // 4, model.time_dim // 2, -1]
    
    fig = plt.figure(figsize=(18, 12))
    
    max_errors = []
    l2_errors = []
    
    for plot_idx, time_idx in enumerate(time_indices):
        t = model.tspan[time_idx]
        
        # Analytical solution
        u_analytical = analytical_solution_2d_dirichlet(X, Y, t, model.current_mu, n_terms=20)
        u_numerical = sol_3d[:, :, time_idx]
        
        # Error
        error = jnp.abs(u_numerical - u_analytical)
        max_error = jnp.max(error)
        l2_error = jnp.sqrt(jnp.sum(error**2) * model.dx * model.dy)
        
        max_errors.append(max_error)
        l2_errors.append(l2_error)
        
        print(f"\nt = {t:.4f}:")
        print(f"  Max error: {max_error:.6e}")
        print(f"  L2 error: {l2_error:.6e}")
        
        # Numerical solution
        ax1 = fig.add_subplot(4, 4, plot_idx*4 + 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_numerical, cmap='hot')
        ax1.set_title(f'Numerical\nt={t:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        
        # Analytical solution
        ax2 = fig.add_subplot(4, 4, plot_idx*4 + 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u_analytical, cmap='hot')
        ax2.set_title(f'Analytical\nt={t:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        
        # Error plot
        ax3 = fig.add_subplot(4, 4, plot_idx*4 + 3)
        contour3 = ax3.contourf(X, Y, error, levels=20, cmap='viridis')
        plt.colorbar(contour3, ax=ax3)
        ax3.set_title(f'Error\nmax={max_error:.2e}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # Center line comparison
        ax4 = fig.add_subplot(4, 4, plot_idx*4 + 4)
        center_y = model.spatial_dim[1] // 2
        ax4.plot(model.xspan, u_numerical[:, center_y], 'b-', 
                label='Numerical', linewidth=2)
        ax4.plot(model.xspan, u_analytical[:, center_y], 'r--', 
                label='Analytical', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('u(x, y=0.5)')
        ax4.set_title(f'Center Line\nt={t:.4f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_analytical_validation.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved: heat2d_analytical_validation.png")
    
    # Error evolution plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    time_points = [model.tspan[i] for i in time_indices]
    
    ax1.plot(time_points, max_errors, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Max Error')
    ax1.set_title('Maximum Error vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(time_points, l2_errors, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('L2 Error')
    ax2.set_title('L2 Error vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_error_evolution.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: heat2d_error_evolution.png")


def example_5_different_initial_conditions():
    """
    Example 5: Test different initial conditions.
    """
    print("\n" + "=" * 70)
    print("Example 5: Different Initial Conditions")
    print("=" * 70)
    
    # Setup model
    model = Heat2DModel(
        spatial_domain=((0.0, 1.0), (0.0, 1.0)),
        time_domain=(0.0, 0.1),
        dx=0.02,
        dy=0.02,
        dt=0.0001,
        diffusion_coeffs=0.2,
        BC=('dirichlet', 'dirichlet')
    )
    
    X, Y = jnp.meshgrid(model.xspan, model.yspan, indexing='ij')
    
    # Different initial conditions
    ics = {
        'Single Gaussian': create_gaussian_ic_2d(X, Y, center=(0.5, 0.5), width=0.1),
        'Off-center Gaussian': create_gaussian_ic_2d(X, Y, center=(0.3, 0.7), width=0.08),
        'Step Function': create_step_ic_2d(X, Y, x_range=(0.3, 0.7), y_range=(0.3, 0.7)),
        'Sinusoidal': create_sinusoidal_ic_2d(X, Y, n_modes=(2, 2))
    }
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, (ic_name, ic_2d) in enumerate(ics.items()):
        print(f"\n{ic_name}:")
        
        model.set_initial_condition(ic_2d)
        solution = model.solve(integrator='CrankNicolson')
        sol_3d = model.reshape_solution(solution)
        
        # Initial condition
        ax1 = fig.add_subplot(4, 4, idx*4 + 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, sol_3d[:, :, 0], cmap='hot')
        ax1.set_title(f'{ic_name}\nInitial')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        
        # Intermediate time
        mid_idx = model.time_dim // 2
        ax2 = fig.add_subplot(4, 4, idx*4 + 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, sol_3d[:, :, mid_idx], cmap='hot')
        ax2.set_title(f't={model.tspan[mid_idx]:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        
        # Final time
        ax3 = fig.add_subplot(4, 4, idx*4 + 3, projection='3d')
        surf3 = ax3.plot_surface(X, Y, sol_3d[:, :, -1], cmap='hot')
        ax3.set_title(f'Final (t={model.tspan[-1]:.4f})')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('u')
        
        # Energy evolution
        ax4 = fig.add_subplot(4, 4, idx*4 + 4)
        energy = jnp.array([jnp.sum(sol_3d[:, :, i]**2) * model.dx * model.dy 
                            for i in range(model.time_dim)])
        ax4.plot(model.tspan, energy, linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Energy')
        ax4.set_title('Energy Decay')
        ax4.grid(True, alpha=0.3)
        
        # Print statistics
        initial_energy = energy[0]
        final_energy = energy[-1]
        print(f"  Initial energy: {initial_energy:.6f}")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Energy ratio: {final_energy/initial_energy:.6f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heat2d_initial_conditions.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved: heat2d_initial_conditions.png")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D HEAT EQUATION - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_1_boundary_conditions()
    example_2_integration_schemes()
    example_3_parameter_sweep()
    example_4_analytical_validation()
    example_5_different_initial_conditions()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  1. heat2d_boundary_conditions.png")
    print("  2. heat2d_integrators.png")
    print("  3. heat2d_energy_comparison.png")
    print("  4. heat2d_parameter_sweep.png")
    print("  5. heat2d_parameter_energy.png")
    print("  6. heat2d_analytical_validation.png")
    print("  7. heat2d_error_evolution.png")
    print("  8. heat2d_initial_conditions.png")
