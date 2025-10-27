"""
Extended examples and validation tests for Heat1D JAX implementation.
Demonstrates all boundary conditions and compares with analytical solutions.
"""

#%%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from heat1d_jax import Heat1DModel, create_gaussian_ic, create_sinusoidal_ic, create_step_ic

#%%
def analytical_solution_dirichlet_zero(x, t, mu, n_terms=50):
    """
    Analytical solution for heat equation with zero Dirichlet BCs.
    Initial condition: u(x,0) = sin(πx) on [0,1]
    """
    L = 1.0
    solution = jnp.zeros_like(x)
    
    for n in range(1, n_terms + 1):
        An = 2.0 if n == 1 else 0.0  # Only first mode for sin(πx) IC
        solution += An * jnp.sin(n * jnp.pi * x / L) * jnp.exp(-(n * jnp.pi / L)**2 * mu * t)
    
    return solution


def test_all_boundary_conditions():
    """Test implementation with all boundary condition types."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Common parameters
    spatial_domain = (0.0, 1.0)
    time_domain = (0.0, 0.2)
    dx = 0.01
    dt = 0.0001
    mu = 0.02
    
    # Test cases for different boundary conditions
    test_cases = [
        ('periodic', {}),
        ('dirichlet', {'same_on_both_ends': True}),
        ('dirichlet', {'same_on_both_ends': False}),
        ('neumann', {}),
        ('mixed', {'order': ['neumann', 'dirichlet']}),
        ('robin', {'alpha': 0.5, 'beta': 0.5})
    ]
    
    for idx, (bc_type, bc_kwargs) in enumerate(test_cases):
        # Create model
        model = Heat1DModel(
            spatial_domain=spatial_domain,
            time_domain=time_domain,
            dx=dx,
            dt=dt,
            diffusion_coeffs=mu,
            BC=bc_type
        )
        
        # Set initial condition
        if bc_type == 'periodic':
            # Use sinusoidal IC for periodic BC
            ic = create_sinusoidal_ic(model.xspan, n_modes=2)
        else:
            # Use Gaussian IC for other BCs
            ic = create_gaussian_ic(model.xspan, center=0.5, width=0.1)
        
        model.set_initial_condition(ic)
        
        # Get finite difference operators
        result = model.finite_diff_model(mu, **bc_kwargs)
        
        # Handle different return types
        if bc_type == 'periodic':
            A = result
            B = None
            system_input = False
            input_data = None
        else:
            A, B = result
            system_input = True
            
            # Set boundary conditions based on type
            if bc_type == 'dirichlet':
                if bc_kwargs.get('same_on_both_ends', True):
                    # Same temperature at both ends
                    input_data = jnp.ones((1, model.time_dim)) * 0.0
                else:
                    # Different temperatures at each end
                    input_data = jnp.array([
                        jnp.ones(model.time_dim) * 0.0,  # Left boundary
                        jnp.ones(model.time_dim) * 0.5   # Right boundary
                    ])
            elif bc_type == 'neumann':
                # Zero flux at boundaries
                input_data = jnp.zeros((2, model.time_dim))
            elif bc_type == 'mixed':
                # Mixed: zero flux (left) and zero temperature (right)
                input_data = jnp.zeros((2, model.time_dim))
            elif bc_type == 'robin':
                # Robin: combination of value and flux
                input_data = jnp.ones((2, model.time_dim)) * 0.1
        
        # Integrate the model
        solution = model.integrate_model(
            tdata=model.tspan,
            u0=model.IC,
            input_data=input_data,
            linear_matrix=A,
            control_matrix=B,
            system_input=system_input,
            integrator_type='CrankNicolson'
        )
        
        # Plot results
        ax = axes[idx]
        
        # Plot at different time steps
        time_indices = [0, model.time_dim // 4, model.time_dim // 2, -1]
        colors = ['blue', 'green', 'orange', 'red']
        
        for t_idx, color in zip(time_indices, colors):
            time_val = model.tspan[t_idx]
            ax.plot(model.xspan, solution[:, t_idx], 
                   label=f't={time_val:.2f}', color=color, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        
        # Format title based on BC type
        if bc_type == 'dirichlet' and not bc_kwargs.get('same_on_both_ends', True):
            title = 'Dirichlet (different ends)'
        elif bc_type == 'mixed':
            order = bc_kwargs.get('order', ['neumann', 'dirichlet'])
            title = f'Mixed ({order[0]}/{order[1]})'
        else:
            title = bc_type.capitalize()
        
        ax.set_title(f'{title} BC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_all_boundary_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_integrator_comparison():
    """Compare different integration schemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Set up model
    model = Heat1DModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.5),
        dx=0.02,
        dt=0.001,  # Larger dt to see differences
        diffusion_coeffs=0.01,
        BC='dirichlet'
    )
    
    # Set initial condition
    ic = create_gaussian_ic(model.xspan, center=0.5, width=0.1)
    model.set_initial_condition(ic)
    
    # Get operators
    mu = 0.01
    A, B = model.finite_diff_model(mu, same_on_both_ends=True)
    
    # Boundary conditions
    boundary_values = jnp.zeros((1, model.time_dim))
    
    # Test different integrators
    integrators = ['ForwardEuler', 'BackwardEuler', 'CrankNicolson']
    colors = ['blue', 'red', 'green']
    
    solutions = {}
    
    for integrator, color in zip(integrators, colors):
        solution = model.integrate_model(
            tdata=model.tspan,
            u0=model.IC,
            input_data=boundary_values,
            linear_matrix=A,
            control_matrix=B,
            system_input=True,
            integrator_type=integrator
        )
        solutions[integrator] = solution
    
    # Plot comparisons at different times
    time_indices = [model.time_dim // 4, model.time_dim // 2, 3 * model.time_dim // 4, -1]
    
    for idx, t_idx in enumerate(time_indices):
        ax = axes.flatten()[idx]
        time_val = model.tspan[t_idx]
        
        for integrator, color in zip(integrators, colors):
            ax.plot(model.xspan, solutions[integrator][:, t_idx],
                   label=integrator, color=color, linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'Comparison at t={time_val:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_integrator_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute and print errors (energy conservation)
    print("\nEnergy Conservation Analysis:")
    print("-" * 40)
    for integrator in integrators:
        energy = jnp.sum(solutions[integrator]**2, axis=0) * model.dx
        energy_loss = (energy[0] - energy[-1]) / energy[0] * 100
        print(f"{integrator:15s}: {energy_loss:6.2f}% energy loss")


def test_parameter_sweep():
    """Test with multiple diffusion coefficients."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Different diffusion coefficients
    diffusion_coeffs = [0.005, 0.01, 0.02, 0.05]
    colors = plt.cm.viridis(np.linspace(0, 1, len(diffusion_coeffs)))
    
    # Set up base model
    spatial_domain = (0.0, 1.0)
    time_domain = (0.0, 0.5)
    dx = 0.01
    dt = 0.0001
    
    # Time points to visualize
    t_final = 0.2
    
    for mu, color in zip(diffusion_coeffs, colors):
        model = Heat1DModel(
            spatial_domain=spatial_domain,
            time_domain=time_domain,
            dx=dx,
            dt=dt,
            diffusion_coeffs=mu,
            BC='neumann'
        )
        
        # Set initial condition
        ic = create_step_ic(model.xspan, left=0.4, right=0.6)
        model.set_initial_condition(ic)
        
        # Get operators for Neumann BC (zero flux)
        A, B = model.finite_diff_model(mu)
        
        # Zero flux boundary conditions
        boundary_values = jnp.zeros((2, model.time_dim))
        
        # Integrate
        solution = model.integrate_model(
            tdata=model.tspan,
            u0=model.IC,
            input_data=boundary_values,
            linear_matrix=A,
            control_matrix=B,
            system_input=True,
            integrator_type='CrankNicolson'
        )
        
        # Find index closest to t_final
        t_idx = jnp.argmin(jnp.abs(model.tspan - t_final))
        
        # Plot solution at t_final
        axes[0].plot(model.xspan, solution[:, t_idx], 
                    label=f'μ={mu:.3f}', color=color, linewidth=2)
        
        # Plot maximum value over time
        max_vals = jnp.max(solution, axis=0)
        axes[1].plot(model.tspan, max_vals, 
                    label=f'μ={mu:.3f}', color=color, linewidth=2)
        
        # Plot total mass (should be conserved for Neumann BC)
        mass = jnp.sum(solution, axis=0) * model.dx
        axes[2].plot(model.tspan, mass, 
                    label=f'μ={mu:.3f}', color=color, linewidth=2)
    
    # Format plots
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_title(f'Solution at t={t_final}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Max value')
    axes[1].set_title('Maximum Temperature Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Total mass')
    axes[2].set_title('Mass Conservation (Neumann BC)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_parameter_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()


def validate_against_analytical():
    """Validate numerical solution against analytical solution."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Set up model for validation
    model = Heat1DModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.5),
        dx=0.01,
        dt=0.0001,
        diffusion_coeffs=0.01,
        BC='dirichlet'
    )
    
    # Use sinusoidal initial condition for analytical solution
    ic = jnp.sin(jnp.pi * model.xspan)
    model.set_initial_condition(ic)
    
    # Get operators
    mu = 0.01
    A, B = model.finite_diff_model(mu, same_on_both_ends=True)
    
    # Zero Dirichlet boundary conditions
    boundary_values = jnp.zeros((1, model.time_dim))
    
    # Numerical solution
    numerical_solution = model.integrate_model(
        tdata=model.tspan,
        u0=model.IC,
        input_data=boundary_values,
        linear_matrix=A,
        control_matrix=B,
        system_input=True,
        integrator_type='CrankNicolson'
    )
    
    # Compare at different times
    test_times = [0.05, 0.1, 0.2]
    
    for idx, t_test in enumerate(test_times):
        t_idx = jnp.argmin(jnp.abs(model.tspan - t_test))
        
        # Analytical solution
        analytical = jnp.sin(jnp.pi * model.xspan) * jnp.exp(-jnp.pi**2 * mu * t_test)
        
        # Numerical solution at this time
        numerical = numerical_solution[:, t_idx]
        
        # Plot comparison
        ax = axes[idx]
        ax.plot(model.xspan, analytical, 'b-', label='Analytical', linewidth=2)
        ax.plot(model.xspan, numerical, 'r--', label='Numerical', linewidth=2)
        
        # Compute error
        error = jnp.abs(analytical - numerical)
        ax_twin = ax.twinx()
        ax_twin.plot(model.xspan, error, 'g:', label='Error', alpha=0.6)
        ax_twin.set_ylabel('Absolute Error', color='g')
        ax_twin.tick_params(axis='y', labelcolor='g')
        
        # Format
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f't = {t_test:.2f}, Max Error = {jnp.max(error):.2e}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_analytical_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print error statistics
    print("\nValidation Against Analytical Solution:")
    print("-" * 50)
    print(f"Spatial resolution (dx): {model.dx}")
    print(f"Temporal resolution (dt): {model.dt}")
    print(f"Diffusion coefficient: {mu}")
    
    for t_test in test_times:
        t_idx = jnp.argmin(jnp.abs(model.tspan - t_test))
        analytical = jnp.sin(jnp.pi * model.xspan) * jnp.exp(-jnp.pi**2 * mu * t_test)
        numerical = numerical_solution[:, t_idx]
        
        error = jnp.abs(analytical - numerical)
        rel_error = error / (jnp.max(jnp.abs(analytical)) + 1e-10)
        
        print(f"\nTime t = {t_test:.2f}:")
        print(f"  Max absolute error: {jnp.max(error):.6e}")
        print(f"  Mean absolute error: {jnp.mean(error):.6e}")
        print(f"  Max relative error: {jnp.max(rel_error):.6e}")


def test_stability_analysis():
    """Test numerical stability for different dt/dx ratios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Base parameters
    spatial_domain = (0.0, 1.0)
    mu = 0.01
    dx = 0.01
    
    # Test different dt values (stability parameter r = mu*dt/dx^2)
    dt_values = [0.00001, 0.00005, 0.0001, 0.0005]  # r = [0.01, 0.05, 0.1, 0.5]
    
    for idx, dt in enumerate(dt_values):
        ax = axes.flatten()[idx]
        
        # Stability parameter
        r = mu * dt / (dx ** 2)
        
        # Create model
        model = Heat1DModel(
            spatial_domain=spatial_domain,
            time_domain=(0.0, 0.1),
            dx=dx,
            dt=dt,
            diffusion_coeffs=mu,
            BC='periodic'
        )
        
        # Initial condition with high frequency component
        ic = create_sinusoidal_ic(model.xspan, n_modes=5)
        model.set_initial_condition(ic)
        
        # Get operators
        A = model.finite_diff_model(mu)
        
        # Integrate with Forward Euler (most stability-sensitive)
        try:
            solution = model.integrate_model(
                tdata=model.tspan,
                u0=model.IC,
                linear_matrix=A,
                system_input=False,
                integrator_type='ForwardEuler'
            )
            
            # Check for instability (NaN or very large values)
            if jnp.any(jnp.isnan(solution)) or jnp.max(jnp.abs(solution)) > 1e10:
                stability = "UNSTABLE"
                color = 'red'
            else:
                stability = "STABLE"
                color = 'green'
                
        except:
            stability = "FAILED"
            color = 'red'
            solution = jnp.zeros((model.spatial_dim, model.time_dim))
        
        # Plot solution evolution
        time_indices = jnp.linspace(0, model.time_dim-1, 5, dtype=int)
        
        for t_idx in time_indices:
            alpha = float(0.3 + 0.7 * (t_idx / model.time_dim))
            ax.plot(model.xspan, solution[:, t_idx], alpha=alpha, color=color)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'dt={dt:.5f}, r={r:.3f} ({stability})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 2])
    
    plt.tight_layout()
    plt.savefig('figures/heat1d_stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nStability Analysis (Forward Euler):")
    print("-" * 50)
    print("Stability criterion: r = μΔt/Δx² ≤ 0.5")
    print(f"μ = {mu}, Δx = {dx}")
    print("-" * 50)
    
    for dt in dt_values:
        r = mu * dt / (dx ** 2)
        expected_stable = r <= 0.5
        print(f"Δt = {dt:.5f}: r = {r:.3f}, Expected: {'STABLE' if expected_stable else 'UNSTABLE'}")

#%%
if __name__ == "__main__":
    print("Running Heat1D JAX validation tests...\n")
    
    # Run all tests
    print("1. Testing all boundary conditions...")
    test_all_boundary_conditions()
    
    print("\n2. Comparing integration schemes...")
    test_integrator_comparison()
    
    print("\n3. Parameter sweep analysis...")
    test_parameter_sweep()
    
    print("\n4. Validating against analytical solution...")
    validate_against_analytical()
    
    print("\n5. Stability analysis...")
    test_stability_analysis()
    
    print("\nAll tests completed!")
# %%
