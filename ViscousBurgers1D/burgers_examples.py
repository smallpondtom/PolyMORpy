"""
Examples demonstrating the Viscous Burgers' equation solver

Includes:
1. Basic periodic boundary condition solving
2. Dirichlet boundary condition with control
3. Comparing conservation types
4. Parameter sensitivity analysis
5. Shock wave formation
"""

#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from burgers_jax import BurgersModel, solve_burgers

#%%
def example_1_periodic_basic():
    """Basic example with periodic BC"""
    print("=" * 60)
    print("Example 1: Basic Periodic Boundary Condition")
    print("=" * 60)
    
    # Setup
    L = 1.0
    T = 0.5
    mu = 0.01
    dx = 0.01
    dt = 0.001
    
    # Create model
    model = BurgersModel(
        spatial_domain=(0.0, L),
        time_domain=(0.0, T),
        dx=dx,
        dt=dt,
        diffusion_coeff=mu,
        BC='periodic',
        conservation_type='NC'
    )
    
    # Initial condition: Gaussian bump
    x = model.xspan
    u0 = jnp.exp(-100 * (x - 0.5)**2)
    
    # Solve
    u = model.solve(u0)
    
    print(f"Grid points: {model.N}")
    print(f"Time steps: {model.T}")
    print(f"Initial max: {jnp.max(u0):.6f}")
    print(f"Final max: {jnp.max(u[:, -1]):.6f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(x, u0, 'b-', label='Initial')
    plt.plot(x, u[:, model.T//4], 'g-', label=f't={model.tspan[model.T//4]:.2f}')
    plt.plot(x, u[:, model.T//2], 'r-', label=f't={model.tspan[model.T//2]:.2f}')
    plt.plot(x, u[:, -1], 'k-', label=f't={model.tspan[-1]:.2f}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(132)
    plt.imshow(u, aspect='auto', extent=[0, T, 0, L], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='u')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Spacetime Heatmap')
    
    plt.subplot(133)
    energy = jnp.sum(u**2, axis=0) * dx
    plt.plot(model.tspan, energy, 'b-')
    plt.xlabel('Time')
    plt.ylabel('Energy (L2 norm)')
    plt.title('Energy Evolution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/burgers_example1.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to figures/burgers_example1.png\n")
    return model, u


def example_2_dirichlet_control():
    """Example with Dirichlet BC and control"""
    print("=" * 60)
    print("Example 2: Dirichlet BC with Control Input")
    print("=" * 60)
    
    # Setup
    L = 1.0
    T = 0.5
    mu = 0.01
    dx = 0.01
    dt = 0.001
    
    model = BurgersModel(
        spatial_domain=(0.0, L),
        time_domain=(0.0, T),
        dx=dx,
        dt=dt,
        diffusion_coeff=mu,
        BC='dirichlet',
        conservation_type='NC'
    )
    
    # Initial condition: zero
    u0 = jnp.zeros(model.N)
    
    # Control: sinusoidal input at boundaries
    control = jnp.zeros((1, model.T))
    control = control.at[0, :].set(0.5 * jnp.sin(2 * jnp.pi * model.tspan / T))
    
    # Solve with and without control
    u_no_control = model.solve(u0)
    u_with_control = model.solve(u0, control=control)
    
    print(f"Max value (no control): {jnp.max(jnp.abs(u_no_control)):.6f}")
    print(f"Max value (with control): {jnp.max(jnp.abs(u_with_control)):.6f}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(u_no_control, aspect='auto', extent=[0, T, 0, L], 
               origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='u')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Without Control')
    
    plt.subplot(122)
    plt.imshow(u_with_control, aspect='auto', extent=[0, T, 0, L], 
               origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='u')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('With Control')
    
    plt.tight_layout()
    plt.savefig('figures/burgers_example2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to figures/burgers_example2.png\n")
    return model, u_with_control


def example_3_conservation_types():
    """Compare different conservation types"""
    print("=" * 60)
    print("Example 3: Comparing Conservation Types")
    print("=" * 60)
    
    # Setup
    L = 1.0
    T = 1.0
    mu = 0.005
    dx = 0.01
    dt = 0.0005
    
    # Initial condition
    x = jnp.arange(0, L, dx)
    u0 = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)
    
    results = {}
    conservation_types = ['NC', 'C', 'EP']
    
    plt.figure(figsize=(15, 4))
    
    for i, ctype in enumerate(conservation_types):
        model = BurgersModel(
            spatial_domain=(0.0, L),
            time_domain=(0.0, T),
            dx=dx,
            dt=dt,
            diffusion_coeff=mu,
            BC='periodic',
            conservation_type=ctype
        )
        
        u = model.solve(u0)
        results[ctype] = u
        
        # Compute energy
        energy = jnp.sum(u**2, axis=0) * dx
        
        print(f"\n{ctype} (", end="")
        if ctype == 'NC':
            print("Non-Conservative)", end="")
        elif ctype == 'C':
            print("Conservative)", end="")
        else:
            print("Energy Preserving)", end="")
        print(f":")
        print(f"  Initial energy: {energy[0]:.6f}")
        print(f"  Final energy: {energy[-1]:.6f}")
        print(f"  Energy change: {(energy[-1] - energy[0]) / energy[0] * 100:.2f}%")
        
        # Plot
        plt.subplot(1, 3, i + 1)
        plt.plot(x, u0, 'k--', alpha=0.5, label='Initial')
        plt.plot(x, u[:, model.T//4], label=f't={model.tspan[model.T//4]:.2f}')
        plt.plot(x, u[:, model.T//2], label=f't={model.tspan[model.T//2]:.2f}')
        plt.plot(x, u[:, -1], label=f't={model.tspan[-1]:.2f}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'{ctype} Type')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/burgers_example3.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to figures/burgers_example3.png\n")
    return results


def example_4_parameter_sensitivity():
    """Demonstrate easy parameter updates"""
    print("=" * 60)
    print("Example 4: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # Setup
    L = 1.0
    T = 0.3
    dx = 0.01
    dt = 0.0005
    
    # Create model (will update mu later)
    model = BurgersModel(
        spatial_domain=(0.0, L),
        time_domain=(0.0, T),
        dx=dx,
        dt=dt,
        diffusion_coeff=0.01,  # Initial value
        BC='periodic',
        conservation_type='NC'
    )
    
    # Initial condition: discontinuity
    x = model.xspan
    u0 = jnp.where(x < 0.5, 1.0, 0.0)
    
    # Test different viscosities
    viscosities = [0.001, 0.005, 0.01, 0.02]
    
    plt.figure(figsize=(15, 4))
    
    for i, mu in enumerate(viscosities):
        # Update parameter and solve
        model.update_parameters(mu=mu)
        u = model.solve(u0)
        
        print(f"μ = {mu:.4f}: max|u| = {jnp.max(jnp.abs(u)):.4f}, "
              f"min(u) = {jnp.min(u):.4f}")
        
        # Plot final state
        plt.subplot(1, 4, i + 1)
        plt.plot(x, u0, 'k--', alpha=0.3, label='Initial')
        plt.plot(x, u[:, -1], 'b-', linewidth=2, label=f't={T}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'μ = {mu:.4f}')
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/burgers_example4.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to figures/burgers_example4.png\n")


def example_5_shock_formation():
    """Demonstrate shock wave formation"""
    print("=" * 60)
    print("Example 5: Shock Wave Formation")
    print("=" * 60)
    
    # Setup for shock formation
    L = 2.0
    T = 1.0
    mu = 0.001  # Very small viscosity to see shock
    dx = 0.01
    dt = 0.0002
    
    model = BurgersModel(
        spatial_domain=(0.0, L),
        time_domain=(0.0, T),
        dx=dx,
        dt=dt,
        diffusion_coeff=mu,
        BC='periodic',
        conservation_type='C'  # Conservative form for shocks
    )
    
    # Initial condition: sine wave (will form shock)
    x = model.xspan
    u0 = jnp.sin(jnp.pi * x)
    
    u = model.solve(u0)
    
    # Plot at multiple times
    times_to_plot = [0, model.T//10, model.T//5, model.T//3, model.T//2, -1]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    for idx in times_to_plot:
        label = f't={model.tspan[idx]:.2f}'
        plt.plot(x, u[:, idx], label=label, linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Shock Wave Formation (μ = 0.001)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.imshow(u, aspect='auto', extent=[0, T, 0, L], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='u')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Spacetime Evolution')
    
    plt.tight_layout()
    plt.savefig('figures/burgers_example5.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Initial condition: sin(πx)")
    print(f"Viscosity: μ = {mu}")
    print(f"Shock becomes visible around t ≈ {T/3:.2f}")
    print(f"Plot saved to figures/burgers_example5.png\n")


def example_6_convenience_function():
    """Show usage of convenience function"""
    print("=" * 60)
    print("Example 6: Using Convenience Function")
    print("=" * 60)
    
    # One-liner solution
    x, t, u = solve_burgers(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.5),
        dx=0.01,
        dt=0.001,
        mu=0.01,
        u0=jnp.exp(-100 * (jnp.arange(0, 1, 0.01) - 0.5)**2),
        BC='periodic',
        conservation_type='EP'
    )
    
    print(f"Solved using convenience function!")
    print(f"Solution shape: {u.shape}")
    print(f"x shape: {x.shape}")
    print(f"t shape: {t.shape}")
    print()

#%%
if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "BURGERS EQUATION EXAMPLES" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Run all examples
    example_1_periodic_basic()
    example_2_dirichlet_control()
    example_3_conservation_types()
    example_4_parameter_sensitivity()
    example_5_shock_formation()
    example_6_convenience_function()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - burgers_example1.png: Basic periodic BC")
    print("  - burgers_example2.png: Dirichlet BC with control")
    print("  - burgers_example3.png: Conservation type comparison")
    print("  - burgers_example4.png: Parameter sensitivity")
    print("  - burgers_example5.png: Shock wave formation")
