"""
Kawahara Equation Solver - Comprehensive Demonstration

This script demonstrates:
1. Different conservation types (NC, C, EP)
2. Different dispersion orders (1, 3)
3. Dynamic parameter updates
4. Visualization capabilities
"""

#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from kawahara_jax import KawaharaModel, example_initial_condition
import time

#%%
def plot_solution(solver, u_solution, title="Kawahara Equation Solution"):
    """Create visualization of the Kawahara solution."""
    
    # Downsample for visualization if needed
    ds = max(1, len(solver.t) // 300)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Space-time evolution (heatmap)
    ax = axes[0, 0]
    im = ax.imshow(u_solution[:, ::ds].T, aspect='auto', origin='lower',
                   extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]],
                   cmap='RdBu_r', interpolation='bilinear')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.set_title('Space-Time Evolution')
    plt.colorbar(im, ax=ax, label='u(x,t)')
    
    # Plot 2: Solution at different times
    ax = axes[0, 1]
    time_indices = [0, len(solver.t)//4, len(solver.t)//2, 3*len(solver.t)//4, -1]
    for idx in time_indices:
        ax.plot(solver.x, u_solution[:, idx], 
                label=f't = {solver.t[idx]:.1f}', linewidth=2, alpha=0.8)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Snapshots at Different Times')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Initial vs Final
    ax = axes[1, 0]
    ax.plot(solver.x, u_solution[:, 0], 'b-', linewidth=2, label='Initial (t=0)')
    ax.plot(solver.x, u_solution[:, -1], 'r-', linewidth=2, 
            label=f'Final (t={solver.t[-1]:.1f})')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('u(x)')
    ax.set_title('Initial vs Final State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Max amplitude over time
    ax = axes[1, 1]
    max_vals = jnp.max(jnp.abs(u_solution), axis=0)
    ax.plot(solver.t, max_vals, 'g-', linewidth=2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Max |u(x,t)|')
    ax.set_title('Maximum Amplitude Over Time')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def demo_basic_usage():
    """Demonstrate basic usage."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)
    
    solver = KawaharaModel(
        spatial_domain=(0.0, 50.0),
        time_domain=(0.0, 100.0),
        nx=256,
        dt=0.01,
        mu=1.0,
        delta=0.15,
        conservation_type='NC',
        dispersion_order=1
    )
    
    u0 = example_initial_condition(solver.x, mode="cosines", 
                                   L=solver.spatial_domain[1])
    
    print("\nSolving...")
    start = time.time()
    u = solver.solve(u0, return_all_steps=True, verbose=False)
    elapsed = time.time() - start
    
    print(f"Computation time: {elapsed:.2f} seconds")
    print(f"Solution shape: {u.shape}")
    print(f"Final state range: [{u[:, -1].min():.4f}, {u[:, -1].max():.4f}]")
    
    fig = plot_solution(solver, u, "Basic Kawahara Solution (NC, Order 1)")
    plt.savefig('figures/kawahara_basic.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/kawahara_basic.png")
    
    return u


def demo_conservation_types():
    """Compare different conservation types."""
    print("\n" + "=" * 70)
    print("DEMO 2: Comparing Conservation Types")
    print("=" * 70)
    
    conservation_types = ['NC', 'C', 'EP']
    results = {}
    
    for ctype in conservation_types:
        print(f"\nSolving with {ctype} conservation...")
        
        solver = KawaharaModel(
            spatial_domain=(0.0, 50.0),
            time_domain=(0.0, 50.0),
            nx=128,
            dt=0.01,
            mu=1.0,
            delta=0.15,
            conservation_type=ctype,
            dispersion_order=1
        )
        
        u0 = example_initial_condition(solver.x, mode="cosines", 
                                       L=solver.spatial_domain[1])
        u = solver.solve(u0, return_all_steps=True, verbose=False)
        results[ctype] = (solver, u)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, ctype in enumerate(conservation_types):
        solver, u = results[ctype]
        ds = max(1, len(solver.t) // 200)
        
        ax = axes[idx]
        im = ax.imshow(u[:, ::ds].T, aspect='auto', origin='lower',
                      extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]],
                      cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        ax.set_title(f'{ctype} Conservation')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('figures/kawahara_conservation_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved: figures/kawahara_conservation_comparison.png")
    
    return results


def demo_dispersion_orders():
    """Compare different dispersion orders."""
    print("\n" + "=" * 70)
    print("DEMO 3: Comparing Dispersion Orders")
    print("=" * 70)
    
    dispersion_orders = [1, 3, 5, 8]
    results = {}
    
    for order in dispersion_orders:
        print(f"\nSolving with dispersion order {order}...")
        
        solver = KawaharaModel(
            spatial_domain=(0.0, 50.0),
            time_domain=(0.0, 50.0),
            nx=128,
            dt=0.01,
            mu=1.0,
            nu=0.08,
            delta=0.15,
            conservation_type='NC',
            dispersion_order=order
        )
        
        u0 = example_initial_condition(solver.x, mode="cosines", 
                                       L=solver.spatial_domain[1])
        u = solver.solve(u0, return_all_steps=True, verbose=False)
        results[order] = (solver, u)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, order in enumerate(dispersion_orders):
        solver, u = results[order]
        ds = max(1, len(solver.t) // 200)
        
        ax = axes[idx]
        im = ax.imshow(u[:, ::ds].T, aspect='auto', origin='lower',
                      extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]],
                      cmap='RdBu_r')
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        if order == 8:
            ax.set_title(f'Dispersion Order 3+5')
        else:
            ax.set_title(f'Dispersion Order {order}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('figures/kawahara_dispersion_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved: figures/kawahara_dispersion_comparison.png")
    
    return results


def demo_parameter_updates():
    """Demonstrate dynamic parameter updates."""
    print("\n" + "=" * 70)
    print("DEMO 4: Dynamic Parameter Updates")
    print("=" * 70)
    
    solver = KawaharaModel(
        spatial_domain=(0.0, 50.0),
        time_domain=(0.0, 30.0),
        nx=128,
        dt=0.01,
        mu=1.0,
        delta=0.15,
        conservation_type='NC',
        dispersion_order=1
    )
    
    u0 = example_initial_condition(solver.x, mode="cosines", 
                                   L=solver.spatial_domain[1])
    
    # Solve with different parameters
    param_sets = [
        {'mu': 1.0, 'delta': 0.15, 'nu': 0.01, 'label': 'μ=1.0, δ=0.15, ν=0.01'},
        {'mu': 0.5, 'delta': 0.15, 'nu': 0.03, 'label': 'μ=0.5, δ=0.15, ν=0.03'},
        {'mu': 1.0, 'delta': 0.30, 'nu': 0.05, 'label': 'μ=1.0, δ=0.30, ν=0.05'},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, params in enumerate(param_sets):
        print(f"\nSolving with {params['label']}...")
        solver.update_parameters(mu=params['mu'], delta=params['delta'],
                                 nu=params['nu'])
        
        u = solver.solve(u0, return_all_steps=True, verbose=False)
        
        ds = max(1, len(solver.t) // 200)
        ax = axes[idx]
        im = ax.imshow(u[:, ::ds].T, aspect='auto', origin='lower',
                      extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]],
                      cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        ax.set_title(params['label'])
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('figures//kawahara_parameter_study.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved: figures/kawahara_parameter_study.png")


def demo_energy_monitoring():
    """Monitor energy for different conservation types."""
    print("\n" + "=" * 70)
    print("DEMO 5: Energy Monitoring")
    print("=" * 70)
    
    conservation_types = ['NC', 'C', 'EP']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ctype in conservation_types:
        print(f"\nComputing with {ctype} conservation...")
        
        solver = KawaharaModel(
            spatial_domain=(0.0, 50.0),
            time_domain=(0.0, 100.0),
            nx=128,
            dt=0.01,
            mu=1.0,
            delta=0.15,
            conservation_type=ctype,
            dispersion_order=1
        )
        
        u0 = example_initial_condition(solver.x, mode="cosines", 
                                       L=solver.spatial_domain[1])
        u = solver.solve(u0, return_all_steps=True, verbose=False)
        
        # Compute energy (L2 norm)
        energy = jnp.sum(u**2, axis=0) * solver.dx
        
        # Compute mass (integral of u)
        mass = jnp.sum(u, axis=0) * solver.dx
        
        # Plot energy evolution
        axes[0].plot(solver.t, energy, linewidth=2, label=ctype, alpha=0.8)
        
        # Plot mass evolution
        axes[1].plot(solver.t, mass, linewidth=2, label=ctype, alpha=0.8)
    
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Energy (L² norm)')
    axes[0].set_title('Energy Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Mass (∫u dx)')
    axes[1].set_title('Mass Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/kawahara_energy_mass.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved: figures/kawahara_energy_mass.png")

#%%
if __name__ == "__main__":
    print("=" * 70)
    print("KAWAHARA EQUATION SOLVER - COMPREHENSIVE DEMONSTRATIONS")
    print("=" * 70)
    
    # Run all demos
    demo_basic_usage()
    demo_conservation_types()
    demo_dispersion_orders()
    demo_parameter_updates()
    demo_energy_monitoring()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - kawahara_basic.png")
    print("  - kawahara_conservation_comparison.png")
    print("  - kawahara_dispersion_comparison.png")
    print("  - kawahara_parameter_study.png")
    print("  - kawahara_energy_mass.png")
