"""
Demonstration of the Allen-Cahn PDE Solver

This script shows how to use the solver and create visualizations.
"""

#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from allen_cahn_jax import AllenCahnModel, example_initial_condition

#%%
def plot_solution(solver, u_solution, initial_condition_mode="random"):
    """Create visualization of the Allen-Cahn solution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Space-time evolution
    ax = axes[0, 0]
    im = ax.imshow(u_solution.T, aspect='auto', origin='lower', 
                   extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]],
                   cmap='RdBu_r')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.set_title('Space-Time Evolution of u(x,t)')
    plt.colorbar(im, ax=ax, label='u')
    
    # Plot 2: Solution at different times
    ax = axes[0, 1]
    time_indices = [0, len(solver.t)//4, len(solver.t)//2, 3*len(solver.t)//4, -1]
    for idx in time_indices:
        ax.plot(solver.x, u_solution[:, idx], label=f't = {solver.t[idx]:.3f}', linewidth=2)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution Profiles at Different Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Initial vs Final
    ax = axes[1, 0]
    ax.plot(solver.x, u_solution[:, 0], 'b-', linewidth=2, label='Initial (t=0)')
    ax.plot(solver.x, u_solution[:, -1], 'r-', linewidth=2, label=f'Final (t={solver.t[-1]:.3f})')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('u(x)')
    ax.set_title('Initial vs Final State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Energy evolution (L2 norm)
    ax = axes[1, 1]
    energy = jnp.sum(u_solution**2, axis=0) * solver.dx
    ax.plot(solver.t, energy, 'g-', linewidth=2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Energy (L² norm)')
    ax.set_title('Energy Evolution Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_methods():
    """Compare SICN and CNAB methods."""
    
    print("Comparing SICN and CNAB methods...")
    
    solver = AllenCahnModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.5),
        dx=0.01,
        dt=0.001,
        mu=0.01,
        epsilon=1.0
    )
    
    u0 = example_initial_condition(solver.x, mode="sine")
    
    # Solve with both methods
    u_sicn = solver.solve(u0, method="sicn", return_all_steps=True)
    u_cnab = solver.solve(u0, method="cnab", return_all_steps=True)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # SICN solution
    im1 = axes[0].imshow(u_sicn.T, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]])
    axes[0].set_xlabel('Space (x)')
    axes[0].set_ylabel('Time (t)')
    axes[0].set_title('SICN Method')
    plt.colorbar(im1, ax=axes[0])
    
    # CNAB solution
    im2 = axes[1].imshow(u_cnab.T, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]])
    axes[1].set_xlabel('Space (x)')
    axes[1].set_ylabel('Time (t)')
    axes[1].set_title('CNAB Method')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = jnp.abs(u_sicn - u_cnab)
    im3 = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='viridis',
                         extent=[solver.x[0], solver.x[-1], solver.t[0], solver.t[-1]])
    axes[2].set_xlabel('Space (x)')
    axes[2].set_ylabel('Time (t)')
    axes[2].set_title(f'Absolute Difference\n(max: {diff.max():.2e})')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig


def phase_separation_demo():
    """Demonstrate phase separation dynamics."""
    
    print("Running phase separation demonstration...")
    
    solver = AllenCahnModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 2.0),
        dx=0.005,
        dt=0.001,
        mu=0.001,  # Small diffusion
        epsilon=5.0  # Strong nonlinearity
    )
    
    # Random initial condition for phase separation
    u0 = example_initial_condition(solver.x, mode="random")
    
    print(f"Solving on {solver.nx} spatial points for {solver.nt} time steps...")
    u_solution = solver.solve(u0, method="cnab", return_all_steps=True)
    
    fig = plot_solution(solver, u_solution, "random")
    fig.suptitle('Phase Separation Dynamics in Allen-Cahn Equation', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    return fig, u_solution


def interface_evolution_demo():
    """Demonstrate interface evolution (sharp transition)."""
    
    print("Running interface evolution demonstration...")
    
    solver = AllenCahnModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 1.0),
        dx=0.005,
        dt=0.0005,
        mu=0.01,
        epsilon=1.0
    )
    
    # Tanh initial condition (sharp interface)
    u0 = example_initial_condition(solver.x, mode="tanh")
    
    u_solution = solver.solve(u0, method="cnab", return_all_steps=True)
    
    fig = plot_solution(solver, u_solution, "tanh")
    fig.suptitle('Interface Evolution in Allen-Cahn Equation', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    return fig, u_solution

#%%
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Allen-Cahn PDE Solver Demonstration")
    print("=" * 60)
    
    # Demo 1: Basic phase separation
    print("\n[1/3] Phase Separation Demo")
    print("-" * 60)
    start = time.time()
    fig1, sol1 = phase_separation_demo()
    print(f"Completed in {time.time() - start:.2f} seconds")
    plt.savefig('figures//phase_separation.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/phase_separation.png")
    
    # Demo 2: Interface evolution
    print("\n[2/3] Interface Evolution Demo")
    print("-" * 60)
    start = time.time()
    fig2, sol2 = interface_evolution_demo()
    print(f"Completed in {time.time() - start:.2f} seconds")
    plt.savefig('figures/interface_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/interface_evolution.png")
    
    # Demo 3: Method comparison
    print("\n[3/3] Method Comparison Demo")
    print("-" * 60)
    start = time.time()
    fig3 = compare_methods()
    print(f"Completed in {time.time() - start:.2f} seconds")
    plt.savefig('figures/method_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/method_comparison.png")
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)
    
    # Show plots
    plt.show()
