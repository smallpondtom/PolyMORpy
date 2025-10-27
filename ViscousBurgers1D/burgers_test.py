"""
Tests for Burgers equation solver

Verifies:
1. Operator construction
2. Integration correctness
3. Conservation properties
4. Parameter updates
5. Control inputs
"""

#%%
import jax.numpy as jnp
import numpy as np
from burgers_jax import BurgersModel, solve_burgers

#%%
def test_operator_shapes():
    """Test that operators have correct shapes"""
    print("Test 1: Operator Shapes")
    print("-" * 40)
    
    # Periodic BC
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.1),
        dx=0.1,
        dt=0.01,
        diffusion_coeff=0.01,
        BC='periodic',
        conservation_type='NC'
    )
    
    N = model.N
    S = N * (N + 1) // 2
    
    assert model.A.shape == (N, N), f"A shape incorrect: {model.A.shape}"
    assert model.F.shape == (N, S), f"F shape incorrect: {model.F.shape}"
    assert model.B is None, "B should be None for periodic BC"
    
    print(f"✓ Periodic BC: A {model.A.shape}, F {model.F.shape}")
    
    # Dirichlet BC
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.1),
        dx=0.1,
        dt=0.01,
        diffusion_coeff=0.01,
        BC='dirichlet',
        conservation_type='NC'
    )
    
    N_dirichlet = model.N  # Dirichlet includes both boundaries
    S_dirichlet = N_dirichlet * (N_dirichlet + 1) // 2
    
    assert model.A.shape == (N_dirichlet, N_dirichlet), f"A shape incorrect: {model.A.shape}"
    assert model.F.shape == (N_dirichlet, S_dirichlet), f"F shape incorrect: {model.F.shape}"
    assert model.B.shape[0] == N_dirichlet, f"B shape incorrect: {model.B.shape}"
    
    print(f"✓ Dirichlet BC: A {model.A.shape}, F {model.F.shape}, B {model.B.shape}")
    print("✓ All operator shapes correct\n")


def test_symmetry_periodic():
    """Test that periodic BC solution has expected behavior for symmetric IC"""
    print("Test 2: Solution Behavior with Periodic BC")
    print("-" * 40)
    
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.1),
        dx=0.01,
        dt=0.001,
        diffusion_coeff=0.01,
        BC='periodic',
        conservation_type='NC'
    )
    
    # Use zero initial condition - solution should remain zero
    u0 = jnp.zeros(model.N)
    u = model.solve(u0)
    
    max_val = jnp.max(jnp.abs(u))
    print(f"Max |u| from zero IC: {max_val:.2e}")
    assert max_val < 1e-10, "Zero IC should remain zero"
    
    # Test with small symmetric perturbation
    x = model.xspan  
    u0_sym = 0.01 * jnp.sin(2 * jnp.pi * x)  # Symmetric about x=0
    u_sym = model.solve(u0_sym)
    
    # Should decay due to diffusion
    final_max = jnp.max(jnp.abs(u_sym[:, -1]))
    initial_max = jnp.max(jnp.abs(u0_sym))
    print(f"Initial max: {initial_max:.6f}, Final max: {final_max:.6f}")
    assert final_max < initial_max, "Solution should decay with diffusion"
    
    print("✓ Solution behaves correctly with periodic BC\n")


def test_conservation_types():
    """Test that different conservation types give different results"""
    print("Test 3: Conservation Types")
    print("-" * 40)
    
    u0 = jnp.sin(2 * jnp.pi * jnp.arange(0, 1, 0.01))
    
    results = {}
    energies = {}
    
    for ctype in ['NC', 'C', 'EP']:
        model = BurgersModel(
            spatial_domain=(0.0, 1.0),
            time_domain=(0.0, 0.2),
            dx=0.01,
            dt=0.0005,
            diffusion_coeff=0.01,
            BC='periodic',
            conservation_type=ctype
        )
        
        u = model.solve(u0)
        results[ctype] = u
        
        # Compute energy
        energy_initial = jnp.sum(u[:, 0]**2) * model.dx
        energy_final = jnp.sum(u[:, -1]**2) * model.dx
        energy_change = (energy_final - energy_initial) / energy_initial * 100
        energies[ctype] = energy_change
        
        print(f"{ctype}: Energy change = {energy_change:+.2f}%")
    
    # All should dissipate energy (negative change)
    for ctype, change in energies.items():
        assert change < 0, f"{ctype} should dissipate energy"
    
    # Check that solutions are different
    diff_NC_C = jnp.max(jnp.abs(results['NC'] - results['C']))
    diff_NC_EP = jnp.max(jnp.abs(results['NC'] - results['EP']))
    
    print(f"\nMax difference NC vs C: {diff_NC_C:.4f}")
    print(f"Max difference NC vs EP: {diff_NC_EP:.4f}")
    
    assert diff_NC_C > 1e-6, "NC and C should give different results"
    assert diff_NC_EP > 1e-6, "NC and EP should give different results"
    
    print("✓ All conservation types work correctly\n")


def test_parameter_update():
    """Test that parameter updates work correctly"""
    print("Test 4: Parameter Updates")
    print("-" * 40)
    
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.1),
        dx=0.01,
        dt=0.001,
        diffusion_coeff=0.01,
        BC='periodic',
        conservation_type='NC'
    )
    
    x = model.xspan
    u0 = jnp.exp(-100 * (x - 0.5)**2)
    
    # Solve with initial mu
    u1 = model.solve(u0)
    
    # Update mu and solve again
    model.update_parameters(mu=0.02)
    u2 = model.solve(u0)
    
    # Solutions should be different
    diff = jnp.max(jnp.abs(u1 - u2))
    print(f"μ=0.01 vs μ=0.02: Max difference = {diff:.4f}")
    assert diff > 1e-6, "Solutions should differ with different parameters"
    
    # Higher viscosity should lead to more diffusion (lower final peak)
    peak1 = jnp.max(u1[:, -1])
    peak2 = jnp.max(u2[:, -1])
    print(f"Peak with μ=0.01: {peak1:.4f}")
    print(f"Peak with μ=0.02: {peak2:.4f}")
    assert peak2 < peak1, "Higher viscosity should reduce peak"
    
    print("✓ Parameter updates work correctly\n")


def test_control_input():
    """Test control input for Dirichlet BC"""
    print("Test 5: Control Input")
    print("-" * 40)
    
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.2),
        dx=0.01,
        dt=0.001,
        diffusion_coeff=0.01,
        BC='dirichlet',
        conservation_type='NC'
    )
    
    u0 = jnp.zeros(model.N)
    
    # Solve without control
    u_no_control = model.solve(u0)
    
    # Solve with control
    control = jnp.zeros((1, model.T))
    control = control.at[0, :].set(0.5)
    u_with_control = model.solve(u0, control=control)
    
    # Control should affect the solution
    diff = jnp.max(jnp.abs(u_with_control - u_no_control))
    print(f"Max difference with/without control: {diff:.4f}")
    assert diff > 1e-6, "Control should affect solution"
    
    # With constant positive control, interior should become positive
    assert jnp.max(u_with_control) > 0.1, "Control should drive solution positive"
    
    print("✓ Control input works correctly\n")


def test_mass_conservation_periodic():
    """Test that mass is conserved for periodic BC with conservative form"""
    print("Test 6: Mass Conservation (Periodic, Conservative)")
    print("-" * 40)
    
    model = BurgersModel(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.5),
        dx=0.01,
        dt=0.0005,
        diffusion_coeff=0.01,
        BC='periodic',
        conservation_type='C'  # Conservative form
    )
    
    x = model.xspan
    u0 = jnp.sin(2 * jnp.pi * x) + 0.5
    
    u = model.solve(u0)
    
    # Compute total mass (integral of u)
    mass_initial = jnp.sum(u[:, 0]) * model.dx
    mass_final = jnp.sum(u[:, -1]) * model.dx
    mass_change = abs(mass_final - mass_initial) / mass_initial * 100
    
    print(f"Initial mass: {mass_initial:.6f}")
    print(f"Final mass: {mass_final:.6f}")
    print(f"Mass change: {mass_change:.4f}%")
    
    # Mass should be approximately conserved
    assert mass_change < 1.0, f"Mass not conserved: {mass_change}% change"
    
    print("✓ Mass approximately conserved\n")


def test_convergence_dt():
    """Test that solution converges with smaller dt"""
    print("Test 7: Temporal Convergence")
    print("-" * 40)
    
    T_final = 0.1
    dx = 0.01
    x = jnp.arange(0, 1, dx)
    u0 = jnp.exp(-100 * (x - 0.5)**2)
    
    dts = [0.002, 0.001, 0.0005]
    solutions = []
    
    for dt in dts:
        model = BurgersModel(
            spatial_domain=(0.0, 1.0),
            time_domain=(0.0, T_final),
            dx=dx,
            dt=dt,
            diffusion_coeff=0.01,
            BC='periodic',
            conservation_type='NC'
        )
        u = model.solve(u0)
        solutions.append(u[:, -1])
        print(f"dt={dt:.4f}: Peak = {jnp.max(u[:, -1]):.6f}")
    
    # Solutions should be getting closer
    diff_coarse = jnp.max(jnp.abs(solutions[0] - solutions[1]))
    diff_fine = jnp.max(jnp.abs(solutions[1] - solutions[2]))
    
    print(f"Difference (dt=0.002 vs 0.001): {diff_coarse:.6f}")
    print(f"Difference (dt=0.001 vs 0.0005): {diff_fine:.6f}")
    
    assert diff_fine < diff_coarse, "Solution should converge with smaller dt"
    print("✓ Solution converges with smaller dt\n")


def test_convenience_function():
    """Test the convenience solve_burgers function"""
    print("Test 8: Convenience Function")
    print("-" * 40)
    
    u0 = jnp.exp(-100 * (jnp.arange(0, 1, 0.01) - 0.5)**2)
    
    x, t, u = solve_burgers(
        spatial_domain=(0.0, 1.0),
        time_domain=(0.0, 0.1),
        dx=0.01,
        dt=0.001,
        mu=0.01,
        u0=u0,
        BC='periodic',
        conservation_type='NC'
    )
    
    assert x.shape[0] == u.shape[0], "x and u spatial dims don't match"
    assert t.shape[0] == u.shape[1], "t and u temporal dims don't match"
    assert u.shape[0] == len(u0), "Solution shape doesn't match IC"
    
    print(f"x shape: {x.shape}")
    print(f"t shape: {t.shape}")
    print(f"u shape: {u.shape}")
    print("✓ Convenience function works correctly\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print(" " * 15 + "BURGERS SOLVER TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_operator_shapes,
        test_symmetry_periodic,
        test_conservation_types,
        test_parameter_update,
        test_control_input,
        test_mass_conservation_periodic,
        test_convergence_dt,
        test_convenience_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All tests passed! 🎉\n")
    
    return failed == 0

#%%
if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
