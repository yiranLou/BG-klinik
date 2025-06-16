#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time
import opensim as osim
from opensim import Coordinate
from tqdm import tqdm


# Get project root directory
def get_project_root():
    """Get the project root directory (newPipeline)"""
    current_file = os.path.abspath(__file__)
    # Navigate up from src/opensim/forward_dynamics.py to newPipeline/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root


PROJECT_ROOT = get_project_root()


def forward_dynamics(
        model_path,
        controls_file_path,
        result_dir,
        start_time=0.0,
        end_time=1.0,
        time_step=0.01,
        integrator_accuracy=1e-1,  # Even more relaxed
        simplified_model=True,
        max_step_attempts=5  # NEW: Maximum attempts per step
):
    """
    Perform forward dynamics simulation using OpenSim 4.5's Manager with extreme performance optimizations.
    - Focuses on getting past the initial step issues
    - Uses extremely relaxed tolerances
    - Implements adaptive step sizing with retry logic
    - Focuses on numerical stability over accuracy
    """

    print("Running super-optimized forward dynamics simulation...")

    # Performance metric: model loading time
    load_start_time = time.time()

    # 1) Load and initialize the model
    osimmodel = osim.Model(model_path)

    # Apply model simplifications for performance
    if simplified_model:
        print("[opt] Applying model simplifications for performance")
        try:
            # Remove unnecessary forces and constraints
            forceSet = osimmodel.getForceSet()
            num_forces = forceSet.getSize()
            print(f"Original model has {num_forces} forces")

            # Disable visualization for speed
            osimmodel.setUseVisualizer(False)

            # Try to simplify by disabling constraints
            constraintSet = osimmodel.getConstraintSet()
            num_constraints = constraintSet.getSize()
            print(f"Model has {num_constraints} constraints")

            # Disable all constraints for maximum performance
            if num_constraints > 0:
                print(f"[opt] Disabling ALL constraints for performance")
                for i in range(num_constraints):
                    try:
                        constraint = constraintSet.get(i)
                        constraint_name = constraint.getName()
                        constraint.set_isEnforced(False)
                        print(f"[opt] Disabled constraint: {constraint_name}")
                    except Exception as e:
                        print(f"[warning] Could not disable constraint: {e}")
        except Exception as e:
            print(f"[warning] Model simplification error: {e}")

    # Initialize the model
    state = osimmodel.initSystem()

    load_time = time.time() - load_start_time
    print(f"[perf] Model loading and initialization: {load_time:.3f} seconds")

    # 2) Lock as many coordinates as possible for stability
    # Start with thorax and add finger joints
    lock_coords = [
        # Thorax 6DOF
        "Thorax_rot1", "Thorax_rot2", "Thorax_rot3",
        "Thorax_tra1", "Thorax_tra2", "Thorax_tra3"
    ]

    # Lock even more coordinates for performance
    if simplified_model:
        print("[opt] Locking most coordinates for performance")
        # Lock all except the main arm movement joints
        keep_unlocked = [
            "ra_sh_elv_angle", "ra_sh_elv", "ra_sh_rot",
            "ra_el_e_f", "ra_wr_sup_pro", "ra_wr_rd_ud"
        ]

        all_coords = osimmodel.getCoordinateSet()
        for i in range(all_coords.getSize()):
            coord = all_coords.get(i)
            cname = coord.getName()

            # Lock all coordinates except those in keep_unlocked
            if cname not in keep_unlocked and cname not in lock_coords:
                lock_coords.append(cname)
                print(f"[opt] Added {cname} to locked coordinates")

    all_coords = osimmodel.getCoordinateSet()
    n_coords = all_coords.getSize()

    locked_count = 0
    unlocked_count = 0

    for i in range(n_coords):
        coord = all_coords.get(i)
        cname = coord.getName()
        if cname in lock_coords:
            coord.set_locked(True)
            locked_count += 1
        else:
            coord.set_locked(False)
            unlocked_count += 1

    print(f"Locked {locked_count} coordinates, unlocked {unlocked_count} coordinates")

    # 3) Set initial pose - just the essential joints
    def set_coord_value(coord_name, angle_deg):
        """Set initial value for coordinates (units: convert degrees to radians)."""
        try:
            coord_obj = osimmodel.getCoordinateSet().get(coord_name)
            coord_obj.setValue(state, math.radians(angle_deg))
        except Exception as e:
            print(f"[warning] Error setting initial angle for {coord_name}: {e}")

    # Set only essential joints for arm pose
    set_coord_value("ra_sh_elv_angle", 30.0)
    set_coord_value("ra_sh_elv", 20.0)
    set_coord_value("ra_sh_rot", 0.0)
    set_coord_value("ra_el_e_f", 45.0)
    set_coord_value("ra_wr_sup_pro", 0.0)
    set_coord_value("ra_wr_rd_ud", 0.0)

    # Assemble model with very relaxed tolerance
    try:
        print("[opt] Assembling model with relaxed tolerance...")

        # Try to set assembly accuracy if the API supports it
        try:
            osimmodel.set_assembly_accuracy(1e-1)  # Extremely relaxed tolerance
        except:
            print("[info] set_assembly_accuracy not available, using default")

        osimmodel.assemble(state)
        print("[info] Model assembly successful")
    except Exception as e:
        print(f"[error] model.assemble() failed: {e}")
        print("[opt] Continuing despite assembly failure...")

    # 4) Create Manager with extremely relaxed settings
    manager = osim.Manager(osimmodel)

    # Super relaxed integrator accuracy
    manager.setIntegratorAccuracy(integrator_accuracy)

    try:
        # Set extremely permissive integrator settings
        manager.setIntegratorMinimumStepSize(1e-10)
        manager.setIntegratorMaximumStepSize(time_step * 10)  # Allow very large steps
    except:
        print("[info] Some integrator settings not available, using defaults")

    # Skip muscle equilibration
    print("[opt] Skipping muscle equilibration for performance")

    # Set initial time
    state.setTime(start_time)
    manager.initialize(state)

    # 5) Run simulation with extreme performance optimizations
    sim_duration = end_time - start_time
    steps = int(sim_duration / time_step)

    # Reduce number of steps for initial testing
    test_steps = min(5, steps)
    print(f"[opt] First trying with just {test_steps} steps for testing")

    print("\n[debug] Using aggressive adaptive step size approach:")

    # NEW: Use adaptive step sizing with retry logic
    current_time = start_time
    target_time = end_time

    # Track successful steps
    successful_steps = 0
    retry_count = 0

    # Use tiny step size for first step to improve stability
    current_step_size = time_step / 10.0

    # Start integration loop with adaptive step size
    try:
        while current_time < target_time:
            # Calculate next time point
            next_time = min(current_time + current_step_size, target_time)

            # Try the step
            step_start = time.time()
            step_success = False

            for attempt in range(max_step_attempts):
                try:
                    print(f"[step] t={current_time:.4f}, stepping to t={next_time:.4f} (attempt {attempt + 1})")
                    manager.integrate(next_time)
                    step_time = time.time() - step_start
                    print(f"[success] Step to t={next_time:.4f} took {step_time:.5f}s")

                    # Step succeeded
                    step_success = True
                    successful_steps += 1

                    # Gradually increase step size on success
                    if successful_steps > 3:
                        current_step_size = min(current_step_size * 1.5, time_step * 5)
                        print(f"[adapt] Increasing step size to {current_step_size:.6f}")

                    break
                except Exception as e:
                    # Step failed
                    retry_count += 1
                    print(f"[retry] Step failed: {e}")

                    # Reduce step size and try again
                    current_step_size /= 2.0
                    next_time = current_time + current_step_size
                    print(f"[adapt] Reducing step size to {current_step_size:.6f}, next target: {next_time:.4f}")

                    # If we've reached a tiny step size, give up
                    if current_step_size < 1e-8:
                        print("[error] Step size too small, giving up")
                        raise Exception("Step size too small")

            if not step_success:
                print(f"[error] Failed to complete step after {max_step_attempts} attempts")
                break

            # Update current time
            current_time = manager.getState().getTime()

            # Break after test steps to see if we're making progress
            if successful_steps >= test_steps:
                print(f"[info] Completed {test_steps} test steps successfully")
                break

        # If we got through the test steps, try to complete the rest
        if successful_steps >= test_steps:
            print(
                f"\n[info] Test steps completed. Proceeding with full simulation from t={current_time:.4f} to t={target_time:.4f}")

            # Continue with standard steps for the remainder
            remaining_time = target_time - current_time
            remaining_steps = int(remaining_time / time_step) + 1

            print(f"[info] Remaining steps: approximately {remaining_steps}")

            # Process remaining steps in small batches
            batch_size = 5
            completed_batches = 0

            while current_time < target_time:
                batch_start_time = current_time
                batch_end_time = min(current_time + (batch_size * time_step), target_time)

                try:
                    print(f"[batch] Processing from t={batch_start_time:.4f} to t={batch_end_time:.4f}")
                    manager.integrate(batch_end_time)
                    completed_batches += 1
                    print(f"[batch] Completed batch {completed_batches}")
                except Exception as e:
                    print(f"[error] Batch failed: {e}")
                    break

                current_time = manager.getState().getTime()

        final_state = manager.getState()
        print(f"[info] Simulation reached t={final_state.getTime():.4f} out of {target_time:.4f}")

        # Try to save results, even if incomplete
        try:
            # Write controls and states even if simulation is incomplete
            osimmodel.printControlStorage(os.path.join(result_dir, "controls.sto"))
            print(f"[info] Saved controls to {os.path.join(result_dir, 'controls.sto')}")

            statesTable = manager.getStateStorage()
            statesTable.print(os.path.join(result_dir, "states.sto"))
            print(f"[info] Saved states to {os.path.join(result_dir, 'states.sto')}")
        except Exception as e:
            print(f"[warning] Could not save results: {e}")

    except Exception as e:
        print(f"[error] Integration failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"[summary] Simulation stats: {successful_steps} successful steps, {retry_count} retries")
    print("Simulation completed to the extent possible.")

    return osimmodel, manager


def main():
    """Main function with super-optimized settings."""
    # Use relative paths based on project structure
    model_path = os.path.join(PROJECT_ROOT, "config", "models", "ms_arm_and_hand-main", "AAH Model",
                              "RightArmAndHand_scaled.osim")
    controls_file_path = os.path.join(PROJECT_ROOT, "data", "processed", "emg_processed", "emg_norm.sto")
    result_dir = os.path.join(PROJECT_ROOT, "results", "forward_dynamics")

    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {model_path}")
    print(f"Controls file: {controls_file_path}")
    print(f"Result directory: {result_dir}")

    # Simulation time
    start_time = 0.0
    end_time = 1.0
    time_step = 0.01

    # Extreme performance optimization settings
    osimmodel, manager = forward_dynamics(
        model_path=model_path,
        controls_file_path=controls_file_path,
        result_dir=result_dir,
        start_time=start_time,
        end_time=end_time,
        time_step=time_step,
        integrator_accuracy=1e-1,  # Extremely relaxed
        simplified_model=True,  # Maximum simplification
        max_step_attempts=5  # Multiple attempts per step
    )

    # Try to analyze whatever results we got
    try:
        print("\nSimulation Analysis:")
        state = manager.getState()

        # Print a few key joint angles
        def get_coord_value_deg(coord_name):
            try:
                coord = osimmodel.getCoordinateSet().get(coord_name)
                return math.degrees(coord.getValue(state))
            except Exception as e:
                return f"Error: {e}"

        print(f"Final shoulder elevation: {get_coord_value_deg('ra_sh_elv'):.1f}°")
        print(f"Final elbow flexion: {get_coord_value_deg('ra_el_e_f'):.1f}°")
        print(f"Final wrist flexion: {get_coord_value_deg('ra_wr_rd_ud'):.1f}°")

        # How far did we get?
        final_time = state.getTime()
        progress_percent = (final_time - start_time) / (end_time - start_time) * 100
        print(f"Simulation progress: {progress_percent:.1f}% ({final_time:.3f}s out of {end_time:.3f}s)")
    except Exception as e:
        print(f"Could not analyze results: {e}")


if __name__ == "__main__":
    main()