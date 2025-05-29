#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import opensim as osim
from opensim import Coordinate
from tqdm import tqdm


def forward_dynamics(
        model_path,
        controls_file_path,
        result_dir,
        start_time=0.0,
        end_time=1.0,
        time_step=0.01
):
    """
    Perform forward dynamics simulation using OpenSim 4.5's ForwardTool and Manager.
    - Only locks: 6 degrees of freedom of the Thorax
    - Unlocks: All other coordinates
    - Sets initial pose to reduce assembly failures
    - Only retains available APIs in ForwardTool
    - Sets integration precision and step size in Manager
    """

    print("=== Starting forward_dynamics execution ===")
    print(f"Model path: {model_path}")
    print(f"Controls file: {controls_file_path}")
    print(f"Results output directory: {result_dir}")
    print(f"Simulation time range: [{start_time}, {end_time}], step size: {time_step}")
    print("================================\n")

    # 1) Load and initialize the model
    osimmodel = osim.Model(model_path)
    state = osimmodel.initSystem()
    print(f"[info] Loaded model: {osimmodel.getName()}")
    print(f"[info] Number of coordinates in model: {osimmodel.getCoordinateSet().getSize()}\n")

    # 2) Create ForwardTool and configure basic settings
    fwd_tool = osim.ForwardTool()
    fwd_tool.setModel(osimmodel)
    fwd_tool.setResultsDir(result_dir)
    fwd_tool.setControlsFileName(controls_file_path)

    # In OpenSim 4.5, APIs like setMaxSteps/setFinalTimeStep don't exist.
    # We can use setSolveForEquilibrium(False) to disable initial equilibrium solving.
    fwd_tool.setSolveForEquilibrium(False)
    # Other fine-grained integration settings can be configured in Manager

    fwd_tool.setStartTime(start_time)
    fwd_tool.setFinalTime(end_time)

    # 3) Only lock thorax coordinates, unlock others
    lock_coords = [
        # Thorax 6DOF:
        "Thorax_rot1",
        "Thorax_rot2",
        "Thorax_rot3",
        "Thorax_tra1",
        "Thorax_tra2",
        "Thorax_tra3"
    ]

    all_coords = osimmodel.getCoordinateSet()
    n_coords = all_coords.getSize()

    # 4) Lock/Unlock - Default is all unlocked, only lock specified thorax coordinates
    for i in range(n_coords):
        coord = all_coords.get(i)
        cname = coord.getName()
        if cname in lock_coords:
            coord.set_locked(True)
            print(f"[info] Locking coordinate: {cname}")
        else:
            coord.set_locked(False)
            print(f"[info] Unlocking coordinate: {cname}")

    # 5) Set initial pose (example)
    def set_coord_value(coord_name, angle_deg):
        """Set initial value for coordinates (units: convert degrees to radians)."""
        try:
            coord_obj = osimmodel.getCoordinateSet().get(coord_name)
            coord_obj.setValue(state, math.radians(angle_deg))
        except Exception as e:
            print(f"[warning] Error setting initial angle for {coord_name}: {e}")

    # Example: Shoulder abduction/elevation, elbow 45° flexion, neutral forearm, radial/ulnar deviation = 0
    set_coord_value("ra_sh_elv_angle", 30.0)
    set_coord_value("ra_sh_elv", 20.0)
    set_coord_value("ra_sh_rot", 0.0)
    set_coord_value("ra_el_e_f", 45.0)
    set_coord_value("ra_wr_sup_pro", 0.0)
    set_coord_value("ra_wr_rd_ud", 0.0)

    # Assemble model - Corrected for proper API call
    try:
        osimmodel.assemble(state)
        print(f"[info] model.assemble() successful, initial pose ready.")
    except Exception as e:
        print(f"[error] model.assemble() failed: {e}")

    # -- 6A) Using ForwardTool.run() --
    print("\n=== (A) Running simulation with ForwardTool.run() ===")
    try:
        fwd_tool.run()
        print("[info] ForwardTool completed. Results written to result directory.")
    except Exception as e:
        print(f"[error] ForwardTool execution failed: {e}")

    # -- 6B) Using Manager + custom integration loop --
    print("\n=== (B) Using Manager integration loop with tqdm visualization ===")
    # Reload
    osimmodel2 = osim.Model(model_path)
    state2 = osimmodel2.initSystem()

    # Apply the same lock/unlock settings
    all_coords2 = osimmodel2.getCoordinateSet()
    for i in range(all_coords2.getSize()):
        coord = all_coords2.get(i)
        if coord.getName() in lock_coords:
            coord.set_locked(True)
        else:
            coord.set_locked(False)

    def set_coord_value_2(coord_name, angle_deg):
        """Set initial values for the second model as well"""
        try:
            coord_obj = osimmodel2.getCoordinateSet().get(coord_name)
            coord_obj.setValue(state2, math.radians(angle_deg))
        except Exception as e:
            print(f"[warning] Error setting initial angle for {coord_name} (second model): {e}")

    set_coord_value_2("ra_sh_elv_angle", 30.0)
    set_coord_value_2("ra_sh_elv", 20.0)
    set_coord_value_2("ra_sh_rot", 0.0)
    set_coord_value_2("ra_el_e_f", 45.0)
    set_coord_value_2("ra_wr_sup_pro", 0.0)
    set_coord_value_2("ra_wr_rd_ud", 0.0)

    try:
        osimmodel2.assemble(state2)
        print("[info] Second model.assemble() successful.")
    except Exception as e:
        print(f"[error] Second assemble failed: {e}")

    # Manager configuration
    manager = osim.Manager(osimmodel2)
    # Integrator precision or max step size can be configured:
    manager.setIntegratorAccuracy(1e-3)  # Reduce integration precision
    try:
        # These settings might not be available in some versions, use try-except to prevent errors
        manager.setIntegratorMinimumStepSize(1e-8)
        manager.setIntegratorMaximumStepSize(time_step)
    except:
        print("[info] Some integrator settings not available, using defaults")

    # Set initial time
    state2.setTime(start_time)
    manager.initialize(state2)

    sim_duration = end_time - start_time
    steps = int(sim_duration / time_step)
    print(f"[info] Custom integration with {steps} steps, each step dt={time_step}")

    try:
        try:
            # Try to equilibrate muscle states before integration, this may help stabilize
            osimmodel2.equilibrateMuscles(state2)
            print("[info] Muscle equilibration completed")
        except:
            print("[info] Muscle equilibration not available, skipping")

        for _ in tqdm(range(steps), desc="Forward Dynamics"):
            current_time = manager.getState().getTime()
            next_time = current_time + time_step
            manager.integrate(next_time)

        final_state = manager.getState()
        print(f"[info] Custom integration completed, t={final_state.getTime():.3f}")
    except Exception as e:
        print(f"[error] Manager integration failed: {e}")

    print("\n=== forward_dynamics execution completed ===\n")


def main():
    """Example main function, modify the following paths before running directly."""
    model_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\RightArmAndHand_scaled.osim"
    controls_file_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\emg_processed\emg_norm.sto"
    result_dir = r"C:\temporary_file\BG_klinik\newPipeline\results\forward_dynamics"

    # Simulation time
    start_time = 0.0
    end_time = 1.0
    time_step = 0.01

    forward_dynamics(
        model_path=model_path,
        controls_file_path=controls_file_path,
        result_dir=result_dir,
        start_time=start_time,
        end_time=end_time,
        time_step=time_step
    )


if __name__ == "__main__":
    main()