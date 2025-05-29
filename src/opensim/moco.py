import opensim as osim
import os
import time
import datetime
import sys


def print_progress(message):
    """Print progress information with timestamp"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")
    # Force flush output buffer to ensure immediate display
    sys.stdout.flush()


# Define optimization iteration callback class
class ProgressCallback():
    def __init__(self):
        self.last_time = time.time()
        self.iteration = 0
        self.start_time = time.time()

    def __call__(self, step, elapsed_time, objective, constraint_violation):
        self.iteration += 1
        current_time = time.time()
        if current_time - self.last_time >= 10.0:  # Update every 10 seconds
            elapsed = current_time - self.start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            print_progress(f"Optimization iteration: {self.iteration}, Objective: {objective:.6f}, "
                           f"Constraint violation: {constraint_violation:.6f}, "
                           f"Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            self.last_time = current_time
        return True  # Continue optimization


# ——— Moco settings ———
MODEL_FILE = r"C:\temporary_file\BG_klinik\opensimarmmodel-main\opensimarmmodel-main\model\right\MOBL_ARMS_right_scaled.osim"
TRC_FILE = "C:/temporary_file/BG_klinik/newPipeline/data/processed/ROM_Ellenbogenflex_R 1.trc"
RESULTS_DIR = "C:/temporary_file/BG_klinik/newPipeline/data/processed/Moco_result"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check files
print_progress("Starting execution...")
for f in (MODEL_FILE, TRC_FILE):
    if not os.path.isfile(f):
        raise FileNotFoundError(f"File not found: {f}")

print_progress("Using Moco Track method...")

# Create and configure ModelProcessor to simplify muscle model
print_progress("Loading model...")
model_processor = osim.ModelProcessor(MODEL_FILE)

# Add muscle processing operators to simplify the model
print_progress("Applying muscle model simplification operations...")
model_processor.append(osim.ModOpIgnoreTendonCompliance())
model_processor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
model_processor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
model_processor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

# Process model and unlock coordinates
print_progress("Processing model and unlocking coordinates...")
processed_model = model_processor.process()
for coord in processed_model.getCoordinateSet():
    coord.set_locked(False)

# Save modified model
temp_model_file = os.path.join(RESULTS_DIR, "processed_model.osim")
processed_model.printToXML(temp_model_file)
print_progress(f"Processed model saved to {temp_model_file}")

# Load marker data to get time range
print_progress("Loading marker data...")
marker_data = osim.MarkerData(TRC_FILE)
t0 = marker_data.getStartFrameTime()
tf = marker_data.getLastFrameTime()
print_progress(f"Marker data time range: {t0:.2f}s to {tf:.2f}s")

# Use MocoTrack for tracking
print_progress("Creating MocoTrack object...")
track = osim.MocoTrack()
track.setName("RightArm_MocoTrack")

# Set processed model
model_processor = osim.ModelProcessor(temp_model_file)
track.setModel(model_processor)

# Set marker file reference
print_progress("Setting marker references...")
track.setMarkersReferenceFromTRC(TRC_FILE)
track.set_allow_unused_references(True)

# Set other MocoTrack options
track.set_track_reference_position_derivatives(True)

# Create and initialize MocoStudy
print_progress("Initializing MocoStudy...")
start_time = time.time()
study = track.initialize()
end_time = time.time()
print_progress(f"MocoStudy initialization complete, took {end_time - start_time:.2f} seconds")

# Get problem and set time range
print_progress("Configuring optimization problem...")
problem = study.updProblem()
problem.setTimeBounds(t0, tf)

# Use simplified solver configuration
print_progress("Configuring solver...")
solver = osim.MocoCasADiSolver()
solver.resetProblem(problem)
solver.set_num_mesh_intervals(20)  # Reduce mesh points to speed up solving
solver.set_optim_max_iterations(300)  # Reduce maximum iterations

# Configure solver output frequency
try:
    # Try to set solver verbosity
    solver.set_verbosity(2)  # Increase output verbosity
    print_progress("Solver verbose output enabled")
except:
    print_progress("Note: Unable to set solver verbosity, will use default")

try:
    # Try to set optimization output frequency
    solver.setPerformanceOutputFrequency(5)  # Output performance info every 5 iterations
except:
    print_progress("Note: Unable to set output frequency, will use default")

study.updSolver().resetProblem(problem)  # Get solver with updSolver() and reset problem

# Solve and save results
print_progress("Starting optimization (this may take minutes to hours, depending on model complexity)...")
print_progress("Start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()

# Create progress callback instance
try:
    callback = ProgressCallback()
    # Try to attach callback to solver
    solver.set_optim_callback(callback)
    print_progress("Optimization progress callback enabled")
except Exception as e:
    print_progress(f"Note: Unable to set optimization callback: {str(e)}")
    print_progress("Will use standard output")

# Output prompt
print_progress("-------- Starting optimization process (please be patient) --------")
print_progress("Note: If no progress is shown, please wait for final results, optimization is still in progress...")

try:
    # Define a background progress monitoring function
    def monitor_progress():
        monitor_start = time.time()
        iteration_counter = 0

        while True:
            time.sleep(30)  # Check every 30 seconds
            current = time.time()
            elapsed = current - monitor_start
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            iteration_counter += 1
            print_progress(f"[Monitor] Optimization still in progress - Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

            # Update progress file for external monitoring
            with open(os.path.join(RESULTS_DIR, "optimization_progress.txt"), "a") as f:
                f.write(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iteration checkpoint {iteration_counter}: Running for {int(hours)}h {int(minutes)}m\n")


    # Try to start monitoring in background thread
    import threading

    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True  # Set as daemon thread so it auto-terminates when main program exits
    monitor_thread.start()
    print_progress("Background progress monitoring started")
except:
    print_progress("Unable to start background progress monitoring")

try:
    solution = study.solve()
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_progress(f"Optimization solve complete! Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    solution_file = os.path.join(RESULTS_DIR, "moco_solution.sto")
    solution.write(solution_file)
    print_progress(f"✅ Moco solution written to: {solution_file}")

    # Display optimization statistics
    print_progress(f"Solve status: {solution.getStatus()}")
    print_progress(f"Success value: {solution.getSuccess()}")
    print_progress(f"Objective function value: {solution.getObjective()}")

    # Use solution to analyze results
    print_progress("Analyzing motion results...")
    try:
        report = osim.report.Report(processed_model, solution_file, RESULTS_DIR)
        report.generate()
        print_progress(f"✅ Analysis report generated, results in: {RESULTS_DIR}")
    except Exception as e:
        print_progress(f"Error generating report: {e}")
        print_progress(f"But solution has been saved to: {solution_file}")
except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_progress(f"Error during solving process! Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print_progress(f"Error message: {str(e)}")
    print_progress("Attempting troubleshooting...")

    # Try using a simpler IK method
    print_progress("Attempting to use Inverse Kinematics (IK) method...")
    try:
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setName("IK_Analysis")
        ik_tool.setModel(processed_model)
        ik_tool.setMarkerDataFileName(TRC_FILE)
        ik_tool.set_report_errors(True)
        ik_tool.setStartTime(t0)
        ik_tool.setEndTime(tf)
        ik_tool.set_output_motion_file(os.path.join(RESULTS_DIR, "ik_result.mot"))
        ik_tool.setResultsDir(RESULTS_DIR)
        ik_tool.run()
        print_progress(f"✅ IK analysis complete, results saved to: {os.path.join(RESULTS_DIR, 'ik_result.mot')}")
    except Exception as e2:
        print_progress(f"IK method also failed: {str(e2)}")

print_progress("Script execution complete.")
print_progress("End time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))