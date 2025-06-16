import os
import opensim as osim
import matplotlib.pyplot as plt

# Get project root directory
def get_project_root():
    """Get the project root directory (newPipeline)"""
    current_file = os.path.abspath(__file__)
    # Navigate up from src/opensim/inverse_kinematics.py to newPipeline/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root

PROJECT_ROOT = get_project_root()

# =========================================================
# STEP 0: User settings that can be modified here
# =========================================================
IK_SETUP_FILE = os.path.join(PROJECT_ROOT, "config", "setup", "IK_setup_N10.xml")
SCALED_MODEL_FILE = os.path.join(PROJECT_ROOT, "config", "models", "ms_arm_and_hand-main", "AAH Model", "RightArmAndHand_scaled.osim")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "Ik_result", "ikTOtest")

# Whether to use custom time range (overrides startTime and endTime in IK_setup.xml)
USE_CUSTOM_TIME_RANGE = False
CUSTOM_START_TIME = 0.0
CUSTOM_END_TIME = 1.0

# After IK runs, assume we want to read the error file named "_ik_marker_errors.sto"
IK_MARKER_ERRORS_FILE = "_ik_marker_errors.sto"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"IK setup file: {IK_SETUP_FILE}")
print(f"Scaled model file: {SCALED_MODEL_FILE}")
print(f"Results directory: {RESULTS_DIR}")

# =========================================================
# STEP 1: Read IK configuration and load model
# =========================================================
print("Loading IK setup from:", IK_SETUP_FILE)
ik_tool = osim.InverseKinematicsTool(IK_SETUP_FILE)

print("Loading scaled model from:", SCALED_MODEL_FILE)
model = osim.Model(SCALED_MODEL_FILE)

ik_tool.setModel(model)

original_start_time = ik_tool.getStartTime()
original_end_time = ik_tool.getEndTime()
print(f"Original IK time range in setup file: [{original_start_time}, {original_end_time}]")

if USE_CUSTOM_TIME_RANGE:
    ik_tool.setStartTime(CUSTOM_START_TIME)
    ik_tool.setEndTime(CUSTOM_END_TIME)
    print(f"Overriding IK time range to: [{CUSTOM_START_TIME}, {CUSTOM_END_TIME}]")
else:
    print("Using time range from the IK setup file.")

# =========================================================
# STEP 2: Configure results output directory and file name
# =========================================================
ik_tool.setResultsDir(RESULTS_DIR)
ik_tool.setOutputMotionFileName("ik_result.sto")

# =========================================================
# STEP 3: Run Inverse Kinematics
# =========================================================
print("Running Inverse Kinematics...")
ik_tool.run()
print("IK finished.\n")

# =========================================================
# STEP 4: Read and plot IK marker error results (test)
# =========================================================
errors_sto_path = os.path.join(RESULTS_DIR, IK_MARKER_ERRORS_FILE)
print("Attempting to read IK marker errors from:", errors_sto_path)

# --- Test logic for reading + plotting ---
try:
    # 1) Read .sto file using TableProcessor
    table_processor = osim.TableProcessor(errors_sto_path)
    table_errors = table_processor.process()

    # 2) Print column labels
    column_labels = table_errors.getColumnLabels()
    print("Columns found:", column_labels)

    # 3) Get time or frame number sequence
    x_values = table_errors.getIndependentColumn()

    # 4) Automatically detect common error columns, use first column if none found
    columns_to_plot = []
    for candidate in ["total_squared_error", "marker_error_RMS", "marker_error_max"]:
        if candidate in column_labels:
            columns_to_plot.append(candidate)

    if not columns_to_plot:  # No standard columns found, plot the first column
        print("No standard error columns found; will just plot the first column.")
        columns_to_plot = [column_labels[0]]

    # 5) Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("IK Marker Errors Test Plot")

    for c in columns_to_plot:
        data_column = table_errors.getDependentColumn(c)
        ax.plot(x_values, data_column.to_numpy(), label=c)

    ax.set_xlabel("Time or Frame")
    ax.set_ylabel("Error (or Value)")
    ax.grid(True)
    ax.legend()
    plt.show()

    print("Success! Plotted data from:", errors_sto_path)

except Exception as e:
    print("Could not load the STO file or plot due to error:\n", e)

print("\nDone. All IK outputs are in:\n", RESULTS_DIR)