import opensim as osim
import matplotlib.pyplot as plt

# =========================================================
# STEP 0: 用户可在此处修改一些通用设置
# =========================================================
IK_SETUP_FILE = r"C:\temporary_file\BG_klinik\newPipeline\config\setup\IK_setup_N10.xml"
SCALED_MODEL_FILE = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\RightArmAndHand_scaled.osim"
RESULTS_DIR = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\Ik_result\ikTOtest"

# 是否使用自定义时间范围 (覆盖 IK_setup.xml 里的 startTime 与 endTime)
USE_CUSTOM_TIME_RANGE = False
CUSTOM_START_TIME = 0.0
CUSTOM_END_TIME = 1.0

# IK 运行后，假设我们想读取的误差文件叫做 "_ik_marker_errors.sto"
IK_MARKER_ERRORS_FILE = r"_ik_marker_errors.sto"

# =========================================================
# STEP 1: 读取 IK 配置并加载模型
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
# STEP 2: 配置结果输出目录与文件名
# =========================================================
ik_tool.setResultsDir(RESULTS_DIR)
ik_tool.setOutputMotionFileName("ik_result.sto")

# =========================================================
# STEP 3: 运行 Inverse Kinematics
# =========================================================
print("Running Inverse Kinematics...")
ik_tool.run()
print("IK finished.\n")

# =========================================================
# STEP 4: 读取并绘制 IK 生成的标记误差结果 (测试)
# =========================================================
errors_sto_path = f"{RESULTS_DIR}\\{IK_MARKER_ERRORS_FILE}"
print("Attempting to read IK marker errors from:", errors_sto_path)

# --- 以下为测试读取 + 绘图逻辑 ---
try:
    # 1) 用 TableProcessor 读取 .sto
    table_processor = osim.TableProcessor(errors_sto_path)
    table_errors = table_processor.process()

    # 2) 打印列标签
    column_labels = table_errors.getColumnLabels()
    print("Columns found:", column_labels)

    # 3) 取时间或帧号序列
    x_values = table_errors.getIndependentColumn()

    # 4) 自动检测常见误差列，如未找到则默认取第一列
    columns_to_plot = []
    for candidate in ["total_squared_error", "marker_error_RMS", "marker_error_max"]:
        if candidate in column_labels:
            columns_to_plot.append(candidate)

    if not columns_to_plot:  # 没发现常见列，就画第一个列
        print("No standard error columns found; will just plot the first column.")
        columns_to_plot = [column_labels[0]]

    # 5) 绘图
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
