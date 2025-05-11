import opensim as osim
import os
import time
import datetime
import sys


def print_progress(message):
    """打印带有时间戳的进度信息"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")
    # 强制刷新输出缓冲区，确保立即显示
    sys.stdout.flush()


# 定义优化迭代回调类
class ProgressCallback():
    def __init__(self):
        self.last_time = time.time()
        self.iteration = 0
        self.start_time = time.time()

    def __call__(self, step, elapsed_time, objective, constraint_violation):
        self.iteration += 1
        current_time = time.time()
        if current_time - self.last_time >= 10.0:  # 每10秒更新一次
            elapsed = current_time - self.start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            print_progress(f"优化迭代: {self.iteration}, 目标值: {objective:.6f}, "
                           f"约束违反: {constraint_violation:.6f}, "
                           f"运行时间: {int(hours)}小时 {int(minutes)}分 {seconds:.2f}秒")
            self.last_time = current_time
        return True  # 继续优化


# ——— Moco settings ———
MODEL_FILE = r"C:\temporary_file\BG_klinik\opensimarmmodel-main\opensimarmmodel-main\model\right\MOBL_ARMS_right_scaled.osim"
TRC_FILE = "C:/temporary_file/BG_klinik/newPipeline/data/processed/ROM_Ellenbogenflex_R 1.trc"
RESULTS_DIR = "C:/temporary_file/BG_klinik/newPipeline/data/processed/Moco_result"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 检查文件
print_progress("开始执行...")
for f in (MODEL_FILE, TRC_FILE):
    if not os.path.isfile(f):
        raise FileNotFoundError(f"找不到文件：{f}")

print_progress("使用 Moco Track 方法...")

# 创建并配置 ModelProcessor 简化肌肉模型
print_progress("加载模型...")
model_processor = osim.ModelProcessor(MODEL_FILE)

# 添加肌肉处理操作符来简化模型
print_progress("应用肌肉模型简化操作...")
model_processor.append(osim.ModOpIgnoreTendonCompliance())
model_processor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
model_processor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
model_processor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

# 处理模型并解锁坐标
print_progress("处理模型并解锁坐标...")
processed_model = model_processor.process()
for coord in processed_model.getCoordinateSet():
    coord.set_locked(False)

# 保存修改后的模型
temp_model_file = os.path.join(RESULTS_DIR, "processed_model.osim")
processed_model.printToXML(temp_model_file)
print_progress(f"处理后的模型已保存至 {temp_model_file}")

# 加载标记数据获取时间范围
print_progress("加载标记数据...")
marker_data = osim.MarkerData(TRC_FILE)
t0 = marker_data.getStartFrameTime()
tf = marker_data.getLastFrameTime()
print_progress(f"标记数据时间范围: {t0:.2f}s 到 {tf:.2f}s")

# 使用 MocoTrack 进行追踪
print_progress("创建 MocoTrack 对象...")
track = osim.MocoTrack()
track.setName("RightArm_MocoTrack")

# 设置已处理的模型
model_processor = osim.ModelProcessor(temp_model_file)
track.setModel(model_processor)

# 设置标记文件参考
print_progress("设置标记参考...")
track.setMarkersReferenceFromTRC(TRC_FILE)
track.set_allow_unused_references(True)

# 设置其他MocoTrack选项
track.set_track_reference_position_derivatives(True)

# 创建并初始化 MocoStudy
print_progress("初始化 MocoStudy...")
start_time = time.time()
study = track.initialize()
end_time = time.time()
print_progress(f"MocoStudy 初始化完成，用时 {end_time - start_time:.2f} 秒")

# 获取问题并设置时间范围
print_progress("配置优化问题...")
problem = study.updProblem()
problem.setTimeBounds(t0, tf)

# 使用简化的求解器配置
print_progress("配置求解器...")
solver = osim.MocoCasADiSolver()
solver.resetProblem(problem)
solver.set_num_mesh_intervals(20)  # 减少网格点数以加快求解
solver.set_optim_max_iterations(300)  # 减少最大迭代次数

# 配置求解器输出频率
try:
    # 尝试设置求解器详细程度
    solver.set_verbosity(2)  # 增加输出详细程度
    print_progress("已设置求解器详细输出")
except:
    print_progress("注意: 无法设置求解器详细程度, 将使用默认值")

try:
    # 尝试设置优化输出频率
    solver.setPerformanceOutputFrequency(5)  # 每5次迭代输出一次性能信息
except:
    print_progress("注意: 无法设置输出频率, 将使用默认值")

study.updSolver().resetProblem(problem)  # 使用updSolver()获取求解器并重置问题

# 求解并保存结果
print_progress("开始优化求解 (这可能需要几分钟到几小时，取决于模型复杂度)...")
print_progress("开始时间: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()

# 创建进度回调实例
try:
    callback = ProgressCallback()
    # 尝试附加回调到求解器
    solver.set_optim_callback(callback)
    print_progress("已启用优化进度回调")
except Exception as e:
    print_progress(f"注意: 无法设置优化回调: {str(e)}")
    print_progress("将使用标准输出")

# 输出提示
print_progress("-------- 开始优化过程 (请耐心等待) --------")
print_progress("注意: 如果没有显示进度，请等待最终结果，优化仍在进行中...")

try:
    # 定义一个后台进度监控函数
    def monitor_progress():
        monitor_start = time.time()
        iteration_counter = 0

        while True:
            time.sleep(30)  # 每30秒检查一次
            current = time.time()
            elapsed = current - monitor_start
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            iteration_counter += 1
            print_progress(f"[监控] 优化仍在进行中 - 已运行: {int(hours)}小时 {int(minutes)}分 {seconds:.2f}秒")

            # 更新进度文件，方便外部监控
            with open(os.path.join(RESULTS_DIR, "optimization_progress.txt"), "a") as f:
                f.write(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 迭代检查点 {iteration_counter}: 已运行 {int(hours)}小时 {int(minutes)}分钟\n")


    # 尝试在后台线程中启动监控
    import threading

    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True  # 设置为守护线程，这样主程序退出时它会自动终止
    monitor_thread.start()
    print_progress("已启动后台进度监控")
except:
    print_progress("无法启动后台进度监控")

try:
    solution = study.solve()
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_progress(f"优化求解完成! 用时: {int(hours)}小时 {int(minutes)}分 {seconds:.2f}秒")

    solution_file = os.path.join(RESULTS_DIR, "moco_solution.sto")
    solution.write(solution_file)
    print_progress(f"✅ Moco solution 写入：{solution_file}")

    # 显示优化统计信息
    print_progress(f"求解状态: {solution.getStatus()}")
    print_progress(f"成功值: {solution.getSuccess()}")
    print_progress(f"目标函数值: {solution.getObjective()}")

    # 使用解决方案分析结果
    print_progress("分析运动结果...")
    try:
        report = osim.report.Report(processed_model, solution_file, RESULTS_DIR)
        report.generate()
        print_progress(f"✅ 分析报告生成完毕，结果位于：{RESULTS_DIR}")
    except Exception as e:
        print_progress(f"生成报告时出错: {e}")
        print_progress(f"但解决方案已保存到: {solution_file}")
except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_progress(f"求解过程中出现错误! 用时: {int(hours)}小时 {int(minutes)}分 {seconds:.2f}秒")
    print_progress(f"错误信息: {str(e)}")
    print_progress("尝试进行故障修复...")

    # 尝试采用更简单的IK方法
    print_progress("尝试使用反向运动学(IK)方法...")
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
        print_progress(f"✅ IK 分析完成，结果保存在: {os.path.join(RESULTS_DIR, 'ik_result.mot')}")
    except Exception as e2:
        print_progress(f"IK方法也失败了: {str(e2)}")

print_progress("脚本执行完毕。")
print_progress("最终时间: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))