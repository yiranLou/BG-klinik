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
    使用OpenSim 4.5的ForwardTool和Manager来进行正向动力学仿真。
    - 解锁：肩带(锁骨/肩胛) + 肩关节3DOF + 肘关节 + 前臂 + 桡尺偏
    - 设置初始姿势减少装配失败
    - ForwardTool中只保留可用API
    - Manager中设置积分精度与步长
    """

    print("=== 开始执行 forward_dynamics ===")
    print(f"模型路径: {model_path}")
    print(f"Controls 文件: {controls_file_path}")
    print(f"结果输出目录: {result_dir}")
    print(f"仿真时间范围: [{start_time}, {end_time}], 步长: {time_step}")
    print("================================\n")

    # 1) 载入模型并初始化
    osimmodel = osim.Model(model_path)
    state = osimmodel.initSystem()
    print(f"[info] 载入模型: {osimmodel.getName()}")
    print(f"[info] 模型中坐标数量: {osimmodel.getCoordinateSet().getSize()}\n")

    # 2) 创建ForwardTool并进行基本配置
    fwd_tool = osim.ForwardTool()
    fwd_tool.setModel(osimmodel)
    fwd_tool.setResultsDir(result_dir)
    fwd_tool.setControlsFileName(controls_file_path)

    # 在OpenSim 4.5中，不存在 setMaxSteps/setFinalTimeStep 等API。
    # 可以使用 setSolveForEquilibrium(False) 来关闭初始平衡解算。
    fwd_tool.setSolveForEquilibrium(False)
    # 其余细粒度积分设置可在 Manager 中配置

    fwd_tool.setStartTime(start_time)
    fwd_tool.setFinalTime(end_time)

    # 3) 解锁坐标列表(肩带+肩+肘+前臂)
    unlock_coords = [
        # 肩带(锁骨/肩胛)常见 DOF:
        "ra_sternoclavicular_r2_d",
        "ra_sternoclavicular_r3_d",
        "ra_acromioclavicular_r2_d",
        "ra_acromioclavicular_r3_d",
        "ra_unrotscap_r2_d",
        "ra_unrotscap_r3_d",
        "ra_unrothum_r1_d",
        "ra_unrothum_r2_d",
        "ra_unrothum_r3_d",

        # 肩关节3DOF
        "ra_sh_elv_angle",
        "ra_sh_elv",
        "ra_sh_rot",

        # 肘
        "ra_el_e_f",

        # 前臂 & 手腕
        "ra_wr_sup_pro",
        "ra_wr_rd_ud"
    ]
    all_coords = osimmodel.getCoordinateSet()
    n_coords = all_coords.getSize()

    # 4) 锁定/解锁
    for i in range(n_coords):
        coord = all_coords.get(i)
        cname = coord.getName()
        if cname in unlock_coords:
            coord.set_locked(False)
            print(f"[info] 解锁坐标: {cname}")
        else:
            coord.set_locked(True)
            print(f"[info] 锁定坐标: {cname}")

    # 5) 设置初始姿势(示例)
    def set_coord_value(coord_name, angle_deg):
        """为坐标设置初始值(单位：角度转弧度)."""
        try:
            coord_obj = osimmodel.getCoordinateSet().get(coord_name)
            coord_obj.setValue(state, math.radians(angle_deg))
        except Exception as e:
            print(f"[warning] 设定 {coord_name} 初始角度时出错: {e}")

    # 示例：肩外展/仰角、肘45度屈曲、前臂中立、桡尺偏=0
    set_coord_value("ra_sh_elv_angle",  30.0)
    set_coord_value("ra_sh_elv",        20.0)
    set_coord_value("ra_sh_rot",         0.0)
    set_coord_value("ra_el_e_f",        45.0)
    set_coord_value("ra_wr_sup_pro",     0.0)
    set_coord_value("ra_wr_rd_ud",       0.0)
    # 若需给锁骨/肩胛 DOF 设值, 同理:
    # set_coord_value("ra_sternoclavicular_r2_d", 5.0)

    # 尝试组装
    try:
        osimmodel.assemble(state)
        print("[info] model.assemble() 成功，初始姿势就绪.")
    except Exception as e:
        print(f"[error] model.assemble() 失败: {e}")

    # -- 6A) 用 ForwardTool.run() --
    print("\n=== (A) 使用ForwardTool.run()进行仿真 ===")
    try:
        fwd_tool.run()
        print("[info] ForwardTool 运行完毕。结果写入到结果目录。")
    except Exception as e:
        print(f"[error] ForwardTool 运行失败: {e}")

    # -- 6B) 使用 Manager + 自定义积分循环 --
    print("\n=== (B) 使用Manager循环积分并tqdm可视化 ===")
    # 重新载入
    osimmodel2 = osim.Model(model_path)
    state2 = osimmodel2.initSystem()

    # 同样解锁
    all_coords2 = osimmodel2.getCoordinateSet()
    for i in range(all_coords2.getSize()):
        coord = all_coords2.get(i)
        if coord.getName() in unlock_coords:
            coord.set_locked(False)
        else:
            coord.set_locked(True)

    def set_coord_value_2(coord_name, angle_deg):
        """第二个模型同样设置初始值"""
        try:
            coord_obj = osimmodel2.getCoordinateSet().get(coord_name)
            coord_obj.setValue(state2, math.radians(angle_deg))
        except Exception as e:
            print(f"[warning] 设定 {coord_name} (第二模型) 初始角度时出错: {e}")

    set_coord_value_2("ra_sh_elv_angle",  30.0)
    set_coord_value_2("ra_sh_elv",        20.0)
    set_coord_value_2("ra_sh_rot",         0.0)
    set_coord_value_2("ra_el_e_f",        45.0)
    set_coord_value_2("ra_wr_sup_pro",     0.0)
    set_coord_value_2("ra_wr_rd_ud",       0.0)

    try:
        osimmodel2.assemble(state2)
        print("[info] 第二次 model.assemble() 成功。")
    except Exception as e:
        print(f"[error] 第二次 assemble 失败: {e}")

    # Manager
    manager = osim.Manager(osimmodel2)
    # 可以配置积分器精度或最大步长:
    manager.setIntegratorAccuracy(1e-3)
    # manager.setIntegratorMaximumStepSize(time_step)

    state2.setTime(start_time)
    manager.initialize(state2)

    sim_duration = end_time - start_time
    steps = int(sim_duration / time_step)
    print(f"[info] 自定义积分共 {steps} 步，每步 dt={time_step}")

    try:
        for _ in tqdm(range(steps), desc="Forward Dynamics"):
            current_time = manager.getState().getTime()
            next_time = current_time + time_step
            manager.integrate(next_time)

        final_state = manager.getState()
        print(f"[info] 自定义积分结束, t={final_state.getTime():.3f}")
    except Exception as e:
        print(f"[error] Manager 积分失败: {e}")

    print("\n=== forward_dynamics执行完毕 ===\n")


def main():
    """示例 main 函数，修改以下路径后直接运行。"""
    model_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\RightArmAndHand_scaled.osim"
    controls_file_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\emg_processed\emg_norm.sto"
    result_dir = r"C:\temporary_file\BG_klinik\newPipeline\results\forward_dynamics"

    # 仿真时间
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
