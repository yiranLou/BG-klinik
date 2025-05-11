"""
C3D文件转换模块。
处理C3D文件与OpenSim格式之间的转换。
"""

import os
import numpy as np
import pyc3dserver as c3d
from scipy.spatial.transform import Rotation as R

# 项目根目录的相对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 处理后数据的输出目录
PROCESSED_DIR = r"/newPipeline/data/processed"

def ensure_dir_exists(directory):
    """
    确保目录存在，如果不存在则创建。

    参数:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def c3dtotrc(src_c3d, output_dir=None):
    """
    将C3D文件转换为TRC和MOT格式。

    参数:
        src_c3d (str): 源C3D文件路径
        output_dir (str, optional): 输出目录，默认为处理后数据目录

    返回:
        tuple: (trc文件路径, mot文件路径)
    """
    try:
        print(f"[DEBUG] c3dtotrc called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        if output_dir is None:
            output_dir = PROCESSED_DIR

        print(f"[DEBUG] Using output directory: {output_dir}")
        ensure_dir_exists(output_dir)

        src_file_name = os.path.basename(src_c3d)
        file_name_without_ext = os.path.splitext(src_file_name)[0]

        trc_path = os.path.join(output_dir, file_name_without_ext + ".trc")
        mot_path = os.path.join(output_dir, file_name_without_ext + ".mot")

        print(f"[DEBUG] Will write TRC to: {trc_path}")
        print(f"[DEBUG] Will write MOT to: {mot_path}")

        # 创建C3Dserver接口
        itf = c3d.c3dserver()
        # 初始化日志
        c3d.init_logger(logger_lvl='DEBUG', c_hdlr_lvl='DEBUG', f_hdlr_lvl='DEBUG', f_hdlr_f_path=None)
        # 打开C3D文件
        ret = c3d.open_c3d(itf, src_c3d)

        # 定义原始坐标轴和目标坐标轴
        axes_src = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axes_tgt = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

        # 计算旋转矩阵
        rot_ret = R.align_vectors(a=axes_src, b=axes_tgt)
        rot_obj = rot_ret[0]
        trf = rot_obj.as_matrix()

        # 导出 TRC 与 MOT
        c3d.export_trc(itf, trc_path, rot_mat=trf)
        c3d.export_mot(itf, mot_path, rot_mat=trf)

        # 关闭C3D
        ret = c3d.close_c3d(itf)

        print(f"已将C3D文件转换为TRC和MOT格式，保存在: {output_dir}")
        return trc_path, mot_path
    except Exception as e:
        print(f"转换C3D文件时出错: {e}")
        return None, None


def c3d_gelenkm(rechts, src_c3d, output_dir=None):
    """
    向C3D文件添加肘部和手部标记点。

    参数:
        rechts (bool): True表示右侧，False表示左侧
        src_c3d (str): 源C3D文件路径
        output_dir (str, optional): 输出目录，默认为处理后数据目录

    返回:
        str: 修改后的C3D文件路径
    """
    try:
        print(f"[DEBUG] c3d_gelenkm called. src_c3d = {src_c3d}, rechts = {rechts}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        if output_dir is None:
            output_dir = PROCESSED_DIR

        print(f"[DEBUG] Using output directory: {output_dir}")
        ensure_dir_exists(output_dir)

        src_file_name = os.path.basename(src_c3d)
        file_name_without_ext = os.path.splitext(src_file_name)[0]
        file_ext = os.path.splitext(src_file_name)[1]

        output_file = os.path.join(output_dir, file_name_without_ext + "_gm" + file_ext)
        print(f"[DEBUG] Will write new C3D (with added markers) to: {output_file}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        dict_markers = c3d.get_dict_markers(itf, blocked_nan=True, resid=True, mask=False, desc=False, frame=False, time=False)
        dict_mkr_pos = dict_markers['DATA']['POS']

        if rechts:
            # 右侧
            new_mkr_name = 'REB'
            new_mkr_pos = (dict_mkr_pos['REL'] + dict_mkr_pos['REM']) * 0.5
            c3d.add_marker(itf, new_mkr_name, new_mkr_pos, mkr_resid=None, mkr_desc='REB', log=False)

            new_mkr_hw = 'HWM'
            hw_mkr_pos = (dict_mkr_pos['RRS'] + dict_mkr_pos['RUS']) * 0.5
            c3d.add_marker(itf, new_mkr_hw, hw_mkr_pos, mkr_resid=None, mkr_desc='HWM', log=False)
        else:
            # 左侧
            new_mkr_name = 'LEB'
            new_mkr_pos = (dict_mkr_pos['LEL'] + dict_mkr_pos['LEM']) * 0.5
            c3d.add_marker(itf, new_mkr_name, new_mkr_pos, mkr_resid=None, mkr_desc='LEB', log=False)

            new_mkr_hw = 'HWM'
            hw_mkr_pos = (dict_mkr_pos['LRS'] + dict_mkr_pos['LUS']) * 0.5
            c3d.add_marker(itf, new_mkr_hw, hw_mkr_pos, mkr_resid=None, mkr_desc='HWM', log=False)

        # 保存修改后的C3D文件
        ret = c3d.save_c3d(itf, output_file, compress_param_blocks=True, log=True)
        ret = c3d.close_c3d(itf, log=True)

        print(f"已添加标记点，保存在: {output_file}")
        return output_file
    except Exception as e:
        print(f"添加标记点到C3D文件时出错: {e}")
        return None


def extract_marker_data(src_c3d):
    """
    从C3D文件中提取标记点数据。

    参数:
        src_c3d (str): 源C3D文件路径

    返回:
        dict: 标记点数据字典
    """
    try:
        print(f"[DEBUG] extract_marker_data called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        marker_names = c3d.get_marker_names(itf)
        marker_data = {}

        for marker in marker_names:
            marker_data[marker] = c3d.get_marker_data(itf, marker)

        ret = c3d.close_c3d(itf)

        return marker_data
    except Exception as e:
        print(f"提取标记点数据时出错: {e}")
        return None


def extract_analog_data(src_c3d):
    """
    从C3D文件中提取模拟通道数据。

    参数:
        src_c3d (str): 源C3D文件路径

    返回:
        dict: 模拟通道数据字典
    """
    try:
        print(f"[DEBUG] extract_analog_data called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        analog_names = c3d.get_analog_names(itf)
        analog_data = {}
        times = c3d.get_analog_times(itf)

        analog_data['time'] = times

        for analog in analog_names:
            analog_data[analog] = c3d.get_analog_data_unscaled(itf, analog)

        ret = c3d.close_c3d(itf)

        return analog_data
    except Exception as e:
        print(f"提取模拟通道数据时出错: {e}")
        return None


def get_c3d_metadata(src_c3d):
    """
    获取C3D文件的元数据信息。

    参数:
        src_c3d (str): 源C3D文件路径

    返回:
        dict: 元数据字典
    """
    try:
        print(f"[DEBUG] get_c3d_metadata called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        metadata = {
            # 'analog_rate': c3d.get_analog_rate(itf),
            # 'video_rate': c3d.get_video_rate(itf),
            'marker_count': len(c3d.get_marker_names(itf)),
            'analog_count': len(c3d.get_analog_names(itf)),
            'first_frame': c3d.get_first_frame(itf),
            'last_frame': c3d.get_last_frame(itf)
        }

        ret = c3d.close_c3d(itf)

        return metadata
    except Exception as e:
        print(f"获取C3D元数据时出错: {e}")
        return None


def add_column_to_file(file_path, new_column_name, new_column_data):
    """
    向文件添加新列。

    参数:
        file_path (str): 要修改的文件路径
        new_column_name (list): 要添加的列名列表
        new_column_data (list of lists): 每列的数据
    """
    try:
        print(f"[DEBUG] add_column_to_file called. file_path = {file_path}")
        print(f"[DEBUG] Does file exist on disk? {os.path.exists(file_path)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[DEBUG] 创建输出目录: {output_dir}")
        
        # 如果文件不存在，创建一个新文件
        if not os.path.exists(file_path):
            print(f"[DEBUG] 文件不存在，创建新文件: {file_path}")
            with open(file_path, 'w') as file:
                # 写入文件头
                file.write("nRows=0\n")
                file.write("nColumns=0\n")
                file.write("endheader\n")
                file.write("\t".join(new_column_name) + "\n")
            
            # 重新打开文件进行读取
            with open(file_path, 'r') as file:
                lines = file.readlines()
        else:
            # 如果文件存在，读取现有内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

        metadata_lines = []
        data_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("endheader"):
                in_header = False
                continue
            if in_header:
                metadata_lines.append(stripped)
            else:
                data_lines.append(stripped)

        metadata = {line.split('=')[0].strip(): line.split('=')[1].strip()
                    for line in metadata_lines if '=' in line}

        nRows = len(data_lines)
        max_rows = max(len(col) for col in new_column_data)

        column_headers = new_column_name

        for i, col_data in enumerate(new_column_data):
            if len(col_data) < max_rows:
                new_column_data[i].extend([None] * (max_rows - len(col_data)))

        nColumns = len(new_column_name)

        metadata["nRows"] = str(max_rows)
        metadata["nColumns"] = str(nColumns)

        fixed_file = []
        fixed_file.extend([f"{key}={value}" for key, value in metadata.items()])
        fixed_file.append("endheader")
        fixed_file.append("\t".join(column_headers))

        for i in range(max_rows):
            row_data = [str(new_column_data[j][i]) if new_column_data[j][i] is not None else ""
                        for j in range(len(new_column_data))]
            fixed_file.append("\t".join(row_data))

        with open(file_path, 'w') as file:
            file.write("\n".join(fixed_file) + "\n")

        print(f"文件头已重建，新列已成功添加。")
    except Exception as e:
        print(f"添加列时出错: {e}")
        import traceback
        traceback.print_exc()


# 示例使用
if __name__ == '__main__':
    # 使用相对路径示例
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'N10_n')
    processed_data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    ensure_dir_exists(processed_data_dir)

    # 示例文件路径
    example_c3d = os.path.join(raw_data_dir, 'right', 'ROM_Ellenbogenflex_R 1.c3d')

    # 调试：转换C3D到TRC和MOT
    print("\n===== 测试C3D到TRC和MOT的转换 =====")
    trc_path, mot_path = c3dtotrc(example_c3d, processed_data_dir)

    # 检查返回值是否为 None
    if trc_path is None or mot_path is None:
        print("转换函数返回了 None，可能在转换过程中发生了错误。")
    else:
        print(f"转换函数返回值:\n TRC路径: {trc_path}\n MOT路径: {mot_path}")

        # 进一步检查文件在不在磁盘上
        trc_exists = os.path.exists(trc_path)
        mot_exists = os.path.exists(mot_path)

        if trc_exists and mot_exists:
            print("调试检查：TRC 文件与 MOT 文件均已在磁盘上找到，转换成功。")
        else:
            print("调试检查：有文件未在磁盘上找到。请检查下列状态：")
            print(f"TRC文件存在：{trc_exists}, MOT文件存在：{mot_exists}")

    # 测试添加标记点功能
    print("\n===== 测试添加标记点功能 =====")
    output_c3d = c3d_gelenkm(True, example_c3d, processed_data_dir)
    if output_c3d is None:
        print("添加标记点函数返回了 None，可能在处理过程中发生了错误。")
    else:
        print(f"添加标记点后的文件保存在: {output_c3d}")

        # 检查文件是否在磁盘上
        if os.path.exists(output_c3d):
            print(f"调试检查：添加标记点后的文件已在磁盘上找到，处理成功。")
        else:
            print(f"调试检查：添加标记点后的文件未在磁盘上找到。")

    # 测试提取标记点数据
    print("\n===== 测试提取标记点数据 =====")
    marker_data = extract_marker_data(example_c3d)
    if marker_data is not None:
        print(f"成功提取了 {len(marker_data)} 个标记点的数据。")
    else:
        print("提取标记点数据失败。")

    # 测试提取模拟通道数据
    print("\n===== 测试提取模拟通道数据 =====")
    analog_data = extract_analog_data(example_c3d)
    if analog_data is not None:
        print(f"成功提取了 {len(analog_data)-1} 个模拟通道的数据。")  # -1 是因为包含了 'time'
    else:
        print("提取模拟通道数据失败。")

    # 测试获取元数据
    print("\n===== 测试获取C3D元数据 =====")
    metadata = get_c3d_metadata(example_c3d)
    if metadata is not None:
        print("C3D文件元数据:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("获取C3D元数据失败。")
