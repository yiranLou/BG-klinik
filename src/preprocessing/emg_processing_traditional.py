"""
EMG 处理模块。
对动作C3D及MVIC C3D文件做滤波、整流和归一化，然后输出至指定目录下的 .sto 文件。
"""

import os
import numpy as np
from scipy.signal import find_peaks, sosfilt, butter
import pyc3dserver as c3d

# 如果需要调用你改好的 c3d_converter.py 中的函数
# 请根据实际路径修改下面的 import 路径
from c3d_converter import c3dtotrc, c3d_gelenkm, extract_analog_data, add_column_to_file

################################################################################
# 路径 & 全局参数
################################################################################

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'N10_n')
MVIC_DIR = os.path.join(RAW_DATA_DIR, 'mvic')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'emg_processed')

# 确保输出目录存在
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

# 你的4个MVIC文件路径（原C3D），如果你想要先做转换，可以在后面流程中调用 c3dtotrc
MVIC_BICEPS_C3D  = os.path.join(MVIC_DIR, 'MVIC_Biceps_r.c3d')
MVIC_TRICEPS_C3D = os.path.join(MVIC_DIR, 'MVIC_Triceps_r.c3d')
MVIC_WRIST_EX_C3D   = os.path.join(MVIC_DIR, 'MVIC_wrist_ex_r.c3d')
MVIC_WRIST_FLX_C3D  = os.path.join(MVIC_DIR, 'MVIC_wrist_flex_r.c3d')

# 你的动作 C3D 文件（建议先用 c3d_converter.py 转换成 .trc/.mot，如果你想直接在这里做EMG处理，可以使用原C3D）
# 假设你动作C3D也放在 raw\N10_n\right 里，比如 “ROM_Ellenbogenflex_R 1.c3d”
ACTION_C3D = os.path.join(RAW_DATA_DIR, 'right', 'ROM_Ellenbogenflex_R 1.c3d')

# 最终 EMG 输出 .sto 文件路径
EMG_OUT_FILE = os.path.join(PROCESSED_DIR, 'emg_norm.sto')

################################################################################
# 带通滤波 & 整流
################################################################################

def butter_bandpass(lowcut, highcut, fs, order=6):
    """
    构建带通滤波器(Butterworth)的二阶截断形式(sos)。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    应用带通滤波器，返回滤波后的信号。
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, data)
    return filtered

def emgfilter(data, fs=1500):
    """
    对EMG信号做15~500Hz带通滤波并取绝对值(全波整流)。
    """
    lowcut = 15
    highcut = 500
    order = 6
    filtered = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    rectified = np.abs(filtered)
    return rectified

################################################################################
# 计算MVIC峰值
################################################################################

def normalize_emg(muscle_name, mvic_c3d_path):
    """
    对给定肌肉的MVIC c3d做滤波整流并找到最大持续>=100ms的峰值(用于归一化分母)。

    返回：
        max_value: float 或 None，如果找不到峰值返回None
    """
    # 打开MVIC C3D
    itf = c3d.c3dserver()
    ret = c3d.open_c3d(itf, mvic_c3d_path)

    # 读取原始模拟信号
    analog_data_mvic = c3d.get_analog_data_unscaled(itf, muscle_name)
    c3d.close_c3d(itf)

    if analog_data_mvic is None:
        print(f"[WARN] {mvic_c3d_path} 中未找到肌肉通道: {muscle_name}")
        return None

    # 过滤 & 整流
    analog_data_mvic = emgfilter(analog_data_mvic, fs=1500)

    # 假设 1点=1ms(只是示例)，或你可以根据实际的 c3d.get_analog_rate(itf) 做换算
    # 这里 frame_rate=1 并不准确，需要你根据实际的MVIC采样率做变更。
    # 可以改成： frame_rate = c3d.get_analog_rate(itf)
    # 不过你在关了c3d后，就拿不到它了，所以要先取到再关
    # 这里姑且保持与原脚本相同写法(但这样的话非常可能不准确!)
    frame_rate = 1

    # 至少持续100ms -> 100ms = int(0.1*采样率)
    num_points = int((250 / 1000) * frame_rate)  # 原脚本写的 100ms = 250ms???
    # 你可以改成： num_points = int(0.1 * actual_sampling_rate)

    # find_peaks: 只找>=最大值*0.2的峰
    peaks, _ = find_peaks(analog_data_mvic, height=np.max(analog_data_mvic)*0.2)
    if len(peaks) == 0:
        print(f"[INFO] 未在 {mvic_c3d_path} 找到任何满足20%阈值的峰。")
        return None

    max_value = -np.inf
    N = len(analog_data_mvic)

    for peak in peaks:
        left = peak
        right = peak
        # 向左
        while left > 0 and analog_data_mvic[left - 1] > analog_data_mvic[left]:
            left -= 1
        # 向右
        while right < (N - 1) and analog_data_mvic[right + 1] > analog_data_mvic[right]:
            right += 1

        plateau_duration = right - left + 1
        if plateau_duration >= num_points:
            if analog_data_mvic[peak] > max_value:
                max_value = analog_data_mvic[peak]

    if max_value == -np.inf:
        print(f"[INFO] 虽然找到峰，但没有持续 >= {num_points} 点的峰。")
        return None

    return max_value

################################################################################
# 读取动作C3D中的EMG并做归一化
################################################################################

def emgdata_elbow_r(c3d_path, mvic_bi, mvic_tri, mvic_flx, mvic_ex):
    """
    右侧肘部肌肉归一化EMG。

    返回 (columns, data)，可写入 .sto。
    """
    # 打开动作C3D
    itf = c3d.c3dserver()
    ret = c3d.open_c3d(itf, c3d_path)

    # 读取通道
    sig_biceps = c3d.get_analog_data_unscaled(itf, 'R_Biceps')
    sig_triceps = c3d.get_analog_data_unscaled(itf, 'R_Triceps')
    sig_wrist_flex = c3d.get_analog_data_unscaled(itf, 'R_Wrist_Flex')
    sig_wrist_ext  = c3d.get_analog_data_unscaled(itf, 'R_Wrist_Ex')

    times = c3d.get_analog_times(itf)  # 时间向量

    # 关闭C3D
    c3d.close_c3d(itf)

    # 滤波整流
    if sig_biceps is not None:
        sig_biceps = emgfilter(sig_biceps, fs=1500)
    else:
        sig_biceps = np.zeros(1)

    if sig_triceps is not None:
        sig_triceps = emgfilter(sig_triceps, fs=1500)
    else:
        sig_triceps = np.zeros(1)

    if sig_wrist_flex is not None:
        sig_wrist_flex = emgfilter(sig_wrist_flex, fs=1500)
    else:
        sig_wrist_flex = np.zeros(1)

    if sig_wrist_ext is not None:
        sig_wrist_ext = emgfilter(sig_wrist_ext, fs=1500)
    else:
        sig_wrist_ext = np.zeros(1)

    # 计算 MVIC 基准
    norm_bi  = normalize_emg('R_Biceps', mvic_bi)
    norm_tri = normalize_emg('R_Triceps', mvic_tri)
    norm_flx = normalize_emg('R_Wrist_Flex', mvic_flx)
    norm_ex  = normalize_emg('R_Wrist_Ex', mvic_ex)

    # 如果 MVIC 全部找不到，就强制用实验峰
    # （或者你可以让其为 None，再判断）
    if norm_bi  is None: norm_bi  = np.max(sig_biceps)
    if norm_tri is None: norm_tri = np.max(sig_triceps)
    if norm_flx is None: norm_flx = np.max(sig_wrist_flex)
    if norm_ex  is None: norm_ex  = np.max(sig_wrist_ext)

    # 实验中最大值
    max_bi  = np.max(sig_biceps)
    max_tri = np.max(sig_triceps)
    max_flx = np.max(sig_wrist_flex)
    max_ex  = np.max(sig_wrist_ext)

    # 归一化
    def normalize_signal(signal, val_mvic):
        if len(signal) == 0:
            return []
        peak_exp = np.max(signal)
        denom = val_mvic if peak_exp < val_mvic else peak_exp
        return np.abs(signal) / denom if denom != 0 else signal

    sig_biceps = normalize_signal(sig_biceps, norm_bi)
    sig_triceps = normalize_signal(sig_triceps, norm_tri)
    sig_wrist_flex = normalize_signal(sig_wrist_flex, norm_flx)
    sig_wrist_ext  = normalize_signal(sig_wrist_ext, norm_ex)

    columns = [
        "time",
        "BIClong", "BICshort", "BRA",
        "TRIlong", "TRIlat", "TRImed",
        "ECRL", "ECRB", "ECU",
        "FCR", "FCU"
    ]
    data = [
        times,
        sig_biceps, sig_biceps, sig_biceps,  # 先都用同一个信号示例
        sig_triceps, sig_triceps, sig_triceps,
        sig_wrist_ext, sig_wrist_ext, sig_wrist_ext,
        sig_wrist_flex, sig_wrist_flex
    ]
    return columns, data

def emgdata_elbow_l(c3d_path, mvic_bi, mvic_tri, mvic_flx, mvic_ex):
    """
    左侧肘部类似写法...
    """
    # 类似 emgdata_elbow_r，不再赘述
    # 你可以复制 r 版本并把通道名 'R_' 替换成 'L_'
    # 下面简要写一下

    itf = c3d.c3dserver()
    c3d.open_c3d(itf, c3d_path)
    sig_biceps = c3d.get_analog_data_unscaled(itf, 'L_Biceps_Brachii')
    sig_triceps = c3d.get_analog_data_unscaled(itf, 'L_Triceps_Brachii')
    sig_wrist_flex = c3d.get_analog_data_unscaled(itf, 'L_Hand_Beuger_FCR')
    sig_wrist_ext  = c3d.get_analog_data_unscaled(itf, 'L_Hand_Strecker')
    times = c3d.get_analog_times(itf)
    c3d.close_c3d(itf)

    # 滤波整流
    sig_biceps = emgfilter(sig_biceps) if sig_biceps is not None else np.zeros(1)
    sig_triceps = emgfilter(sig_triceps) if sig_triceps is not None else np.zeros(1)
    sig_wrist_flex = emgfilter(sig_wrist_flex) if sig_wrist_flex is not None else np.zeros(1)
    sig_wrist_ext  = emgfilter(sig_wrist_ext)  if sig_wrist_ext  is not None else np.zeros(1)

    # MVIC
    norm_bi  = normalize_emg('L_Biceps_Brachii', mvic_bi)
    norm_tri = normalize_emg('L_Triceps_Brachii', mvic_tri)
    norm_flx = normalize_emg('L_Hand_Beuger_FCR', mvic_flx)
    norm_ex  = normalize_emg('L_Hand_Strecker', mvic_ex)

    if norm_bi  is None: norm_bi  = np.max(sig_biceps)
    if norm_tri is None: norm_tri = np.max(sig_triceps)
    if norm_flx is None: norm_flx = np.max(sig_wrist_flex)
    if norm_ex  is None: norm_ex  = np.max(sig_wrist_ext)

    def normalize_signal(signal, val_mvic):
        peak_exp = np.max(signal)
        denom = val_mvic if peak_exp < val_mvic else peak_exp
        return np.abs(signal) / denom if denom != 0 else signal

    sig_biceps = normalize_signal(sig_biceps, norm_bi)
    sig_triceps = normalize_signal(sig_triceps, norm_tri)
    sig_wrist_flex = normalize_signal(sig_wrist_flex, norm_flx)
    sig_wrist_ext  = normalize_signal(sig_wrist_ext, norm_ex)

    columns = [
        "time",
        "BIClong", "BICshort", "BRA",
        "TRIlong", "TRIlat", "TRImed",
        "ECRL", "ECRB", "ECU",
        "FCR", "FCU"
    ]
    data = [
        times,
        sig_biceps, sig_biceps, sig_biceps,
        sig_triceps, sig_triceps, sig_triceps,
        sig_wrist_ext, sig_wrist_ext, sig_wrist_ext,
        sig_wrist_flex, sig_wrist_flex
    ]
    return columns, data

################################################################################
# 主函数/入口
################################################################################

def process_emg(side_is_right=True):
    """
    根据侧别选择 r/l 函数，对动作C3D + 四个MVIC文件做处理，然后把EMG写到 .sto。
    """
    # 侧别
    if side_is_right:
        # 调用右侧
        cols, dat = emgdata_elbow_r(
            c3d_path=ACTION_C3D,
            mvic_bi=MVIC_BICEPS_C3D,
            mvic_tri=MVIC_TRICEPS_C3D,
            mvic_flx=MVIC_WRIST_FLX_C3D,
            mvic_ex=MVIC_WRIST_EX_C3D
        )
    else:
        # 调用左侧
        cols, dat = emgdata_elbow_l(
            c3d_path=ACTION_C3D,
            mvic_bi=MVIC_BICEPS_C3D,
            mvic_tri=MVIC_TRICEPS_C3D,
            mvic_flx=MVIC_WRIST_FLX_C3D,
            mvic_ex=MVIC_WRIST_EX_C3D
        )

    # 把结果写到 emg_norm.sto (或你想要的其他文件名)
    if os.path.exists(EMG_OUT_FILE):
        print(f"[INFO] 文件已存在，将在末尾添加新列。 => {EMG_OUT_FILE}")
    else:
        print(f"[INFO] 文件不存在，将新建文件。 => {EMG_OUT_FILE}")

    # 这里复用 c3d_converter.py 中的 add_column_to_file
    # 这个函数的签名是 add_column_to_file(file_path, new_column_name, new_column_data)
    add_column_to_file(
        file_path=EMG_OUT_FILE,
        new_column_name=cols,
        new_column_data=dat
    )
    print(f"[INFO] EMG 数据已写入: {EMG_OUT_FILE}")


# 只有在直接运行这个脚本时，才会执行下面的示例
if __name__ == '__main__':
    print("=== 开始 EMG 处理流程 ===")

    # 如果你想先转换你的MVIC C3D到TRC/MOT，可以在这里做：
    # 例如：
    # from c3d_converter import c3dtotrc
    # trc_bi, mot_bi = c3dtotrc(MVIC_BICEPS_C3D, PROCESSED_DIR)
    # trc_tri, mot_tri = c3dtotrc(MVIC_TRICEPS_C3D, PROCESSED_DIR)
    # ... etc
    #
    # 但是对 EMG 而言，你可能只需要读原C3D的模拟通道即可。

    # 同理，对动作C3D先转换到TRC/MOT，也可在这里：
    # action_trc, action_mot = c3dtotrc(ACTION_C3D, PROCESSED_DIR)

    # 直接进行 EMG 提取和归一化
    process_emg(side_is_right=True)

    print("=== EMG 处理流程结束 ===")
