"""
EMG 处理模块 - 按照学术论文标准实现完整的EMG信号处理流程。
对动作C3D及MVIC C3D文件做完整的6步处理：DC偏移移除、陷波滤波、高通滤波、小波去噪、全波整流、低通滤波包络提取和MVIC归一化。
"""

import os
import numpy as np
from scipy.signal import find_peaks, sosfilt, butter, iirnotch, filtfilt
import pywt
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

# EMG处理参数（符合学术标准）
EMG_PARAMS = {
    'sampling_rate': 1500,          # 采样率 Hz
    'notch_freq': 50,               # 陷波滤波频率 Hz (50Hz for Europe, 60Hz for North America)
    'highpass_cutoff': 20,          # 高通滤波截止频率 Hz
    'lowpass_cutoff': 6,            # 低通滤波截止频率 Hz (包络提取)
    'filter_order': 4,              # 滤波器阶数
    'wavelet_name': 'db4',          # 小波去噪使用的小波基
    'wavelet_mode': 'symmetric'     # 小波边界模式
}

# MVIC文件路径
MVIC_BICEPS_C3D  = os.path.join(MVIC_DIR, 'MVIC_Biceps_r.c3d')
MVIC_TRICEPS_C3D = os.path.join(MVIC_DIR, 'MVIC_Triceps_r.c3d')
MVIC_WRIST_EX_C3D   = os.path.join(MVIC_DIR, 'MVIC_wrist_ex_r.c3d')
MVIC_WRIST_FLX_C3D  = os.path.join(MVIC_DIR, 'MVIC_wrist_flex_r.c3d')

# 动作C3D文件
ACTION_C3D = os.path.join(RAW_DATA_DIR, 'right', 'ROM_Ellenbogenflex_R 1.c3d')

# EMG输出文件
EMG_OUT_FILE = os.path.join(PROCESSED_DIR, 'emg_norm.sto')

################################################################################
# 学术标准EMG信号处理流程（6步法）
################################################################################

def remove_dc_offset(signal):
    """
    步骤1: 移除DC偏移 - 减去信号的均值
    
    Args:
        signal (np.array): 原始EMG信号
        
    Returns:
        np.array: 移除DC偏移后的信号
    """
    return signal - np.mean(signal)

def notch_filter(signal, fs, notch_freq=50, quality_factor=30):
    """
    步骤2: 陷波滤波 - 去除电力线干扰（50/60Hz）
    
    Args:
        signal (np.array): 输入信号
        fs (float): 采样频率
        notch_freq (float): 陷波频率（50Hz欧洲，60Hz北美）
        quality_factor (float): 品质因数，控制陷波宽度
        
    Returns:
        np.array: 陷波滤波后的信号
    """
    # 设计IIR陷波滤波器
    b, a = iirnotch(notch_freq, quality_factor, fs)
    # 使用零相位滤波避免相位延迟
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def highpass_filter(signal, fs, cutoff=20, order=4):
    """
    步骤3: 高通滤波 - 去除运动伪影和低频干扰
    
    Args:
        signal (np.array): 输入信号
        fs (float): 采样频率
        cutoff (float): 截止频率 Hz
        order (int): 滤波器阶数
        
    Returns:
        np.array: 高通滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # 设计Butterworth高通滤波器
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # 使用零相位滤波
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def wavelet_denoise(signal, wavelet='db4', sigma=None):
    """
    步骤4: 小波去噪 - 去除随机噪声同时保持信号特征
    
    Args:
        signal (np.array): 输入信号
        wavelet (str): 小波基函数
        sigma (float): 噪声标准差估计，None时自动估计
        
    Returns:
        np.array: 去噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric')
    
    # 估计噪声标准差（使用最高频系数）
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # 软阈值去噪
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') 
                        for detail in coeffs_thresh[1:]]
    
    # 小波重构
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
    
    # 确保输出长度与输入相同
    if len(denoised_signal) != len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    
    return denoised_signal

def full_wave_rectification(signal):
    """
    步骤5: 全波整流 - 取信号的绝对值
    
    Args:
        signal (np.array): 输入信号
        
    Returns:
        np.array: 整流后的信号
    """
    return np.abs(signal)

def lowpass_envelope(signal, fs, cutoff=6, order=4):
    """
    步骤6: 低通滤波 - 创建线性包络
    
    Args:
        signal (np.array): 整流后的信号
        fs (float): 采样频率
        cutoff (float): 截止频率 Hz
        order (int): 滤波器阶数
        
    Returns:
        np.array: 包络信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # 设计Butterworth低通滤波器
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # 使用零相位滤波
    envelope = filtfilt(b, a, signal)
    return envelope

def emg_process_academic_standard(signal, fs=1500):
    """
    按照学术论文标准的完整EMG信号处理流程
    
    实现步骤：
    1. 移除DC偏移
    2. 陷波滤波（50/60Hz）
    3. 高通滤波（20Hz）
    4. 小波去噪
    5. 全波整流
    6. 低通滤波包络（6Hz）
    
    Args:
        signal (np.array): 原始EMG信号
        fs (float): 采样频率
        
    Returns:
        dict: 包含各步骤处理结果的字典
    """
    if signal is None or len(signal) == 0:
        return {
            'raw': np.array([]),
            'dc_removed': np.array([]),
            'notch_filtered': np.array([]),
            'highpass_filtered': np.array([]),
            'denoised': np.array([]),
            'rectified': np.array([]),
            'envelope': np.array([])
        }
    
    results = {'raw': signal}
    
    # 步骤1: 移除DC偏移
    print("  - 步骤1: 移除DC偏移")
    dc_removed = remove_dc_offset(signal)
    results['dc_removed'] = dc_removed
    
    # 步骤2: 陷波滤波
    print("  - 步骤2: 陷波滤波 (50Hz)")
    notch_filtered = notch_filter(dc_removed, fs, EMG_PARAMS['notch_freq'])
    results['notch_filtered'] = notch_filtered
    
    # 步骤3: 高通滤波
    print("  - 步骤3: 高通滤波 (20Hz)")
    highpass_filtered = highpass_filter(notch_filtered, fs, EMG_PARAMS['highpass_cutoff'])
    results['highpass_filtered'] = highpass_filtered
    
    # 步骤4: 小波去噪
    print("  - 步骤4: 小波去噪")
    denoised = wavelet_denoise(highpass_filtered, EMG_PARAMS['wavelet_name'])
    results['denoised'] = denoised
    
    # 步骤5: 全波整流
    print("  - 步骤5: 全波整流")
    rectified = full_wave_rectification(denoised)
    results['rectified'] = rectified
    
    # 步骤6: 低通滤波包络
    print("  - 步骤6: 低通滤波包络 (6Hz)")
    envelope = lowpass_envelope(rectified, fs, EMG_PARAMS['lowpass_cutoff'])
    results['envelope'] = envelope
    
    return results

################################################################################
# 计算MVIC峰值（改进版）
################################################################################

def normalize_emg_academic(muscle_name, mvic_c3d_path):
    """
    使用学术标准处理MVIC数据并找到最大持续峰值用于归一化
    
    Args:
        muscle_name (str): 肌肉通道名称
        mvic_c3d_path (str): MVIC C3D文件路径
        
    Returns:
        float or None: MVIC最大值，用于归一化
    """
    print(f"处理MVIC数据: {muscle_name}")
    
    # 打开MVIC C3D
    itf = c3d.c3dserver()
    ret = c3d.open_c3d(itf, mvic_c3d_path)

    # 读取原始模拟信号
    analog_data_mvic = c3d.get_analog_data_unscaled(itf, muscle_name)
    c3d.close_c3d(itf)

    if analog_data_mvic is None:
        print(f"[WARN] {mvic_c3d_path} 中未找到肌肉通道: {muscle_name}")
        return None

    # 使用学术标准处理EMG信号
    processed_results = emg_process_academic_standard(analog_data_mvic, EMG_PARAMS['sampling_rate'])
    
    # 使用最终的包络信号进行峰值检测
    envelope_signal = processed_results['envelope']
    
    if len(envelope_signal) == 0:
        print(f"[WARN] 处理后的信号为空: {muscle_name}")
        return None

    # 寻找持续时间>=100ms的峰值
    frame_rate = EMG_PARAMS['sampling_rate']
    min_duration_ms = 100
    num_points = int((min_duration_ms / 1000) * frame_rate)

    # 寻找峰值（至少为最大值的20%）
    peaks, _ = find_peaks(envelope_signal, height=np.max(envelope_signal) * 0.2)
    
    if len(peaks) == 0:
        print(f"[INFO] 未找到满足阈值的峰值: {muscle_name}")
        return np.max(envelope_signal)  # 返回整个信号的最大值

    max_value = -np.inf
    N = len(envelope_signal)

    # 检查每个峰值的持续时间
    for peak in peaks:
        left = peak
        right = peak
        
        # 向左寻找下降点
        while left > 0 and envelope_signal[left - 1] >= envelope_signal[left]:
            left -= 1
        
        # 向右寻找下降点
        while right < (N - 1) and envelope_signal[right + 1] >= envelope_signal[right]:
            right += 1

        plateau_duration = right - left + 1
        
        if plateau_duration >= num_points:
            if envelope_signal[peak] > max_value:
                max_value = envelope_signal[peak]
                print(f"  找到有效峰值: {envelope_signal[peak]:.4f}, 持续时间: {plateau_duration}点 ({plateau_duration/frame_rate*1000:.1f}ms)")

    if max_value == -np.inf:
        print(f"[INFO] 未找到持续时间>=100ms的峰值，使用全局最大值: {muscle_name}")
        return np.max(envelope_signal)

    return max_value

################################################################################
# 读取动作C3D中的EMG并做归一化（改进版）
################################################################################

def emgdata_elbow_r_academic(c3d_path, mvic_bi, mvic_tri, mvic_flx, mvic_ex):
    """
    使用学术标准处理右侧肘部肌肉EMG信号并进行MVIC归一化
    
    Args:
        c3d_path (str): 动作C3D文件路径
        mvic_bi (str): 二头肌MVIC文件路径
        mvic_tri (str): 三头肌MVIC文件路径
        mvic_flx (str): 腕屈肌MVIC文件路径
        mvic_ex (str): 腕伸肌MVIC文件路径
        
    Returns:
        tuple: (columns, data) - 列名和数据，可写入.sto文件
    """
    print("处理右侧肘部EMG数据...")
    
    # 打开动作C3D
    itf = c3d.c3dserver()
    ret = c3d.open_c3d(itf, c3d_path)

    # 读取各肌肉通道
    muscle_channels = {
        'biceps': 'R_Biceps',
        'triceps': 'R_Triceps', 
        'wrist_flex': 'R_Wrist_Flex',
        'wrist_ext': 'R_Wrist_Ex'
    }
    
    raw_signals = {}
    for muscle, channel in muscle_channels.items():
        raw_signals[muscle] = c3d.get_analog_data_unscaled(itf, channel)
        if raw_signals[muscle] is None:
            print(f"[WARN] 未找到通道: {channel}")
            raw_signals[muscle] = np.array([])

    times = c3d.get_analog_times(itf)
    c3d.close_c3d(itf)

    # 使用学术标准处理每个肌肉信号
    processed_signals = {}
    for muscle, signal in raw_signals.items():
        print(f"处理 {muscle} 信号...")
        processed_results = emg_process_academic_standard(signal, EMG_PARAMS['sampling_rate'])
        processed_signals[muscle] = processed_results['envelope']  # 使用最终的包络信号

    # 计算MVIC归一化因子
    print("计算MVIC归一化因子...")
    mvic_values = {
        'biceps': normalize_emg_academic('R_Biceps', mvic_bi),
        'triceps': normalize_emg_academic('R_Triceps', mvic_tri),
        'wrist_flex': normalize_emg_academic('R_Wrist_Flex', mvic_flx),
        'wrist_ext': normalize_emg_academic('R_Wrist_Ex', mvic_ex)
    }

    # 归一化信号
    def normalize_signal_academic(signal, mvic_value):
        """改进的归一化函数"""
        if len(signal) == 0 or mvic_value is None or mvic_value == 0:
            return signal
        
        exp_max = np.max(signal)
        # 如果实验中的最大值超过MVIC，使用实验最大值作为分母
        denominator = max(mvic_value, exp_max)
        
        normalized = signal / denominator
        print(f"    归一化: MVIC={mvic_value:.4f}, 实验最大值={exp_max:.4f}, 使用分母={denominator:.4f}")
        
        return normalized

    # 执行归一化
    normalized_signals = {}
    for muscle in processed_signals.keys():
        print(f"归一化 {muscle} 信号...")
        normalized_signals[muscle] = normalize_signal_academic(
            processed_signals[muscle], 
            mvic_values[muscle]
        )

    # 构建输出数据结构
    columns = [
        "time",
        "BIClong", "BICshort", "BRA",           # 肘屈肌群
        "TRIlong", "TRIlat", "TRImed",          # 肘伸肌群  
        "ECRL", "ECRB", "ECU",                  # 腕伸肌群
        "FCR", "FCU"                            # 腕屈肌群
    ]
    
    data = [
        times,
        normalized_signals['biceps'], normalized_signals['biceps'], normalized_signals['biceps'],
        normalized_signals['triceps'], normalized_signals['triceps'], normalized_signals['triceps'],
        normalized_signals['wrist_ext'], normalized_signals['wrist_ext'], normalized_signals['wrist_ext'],
        normalized_signals['wrist_flex'], normalized_signals['wrist_flex']
    ]
    
    return columns, data

def emgdata_elbow_l_academic(c3d_path, mvic_bi, mvic_tri, mvic_flx, mvic_ex):
    """
    使用学术标准处理左侧肘部肌肉EMG信号并进行MVIC归一化
    
    Args:
        c3d_path (str): 动作C3D文件路径
        mvic_bi (str): 二头肌MVIC文件路径
        mvic_tri (str): 三头肌MVIC文件路径
        mvic_flx (str): 腕屈肌MVIC文件路径
        mvic_ex (str): 腕伸肌MVIC文件路径
        
    Returns:
        tuple: (columns, data) - 列名和数据，可写入.sto文件
    """
    print("处理左侧肘部EMG数据...")
    
    # 打开动作C3D
    itf = c3d.c3dserver()
    ret = c3d.open_c3d(itf, c3d_path)

    # 读取各肌肉通道（左侧通道名可能不同）
    muscle_channels = {
        'biceps': 'L_Biceps_Brachii',
        'triceps': 'L_Triceps_Brachii', 
        'wrist_flex': 'L_Hand_Beuger_FCR',
        'wrist_ext': 'L_Hand_Strecker'
    }
    
    raw_signals = {}
    for muscle, channel in muscle_channels.items():
        raw_signals[muscle] = c3d.get_analog_data_unscaled(itf, channel)
        if raw_signals[muscle] is None:
            print(f"[WARN] 未找到通道: {channel}")
            raw_signals[muscle] = np.array([])

    times = c3d.get_analog_times(itf)
    c3d.close_c3d(itf)

    # 使用学术标准处理每个肌肉信号
    processed_signals = {}
    for muscle, signal in raw_signals.items():
        print(f"处理 {muscle} 信号...")
        processed_results = emg_process_academic_standard(signal, EMG_PARAMS['sampling_rate'])
        processed_signals[muscle] = processed_results['envelope']  # 使用最终的包络信号

    # 计算MVIC归一化因子（使用对应的左侧通道名）
    print("计算MVIC归一化因子...")
    mvic_values = {
        'biceps': normalize_emg_academic('L_Biceps_Brachii', mvic_bi),
        'triceps': normalize_emg_academic('L_Triceps_Brachii', mvic_tri),
        'wrist_flex': normalize_emg_academic('L_Hand_Beuger_FCR', mvic_flx),
        'wrist_ext': normalize_emg_academic('L_Hand_Strecker', mvic_ex)
    }

    # 归一化信号
    def normalize_signal_academic(signal, mvic_value):
        """改进的归一化函数"""
        if len(signal) == 0 or mvic_value is None or mvic_value == 0:
            return signal
        
        exp_max = np.max(signal)
        # 如果实验中的最大值超过MVIC，使用实验最大值作为分母
        denominator = max(mvic_value, exp_max)
        
        normalized = signal / denominator
        print(f"    归一化: MVIC={mvic_value:.4f}, 实验最大值={exp_max:.4f}, 使用分母={denominator:.4f}")
        
        return normalized

    # 执行归一化
    normalized_signals = {}
    for muscle in processed_signals.keys():
        print(f"归一化 {muscle} 信号...")
        normalized_signals[muscle] = normalize_signal_academic(
            processed_signals[muscle], 
            mvic_values[muscle]
        )

    # 构建输出数据结构
    columns = [
        "time",
        "BIClong", "BICshort", "BRA",           # 肘屈肌群
        "TRIlong", "TRIlat", "TRImed",          # 肘伸肌群  
        "ECRL", "ECRB", "ECU",                  # 腕伸肌群
        "FCR", "FCU"                            # 腕屈肌群
    ]
    
    data = [
        times,
        normalized_signals['biceps'], normalized_signals['biceps'], normalized_signals['biceps'],
        normalized_signals['triceps'], normalized_signals['triceps'], normalized_signals['triceps'],
        normalized_signals['wrist_ext'], normalized_signals['wrist_ext'], normalized_signals['wrist_ext'],
        normalized_signals['wrist_flex'], normalized_signals['wrist_flex']
    ]
    
    return columns, data

################################################################################
# 主函数/入口
################################################################################

def process_emg_academic_standard(side_is_right=True):
    """
    使用学术标准的EMG处理流程：根据侧别选择处理函数，对动作C3D + 四个MVIC文件做完整的6步处理
    
    处理步骤：
    1. DC偏移移除
    2. 陷波滤波（50/60Hz）
    3. 高通滤波（20Hz）
    4. 小波去噪
    5. 全波整流
    6. 低通滤波包络（6Hz）
    7. MVIC归一化
    
    Args:
        side_is_right (bool): True为右侧，False为左侧
    """
    print(f"=== 开始学术标准EMG处理流程 ({'右侧' if side_is_right else '左侧'}) ===")
    
    # 打印处理参数
    print("EMG处理参数:")
    for key, value in EMG_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    
    # 侧别选择
    if side_is_right:
        print("使用右侧肘部EMG处理函数...")
        cols, dat = emgdata_elbow_r_academic(
            c3d_path=ACTION_C3D,
            mvic_bi=MVIC_BICEPS_C3D,
            mvic_tri=MVIC_TRICEPS_C3D,
            mvic_flx=MVIC_WRIST_FLX_C3D,
            mvic_ex=MVIC_WRIST_EX_C3D
        )
    else:
        print("使用左侧肘部EMG处理函数...")
        cols, dat = emgdata_elbow_l_academic(
            c3d_path=ACTION_C3D,
            mvic_bi=MVIC_BICEPS_C3D,
            mvic_tri=MVIC_TRICEPS_C3D,
            mvic_flx=MVIC_WRIST_FLX_C3D,
            mvic_ex=MVIC_WRIST_EX_C3D
        )

    # 写入结果到.sto文件
    print(f"\n写入EMG数据到: {EMG_OUT_FILE}")
    if os.path.exists(EMG_OUT_FILE):
        print("[INFO] 文件已存在，将添加新列")
    else:
        print("[INFO] 创建新的EMG文件")

    # 使用c3d_converter中的函数写入数据
    add_column_to_file(
        file_path=EMG_OUT_FILE,
        new_column_name=cols,
        new_column_data=dat
    )
    print(f"[SUCCESS] EMG数据已成功写入: {EMG_OUT_FILE}")

# 保持向后兼容的旧函数（已弃用）
def process_emg(side_is_right=True):
    """
    向后兼容函数 - 建议使用 process_emg_academic_standard()
    """
    print("[DEPRECATED] 此函数已弃用，建议使用 process_emg_academic_standard()")
    print("自动切换到学术标准处理流程...")
    process_emg_academic_standard(side_is_right)

# 只有在直接运行这个脚本时，才会执行下面的示例
if __name__ == '__main__':
    print("=== EMG信号处理模块 - 学术标准实现 ===")
    print("实现论文中描述的完整6步EMG处理流程:")
    print("1. DC偏移移除")
    print("2. 陷波滤波（50/60Hz电力线干扰）")
    print("3. 高通滤波（20Hz截止频率）")
    print("4. 小波去噪")
    print("5. 全波整流")
    print("6. 低通滤波包络（6Hz截止频率）")
    print("7. MVIC归一化")
    print()

    # 检查依赖包
    try:
        import pywt
        print("✓ PyWavelets (小波分析包) 已安装")
    except ImportError:
        print("✗ 缺少依赖: PyWavelets")
        print("请安装: pip install PyWavelets")
        exit(1)

    # 执行学术标准EMG处理
    process_emg_academic_standard(side_is_right=True)

    print("=== EMG处理流程完成 ===")
