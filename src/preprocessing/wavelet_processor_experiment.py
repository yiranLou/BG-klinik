"""
小波去噪EMG处理模块
用于与传统带通滤波方法进行对比实验
"""

import numpy as np
import pywt
from scipy.signal import butter, sosfilt, find_peaks
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pyc3dserver as c3d


class WaveletEMGProcessor:
    """小波去噪EMG处理器"""

    def __init__(self, wavelet_type: str = 'db4', decomposition_level: int = 5):
        """
        初始化小波处理器

        Args:
            wavelet_type: 小波基类型 (如 'db4', 'sym6', 'coif3', 'bior3.5')
            decomposition_level: 分解层数
        """
        self.wavelet_type = wavelet_type
        self.decomposition_level = decomposition_level
        self.threshold_methods = ['soft', 'hard']  # 软阈值和硬阈值

    def wavelet_denoise(self, signal: np.ndarray, fs: int = 1500,
                        threshold_method: str = 'soft') -> np.ndarray:
        """
        使用小波变换对EMG信号去噪

        Args:
            signal: 原始EMG信号
            fs: 采样频率
            threshold_method: 阈值处理方法 ('soft' 或 'hard')

        Returns:
            去噪后的信号
        """
        # 小波分解
        coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.decomposition_level)

        # 估计噪声水平（使用MAD方法）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # 计算阈值（使用universal threshold）
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # 对细节系数进行阈值处理
        coeffs_thresh = coeffs.copy()
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_method)

        # 小波重构
        denoised_signal = pywt.waverec(coeffs_thresh, self.wavelet_type)

        # 确保返回信号长度与原信号相同
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]

        return denoised_signal

    def emg_filter_and_rectify(self, signal: np.ndarray, fs: int = 1500,
                               threshold_method: str = 'soft') -> np.ndarray:
        """
        对EMG信号进行去噪和整流处理

        Args:
            signal: 原始EMG信号
            fs: 采样频率
            threshold_method: 阈值处理方法

        Returns:
            处理后的信号
        """
        # 小波去噪
        denoised = self.wavelet_denoise(signal, fs, threshold_method)

        # 全波整流
        rectified = np.abs(denoised)

        # 低通滤波获取包络（可选）
        # lowpass_sos = butter(4, 10, btype='low', fs=fs, output='sos')
        # envelope = sosfilt(lowpass_sos, rectified)

        return rectified

    def normalize_emg(self, muscle_name: str, mvic_c3d_path: str) -> Optional[float]:
        """
        计算MVIC值用于归一化（与原代码相同的逻辑）

        Returns:
            max_value: MVIC值
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

        # 使用小波去噪处理
        analog_data_mvic = self.emg_filter_and_rectify(analog_data_mvic)

        # 找峰值逻辑与原代码相同
        frame_rate = 1500  # 假设采样率
        num_points = int(0.1 * frame_rate)  # 100ms

        peaks, _ = find_peaks(analog_data_mvic, height=np.max(analog_data_mvic) * 0.2)
        if len(peaks) == 0:
            return None

        max_value = -np.inf
        N = len(analog_data_mvic)

        for peak in peaks:
            left = peak
            right = peak
            while left > 0 and analog_data_mvic[left - 1] > analog_data_mvic[left]:
                left -= 1
            while right < (N - 1) and analog_data_mvic[right + 1] > analog_data_mvic[right]:
                right += 1

            plateau_duration = right - left + 1
            if plateau_duration >= num_points:
                if analog_data_mvic[peak] > max_value:
                    max_value = analog_data_mvic[peak]

        if max_value == -np.inf:
            return None

        return max_value


# 传统带通滤波处理器（用于对比）
class TraditionalEMGProcessor:
    """传统带通滤波EMG处理器"""

    def __init__(self, lowcut: float = 15, highcut: float = 500, order: int = 6):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def butter_bandpass(self, fs: int):
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = butter(self.order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def emgfilter(self, data: np.ndarray, fs: int = 1500) -> np.ndarray:
        """带通滤波并整流"""
        sos = self.butter_bandpass(fs)
        filtered = sosfilt(sos, data)
        rectified = np.abs(filtered)
        return rectified


# 评估指标计算
class EMGEvaluator:
    """EMG处理方法评估器"""

    @staticmethod
    def calculate_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """计算信噪比（SNR）"""
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean((clean_signal - noisy_signal) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_rmse(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算均方根误差（RMSE）"""
        return np.sqrt(np.mean((signal1 - signal2) ** 2))

    @staticmethod
    def calculate_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算相关系数"""
        return np.corrcoef(signal1, signal2)[0, 1]

    @staticmethod
    def calculate_smoothness(signal: np.ndarray) -> float:
        """计算信号平滑度（使用二阶导数）"""
        second_derivative = np.diff(signal, n=2)
        smoothness = np.mean(second_derivative ** 2)
        return smoothness

    @staticmethod
    def calculate_zero_crossing_rate(signal: np.ndarray) -> float:
        """计算过零率"""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)

    @staticmethod
    def calculate_power_spectrum_density(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """计算功率谱密度"""
        from scipy import signal as sig
        f, psd = sig.welch(signal, fs, nperseg=1024)
        return f, psd

    @staticmethod
    def calculate_median_frequency(signal: np.ndarray, fs: int) -> float:
        """计算中频"""
        f, psd = EMGEvaluator.calculate_power_spectrum_density(signal, fs)
        cumsum_psd = np.cumsum(psd)
        total_power = cumsum_psd[-1]
        median_idx = np.where(cumsum_psd >= total_power / 2)[0][0]
        return f[median_idx]


# 可视化比较工具
def plot_comparison(time: np.ndarray, signals: Dict[str, np.ndarray],
                    title: str = "EMG Signal Comparison"):
    """
    绘制多个信号的对比图

    Args:
        time: 时间轴
        signals: 信号字典，键为信号名称，值为信号数据
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))

    # 原始信号子图
    plt.subplot(len(signals), 1, 1)
    if 'raw' in signals:
        plt.plot(time, signals['raw'], 'k-', alpha=0.5, label='Raw EMG')
        plt.legend()
        plt.ylabel('Amplitude')
        plt.title(title)

    # 处理后的信号子图
    plot_idx = 2
    for name, signal in signals.items():
        if name != 'raw':
            plt.subplot(len(signals), 1, plot_idx)
            plt.plot(time, signal, label=name)
            plt.legend()
            plt.ylabel('Amplitude')
            plot_idx += 1

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_frequency_comparison(signals: Dict[str, np.ndarray], fs: int = 1500):
    """绘制频域对比图"""
    plt.figure(figsize=(12, 6))

    for name, signal in signals.items():
        f, psd = EMGEvaluator.calculate_power_spectrum_density(signal, fs)
        plt.semilogy(f, psd, label=name)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Frequency Domain Comparison')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 500)
    plt.show()


# 主要的比较函数
def compare_wavelets(signal: np.ndarray, fs: int = 1500) -> Dict[str, Dict[str, float]]:
    """
    比较不同小波基的去噪效果

    Args:
        signal: 原始EMG信号
        fs: 采样频率

    Returns:
        results: 包含各种评估指标的字典
    """
    # 定义要测试的小波基
    wavelets = ['db4', 'db8', 'sym6', 'sym8', 'coif3', 'coif5', 'bior3.5', 'bior3.7']
    results = {}

    # 传统方法作为基准
    trad_processor = TraditionalEMGProcessor()
    trad_filtered = trad_processor.emgfilter(signal, fs)

    evaluator = EMGEvaluator()

    # 计算传统方法的指标
    results['traditional'] = {
        'smoothness': evaluator.calculate_smoothness(trad_filtered),
        'zero_crossing': evaluator.calculate_zero_crossing_rate(trad_filtered),
        'median_freq': evaluator.calculate_median_frequency(trad_filtered, fs)
    }

    # 测试每个小波基
    for wavelet in wavelets:
        wavelet_processor = WaveletEMGProcessor(wavelet_type=wavelet)

        # 尝试软阈值和硬阈值
        for threshold_method in ['soft', 'hard']:
            key = f"{wavelet}_{threshold_method}"

            try:
                denoised = wavelet_processor.emg_filter_and_rectify(signal, fs, threshold_method)

                results[key] = {
                    'rmse_to_trad': evaluator.calculate_rmse(trad_filtered, denoised),
                    'correlation_to_trad': evaluator.calculate_correlation(trad_filtered, denoised),
                    'smoothness': evaluator.calculate_smoothness(denoised),
                    'zero_crossing': evaluator.calculate_zero_crossing_rate(denoised),
                    'median_freq': evaluator.calculate_median_frequency(denoised, fs)
                }
            except Exception as e:
                print(f"Error processing {key}: {e}")
                results[key] = None

    return results