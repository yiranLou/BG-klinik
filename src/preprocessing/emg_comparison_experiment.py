"""
EMG处理方法对比实验主程序
比较传统带通滤波与多种小波去噪方法的效果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyc3dserver as c3d
from typing import Dict, List, Tuple
from wavelet_emg_processor import (
    WaveletEMGProcessor,
    TraditionalEMGProcessor,
    EMGEvaluator,
    compare_wavelets,
    plot_comparison,
    plot_frequency_comparison
)
from c3d_converter import add_column_to_file

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EMGComparison:
    """EMG处理方法比较实验类"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.raw_data_dir = os.path.join(project_root, 'data', 'raw', 'N10_n')
        self.mvic_dir = os.path.join(self.raw_data_dir, 'mvic')
        self.output_dir = os.path.join(project_root, 'data', 'processed', 'emg_comparison')

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 初始化评估器
        self.evaluator = EMGEvaluator()

    def load_emg_data(self, c3d_path: str, muscle_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        从C3D文件加载EMG数据

        Returns:
            signal: EMG信号
            times: 时间轴
            fs: 采样频率
        """
        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, c3d_path)

        # 获取header信息
        dict_header = c3d.get_dict_header(itf)

        # 从header中获取采样率
        fs = dict_header['ANALOG_FRAME_RATE']

        # 直接使用get_analog_data_unscaled获取特定通道的数据
        signal = c3d.get_analog_data_unscaled(itf, muscle_name)

        # 如果没找到通道，尝试查看所有通道名称
        if signal is None:
            dict_analogs = c3d.get_dict_analogs(itf)
            if 'LABELS' in dict_analogs:
                available_channels = dict_analogs['LABELS']
                print(f"Available EMG channels: {available_channels}")

                # 尝试模糊匹配
                for channel_name in available_channels:
                    if muscle_name.lower() in channel_name.lower():
                        signal = c3d.get_analog_data_unscaled(itf, channel_name)
                        if signal is not None:
                            print(f"Found channel '{channel_name}' for muscle '{muscle_name}'")
                            break

            if signal is None:
                raise KeyError(f"Muscle '{muscle_name}' not found in analog channels")

        # 获取时间轴
        times = c3d.get_analog_times(itf)

        c3d.close_c3d(itf)

        # 确保信号是一维的numpy数组
        signal = np.array(signal).flatten()

        return signal, times, int(fs)

    def process_single_muscle(self, signal: np.ndarray, times: np.ndarray,
                              fs: int, muscle_name: str) -> Dict:
        """
        使用不同方法处理单个肌肉的EMG信号

        Returns:
            results: 包含所有处理结果的字典
        """
        results = {'times': times, 'raw': signal}

        # 传统带通滤波
        trad_processor = TraditionalEMGProcessor()
        results['traditional'] = trad_processor.emgfilter(signal, fs)

        # 不同小波基的处理结果
        wavelets = ['db4', 'db8', 'sym6', 'coif3', 'bior3.5']

        for wavelet in wavelets:
            processor = WaveletEMGProcessor(wavelet_type=wavelet)

            # 软阈值
            try:
                results[f'{wavelet}_soft'] = processor.emg_filter_and_rectify(
                    signal, fs, threshold_method='soft')
            except Exception as e:
                print(f"Error with {wavelet} soft threshold: {e}")

            # 硬阈值
            try:
                results[f'{wavelet}_hard'] = processor.emg_filter_and_rectify(
                    signal, fs, threshold_method='hard')
            except Exception as e:
                print(f"Error with {wavelet} hard threshold: {e}")

        return results

    def calculate_metrics(self, results: Dict) -> pd.DataFrame:
        """计算所有方法的评估指标"""
        metrics_data = []

        # 使用传统方法作为参考
        reference = results['traditional']

        for method, signal in results.items():
            if method in ['times', 'raw']:
                continue

            if method == 'traditional':
                # 传统方法与自身比较
                metrics = {
                    'method': method,
                    'rmse_to_ref': 0.0,
                    'correlation': 1.0,
                    'smoothness': self.evaluator.calculate_smoothness(signal),
                    'zero_crossing': self.evaluator.calculate_zero_crossing_rate(signal),
                    'median_freq': self.evaluator.calculate_median_frequency(signal, 1500)
                }
            else:
                metrics = {
                    'method': method,
                    'rmse_to_ref': self.evaluator.calculate_rmse(reference, signal),
                    'correlation': self.evaluator.calculate_correlation(reference, signal),
                    'smoothness': self.evaluator.calculate_smoothness(signal),
                    'zero_crossing': self.evaluator.calculate_zero_crossing_rate(signal),
                    'median_freq': self.evaluator.calculate_median_frequency(signal, 1500)
                }

            metrics_data.append(metrics)

        return pd.DataFrame(metrics_data)

    def plot_time_domain_comparison(self, results: Dict, muscle_name: str, save: bool = True):
        """绘制时域对比图"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        times = results['times']

        # 原始信号
        axes[0].plot(times, results['raw'], 'k-', alpha=0.7, linewidth=0.5)
        axes[0].set_title(f'{muscle_name} - Raw EMG')
        axes[0].set_ylabel('Amplitude')

        # 传统滤波
        axes[1].plot(times, results['traditional'], 'b-', alpha=0.7)
        axes[1].set_title('Traditional Bandpass Filter (15-500 Hz)')
        axes[1].set_ylabel('Amplitude')

        # 最佳小波方法（示例：db4软阈值）
        if 'db4_soft' in results:
            axes[2].plot(times, results['db4_soft'], 'r-', alpha=0.7)
            axes[2].set_title('Wavelet Denoising (db4, soft threshold)')
            axes[2].set_ylabel('Amplitude')

        # 对比图
        axes[3].plot(times, results['traditional'], 'b-', alpha=0.7, label='Traditional')
        if 'db4_soft' in results:
            axes[3].plot(times, results['db4_soft'], 'r-', alpha=0.7, label='Wavelet (db4)')
        axes[3].set_title('Comparison')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, f'{muscle_name}_time_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_frequency_domain_comparison(self, results: Dict, muscle_name: str, save: bool = True):
        """绘制频域对比图"""
        plt.figure(figsize=(12, 8))

        # 选择几个代表性方法进行比较
        methods_to_plot = ['raw', 'traditional', 'db4_soft', 'sym6_soft', 'coif3_hard']

        for method in methods_to_plot:
            if method in results:
                f, psd = self.evaluator.calculate_power_spectrum_density(results[method], 1500)
                label = method.replace('_', ' ').title()
                if method == 'raw':
                    plt.semilogy(f, psd, 'k-', alpha=0.5, linewidth=0.8, label=label)
                else:
                    plt.semilogy(f, psd, label=label, linewidth=1.5)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'{muscle_name} - Frequency Domain Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 500)

        if save:
            filename = os.path.join(self.output_dir, f'{muscle_name}_frequency_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame, muscle_name: str, save: bool = True):
        """绘制评估指标对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 相关系数条形图
        correlation_data = metrics_df.sort_values('correlation', ascending=False)
        sns.barplot(data=correlation_data, x='method', y='correlation', ax=axes[0])
        axes[0].set_title('Correlation with Traditional Method')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0].set_ylim(0.9, 1.0)

        # RMSE条形图
        rmse_data = metrics_df.sort_values('rmse_to_ref')
        sns.barplot(data=rmse_data, x='method', y='rmse_to_ref', ax=axes[1])
        axes[1].set_title('RMSE from Traditional Method')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 平滑度条形图（越小越平滑）
        smoothness_data = metrics_df.sort_values('smoothness')
        sns.barplot(data=smoothness_data, x='method', y='smoothness', ax=axes[2])
        axes[2].set_title('Smoothness (lower is smoother)')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 过零率条形图
        zc_data = metrics_df.sort_values('zero_crossing')
        sns.barplot(data=zc_data, x='method', y='zero_crossing', ax=axes[3])
        axes[3].set_title('Zero Crossing Rate')
        plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 中频条形图
        mf_data = metrics_df.sort_values('median_freq')
        sns.barplot(data=mf_data, x='method', y='median_freq', ax=axes[4])
        axes[4].set_title('Median Frequency (Hz)')
        plt.setp(axes[4].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 综合评分雷达图（需要归一化）
        # 在第6个子图位置创建极坐标
        ax_radar = plt.subplot(2, 3, 6, projection='polar')

        # 选择几个代表性方法
        methods_to_compare = ['traditional', 'db4_soft', 'sym6_soft', 'coif3_hard']
        colors = ['blue', 'red', 'green', 'purple']

        # 准备雷达图数据
        metrics = ['correlation', 'smoothness', 'zero_crossing', 'median_freq']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 完成闭合

        for method, color in zip(methods_to_compare, colors):
            if method in metrics_df['method'].values:
                values = []
                method_data = metrics_df[metrics_df['method'] == method].iloc[0]

                # 归一化到0-1（根据最好的方向）
                values.append(method_data['correlation'])  # 越大越好

                # 平滑度归一化（越小越好）
                smoothness_range = metrics_df['smoothness'].max() - metrics_df['smoothness'].min()
                if smoothness_range > 0:
                    smoothness_norm = 1 - (method_data['smoothness'] - metrics_df['smoothness'].min()) / smoothness_range
                else:
                    smoothness_norm = 0.5
                values.append(smoothness_norm)

                # 过零率归一化（越小越好）
                zc_range = metrics_df['zero_crossing'].max() - metrics_df['zero_crossing'].min()
                if zc_range > 0:
                    zc_norm = 1 - (method_data['zero_crossing'] - metrics_df['zero_crossing'].min()) / zc_range
                else:
                    zc_norm = 0.5
                values.append(zc_norm)

                # 中频归一化（保持中等最好）
                mf_range = metrics_df['median_freq'].max() - metrics_df['median_freq'].min()
                if mf_range > 0:
                    mf_norm = (method_data['median_freq'] - metrics_df['median_freq'].min()) / mf_range
                else:
                    mf_norm = 0.5
                values.append(mf_norm)

                values += values[:1]  # 完成闭合

                ax_radar.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
                ax_radar.fill(angles, values, alpha=0.25, color=color)
            else:
                print(f"警告：方法 '{method}' 在数据中不存在")

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Comprehensive Comparison')
        ax_radar.legend()

        plt.suptitle(f'{muscle_name} - Metrics Comparison', fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, f'{muscle_name}_metrics_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    def run_comparison_experiment(self, c3d_path: str, muscle_names: List[str]):
        """
        运行完整的比较实验

        Args:
            c3d_path: C3D文件路径
            muscle_names: 要处理的肌肉通道名称列表
        """
        all_metrics = []

        for muscle_name in muscle_names:
            print(f"\nProcessing muscle: {muscle_name}")

            # 加载数据
            try:
                signal, times, fs = self.load_emg_data(c3d_path, muscle_name)
            except Exception as e:
                print(f"Error loading {muscle_name}: {e}")
                continue

            # 处理信号
            results = self.process_single_muscle(signal, times, fs, muscle_name)

            # 计算评估指标
            metrics_df = self.calculate_metrics(results)
            metrics_df['muscle'] = muscle_name
            all_metrics.append(metrics_df)

            # 绘制对比图
            self.plot_time_domain_comparison(results, muscle_name)
            self.plot_frequency_domain_comparison(results, muscle_name)
            self.plot_metrics_comparison(metrics_df, muscle_name)

            # 保存最佳处理结果
            self.save_best_results(results, metrics_df, muscle_name)

        # 汇总所有肌肉的评估结果
        if all_metrics:
            full_metrics = pd.concat(all_metrics, ignore_index=True)
            self.plot_overall_comparison(full_metrics)

            # 保存评估报告
            report_path = os.path.join(self.output_dir, 'emg_comparison_report.csv')
            full_metrics.to_csv(report_path, index=False)
            print(f"\nEvaluation report saved to: {report_path}")

    def save_best_results(self, results: Dict, metrics_df: pd.DataFrame, muscle_name: str):
        """保存最佳处理结果"""
        # 基于综合评分选择最佳方法
        # 处理除零错误
        smoothness_range = metrics_df['smoothness'].max() - metrics_df['smoothness'].min()

        if smoothness_range > 0:
            smoothness_normalized = (metrics_df['smoothness'] - metrics_df['smoothness'].min()) / smoothness_range
            metrics_df['composite_score'] = (
                metrics_df['correlation'] * 0.5 +  # 与传统方法的相关性
                (1 - smoothness_normalized) * 0.5  # 平滑度（归一化，越小越好）
            )
        else:
            # 如果所有方法的平滑度相同，只基于相关性评分
            metrics_df['composite_score'] = metrics_df['correlation']

        best_method = metrics_df.loc[metrics_df['composite_score'].idxmax(), 'method']
        print(f"Best method for {muscle_name}: {best_method}")

        # 保存最佳结果到文件
        if best_method in results:
            output_data = np.column_stack((results['times'], results[best_method]))
            output_file = os.path.join(self.output_dir, f'{muscle_name}_{best_method}.sto')

            # 创建sto格式的header
            header = f"{muscle_name}_EMG\nversion=1\nnRows={len(output_data)}\nnColumns=2\n"
            header += f"inDegrees=no\nendheader\ntime\t{muscle_name}\n"

            np.savetxt(output_file, output_data, delimiter='\t', header=header, comments='')
            print(f"Best result saved to: {output_file}")

    def plot_overall_comparison(self, metrics_df: pd.DataFrame, save: bool = True):
        """绘制所有肌肉的整体对比图"""
        # 计算每种方法在所有肌肉上的平均表现
        avg_metrics = metrics_df.groupby('method').agg({
            'correlation': 'mean',
            'rmse_to_ref': 'mean',
            'smoothness': 'mean',
            'zero_crossing': 'mean',
            'median_freq': 'mean'
        }).reset_index()

        # 创建4个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 绘制平均相关系数
        sns.barplot(data=avg_metrics.sort_values('correlation', ascending=False),
                    x='method', y='correlation', ax=axes[0])
        axes[0].set_title('Average Correlation Across All Muscles')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0].set_ylim(0.9, 1.0)

        # 绘制平均RMSE
        sns.barplot(data=avg_metrics.sort_values('rmse_to_ref'),
                    x='method', y='rmse_to_ref', ax=axes[1])
        axes[1].set_title('Average RMSE Across All Muscles')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 绘制平均平滑度
        sns.barplot(data=avg_metrics.sort_values('smoothness'),
                    x='method', y='smoothness', ax=axes[2])
        axes[2].set_title('Average Smoothness Across All Muscles')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 绘制综合评分
        # 避免除零错误
        smoothness_range = avg_metrics['smoothness'].max() - avg_metrics['smoothness'].min()
        zc_range = avg_metrics['zero_crossing'].max() - avg_metrics['zero_crossing'].min()

        if smoothness_range > 0 and zc_range > 0:
            avg_metrics['composite_score'] = (
                avg_metrics['correlation'] * 0.4 +
                (1 - (avg_metrics['smoothness'] - avg_metrics['smoothness'].min()) / smoothness_range) * 0.3 +
                (1 - (avg_metrics['zero_crossing'] - avg_metrics['zero_crossing'].min()) / zc_range) * 0.3
            )
        else:
            # 如果范围为0，只基于相关性评分
            avg_metrics['composite_score'] = avg_metrics['correlation']

        sns.barplot(data=avg_metrics.sort_values('composite_score', ascending=False),
                    x='method', y='composite_score', ax=axes[3])
        axes[3].set_title('Composite Score (Higher is Better)')
        plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('Overall Comparison Across All Muscles', fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, 'overall_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

        # 打印最佳方法
        best_overall = avg_metrics.loc[avg_metrics['composite_score'].idxmax(), 'method']
        print(f"\nBest overall method: {best_overall}")
        print(f"Composite score: {avg_metrics.loc[avg_metrics['composite_score'].idxmax(), 'composite_score']:.3f}")


# 主函数
def main():
    """运行EMG处理比较实验"""
    # 设置项目路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 初始化比较实验
    comparison = EMGComparison(project_root)

    # 设置要处理的C3D文件和肌肉通道
    action_c3d = os.path.join(project_root, 'data', 'raw', 'N10_n', 'right', 'ROM_Ellenbogenflex_R 1.c3d')
    muscle_names = ['R_Biceps', 'R_Triceps', 'R_Wrist_Flex', 'R_Wrist_Ex']

    # 运行比较实验
    comparison.run_comparison_experiment(action_c3d, muscle_names)

    print("\nComparison experiment completed!")


if __name__ == '__main__':
    main()