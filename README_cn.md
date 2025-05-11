我会将这个文档翻译成中文。

# OpenSim上肢生物力学分析流水线

## 项目概述

本项目实现了一个完整的OpenSim数据处理流水线，专注于上肢生物力学分析，特别是手腕和手指肌肉的肌电图(EMG)和运动学分析。该项目整合了数据收集、处理、分析和可视化的整个过程，结合了肌肉协同理论和神经网络方法。

## 功能特点

- **C3D数据转换**：将动作捕捉数据转换为OpenSim兼容格式
- **EMG信号处理**：小波变换预处理和MVIC标准化
- **OpenSim核心功能**：模型缩放、反向运动学、反向动力学、正向动力学
- **肌肉协同分析**：基于非负矩阵分解(NMF)的肌肉协同提取
- **神经网络集成**：训练神经网络将肌肉协同映射到关节动力学
- **痉挛分析**：分析慢速和快速被动拉伸期间的EMG差异

## 项目结构

```
opensim_pipeline/
│
├── config/                          # 配置文件
│   ├── __init__.py
│   ├── opensim_config.yaml          # OpenSim配置参数
│   ├── emg_config.yaml              # EMG处理参数  
│   ├── models/                      # OpenSim模型文件
│   │   ├── upper_limb_hand.osim     # 上肢和手部模型
│   │   └── markers.xml              # 标记定义
│   └── setup                        # 配置文件(.xml)
│
├── data/                            # 数据存储
│   ├── raw/                         # 原始实验数据
│   │   ├── emg/                     # 原始EMG数据
│   │   ├── motion_capture/          # 原始动作捕捉数据(C3D)
│   │   └── mvic/                    # 最大随意等长收缩数据
│   ├── processed/                   # 处理后的数据
│   │   ├── emg_processed/                     # 处理后的EMG
│   │   ├── kinematics/              # 处理后的运动学数据
│   │   └── synergies/               # 提取的肌肉协同
│   └── results/                     # 分析结果
│       ├── figures/                 # 生成的图表
│       └── tables/                  # 生成的表格
│
├── src/                             # 源代码
│   ├── __init__.py
│   │
│   ├── preprocessing/               # 数据预处理
│   │   ├── __init__.py
│   │   ├── c3d_converter.py         # 转换为trc和mot
│   │   ├── emg_processing.py        # EMG信号处理
│   │   ├── wavelet_transform.py     # EMG小波变换
│   │   ├── normalization.py         # 数据标准化(MVIC)
│   │   └── kinematics_processing.py # 运动学数据处理
│   │
│   ├── opensim/                     # OpenSim封装
│   │   ├── __init__.py
│   │   ├── scaling.py               # 模型缩放工具
│   │   ├── inverse_kinematics.py    # 反向运动学
│   │   ├── inverse_dynamics.py      # 反向动力学
│   │   ├── forward_dynamics.py      # 正向动力学
│   │   └── visualization.py         # OpenSim可视化
│   │
│   ├── synergy/                     # 肌肉协同分析
│   │   ├── __init__.py
│   │   ├── nmf.py                   # 非负矩阵分解
│   │   ├── synergy_extraction.py    # 肌肉协同提取
│   │   └── synergy_analysis.py      # 协同分析工具
│   │
│   ├── neural_network/              # 神经网络模型
│   │   ├── __init__.py
│   │   ├── models.py                # 神经网络模型定义
│   │   ├── training.py              # 模型训练工具
│   │   └── evaluation.py            # 模型评估
│   │
│   ├── analysis/                    # 数据分析
│   │   ├── __init__.py
│   │   ├── spasticity_analysis.py   # 痉挛分析(EMG_HV - EMG_LV)
│   │   ├── range_of_motion.py       # 运动范围分析
│   │   └── statistical_analysis.py  # 统计分析
│   │
│   └── visualization/               # 可视化工具
│       ├── __init__.py
│       ├── plot_emg.py              # EMG绘图
│       ├── plot_kinematics.py       # 运动学绘图
│       └── plot_synergies.py        # 协同绘图
│
├── scripts/                         # 可执行脚本
│   ├── process_c3d.py               # 处理C3D文件
│   ├── preprocess_emg.py            # 预处理EMG数据
│   ├── extract_synergies.py         # 提取肌肉协同
│   ├── scale_model.py               # 缩放OpenSim模型
│   ├── run_ik.py                    # 运行反向运动学
│   ├── run_id.py                    # 运行反向动力学
│   ├── train_nn.py                  # 训练神经网络
│   └── run_fd.py                    # 运行正向动力学
│
├── notebooks/                       # Jupyter笔记本
│   ├── emg_preprocessing.ipynb      # EMG预处理
│   ├── kinematic_analysis.ipynb     # 运动学分析
│   ├── synergy_extraction.ipynb     # 协同提取分析
│   └── neural_network_training.ipynb # 神经网络训练
│
├── tests/                           # 单元测试
│   ├── __init__.py
│   ├── test_c3d_converter.py        # 测试C3D转换
│   ├── test_emg_processing.py       # 测试EMG处理
│   ├── test_scaling.py              # 测试模型缩放
│   └── test_nmf.py                  # 测试NMF实现
│
├── requirements.txt                 # Python依赖
├── setup.py                         # 包安装脚本
└── README.md                        # 项目文档
```

## 安装指南

### 先决条件

- Python 3.8+
- OpenSim 4.5+
- Numpy, Scipy, Pandas, Matplotlib
- PyTorch或TensorFlow
- PyWavelets

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/opensim_pipeline.git
cd opensim_pipeline
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 安装OpenSim Python API
```bash
# 根据官方OpenSim指南按照您的操作系统安装适当的版本
```

5. 安装此项目包
```bash
pip install -e .
```

## 实验方法和参数

### 表面肌电图(sEMG)

#### 测量的肌肉群
- **腕屈肌**：桡侧腕屈肌(FCR)，尺侧腕屈肌(FCU)
- **腕伸肌**：桡侧腕长伸肌(ECRL)，桡侧腕短伸肌(ECRB)，尺侧腕伸肌(ECU)
- **手指(包括拇指)长屈肌和伸肌**：
  - 指浅屈肌(FDS)，指深屈肌(FDP)，总指伸肌(EDC)
  - 拇长屈肌(FPL)和拇长伸肌(EPL)

#### 放置指南
- 遵循SENIAM指南进行电极放置和程序

#### 标准化
- 使用最大随意等长收缩(MVIC)测试
- 将原始EMG值转换为%MVIC进行比较

#### 避免主动参与
- 监测拮抗肌活动以排除主动努力的干扰

### 3D动作捕捉

#### 基本原理
- 使用红外反射标记
- 多个摄像机同时捕捉标记空间位置的变化

#### 采用的模型
- **U.L.E.M.A.模型**：用于大型上肢关节(肩、肘、腕)
- **HAWK 26标记集**：用于手指关节的三维运动学信息

#### 虚拟标记和指针校准技术
- 使用"集群"减少皮肤移动引起的误差
- 更精确地识别关节中心和运动轴

#### 测量参数
- 关节角度(例如，掌指关节，近指关节，远指关节，腕关节)
- 角速度(一阶导数)

### 实验流程

#### 被动拉伸(慢速vs.快速)
- **慢速**：目标30°/s
- **快速**：目标180°/s
- 每种速度3次重复，间隔5秒休息

#### 测量顺序
- 不同条件下的被动腕伸展
- 每个手指或拇指的单独拉伸

### 关键输出指标

#### EMG参数
- **EMG_LV**：慢速被动拉伸期间的平均EMG(%MVIC)
- **EMG_HV**：快速拉伸期间的平均EMG
- **EMG_change = EMG_HV - EMG_LV**：反映痉挛程度

#### 最大伸展限制角度
- 每个关节可达到的最大伸展范围

## 使用指南

### 数据预处理

1. 处理C3D文件
```bash
python scripts/process_c3d.py --input data/raw/motion_capture --output data/processed/kinematics
```

2. 预处理EMG数据
```bash
python scripts/preprocess_emg.py --input data/raw/emg --output data/processed/emg --config config/emg_config.yaml
```

3. 提取肌肉协同
```bash
python scripts/extract_synergies.py --input data/processed/emg --output data/processed/synergies --num-synergies 4
```

### OpenSim工作流

1. 缩放模型
```bash
python scripts/scale_model.py --model config/models/upper_limb_hand.osim --markers data/processed/kinematics/static_trial.trc --output models/scaled_model.osim
```

2. 运行反向运动学
```bash
python scripts/run_ik.py --model models/scaled_model.osim --markers data/processed/kinematics/dynamic_trial.trc --output data/processed/kinematics/joint_angles.mot
```

3. 训练神经网络
```bash
python scripts/train_nn.py --synergies data/processed/synergies --kinematics data/processed/kinematics/joint_angles.mot --output models/nn_model.pt
```

4. 运行正向动力学
```bash
python scripts/run_fd.py --model models/scaled_model.osim --nn-model models/nn_model.pt --synergies data/processed/synergies/test_data.csv --output data/results/fd_results.mot
```

### 分析和可视化

1. 痉挛分析
```bash
python scripts/analyze_spasticity.py --emg-slow data/processed/emg/slow_stretch.csv --emg-fast data/processed/emg/fast_stretch.csv --output data/results/spasticity_analysis.csv
```

2. 使用Jupyter笔记本进行交互式分析
```bash
jupyter notebook notebooks/
```

## 技术实现亮点

### 1. 小波变换预处理

- 选择适当的小波基函数(Daubechies db4)
- 3-5级小波分解
- 阈值去噪
- 使用PyWavelets库实现

### 2. 肌肉协同提取(NMF)

- 使用sklearn.decomposition.NMF或nimfa库
- EMG ≈ W × H矩阵分解
- 选择适当的协同数R(2-4)

### 3. 神经网络映射

- 输入：肌肉协同激活系数h(t)
- 输出：关节角度、角速度或肌肉激活值
- 网络规模：2-4层MLP，每层32-128个神经元
- 避免过拟合：数据增强，交叉验证

### 4. OpenSim集成

- 离线预测 + OpenSim驱动
- 通过Python API创建自定义控制器
- 评估：RMSE，相关系数，DTW误差

## 开发者指南

### 代码标准
- 遵循PEP 8
- 使用类型注解
- 为所有函数编写文档字符串

### 测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_emg_processing.py
```

### 贡献指南
1. Fork项目
2. 创建特性分支
3. 提交您的更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详情见LICENSE文件

## 作者

- 您的姓名 - [您的电子邮件]

# OpenSim模型缩放工具

此目录包含将OpenSim模型缩放并将结果保存到指定位置的脚本。

## 文件

- `scale_model.py`：使用Scale类缩放OpenSim模型的主Python脚本。
- `run_scale_model.bat`：一个Windows批处理文件，简化带参数运行Python脚本的过程。

## 要求

- Python 3.6+
- OpenSim Python API
- opensimarmmodel-main模块中的Scale类

## 使用方法

### 使用批处理文件

1. 编辑`run_scale_model.bat`文件以设置正确的路径和参数：
   - `MODEL_INPUT`：通用OpenSim模型的路径(.osim文件)
   - `XML_INPUT`：通用缩放XML配置的路径
   - `STATIC_PATH`：静态试验的路径(.trc文件)
   - `MASS`：参与者的质量(kg)
   - `HEIGHT`：参与者的身高(mm)
   - `AGE`：参与者的年龄(岁)
   - `OUTPUT_NAME`：输出文件的基本名称

2. 通过双击或从命令提示符运行批处理文件。

### 直接使用Python脚本

您也可以直接运行Python脚本，并带有所需的参数：

```
python scale_model.py --model_input <模型路径> --xml_input <XML路径> --static_path <TRC路径> --mass <体重kg> [--height <身高mm>] [--age <年龄>] [--add_model <附加模型路径>] [--remove_unused] [--output_name <基本名称>]
```

### 必需参数

- `--model_input`：通用模型的路径(.osim文件)
- `--xml_input`：通用缩放XML配置的路径
- `--static_path`：静态试验的路径(.trc文件)
- `--mass`：参与者的质量(kg)

### 可选参数

- `--height`：参与者的身高(mm)(默认：-1)
- `--age`：参与者的年龄(岁)(默认：-1)
- `--add_model`：要附加的模型路径
- `--remove_unused`：移除未使用标记的标志
- `--output_name`：输出文件的基本名称(默认：'scaled_model')

## 输出

所有输出文件将保存到：
```
C:/temporary_file/BG_klinik/newPipeline/data/processed
```

该脚本将生成：
- 缩放后的OpenSim模型(.osim文件)
- 缩放配置XML文件
- 可能基于Scale类功能的其他文件

## 故障排除

如果遇到错误：

1. 确保正确安装Python和OpenSim
2. 检查所有输入文件的路径是否正确
3. 验证Python环境中是否可访问Scale类
4. 确保输出目录存在或可以创建