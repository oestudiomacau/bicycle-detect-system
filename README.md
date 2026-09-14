# 校园非机动车异常监测 GUI 原型

这是根据开题报告搭建的未接入海康威视 SDK 的桌面端框架。系统既支持模拟视频源，也可以读取本地视频并使用 Hugging Face RT-DETR 检测自行车、电动车和摩托车。

## 已实现

- 道路观察位与停车观察位切换
- 双虚拟线计时与区间平均速度计算
- 超速阈值配置和告警截图
- 停车区域越界判断
- 倒地、异常堆放模拟结果和异常分数展示
- 实时指标、事件表格、JSONL 历史记录
- 手动截图和告警自动截图
- 海康 HCNetSDK、Anomalib 模型适配接口
- 三段开放许可示例视频和快速加载菜单
- 本地视频播放、真实两轮车检测和简单质心跟踪
- 自定义启动示范视频，保存视频路径和监测模式并在下次启动时自动播放
- 在停车画面中拖拽标定规定停车区域，并自动保存配置
- 在测速画面中拖拽绘制任意方向的虚拟线 A、B
- 在违停画面中逐点绘制并闭合不规则停车边界
- Anomalib 2.6.2 独立训练页面，支持 PatchCore、PaDiM、CUDA/CPU 和模型导出

模拟源中的检测框由模拟引擎生成。本地视频在已安装时优先使用 `PekingU/rtdetr_r50vd_coco_o365`，模型通过独立 CUDA 进程异步运行；加载期间或不可用时自动回退到 YOLOX-Tiny 和 MobileNet SSD。速度结果仍依赖现场距离标定和稳定跟踪，停车模块当前只对真实检测框执行越界规则，倒地与异常堆放需要接入 Anomalib。

## 运行

可直接双击 `run_gui.bat`。第一次运行会自动创建 Python 环境并安装依赖。

也可以在 PowerShell 中运行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

数据会写入 `runtime/events.jsonl`，截图保存在 `runtime/snapshots/`。
人工标定的停车区域保存在 `runtime/settings.json`。

### 安装 RT-DETR 高精度检测模型

RT-DETR R50 权重约 172MB，不会提交到 GitHub。首次使用执行：

```powershell
.\.venv\Scripts\python.exe scripts\setup_rtdetr_model.py
```

RTX 2060 默认使用 CUDA 12.6。安装后直接启动 GUI，右侧“车辆检测模型”会从
“RT-DETR R50 正在加载”切换为“Hugging Face RT-DETR R50 · CUDA”。未安装时软件仍会
使用内置 YOLOX-Tiny。

### 自定义启动示范视频

首次运行默认自动播放内置测速示范。需要替换时：

1. 先切换到“道路观察位”或“停车观察位”。
2. 点击工具栏“自定义示范 → 选择并设为启动示范”。
3. 此后正常双击 `run_gui.bat`，程序会自动进入该模式并播放所选视频。

自定义视频被移动或删除后会自动使用内置测速示范；菜单中的“取消启动自动播放”可让
程序下次从模拟源启动。

工具栏中的“加载示例视频”可以直接选择：

- 固定机位公路车队通过视频，用于测速测试
- 密集自行车停车区视频，用于停车越界测试
- 停车区车辆清理视频，用于状态变化测试

视频来源与许可见 `sample_videos/SOURCES.md`。

也可以直接从命令行加载视频：

```powershell
.\.venv\Scripts\python.exe main.py --video sample_videos\road_cyclists.mp4
.\.venv\Scripts\python.exe main.py --parking --video sample_videos\parking_dense.mp4
```

## 后续接入点

海康设备接入位于 `campus_monitor/integrations.py` 的 `HikvisionSdkAdapter`：

1. 初始化 HCNetSDK 并完成设备登录。
2. 将实时预览回调中的码流解码为图像帧。
3. 实现道路和停车观察位对应的云台预置点切换。
4. 切换观察位期间暂停分析，并重置目标跟踪状态。

Anomalib 接入位于同文件的 `ParkingAnomalyAdapter`。真实检测阶段可将模型输出的异常分数、标签和掩膜转换为统一事件。

## 停车区域人工标定

1. 切换到“违规停放监测”。
2. 点击右侧“标定区域”，视频会自动暂停。
3. 在画面中按住鼠标左键拖出矩形，点击“确认区域”。

也可以点击“绘制边界”，依次点击多个顶点，用连续线段围出不规则停车区；右键撤销上一点，确认后自动闭合。

## 测速线人工标定

1. 切换到“行驶速度检测”。
2. 点击右侧“绘制 A / B”。
3. 先拖拽虚拟线 A，再拖拽虚拟线 B，最后确认。

目标中心轨迹必须与两条有限线段依次相交才会测速。线段位置会自动保存，但“双线实际距离”仍需按现场真实距离填写。

停车边界和测速线会同时作用于模拟源和本地视频检测。“恢复默认”可还原初始标定。

## Anomalib 训练

侧边栏进入“异常模型训练”，首次使用时点击“安装 / 更新 Anomalib”。训练依赖安装在独立的 `.venv-anomalib` 中，不会改变主 GUI 环境。

- 默认模型：PatchCore + ResNet18
- 轻量备选：PaDiM + ResNet18
- 推荐设备：RTX 2060 使用 CUDA 12.6
- 已在 RTX 2060 6GB 上验证；默认 `256 px / batch 4`，显存不足时将 batch 调到 1 或 2
- 输入数据：正常停车图片必需，异常测试图片可选
- 输出位置：`artifacts/anomalib/<训练时间>/`
- 导出结果：Torch `.pt`，以及可选的 OpenVINO 模型

数据目录示例和采集建议见 `datasets/README.md`。训练环境也可在命令行安装：

```powershell
.\.venv\Scripts\python.exe scripts\setup_anomalib_env.py --env .venv-anomalib --backend cu126
```

Hugging Face 现成模型的许可证、成熟度和适用性对比见 `docs/HUGGINGFACE_MODELS.md`。

速度检测的核心计算和停车规则位于 `campus_monitor/domain.py`，模拟数据流位于 `campus_monitor/simulation.py`，便于后续用真实检测跟踪结果替换。
