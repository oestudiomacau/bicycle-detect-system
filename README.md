# 校园非机动车异常监测 GUI 原型

这是根据开题报告搭建的未接入海康威视 SDK 的桌面端框架。系统既支持模拟视频源，也可以读取本地 MP4 视频并使用 MobileNet SSD 检测自行车。

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
- 本地视频播放、真实自行车检测和简单质心跟踪

模拟源中的检测框由模拟引擎生成。本地视频中的自行车检测框来自 MobileNet SSD；速度结果仍依赖现场距离标定和稳定跟踪，停车模块当前只对真实检测框执行越界规则，倒地与异常堆放需要接入 Anomalib。

## 运行

可直接双击 `run_gui.bat`。第一次运行会自动创建 Python 环境并安装依赖。

也可以在 PowerShell 中运行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

数据会写入 `runtime/events.jsonl`，截图保存在 `runtime/snapshots/`。

工具栏中的“加载示例视频”可以直接选择：

- 固定机位公路车队通过视频，用于测速测试
- 密集自行车停车区视频，用于停车越界测试
- 停车区车辆清理视频，用于状态变化测试

视频来源与许可见 `sample_videos/SOURCES.md`。

## 后续接入点

海康设备接入位于 `campus_monitor/integrations.py` 的 `HikvisionSdkAdapter`：

1. 初始化 HCNetSDK 并完成设备登录。
2. 将实时预览回调中的码流解码为图像帧。
3. 实现道路和停车观察位对应的云台预置点切换。
4. 切换观察位期间暂停分析，并重置目标跟踪状态。

Anomalib 接入位于同文件的 `ParkingAnomalyAdapter`。真实检测阶段可将模型输出的异常分数、标签和掩膜转换为统一事件。

速度检测的核心计算和停车规则位于 `campus_monitor/domain.py`，模拟数据流位于 `campus_monitor/simulation.py`，便于后续用真实检测跟踪结果替换。
