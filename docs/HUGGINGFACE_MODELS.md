# Hugging Face 模型调研

调研日期：2026-09-15。下载量为调研时 Hugging Face API 返回值，会随时间变化。

## 推荐候选

| 模型 | 用途 | 下载量 | 许可证 | 结论 |
| --- | --- | ---: | --- | --- |
| [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) | COCO 目标检测，包含 `bicycle` | 669,647 | Apache-2.0 | 成熟、许可证友好，可直接替换当前检测器，但速度和显存占用高于 MobileNet SSD |
| [PekingU/rtdetr_r50vd_coco_o365](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) | 实时目标检测，包含 COCO 类别 | 297,358 | Apache-2.0 | 已接入当前 GUI，使用独立 CUDA worker，YOLOX-Tiny 自动回退 |
| [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11) | COCO 目标检测，包含 `bicycle` | 15,354 | AGPL-3.0 | 接入简单、速度快，但发布和商用前必须确认 AGPL 义务 |
| [timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k) | Anomalib PatchCore/PaDiM 特征骨干 | 1,820,626 | Apache-2.0 | 成熟的预训练特征提取器；仍需用本机位正常停车图片建立异常记忆库 |

## 不建议直接接入

| 模型 | 原因 |
| --- | --- |
| [bmombie/bicycle_rider_detector_002](https://huggingface.co/bmombie/bicycle_rider_detector_002) | YOLOv8n 单类 `bicycle_rider`，只有约 32 次下载，模型卡未声明许可证，成熟度不足 |
| [mcity-data-engine/fisheye8k_anomalib_Padim](https://huggingface.co/mcity-data-engine/fisheye8k_anomalib_Padim) | 是 Anomalib 权重，但训练目标是鱼眼相机中的 `Truck` 异常，不是自行车违停 |
| [mcity-data-engine/fisheye8k_anomalib_EfficientAd](https://huggingface.co/mcity-data-engine/fisheye8k_anomalib_EfficientAd) | 同样针对 Fisheye8K 的卡车异常，机位与业务不匹配 |
| [Wimflorijn/parking-yolo-v8](https://huggingface.co/Wimflorijn/parking-yolo-v8) | 类别是 `parkeergarage`，用于识别停车场场景，不是识别违停车辆 |

## 采用建议

1. 当前版本优先使用 RT-DETR R50 检测自行车和摩托车类别，在 RTX 2060 上完成验证。
2. RT-DETR 未安装、正在加载或运行失败时，自动回退到 YOLOX-Tiny 和 MobileNet SSD。
3. “越界停放”使用 GUI 绘制的停车边界和几何规则判断，不需要异常模型。
4. “倒地、堆放、姿态异常”没有找到成熟且业务匹配的现成权重，应使用当前 Anomalib 训练页，以现场固定机位图片微调 PatchCore 或 PaDiM。

实测固定帧中，RT-DETR 在密集停车、停车清理、道路骑行三个样例分别检测到 20、3、8 个两轮车目标（阈值 0.30），高于旧检测链路。结论：Hugging Face 上有成熟的自行车目标检测基础模型，但没有找到可直接可靠判断校园自行车违停、倒地和堆放的成熟模型。
