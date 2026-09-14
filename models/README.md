# 两轮车检测模型

安装 RT-DETR 后，默认优先使用 Hugging Face RT-DETR R50 检测 COCO 中的 `bicycle`
和 `motorbike` 类别。模型在独立的 CUDA 进程中异步推理，未就绪或不可用时自动回退
到 YOLOX-Tiny，再由 MobileNet SSD 补充检测。

## Hugging Face RT-DETR R50

- 模型：https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365
- 许可：Apache-2.0
- 参数量：约 42M
- 本地目录：`rtdetr_r50vd_coco_o365/`
- 安装命令：`.\.venv\Scripts\python.exe scripts\setup_rtdetr_model.py`（在项目根目录执行）

权重文件约 172MB，超过 GitHub 普通单文件限制，因此该目录不会提交到仓库。

## YOLOX-Tiny

- 项目：https://github.com/Megvii-BaseDetection/YOLOX
- 许可：Apache-2.0
- 许可文件：`YOLOX_LICENSE`
- 模型文件：`yolox_tiny.onnx`
- 来源：https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0

## MobileNet SSD

本目录使用 `chuanqi305/MobileNet-SSD` 提供的 VOC0712 预训练模型，检测 `bicycle`
和 `motorbike` 类别。

- 项目：https://github.com/chuanqi305/MobileNet-SSD
- 许可：MIT
- 模型文件：`mobilenet_ssd.caffemodel`
- 网络定义：`mobilenet_ssd.prototxt`

正式部署仍建议使用现场采集的校园非机动车数据进行微调。
