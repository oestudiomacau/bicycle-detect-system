from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .anomalib_training import dataset_summary, validate_dataset


class PathPicker(QWidget):
    def __init__(self, title: str, initial: Path | None, optional: bool = False) -> None:
        super().__init__()
        self.title = title
        self.optional = optional
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QLineEdit(str(initial) if initial else "")
        self.line_edit.setClearButtonEnabled(optional)
        browse = QPushButton()
        browse.setToolTip(f"选择{title}")
        browse.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DirOpenIcon))
        browse.setFixedWidth(38)
        browse.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(browse)

    def path(self) -> Path | None:
        value = self.line_edit.text().strip()
        return Path(value) if value else None

    def _browse(self) -> None:
        initial = self.line_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, f"选择{self.title}", initial)
        if selected:
            self.line_edit.setText(selected)


class AnomalibTrainingPage(QWidget):
    status_changed = Signal(str)

    def __init__(self, root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root
        self.environment = root / ".venv-anomalib"
        self.environment_python = self.environment / "Scripts" / "python.exe"
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process_environment = QProcessEnvironment.systemEnvironment()
        process_environment.insert("PYTHONUTF8", "1")
        process_environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(process_environment)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process_kind = ""
        self._build_ui()
        QTimer.singleShot(0, self.check_environment)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Anomalib 异常模型训练")
        title.setObjectName("pageTitle")
        subtitle = QLabel("停车场景单类异常检测 · 独立训练环境 · 模型导出")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        environment_bar = QFrame()
        environment_bar.setObjectName("toolbar")
        environment_layout = QHBoxLayout(environment_bar)
        environment_layout.setContentsMargins(12, 9, 12, 9)
        self.environment_status = QLabel("正在检查训练环境...")
        self.environment_status.setObjectName("environmentStatus")
        environment_layout.addWidget(self.environment_status)
        environment_layout.addStretch()
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("CUDA 12.6 · RTX 2060", "cu126")
        self.backend_combo.addItem("CPU", "cpu")
        environment_layout.addWidget(self.backend_combo)
        self.check_button = QPushButton("检查环境")
        self.check_button.clicked.connect(self.check_environment)
        environment_layout.addWidget(self.check_button)
        self.setup_button = QPushButton("安装 / 更新 Anomalib")
        self.setup_button.setObjectName("primaryButton")
        self.setup_button.clicked.connect(self.setup_environment)
        environment_layout.addWidget(self.setup_button)
        layout.addWidget(environment_bar)

        content = QHBoxLayout()
        content.setSpacing(14)
        config_panel = QFrame()
        config_panel.setObjectName("panel")
        config_panel.setFixedWidth(430)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(18, 16, 18, 16)
        config_layout.setSpacing(12)
        config_title = QLabel("训练配置")
        config_title.setObjectName("panelTitle")
        config_layout.addWidget(config_title)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.normal_picker = PathPicker("正常停车图片目录", self.root / "datasets" / "parking" / "normal")
        self.abnormal_picker = PathPicker(
            "异常测试图片目录",
            None,
            optional=True,
        )
        self.output_picker = PathPicker("训练输出目录", self.root / "artifacts" / "anomalib")
        self.model_combo = QComboBox()
        self.model_combo.addItem("PatchCore · ResNet18（推荐）", "patchcore")
        self.model_combo.addItem("PaDiM · ResNet18（更省资源）", "padim")
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择", "auto")
        self.device_combo.addItem("CUDA 显卡", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.export_combo = QComboBox()
        self.export_combo.addItem("Torch + OpenVINO", "both")
        self.export_combo.addItem("仅 Torch", "torch")
        self.export_combo.addItem("仅 OpenVINO", "openvino")
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(128, 512)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(256)
        self.image_size_spin.setSuffix(" px")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)

        form.addRow("正常训练图片", self.normal_picker)
        form.addRow("异常测试图片", self.abnormal_picker)
        form.addRow("输出目录", self.output_picker)
        form.addRow("模型", self.model_combo)
        form.addRow("训练设备", self.device_combo)
        form.addRow("图像尺寸", self.image_size_spin)
        form.addRow("批量大小", self.batch_size_spin)
        form.addRow("导出格式", self.export_combo)
        config_layout.addLayout(form)

        self.dataset_status = QLabel("尚未检查数据集")
        self.dataset_status.setObjectName("datasetStatus")
        self.dataset_status.setWordWrap(True)
        config_layout.addWidget(self.dataset_status)
        config_layout.addStretch()

        actions = QGridLayout()
        self.dataset_button = QPushButton("检查数据集")
        self.dataset_button.clicked.connect(self.check_dataset)
        self.start_button = QPushButton("开始训练")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.clicked.connect(self.open_output_directory)
        actions.addWidget(self.dataset_button, 0, 0)
        actions.addWidget(self.start_button, 0, 1)
        actions.addWidget(self.stop_button, 1, 0)
        actions.addWidget(self.open_output_button, 1, 1)
        config_layout.addLayout(actions)
        content.addWidget(config_panel)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(10)
        log_header = QHBoxLayout()
        log_title = QLabel("训练日志")
        log_title.setObjectName("panelTitle")
        log_header.addWidget(log_title)
        log_header.addStretch()
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(lambda: self.log.clear())
        log_header.addWidget(clear_button)
        log_layout.addLayout(log_header)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        log_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("trainingLog")
        self.log.setPlaceholderText("环境检查、训练进度和模型导出结果会显示在这里。")
        log_layout.addWidget(self.log, 1)
        content.addWidget(log_panel, 1)
        layout.addLayout(content, 1)

    def check_environment(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        if not self.environment_python.exists():
            self.environment_status.setText("Anomalib 训练环境未安装")
            self.environment_status.setProperty("ready", False)
            self.environment_status.style().unpolish(self.environment_status)
            self.environment_status.style().polish(self.environment_status)
            return
        command = (
            "from importlib.metadata import version; import torch; "
            "print('Anomalib ' + version('anomalib')); "
            "print('PyTorch ' + torch.__version__); "
            "print('CUDA=' + str(torch.cuda.is_available())); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); "
            + (
                "assert torch.cuda.is_available(), 'CUDA 12.6 environment is not active'"
                if self.backend_combo.currentData() == "cu126"
                else ""
            )
        )
        self._start_process(str(self.environment_python), ["-c", command], "check")

    def setup_environment(self) -> None:
        backend = str(self.backend_combo.currentData())
        script = self.root / "scripts" / "setup_anomalib_env.py"
        self.log.appendPlainText(f"\n安装 Anomalib 2.6.2，计算后端：{backend}")
        self._start_process(
            sys.executable,
            [str(script), "--env", str(self.environment), "--backend", backend],
            "setup",
        )

    def check_dataset(self, show_message: bool = False) -> bool:
        normal_dir = self.normal_picker.path()
        abnormal_dir = self.abnormal_picker.path()
        if normal_dir is None:
            self.dataset_status.setText("请选择正常停车图片目录")
            return False
        try:
            summary = validate_dataset(normal_dir, abnormal_dir)
        except ValueError as error:
            summary = dataset_summary(normal_dir, abnormal_dir)
            self.dataset_status.setText(
                f"数据不可用：{error}（正常 {summary['normal']} 张，异常 {summary['abnormal']} 张）"
            )
            if show_message:
                QMessageBox.warning(self, "数据集检查", self.dataset_status.text())
            return False
        note = "使用真实异常图片评估" if abnormal_dir else "未提供异常图片，将自动合成测试异常"
        self.dataset_status.setText(
            f"数据可用：正常 {summary['normal']} 张，异常 {summary['abnormal']} 张；{note}"
        )
        return True

    def start_training(self) -> None:
        if not self.environment_python.exists():
            QMessageBox.warning(self, "训练环境", "请先安装 Anomalib 训练环境。")
            return
        if not self.check_dataset(show_message=True):
            return
        normal_dir = self.normal_picker.path()
        output_dir = self.output_picker.path()
        abnormal_dir = self.abnormal_picker.path()
        if normal_dir is None or output_dir is None:
            QMessageBox.warning(self, "训练配置", "正常图片目录和输出目录不能为空。")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = [
            "-m",
            "campus_monitor.anomalib_training",
            "--normal-dir",
            str(normal_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--model",
            str(self.model_combo.currentData()),
            "--device",
            str(self.device_combo.currentData()),
            "--image-size",
            str(self.image_size_spin.value()),
            "--batch-size",
            str(self.batch_size_spin.value()),
            "--export",
            str(self.export_combo.currentData()),
        ]
        if abnormal_dir is not None:
            arguments.extend(["--abnormal-dir", str(abnormal_dir.resolve())])
        self.log.appendPlainText("\n开始新的训练任务")
        self._start_process(str(self.environment_python), arguments, "training")

    def stop_process(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.log.appendPlainText("正在停止当前任务...")
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def open_output_directory(self) -> None:
        output = self.output_picker.path()
        if output is None:
            return
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.resolve())))

    def shutdown(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()
            self.process.waitForFinished(1000)

    def _start_process(self, program: str, arguments: list[str], kind: str) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "任务运行中", "请先等待或停止当前任务。")
            return
        self.process_kind = kind
        self.progress.setRange(0, 0)
        self._set_running(True)
        if kind == "check":
            self.environment_status.setText("正在检查训练环境...")
        self.process.setWorkingDirectory(str(self.root))
        self.process.start(program, arguments)

    def _read_process_output(self) -> None:
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(output)
            self.log.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        kind = self.process_kind
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        self._set_running(False)
        if kind == "check":
            ready = exit_code == 0
            self.environment_status.setText("Anomalib 2.6.2 已就绪" if ready else "训练环境检查失败")
            self.environment_status.setProperty("ready", ready)
            self.environment_status.style().unpolish(self.environment_status)
            self.environment_status.style().polish(self.environment_status)
        elif kind == "setup":
            if exit_code == 0:
                self.environment_status.setText("安装完成，正在验证...")
                QTimer.singleShot(100, self.check_environment)
            else:
                self.environment_status.setText("Anomalib 安装失败，请查看日志")
        elif kind == "training":
            message = "训练完成，模型已写入输出目录" if exit_code == 0 else "训练失败，请查看日志末尾"
            self.status_changed.emit(message)
            self.log.appendPlainText("\n" + message)
        self.process_kind = ""

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        self.log.appendPlainText(f"无法启动进程：{self.process.errorString()}")
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.setup_button.setEnabled(not running)
        self.check_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.dataset_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
