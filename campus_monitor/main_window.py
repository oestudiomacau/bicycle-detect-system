from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .domain import (
    DEFAULT_PARKING_POLYGON,
    DEFAULT_SPEED_LINES,
    EventStore,
    LineSegment,
    MonitorEvent,
    Point,
)
from .integrations import HikvisionSdkAdapter, ParkingAnomalyAdapter
from .settings import SettingsStore
from .simulation import SimulationEngine
from .training_page import AnomalibTrainingPage
from .video_analysis import RTDetrWorkerDetector, VideoAnalysisController
from .widgets import MetricCard, VideoCanvas


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
SNAPSHOT_DIR = RUNTIME_DIR / "snapshots"
SAMPLE_VIDEO_DIR = ROOT / "sample_videos"
MODEL_DIR = ROOT / "models"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("校园非机动车行驶与停放异常监测系统")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)

        self.snapshot_dir = SNAPSHOT_DIR
        self.engine = SimulationEngine(self)
        rtdetr = RTDetrWorkerDetector(
            ROOT / ".venv-anomalib" / "Scripts" / "python.exe",
            ROOT / "scripts" / "rtdetr_worker.py",
            MODEL_DIR / "rtdetr_r50vd_coco_o365",
            RUNTIME_DIR,
        )
        self.video_controller = VideoAnalysisController(
            MODEL_DIR / "mobilenet_ssd.prototxt",
            MODEL_DIR / "mobilenet_ssd.caffemodel",
            yolox_weights=MODEL_DIR / "yolox_tiny.onnx",
            rtdetr=rtdetr,
            parent=self,
        )
        self.active_source = "simulation"
        self.event_store = EventStore(RUNTIME_DIR / "events.jsonl")
        self.settings_store = SettingsStore(RUNTIME_DIR / "settings.json")
        self.hikvision = HikvisionSdkAdapter()
        self.anomalib = ParkingAnomalyAdapter()
        self.events: list[MonitorEvent] = self.event_store.load()
        self.warning_count = sum(1 for item in self.events if item.severity == "warning")
        self._calibration_was_running = False
        parking_polygon = self.settings_store.load_parking_polygon()
        speed_lines = self.settings_store.load_speed_lines()
        self.engine.set_parking_polygon(parking_polygon)
        self.video_controller.set_parking_polygon(parking_polygon)
        self.engine.set_speed_lines(speed_lines)
        self.video_controller.set_speed_lines(speed_lines)

        self._apply_theme()
        self._build_ui()
        self.engine.event_raised.connect(self._handle_event)
        self.engine.metrics_changed.connect(self._update_metrics)
        self.video_controller.frame_ready.connect(self.canvas.set_video_frame)
        self.video_controller.image_ready.connect(self._show_image_result)
        self.video_controller.event_raised.connect(self._handle_event)
        self.video_controller.metrics_changed.connect(self._update_metrics)
        self.video_controller.status_changed.connect(self._update_source_status)
        self.video_controller.detector_status_changed.connect(
            self._update_detector_status
        )
        self.canvas.zone_changed.connect(self._apply_parking_zone)
        self.canvas.speed_lines_changed.connect(self._apply_speed_lines)
        self.canvas.parking_polygon_changed.connect(self._apply_parking_polygon)
        self.training_page.status_changed.connect(
            lambda message: self.statusBar().showMessage(message, 7000)
        )
        self._load_event_table()
        self._refresh_mode_panel()

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        self.monitor_workspace = self._build_workspace()
        self.events_page = self._build_events_page()
        self.training_page = AnomalibTrainingPage(ROOT)
        self.pages.addWidget(self.monitor_workspace)
        self.pages.addWidget(self.events_page)
        self.pages.addWidget(self.training_page)
        shell_layout.addWidget(self.pages, 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 20)
        layout.setSpacing(7)

        brand = QLabel("校园非机动车\n监测系统")
        brand.setObjectName("brand")
        sub = QLabel("26 教学楼 · 本地原型")
        sub.setObjectName("brandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(25)

        section = QLabel("监测模块")
        section.setObjectName("sideSection")
        layout.addWidget(section)

        self.road_nav = self._nav_button("行驶速度检测")
        self.parking_nav = self._nav_button("违规停放监测")
        self.events_nav = self._nav_button("事件与截图")
        self.training_nav = self._nav_button("异常模型训练")
        self.settings_nav = self._nav_button("系统接口设置")
        self.road_nav.clicked.connect(lambda: self.select_mode("road"))
        self.parking_nav.clicked.connect(lambda: self.select_mode("parking"))
        self.events_nav.clicked.connect(self._focus_events)
        self.training_nav.clicked.connect(self.show_training)
        self.settings_nav.clicked.connect(self._show_sdk_info)
        for button in (
            self.road_nav,
            self.parking_nav,
            self.events_nav,
            self.training_nav,
            self.settings_nav,
        ):
            layout.addWidget(button)
        self.road_nav.setChecked(True)

        layout.addStretch()
        status = QFrame()
        status.setObjectName("sdkStatus")
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(12, 11, 12, 11)
        sdk_title = QLabel("海康 SDK")
        sdk_title.setObjectName("sdkTitle")
        sdk_value = QLabel("未接入")
        sdk_value.setObjectName("sdkValue")
        sdk_hint = QLabel("当前使用本地模拟视频源")
        sdk_hint.setWordWrap(True)
        sdk_hint.setObjectName("sdkHint")
        status_layout.addWidget(sdk_title)
        status_layout.addWidget(sdk_value)
        status_layout.addWidget(sdk_hint)
        layout.addWidget(status)
        return sidebar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        self.page_title = QLabel("行驶速度检测")
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel("固定道路观察位 · 目标跟踪 · 双线计时 · 区间平均速度")
        self.page_subtitle.setObjectName("pageSubtitle")
        header_text.addWidget(self.page_title)
        header_text.addWidget(self.page_subtitle)
        header.addLayout(header_text)
        header.addStretch()
        self.header_model_status = QLabel(
            self._short_detector_status(self.video_controller.detector.status_text)
        )
        self.header_model_status.setObjectName("modelStatus")
        self.header_model_status.setToolTip(self.video_controller.detector.status_text)
        header.addWidget(self.header_model_status)
        self.source_status = QLabel("模拟源已连接")
        self.source_status.setObjectName("sourceStatus")
        self.source_status.setMaximumWidth(310)
        self.source_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.source_status)
        layout.addLayout(header)

        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.track_metric = MetricCard("画面目标", "3", "个")
        self.speed_metric = MetricCard("最近最高速度", "0.0", "km/h")
        self.parking_metric = MetricCard("停车异常", "3", "处")
        self.event_metric = MetricCard("累计告警", str(self.warning_count), "条")
        for card in (self.track_metric, self.speed_metric, self.parking_metric, self.event_metric):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(14)
        video_frame = QFrame()
        video_frame.setObjectName("panel")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(10, 10, 10, 10)
        self.canvas = VideoCanvas(self.engine)
        video_layout.addWidget(self.canvas)
        monitor_row.addWidget(video_frame, 1)
        monitor_row.addWidget(self._build_parameter_panel())
        layout.addLayout(monitor_row, 1)

        events_panel = QFrame()
        events_panel.setObjectName("panel")
        events_layout = QVBoxLayout(events_panel)
        events_layout.setContentsMargins(14, 12, 14, 12)
        events_layout.setSpacing(8)
        events_header = QHBoxLayout()
        events_title = QLabel("最近事件")
        events_title.setObjectName("panelTitle")
        events_hint = QLabel("告警仅记录事件，截图由工具栏手动保存")
        events_hint.setObjectName("panelHint")
        events_header.addWidget(events_title)
        events_header.addWidget(events_hint)
        events_header.addStretch()
        events_layout.addLayout(events_header)
        self.event_table = self._build_event_table()
        events_layout.addWidget(self.event_table)
        layout.addWidget(events_panel, 0)
        return workspace

    def _build_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)

        self.road_mode_button = QPushButton("道路观察位")
        self.parking_mode_button = QPushButton("停车观察位")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for button in (self.road_mode_button, self.parking_mode_button):
            button.setCheckable(True)
            button.setObjectName("modeButton")
            self.mode_group.addButton(button)
        self.road_mode_button.setChecked(True)
        self.road_mode_button.clicked.connect(lambda: self.select_mode("road"))
        self.parking_mode_button.clicked.connect(lambda: self.select_mode("parking"))
        layout.addWidget(self.road_mode_button)
        layout.addWidget(self.parking_mode_button)
        layout.addSpacing(8)

        self.run_button = QPushButton("暂停分析")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))
        self.run_button.clicked.connect(self._toggle_running)
        layout.addWidget(self.run_button)

        self.reset_button = QPushButton("重置模拟")
        self.reset_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload))
        self.reset_button.clicked.connect(self._reset_source)
        layout.addWidget(self.reset_button)
        layout.addStretch()

        sample_button = QPushButton("加载示例视频")
        sample_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogContentsView))
        sample_menu = QMenu(sample_button)
        samples = [
            ("测速：固定机位车队通过", SAMPLE_VIDEO_DIR / "road_cyclists.mp4", "road"),
            ("停车：密集停车区域", SAMPLE_VIDEO_DIR / "parking_dense.mp4", "parking"),
            ("停车：车辆清理场景", SAMPLE_VIDEO_DIR / "parking_removal.mp4", "parking"),
        ]
        for label, path, mode in samples:
            action = sample_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, video_path=path, selected_mode=mode: self.load_video(
                    video_path, selected_mode
                )
            )
        sample_menu.addSeparator()
        sample_menu.addAction("返回模拟视频源").triggered.connect(self._use_simulation_source)
        sample_button.setMenu(sample_menu)
        layout.addWidget(sample_button)

        import_button = QPushButton("选择本地视频")
        import_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DirOpenIcon))
        import_button.clicked.connect(self._choose_video)
        layout.addWidget(import_button)
        image_button = QPushButton("检测图片")
        image_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon)
        )
        image_button.clicked.connect(self._choose_image)
        layout.addWidget(image_button)
        startup_button = QPushButton("自定义示范")
        startup_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        )
        startup_menu = QMenu(startup_button)
        startup_menu.addAction("选择并设为启动示范").triggered.connect(
            self._choose_startup_video
        )
        startup_menu.addAction("取消启动自动播放").triggered.connect(
            self._clear_startup_video
        )
        startup_button.setMenu(startup_menu)
        startup_button.setToolTip("保存当前监测模式，并在下次启动时自动播放")
        layout.addWidget(startup_button)
        snapshot_button = QPushButton("保存截图")
        snapshot_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        snapshot_button.clicked.connect(self._save_manual_snapshot)
        layout.addWidget(snapshot_button)
        return toolbar

    def _build_parameter_panel(self) -> QFrame:
        scroll = QScrollArea()
        scroll.setObjectName("parameterScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(306)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(286)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(17, 16, 17, 16)
        layout.setSpacing(10)

        title = QLabel("分析参数")
        title.setObjectName("panelTitle")
        self.mode_description = QLabel()
        self.mode_description.setObjectName("panelHint")
        self.mode_description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(self.mode_description)
        layout.addSpacing(5)

        self.distance_label = QLabel("双线实际距离")
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.5, 100.0)
        self.distance_spin.setDecimals(1)
        self.distance_spin.setValue(12.0)
        self.distance_spin.setSuffix(" m")
        self.distance_spin.valueChanged.connect(self._speed_config_changed)

        self.threshold_label = QLabel("超速实验阈值")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 60.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setValue(15.0)
        self.threshold_spin.setSuffix(" km/h")
        self.threshold_spin.valueChanged.connect(self._speed_config_changed)

        self.detector_label = QLabel("车辆检测模型")
        self.detector_value = QLabel(self.video_controller.detector.status_text)
        self.detector_value.setObjectName("readOnlyValue")
        self.detector_value.setWordWrap(True)
        self.detector_value.setMinimumHeight(42)
        self.detector_value.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.speed_line_label = QLabel("测速虚拟线")
        self.speed_line_value = QLabel("虚拟线 A / B · 已配置")
        self.speed_line_value.setObjectName("readOnlyValue")
        self.speed_line_value.setMinimumHeight(42)
        self.speed_line_value.setWordWrap(True)
        speed_line_actions = QHBoxLayout()
        speed_line_actions.setSpacing(6)
        self.calibrate_speed_button = QPushButton("绘制 A / B")
        self.calibrate_speed_button.clicked.connect(self._start_speed_line_calibration)
        self.reset_speed_button = QPushButton("恢复默认")
        self.reset_speed_button.clicked.connect(self._reset_default_speed_lines)
        self.confirm_speed_button = QPushButton("确认两条线")
        self.confirm_speed_button.setObjectName("primaryButton")
        self.confirm_speed_button.clicked.connect(self._confirm_speed_line_calibration)
        self.cancel_speed_button = QPushButton("取消")
        self.cancel_speed_button.clicked.connect(self._cancel_calibration)
        for button in (
            self.calibrate_speed_button,
            self.reset_speed_button,
            self.confirm_speed_button,
            self.cancel_speed_button,
        ):
            speed_line_actions.addWidget(button)

        self.zone_label = QLabel("停车边界")
        self.zone_value = QLabel("ROI 01 · 已配置")
        self.zone_value.setObjectName("readOnlyValue")
        self.zone_value.setMinimumHeight(42)
        self.zone_value.setWordWrap(True)
        zone_actions = QGridLayout()
        zone_actions.setSpacing(6)
        self.calibrate_zone_button = QPushButton("框选矩形")
        self.calibrate_zone_button.clicked.connect(self._start_zone_calibration)
        self.draw_boundary_button = QPushButton("绘制边界")
        self.draw_boundary_button.clicked.connect(self._start_parking_boundary_calibration)
        self.reset_zone_button = QPushButton("恢复默认")
        self.reset_zone_button.clicked.connect(self._reset_default_zone)
        self.confirm_zone_button = QPushButton("确认区域")
        self.confirm_zone_button.setObjectName("primaryButton")
        self.confirm_zone_button.clicked.connect(self._confirm_zone_calibration)
        self.cancel_zone_button = QPushButton("取消")
        self.cancel_zone_button.clicked.connect(self._cancel_calibration)
        zone_actions.addWidget(self.calibrate_zone_button, 0, 0)
        zone_actions.addWidget(self.draw_boundary_button, 0, 1)
        zone_actions.addWidget(self.reset_zone_button, 1, 0, 1, 2)
        zone_actions.addWidget(self.confirm_zone_button, 0, 0)
        zone_actions.addWidget(self.cancel_zone_button, 0, 1)
        self.model_label = QLabel("状态异常模型")
        self.model_value = QLabel("Anomalib 2.6.2 · 可训练")
        self.model_value.setObjectName("readOnlyValueWarning")
        self.model_value.setMinimumHeight(42)
        self.model_value.setWordWrap(True)

        for label, widget in (
            (self.detector_label, self.detector_value),
            (self.distance_label, self.distance_spin),
            (self.threshold_label, self.threshold_spin),
            (self.speed_line_label, self.speed_line_value),
            (self.zone_label, self.zone_value),
            (self.model_label, self.model_value),
        ):
            label.setObjectName("fieldLabel")
            layout.addWidget(label)
            layout.addWidget(widget)
            if widget is self.speed_line_value:
                layout.addLayout(speed_line_actions)
            if widget is self.zone_value:
                layout.addLayout(zone_actions)

        layout.addStretch()
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        layout.addWidget(divider)

        self.pipeline_label = QLabel()
        self.pipeline_label.setObjectName("pipeline")
        self.pipeline_label.setWordWrap(True)
        layout.addWidget(self.pipeline_label)

        sdk_button = QPushButton("查看 SDK 接口预留")
        sdk_button.clicked.connect(self._show_sdk_info)
        layout.addWidget(sdk_button)
        panel.setMinimumHeight(panel.sizeHint().height())
        scroll.setWidget(panel)
        return scroll

    def _build_events_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("workspace")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("事件与截图")
        title.setObjectName("pageTitle")
        subtitle = QLabel("查看历史告警记录和手动保存的监测截图")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.addWidget(QLabel("事件筛选"))
        self.event_filter = QComboBox()
        self.event_filter.addItem("全部事件", "all")
        self.event_filter.addItem("仅告警", "warning")
        self.event_filter.addItem("仅普通记录", "info")
        self.event_filter.currentIndexChanged.connect(self._refresh_event_history)
        toolbar_layout.addWidget(self.event_filter)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self._refresh_events_page)
        toolbar_layout.addWidget(refresh_button)
        toolbar_layout.addStretch()
        open_log_button = QPushButton("打开事件日志")
        open_log_button.clicked.connect(self._open_event_log)
        toolbar_layout.addWidget(open_log_button)
        open_folder_button = QPushButton("打开截图目录")
        open_folder_button.clicked.connect(self._open_snapshot_folder)
        toolbar_layout.addWidget(open_folder_button)
        layout.addWidget(toolbar)

        history_panel = QFrame()
        history_panel.setObjectName("panel")
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(14, 12, 14, 12)
        history_header = QHBoxLayout()
        history_title = QLabel("事件历史")
        history_title.setObjectName("panelTitle")
        self.event_history_summary = QLabel()
        self.event_history_summary.setObjectName("panelHint")
        history_header.addWidget(history_title)
        history_header.addStretch()
        history_header.addWidget(self.event_history_summary)
        history_layout.addLayout(history_header)
        self.event_history_table = self._build_event_table()
        self.event_history_table.setMinimumHeight(230)
        history_layout.addWidget(self.event_history_table)
        layout.addWidget(history_panel, 3)

        snapshots_row = QHBoxLayout()
        snapshots_row.setSpacing(14)
        list_panel = QFrame()
        list_panel.setObjectName("panel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(14, 12, 14, 12)
        list_title = QLabel("手动截图")
        list_title.setObjectName("panelTitle")
        self.snapshot_summary = QLabel()
        self.snapshot_summary.setObjectName("panelHint")
        list_layout.addWidget(list_title)
        list_layout.addWidget(self.snapshot_summary)
        self.snapshot_table = QTableWidget(0, 3)
        self.snapshot_table.setHorizontalHeaderLabels(["文件名", "保存时间", "大小"])
        self.snapshot_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.snapshot_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.snapshot_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.snapshot_table.setAlternatingRowColors(True)
        self.snapshot_table.verticalHeader().setVisible(False)
        snapshot_header = self.snapshot_table.horizontalHeader()
        snapshot_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        snapshot_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        snapshot_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.snapshot_table.itemSelectionChanged.connect(self._preview_selected_snapshot)
        self.snapshot_table.itemDoubleClicked.connect(lambda _item: self._open_selected_snapshot())
        list_layout.addWidget(self.snapshot_table)
        self.open_snapshot_button = QPushButton("打开所选截图")
        self.open_snapshot_button.setEnabled(False)
        self.open_snapshot_button.clicked.connect(self._open_selected_snapshot)
        list_layout.addWidget(self.open_snapshot_button)
        snapshots_row.addWidget(list_panel, 3)

        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 12, 14, 12)
        preview_title = QLabel("截图预览")
        preview_title.setObjectName("panelTitle")
        preview_layout.addWidget(preview_title)
        self.snapshot_preview = QLabel("选择左侧截图后在此预览")
        self.snapshot_preview.setObjectName("snapshotPreview")
        self.snapshot_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.snapshot_preview.setMinimumSize(360, 190)
        preview_layout.addWidget(self.snapshot_preview, 1)
        snapshots_row.addWidget(preview_panel, 2)
        layout.addLayout(snapshots_row, 2)

        self._refresh_event_history()
        self._refresh_snapshots()
        return page

    def _build_event_table(self) -> QTableWidget:
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["时间", "事件类型", "观察位置", "详情", "检测值", "状态"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(165)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        return table

    def select_mode(self, mode: str) -> None:
        if (
            hasattr(self, "canvas")
            and self.canvas.calibration_active
            and mode != self.engine.mode
        ):
            self._cancel_calibration()
        if hasattr(self, "pages"):
            self.pages.setCurrentWidget(self.monitor_workspace)
        self.engine.set_mode(mode)
        self.video_controller.set_mode(mode)
        road = mode == "road"
        self.road_mode_button.setChecked(road)
        self.parking_mode_button.setChecked(not road)
        self.road_nav.setChecked(road)
        self.parking_nav.setChecked(not road)
        self.page_title.setText("行驶速度检测" if road else "违规停放监测")
        self.page_subtitle.setText(
            "固定道路观察位 · 目标跟踪 · 双线计时 · 区间平均速度"
            if road
            else "固定停车观察位 · 区域越界 · 倒地与异常堆放"
        )
        self._refresh_mode_panel()

    def _refresh_mode_panel(self) -> None:
        road = self.engine.mode == "road"
        self.mode_description.setText(
            "目标依次通过虚拟线 A、B 后，根据实测距离和时间差计算平均速度。"
            if road
            else "当前先用几何规则模拟停车越界，并预留 Anomalib 异常分数与热力图接口。"
        )
        self.distance_label.setVisible(road)
        self.distance_spin.setVisible(road)
        self.threshold_label.setVisible(road)
        self.threshold_spin.setVisible(road)
        self.speed_line_label.setVisible(road)
        self.speed_line_value.setVisible(road)
        self.zone_label.setVisible(not road)
        self.zone_value.setVisible(not road)
        self.model_label.setVisible(not road)
        self.model_value.setVisible(not road)
        self.speed_line_value.setText(self._speed_lines_summary(self.engine.speed_service.lines))
        self.zone_value.setText(self._polygon_summary(self.engine.parking_service.polygon))
        self.detector_value.setText(self.video_controller.detector.status_text)
        self._update_calibration_controls()
        self.pipeline_label.setText(
            "处理链路\n模拟取流 → 目标跟踪 → 双线计时 → 速度计算 → 阈值告警"
            if road
            else "处理链路\n模拟取流 → 停车区检测 → 越界规则 → Anomalib 接口 → 事件融合"
        )

    def _toggle_running(self) -> None:
        if self.active_source == "image":
            self.video_controller.restart_image()
            self.canvas.set_video_running(self.video_controller.running)
            self._set_image_button_state()
            return
        if self.active_source == "video":
            running = not self.video_controller.running
            self.video_controller.set_running(running)
            self.canvas.set_video_running(running)
        else:
            running = not self.engine.running
            self.engine.set_running(running)
        if running:
            self.run_button.setText("暂停分析")
            self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))
        else:
            self.run_button.setText("继续分析")
            self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay))
        self.canvas.update()

    def _speed_config_changed(self) -> None:
        self.engine.set_speed_config(self.distance_spin.value(), self.threshold_spin.value())
        self.video_controller.set_speed_config(self.distance_spin.value(), self.threshold_spin.value())
        self.canvas.update()

    def _prepare_calibration(self, mode: str) -> bool:
        self.select_mode(mode)
        if self.canvas.calibration_active:
            return False
        self._calibration_was_running = (
            self.video_controller.running
            if self.active_source in {"video", "image"}
            else self.engine.running
        )
        if self.active_source in {"video", "image"}:
            self.video_controller.set_running(False)
            self.canvas.set_video_running(False)
        else:
            self.engine.set_running(False)
        self.run_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        return True

    def _start_speed_line_calibration(self) -> None:
        if not self._prepare_calibration("road"):
            return
        self.canvas.begin_speed_line_calibration()
        self.speed_line_value.setText("请依次拖拽绘制虚拟线 A、B")
        self._update_calibration_controls()

    def _start_zone_calibration(self) -> None:
        if not self._prepare_calibration("parking"):
            return
        self.canvas.begin_zone_calibration()
        self.zone_value.setText("请在画面中拖拽一个矩形区域")
        self._update_calibration_controls()

    def _start_parking_boundary_calibration(self) -> None:
        if not self._prepare_calibration("parking"):
            return
        self.canvas.begin_parking_boundary_calibration()
        self.zone_value.setText("依次点击顶点，用线段围出停车区域")
        self._update_calibration_controls()

    def _confirm_speed_line_calibration(self) -> None:
        if not self.canvas.confirm_speed_line_calibration():
            QMessageBox.information(self, "测速线标定", "请先依次拖拽绘制虚拟线 A 和 B。")
            return
        self._finish_calibration_session()

    def _confirm_zone_calibration(self) -> None:
        if self.canvas.calibration_mode == "parking_polygon":
            if not self.canvas.confirm_parking_boundary_calibration():
                QMessageBox.information(self, "停车边界标定", "请至少点击 3 个边界顶点。")
                return
        elif not self.canvas.confirm_zone_calibration():
            QMessageBox.information(self, "停车区域标定", "请先在画面中按住鼠标拖拽停车区域。")
            return
        self._finish_calibration_session()

    def _cancel_calibration(self) -> None:
        if not self.canvas.calibration_active:
            return
        self.canvas.cancel_calibration()
        self.speed_line_value.setText(self._speed_lines_summary(self.engine.speed_service.lines))
        self.zone_value.setText(self._polygon_summary(self.engine.parking_service.polygon))
        self._finish_calibration_session()

    def _finish_calibration_session(self) -> None:
        self.run_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        if self.active_source == "image":
            if self._calibration_was_running:
                self.video_controller.set_running(True)
            self.canvas.set_video_running(self.video_controller.running)
            self._set_image_button_state()
        elif self._calibration_was_running:
            if self.active_source == "video":
                self.video_controller.set_running(True)
                self.canvas.set_video_running(True)
            else:
                self.engine.set_running(True)
        if self.active_source != "image":
            self._set_run_button_state(
                self.video_controller.running
                if self.active_source == "video"
                else self.engine.running
            )
        self._update_calibration_controls()

    def _apply_speed_lines(self, lines: tuple[LineSegment, LineSegment]) -> None:
        self.engine.set_speed_lines(lines)
        self.video_controller.set_speed_lines(lines)
        self.settings_store.save_speed_lines(lines)
        self.speed_line_value.setText(self._speed_lines_summary(lines))
        self.statusBar().showMessage("测速虚拟线 A、B 已保存并开始参与过线判断", 6000)
        self.canvas.update()

    def _apply_parking_zone(self, zone: tuple[float, float, float, float]) -> None:
        self.engine.set_parking_zone(zone)
        self.video_controller.set_parking_zone(zone)
        self.settings_store.save_parking_zone(zone)
        self.zone_value.setText(self._polygon_summary(self.engine.parking_service.polygon))
        self.statusBar().showMessage("停车区域已保存并同步到模拟与本地视频检测", 6000)
        self.canvas.update()

    def _apply_parking_polygon(self, polygon: tuple[Point, ...]) -> None:
        self.engine.set_parking_polygon(polygon)
        self.video_controller.set_parking_polygon(polygon)
        self.settings_store.save_parking_polygon(polygon)
        self.zone_value.setText(self._polygon_summary(polygon))
        self.statusBar().showMessage("停车边界线已保存，闭合区域外将判为越界停放", 6000)
        self.canvas.update()

    def _reset_default_speed_lines(self) -> None:
        if self.canvas.calibration_active:
            self._cancel_calibration()
        self._apply_speed_lines(DEFAULT_SPEED_LINES)

    def _reset_default_zone(self) -> None:
        if self.canvas.calibration_active:
            self._cancel_calibration()
        self._apply_parking_polygon(DEFAULT_PARKING_POLYGON)

    def _update_calibration_controls(self) -> None:
        road = self.engine.mode == "road"
        speed_calibrating = self.canvas.calibration_mode == "speed_lines"
        parking_calibrating = self.canvas.calibration_mode in {
            "parking_rectangle",
            "parking_polygon",
        }
        self.calibrate_speed_button.setVisible(road and not speed_calibrating)
        self.reset_speed_button.setVisible(road and not speed_calibrating)
        self.confirm_speed_button.setVisible(road and speed_calibrating)
        self.cancel_speed_button.setVisible(road and speed_calibrating)
        self.calibrate_zone_button.setVisible(not road and not parking_calibrating)
        self.draw_boundary_button.setVisible(not road and not parking_calibrating)
        self.reset_zone_button.setVisible(not road and not parking_calibrating)
        self.confirm_zone_button.setVisible(not road and parking_calibrating)
        self.cancel_zone_button.setVisible(not road and parking_calibrating)

    @staticmethod
    def _speed_lines_summary(lines: tuple[LineSegment, LineSegment]) -> str:
        line_a, line_b = lines
        return f"A 长度 {MainWindow._line_length(line_a):.0%} · B 长度 {MainWindow._line_length(line_b):.0%}"

    @staticmethod
    def _line_length(line: LineSegment) -> float:
        return ((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2) ** 0.5

    @staticmethod
    def _polygon_summary(polygon: tuple[Point, ...]) -> str:
        return f"{len(polygon)} 个边界点 · 闭合区域已配置"

    def _handle_event(self, event: MonitorEvent) -> None:
        self.events.append(event)
        self.event_store.append(event)
        if event.severity == "warning":
            self.warning_count += 1
        self.event_metric.set_value(str(self.warning_count))
        self._insert_event_row(event, at_top=True)
        if hasattr(self, "events_page") and self.pages.currentWidget() is self.events_page:
            self._append_event_history(event)

    def _insert_event_row(self, event: MonitorEvent, at_top: bool = True) -> None:
        row = 0 if at_top else self.event_table.rowCount()
        self.event_table.insertRow(row)
        self._write_event_row(self.event_table, row, event)
        while self.event_table.rowCount() > 30:
            self.event_table.removeRow(self.event_table.rowCount() - 1)

    @staticmethod
    def _write_event_row(table: QTableWidget, row: int, event: MonitorEvent) -> None:
        values = [
            event.occurred_at,
            event.event_type,
            event.source,
            event.detail,
            event.value,
            event.status,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setForeground(QColor("#344147"))
            if event.severity == "warning" and column in {1, 5}:
                item.setForeground(QColor("#d84e47"))
            if column == 5:
                item.setData(Qt.ItemDataRole.UserRole, event.severity)
            table.setItem(row, column, item)

    def _load_event_table(self) -> None:
        for event in reversed(self.events[-30:]):
            self._insert_event_row(event, at_top=False)

    def _update_metrics(self, metrics: dict) -> None:
        self.track_metric.set_value(str(metrics["active_tracks"]))
        self.speed_metric.set_value(f"{metrics['current_speed']:.1f}")
        self.parking_metric.set_value(str(metrics["parking_violations"]))

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择测试视频",
            str(SAMPLE_VIDEO_DIR),
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if path:
            self.load_video(Path(path))

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待检测图片",
            str(Path.home()),
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if path:
            self.load_image(Path(path))

    def _choose_startup_video(self) -> None:
        configured = self.settings_store.load_startup_video()
        initial = configured[0].parent if configured else SAMPLE_VIDEO_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择启动示范视频",
            str(initial),
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if not path:
            return
        video_path = Path(path)
        mode = self.engine.mode
        if self.load_video(video_path, mode):
            self.settings_store.save_startup_video(video_path, mode)
            mode_name = "停车检测" if mode == "parking" else "速度检测"
            self.statusBar().showMessage(
                f"已设为启动示范：{video_path.name} · {mode_name}",
                6000,
            )

    def _clear_startup_video(self) -> None:
        self.settings_store.clear_startup_video()
        self.statusBar().showMessage("已取消启动示范视频", 5000)

    def load_startup_video(self) -> bool:
        if not self.settings_store.startup_video_enabled():
            return False
        configured = self.settings_store.load_startup_video()
        if configured is None:
            path, mode = SAMPLE_VIDEO_DIR / "road_cyclists.mp4", "road"
        else:
            path, mode = configured
        return self.load_video(path, mode)

    def load_video(self, path: Path, mode: str | None = None) -> bool:
        if mode:
            self.select_mode(mode)
        try:
            self.video_controller.open(path)
        except (OSError, RuntimeError) as error:
            QMessageBox.warning(self, "无法加载视频", str(error))
            return False
        self.active_source = "video"
        self.engine.set_running(False)
        self.canvas.set_video_running(True)
        self.reset_button.setText("重新播放")
        self.reset_button.setVisible(True)
        self.run_button.setEnabled(True)
        self.run_button.setText("暂停分析")
        self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))
        return True

    def load_image(self, path: Path, mode: str | None = None) -> bool:
        if mode:
            self.select_mode(mode)
        try:
            self.video_controller.open_image(path)
        except (OSError, RuntimeError) as error:
            QMessageBox.warning(self, "无法检测图片", str(error))
            return False
        self.active_source = "image"
        self.engine.set_running(False)
        self.reset_button.setText("重新检测")
        self.reset_button.setVisible(False)
        self._set_image_button_state()
        return True

    def _use_simulation_source(self) -> None:
        self.video_controller.close()
        self.active_source = "simulation"
        self.canvas.clear_video()
        self.engine.set_running(True)
        self.reset_button.setText("重置模拟")
        self.reset_button.setVisible(True)
        self.source_status.setText("模拟源已连接")
        self.source_status.setToolTip("模拟源已连接")
        self.run_button.setEnabled(True)
        self.run_button.setText("暂停分析")
        self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))

    def _reset_source(self) -> None:
        if self.active_source == "video":
            self.video_controller.restart()
            self.canvas.set_video_running(True)
        elif self.active_source == "image":
            self.video_controller.restart_image()
            self.canvas.set_video_running(self.video_controller.running)
            self._set_image_button_state()
        else:
            self.engine.reset_scene()

    def _update_source_status(self, text: str) -> None:
        self.source_status.setText(text)
        self.source_status.setToolTip(text)

    def _update_detector_status(self, text: str) -> None:
        self.detector_value.setText(text)
        self.header_model_status.setText(self._short_detector_status(text))
        self.header_model_status.setToolTip(text)

    def _show_image_result(
        self,
        image,
        detections,
        source_name: str,
        running: bool,
    ) -> None:
        self.canvas.set_video_frame(image, detections, source_name)
        self.canvas.set_video_running(running)
        self._set_image_button_state()

    def _set_image_button_state(self) -> None:
        if self.active_source != "image":
            return
        running = self.video_controller.running
        self.run_button.setEnabled(not running)
        self.run_button.setText("检测中..." if running else "重新检测")
        icon = (
            self.style().StandardPixmap.SP_BrowserReload
            if not running
            else self.style().StandardPixmap.SP_MediaPlay
        )
        self.run_button.setIcon(self.style().standardIcon(icon))

    @staticmethod
    def _short_detector_status(text: str) -> str:
        normalized = text.upper()
        if "未安装" in text:
            return "RT-DETR · 未安装"
        if "失败" in text or "ERROR" in normalized:
            return "RT-DETR · 启动失败"
        if "RT-DETR" in text:
            suffix = (
                "CUDA"
                if "CUDA" in normalized
                else "CPU"
                if "CPU" in normalized
                else "加载中"
                if "加载" in text
                else "待启动"
            )
            return f"RT-DETR R50 · {suffix}"
        if "YOLOX" in text:
            return "YOLOX-Tiny · 回退"
        return "MobileNet-SSD · 回退"

    def _save_manual_snapshot(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"manual_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.canvas.grab().save(str(path))
        self.statusBar().showMessage(f"截图已保存：{path}", 5000)

    def _focus_events(self) -> None:
        if self.canvas.calibration_active:
            self._cancel_calibration()
        self.events_nav.setChecked(True)
        self.pages.setCurrentWidget(self.events_page)
        self._refresh_events_page()

    def _refresh_events_page(self) -> None:
        self._refresh_event_history()
        self._refresh_snapshots()

    def _refresh_event_history(self) -> None:
        if not hasattr(self, "event_history_table"):
            return
        events = self.event_store.load(limit=5000)
        selected_filter = self.event_filter.currentData()
        if selected_filter != "all":
            events = [item for item in events if item.severity == selected_filter]
        self.event_history_table.setRowCount(0)
        for event in reversed(events):
            row = self.event_history_table.rowCount()
            self.event_history_table.insertRow(row)
            self._write_event_row(self.event_history_table, row, event)
        warning_count = sum(1 for item in events if item.severity == "warning")
        self._event_history_warning_count = warning_count
        self.event_history_summary.setText(
            f"当前显示 {len(events)} 条 · 告警 {warning_count} 条"
        )

    def _append_event_history(self, event: MonitorEvent) -> None:
        selected_filter = self.event_filter.currentData()
        if selected_filter != "all" and event.severity != selected_filter:
            return
        self.event_history_table.insertRow(0)
        self._write_event_row(self.event_history_table, 0, event)
        warning_count = self._event_history_warning_count
        if event.severity == "warning":
            warning_count += 1
        if self.event_history_table.rowCount() > 5000:
            removed_row = self.event_history_table.rowCount() - 1
            removed_status = self.event_history_table.item(removed_row, 5)
            if (
                removed_status is not None
                and removed_status.data(Qt.ItemDataRole.UserRole) == "warning"
            ):
                warning_count -= 1
            self.event_history_table.removeRow(removed_row)
        visible_count = self.event_history_table.rowCount()
        self._event_history_warning_count = warning_count
        self.event_history_summary.setText(
            f"当前显示 {visible_count} 条 · 告警 {warning_count} 条"
        )

    def _refresh_snapshots(self) -> None:
        if not hasattr(self, "snapshot_table"):
            return
        selected_path = self._selected_snapshot_path()
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths = []
        if self.snapshot_dir.is_dir():
            paths = sorted(
                (
                    path
                    for path in self.snapshot_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in extensions
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        self.snapshot_table.setRowCount(0)
        selected_row = -1
        for path in paths:
            row = self.snapshot_table.rowCount()
            self.snapshot_table.insertRow(row)
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = path.stat().st_size / 1024
            name_item = QTableWidgetItem(path.name)
            name_item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.snapshot_table.setItem(row, 0, name_item)
            self.snapshot_table.setItem(row, 1, QTableWidgetItem(modified))
            self.snapshot_table.setItem(row, 2, QTableWidgetItem(f"{size_kb:.1f} KB"))
            if path == selected_path:
                selected_row = row
        self.snapshot_summary.setText(f"共 {len(paths)} 张 · 仅显示手动保存截图")
        if selected_row >= 0:
            self.snapshot_table.selectRow(selected_row)
        elif paths:
            self.snapshot_table.selectRow(0)
        else:
            self.snapshot_preview.clear()
            self.snapshot_preview.setText("暂无手动截图")
            self.open_snapshot_button.setEnabled(False)

    def _selected_snapshot_path(self) -> Path | None:
        if not hasattr(self, "snapshot_table"):
            return None
        row = self.snapshot_table.currentRow()
        item = self.snapshot_table.item(row, 0) if row >= 0 else None
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def _preview_selected_snapshot(self) -> None:
        path = self._selected_snapshot_path()
        if path is None or not path.is_file():
            self.snapshot_preview.clear()
            self.snapshot_preview.setText("选择左侧截图后在此预览")
            self.open_snapshot_button.setEnabled(False)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.snapshot_preview.clear()
            self.snapshot_preview.setText("截图无法读取")
            self.open_snapshot_button.setEnabled(False)
            return
        preview_size = self.snapshot_preview.size()
        self.snapshot_preview.setPixmap(
            pixmap.scaled(
                preview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.open_snapshot_button.setEnabled(True)

    def _open_selected_snapshot(self) -> None:
        path = self._selected_snapshot_path()
        if path is not None and path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _open_event_log(self) -> None:
        if not self.event_store.path.exists():
            self.statusBar().showMessage("当前还没有事件日志", 4000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.event_store.path.resolve())))

    def _open_snapshot_folder(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.snapshot_dir.resolve())))

    def show_training(self) -> None:
        if self.canvas.calibration_active:
            self._cancel_calibration()
        if self.active_source in {"video", "image"}:
            self.video_controller.set_running(False)
            self.canvas.set_video_running(False)
        else:
            self.engine.set_running(False)
        if self.active_source == "image":
            self._set_image_button_state()
        else:
            self._set_run_button_state(False)
        self.training_nav.setChecked(True)
        self.pages.setCurrentWidget(self.training_page)

    def _show_sdk_info(self) -> None:
        QMessageBox.information(
            self,
            "海康 SDK 接入位置",
            "已在 campus_monitor/integrations.py 中预留 HikvisionSdkAdapter。\n\n"
            "后续需要实现：\n"
            "1. NET_DVR_Init 与设备登录\n"
            "2. 实时预览回调和视频帧转换\n"
            "3. 道路/停车预置点切换\n"
            "4. 切换期间暂停分析并清空跟踪状态",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.video_controller.shutdown()
        self.training_page.shutdown()
        super().closeEvent(event)

    def _set_run_button_state(self, running: bool) -> None:
        if running:
            self.run_button.setText("暂停分析")
            self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))
        else:
            self.run_button.setText("继续分析")
            self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay))

    @staticmethod
    def _nav_button(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.setMinimumHeight(40)
        return button

    def _apply_theme(self) -> None:
        QApplication.setStyle("Fusion")
        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
        QApplication.setFont(QFont("Microsoft YaHei UI", 10))
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f3f5f6"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#20292e"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#263138"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            * { font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; font-size: 13px; }
            QMainWindow, #shell, #workspace, QStackedWidget { background: #f3f5f6; }
            #sidebar { background: #20292e; border: none; }
            #brand { color: #ffffff; font-size: 20px; font-weight: 700; line-height: 1.35; }
            #brandSub { color: #91a0a7; font-size: 11px; }
            #sideSection { color: #718087; font-size: 11px; margin: 4px 8px 5px; }
            #navButton { background: transparent; color: #c5ced2; border: none; border-radius: 5px;
                         text-align: left; padding: 10px 12px; }
            #navButton:hover { background: #2b373d; color: #ffffff; }
            #navButton:checked { background: #33444b; color: #ffffff; border-left: 3px solid #35b99a; padding-left: 9px; }
            #sdkStatus { background: #172025; border: 1px solid #354249; border-radius: 6px; }
            #sdkTitle { color: #8d9aa0; font-size: 11px; }
            #sdkValue { color: #efb44c; font-size: 16px; font-weight: 700; }
            #sdkHint { color: #7d8b92; font-size: 10px; }
            #pageTitle { color: #1d272c; font-size: 23px; font-weight: 700; }
            #pageSubtitle { color: #748087; font-size: 12px; }
            #sourceStatus { color: #198a70; background: #e3f4ef; border: 1px solid #c4e6dc;
                            border-radius: 5px; padding: 7px 11px; font-size: 11px; }
            #modelStatus { color: #2d6472; background: #e7f2f5; border: 1px solid #c7dfe5;
                           border-radius: 5px; padding: 7px 11px; font-size: 11px; font-weight: 700; }
            #environmentStatus { color: #a06a12; font-size: 12px; font-weight: 700; }
            #environmentStatus[ready="true"] { color: #198a70; }
            #toolbar, #panel, #metricCard { background: #ffffff; border: 1px solid #dce2e5; border-radius: 6px; }
            #parameterScroll { background: transparent; border: none; }
            #parameterScroll > QWidget > QWidget { background: transparent; }
            #metricTitle { color: #78858b; font-size: 11px; }
            #metricValue { color: #1e282d; font-size: 23px; font-weight: 700; }
            #metricSuffix { color: #7b878d; font-size: 10px; padding-bottom: 3px; }
            QPushButton { background: #ffffff; color: #344147; border: 1px solid #ced6da;
                          border-radius: 5px; padding: 7px 11px; min-height: 18px; }
            QPushButton:hover { background: #f2f6f7; border-color: #aebbc1; }
            QPushButton:pressed { background: #e8edef; }
            #primaryButton { background: #267e6b; border-color: #267e6b; color: #ffffff; }
            #primaryButton:hover { background: #206d5d; }
            #modeButton { border: none; background: transparent; color: #6c797f; }
            #modeButton:checked { background: #e8f3f0; color: #176f5d; font-weight: 700; }
            #panelTitle { color: #29343a; font-size: 15px; font-weight: 700; }
            #panelHint { color: #7a878d; font-size: 11px; }
            #fieldLabel { color: #606e75; font-size: 11px; margin-top: 3px; }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #f8fafb; color: #20292e;
                             border: 1px solid #d4dcdf; border-radius: 4px;
                             padding: 7px 9px; min-height: 22px; }
            QLineEdit:disabled, QLineEdit:read-only { color: #4f5d63; }
            QLineEdit { selection-background-color: #267e6b; selection-color: #ffffff; }
            #readOnlyValue, #readOnlyValueWarning { background: #f5f7f8; border: 1px solid #d7dee1;
                                                   border-radius: 4px; padding: 9px; color: #425158; }
            #readOnlyValueWarning { color: #a06a12; background: #fff7e5; border-color: #ead6aa; }
            #pipeline { color: #66757c; font-size: 11px; line-height: 1.5; }
            #datasetStatus { color: #5f6d74; background: #f3f6f7; border: 1px solid #d9e0e3;
                             border-radius: 4px; padding: 9px; }
            #trainingLog { background: #172026; color: #d7e0e3; border: none;
                           border-radius: 4px; padding: 10px;
                           font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; }
            #divider { color: #dfe4e6; }
            QTableWidget { background: #ffffff; alternate-background-color: #f7f9fa; border: none;
                           gridline-color: #e4e9eb; selection-background-color: #e2f2ee;
                           selection-color: #243036; }
            QHeaderView::section { background: #f0f3f4; color: #59666c; border: none;
                                   border-bottom: 1px solid #d8e0e3; padding: 7px; font-size: 11px; }
            QTableWidget::item { padding: 5px; }
            #snapshotPreview { background: #172025; color: #8d9aa0; border: 1px solid #354249;
                               border-radius: 4px; padding: 8px; }
            QStatusBar { background: #ffffff; color: #647279; }
            """
        )
