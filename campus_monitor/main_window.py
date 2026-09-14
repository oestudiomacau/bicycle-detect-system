from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .domain import EventStore, MonitorEvent
from .integrations import HikvisionSdkAdapter, ParkingAnomalyAdapter
from .simulation import SimulationEngine
from .video_analysis import VideoAnalysisController
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

        self.engine = SimulationEngine(self)
        self.video_controller = VideoAnalysisController(
            MODEL_DIR / "mobilenet_ssd.prototxt",
            MODEL_DIR / "mobilenet_ssd.caffemodel",
            self,
        )
        self.active_source = "simulation"
        self.event_store = EventStore(RUNTIME_DIR / "events.jsonl")
        self.hikvision = HikvisionSdkAdapter()
        self.anomalib = ParkingAnomalyAdapter()
        self.events: list[MonitorEvent] = self.event_store.load()
        self.warning_count = sum(1 for item in self.events if item.severity == "warning")

        self._apply_theme()
        self._build_ui()
        self.engine.event_raised.connect(self._handle_event)
        self.engine.metrics_changed.connect(self._update_metrics)
        self.video_controller.frame_ready.connect(self.canvas.set_video_frame)
        self.video_controller.event_raised.connect(self._handle_event)
        self.video_controller.metrics_changed.connect(self._update_metrics)
        self.video_controller.status_changed.connect(self._update_source_status)
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
        shell_layout.addWidget(self._build_workspace(), 1)

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
        self.settings_nav = self._nav_button("系统接口设置")
        self.road_nav.clicked.connect(lambda: self.select_mode("road"))
        self.parking_nav.clicked.connect(lambda: self.select_mode("parking"))
        self.events_nav.clicked.connect(self._focus_events)
        self.settings_nav.clicked.connect(self._show_sdk_info)
        for button in (self.road_nav, self.parking_nav, self.events_nav, self.settings_nav):
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
        self.source_status = QLabel("●  模拟源已连接")
        self.source_status.setObjectName("sourceStatus")
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
        events_hint = QLabel("告警触发时自动保存当前画面")
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
        snapshot_button = QPushButton("保存截图")
        snapshot_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        snapshot_button.clicked.connect(self._save_manual_snapshot)
        layout.addWidget(snapshot_button)
        return toolbar

    def _build_parameter_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(292)
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

        self.zone_label = QLabel("停车区域")
        self.zone_value = QLabel("ROI 01 · 已配置")
        self.zone_value.setObjectName("readOnlyValue")
        self.model_label = QLabel("状态异常模型")
        self.model_value = QLabel("Anomalib · 待接入")
        self.model_value.setObjectName("readOnlyValueWarning")

        for label, widget in (
            (self.distance_label, self.distance_spin),
            (self.threshold_label, self.threshold_spin),
            (self.zone_label, self.zone_value),
            (self.model_label, self.model_value),
        ):
            label.setObjectName("fieldLabel")
            layout.addWidget(label)
            layout.addWidget(widget)

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
        return panel

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
        self.zone_label.setVisible(not road)
        self.zone_value.setVisible(not road)
        self.model_label.setVisible(not road)
        self.model_value.setVisible(not road)
        self.pipeline_label.setText(
            "处理链路\n模拟取流 → 目标跟踪 → 双线计时 → 速度计算 → 阈值告警"
            if road
            else "处理链路\n模拟取流 → 停车区检测 → 越界规则 → Anomalib 接口 → 事件融合"
        )

    def _toggle_running(self) -> None:
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

    def _handle_event(self, event: MonitorEvent) -> None:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot = SNAPSHOT_DIR / f"{stamp}_{event.event_type}.png"
        self.canvas.grab().save(str(snapshot))
        event.snapshot = str(snapshot)
        self.events.append(event)
        self.event_store.append(event)
        if event.severity == "warning":
            self.warning_count += 1
        self.event_metric.set_value(str(self.warning_count))
        self._insert_event_row(event, at_top=True)

    def _insert_event_row(self, event: MonitorEvent, at_top: bool = True) -> None:
        row = 0 if at_top else self.event_table.rowCount()
        self.event_table.insertRow(row)
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
            self.event_table.setItem(row, column, item)
        while self.event_table.rowCount() > 30:
            self.event_table.removeRow(self.event_table.rowCount() - 1)

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

    def load_video(self, path: Path, mode: str | None = None) -> None:
        if mode:
            self.select_mode(mode)
        try:
            self.video_controller.open(path)
        except (OSError, RuntimeError) as error:
            QMessageBox.warning(self, "无法加载视频", str(error))
            return
        self.active_source = "video"
        self.engine.set_running(False)
        self.canvas.set_video_running(True)
        self.reset_button.setText("重新播放")
        self.run_button.setText("暂停分析")
        self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))

    def _use_simulation_source(self) -> None:
        self.video_controller.close()
        self.active_source = "simulation"
        self.canvas.clear_video()
        self.engine.set_running(True)
        self.reset_button.setText("重置模拟")
        self.source_status.setText("●  模拟源已连接")
        self.run_button.setText("暂停分析")
        self.run_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))

    def _reset_source(self) -> None:
        if self.active_source == "video":
            self.video_controller.restart()
            self.canvas.set_video_running(True)
        else:
            self.engine.reset_scene()

    def _update_source_status(self, text: str) -> None:
        self.source_status.setText(f"●  {text}")

    def _save_manual_snapshot(self) -> None:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"manual_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.canvas.grab().save(str(path))
        self.statusBar().showMessage(f"截图已保存：{path}", 5000)

    def _focus_events(self) -> None:
        self.event_table.setFocus()
        self.event_table.scrollToTop()

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
        QApplication.setFont(QFont("Noto Sans SC", 10))
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f3f5f6"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#20292e"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#263138"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            * { font-family: "Noto Sans SC"; font-size: 13px; }
            QMainWindow, #shell, #workspace { background: #f3f5f6; }
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
            #toolbar, #panel, #metricCard { background: #ffffff; border: 1px solid #dce2e5; border-radius: 6px; }
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
            QDoubleSpinBox { background: #f8fafb; border: 1px solid #d4dcdf; border-radius: 4px;
                             padding: 7px 9px; min-height: 22px; }
            #readOnlyValue, #readOnlyValueWarning { background: #f5f7f8; border: 1px solid #d7dee1;
                                                   border-radius: 4px; padding: 9px; color: #425158; }
            #readOnlyValueWarning { color: #a06a12; background: #fff7e5; border-color: #ead6aa; }
            #pipeline { color: #66757c; font-size: 11px; line-height: 1.5; }
            #divider { color: #dfe4e6; }
            QTableWidget { background: #ffffff; alternate-background-color: #f7f9fa; border: none;
                           gridline-color: #e4e9eb; selection-background-color: #e2f2ee;
                           selection-color: #243036; }
            QHeaderView::section { background: #f0f3f4; color: #59666c; border: none;
                                   border-bottom: 1px solid #d8e0e3; padding: 7px; font-size: 11px; }
            QTableWidget::item { padding: 5px; }
            QStatusBar { background: #ffffff; color: #647279; }
            """
        )
