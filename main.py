from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from campus_monitor.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校园非机动车异常监测 GUI 原型")
    parser.add_argument("--screenshot", type=Path, help="启动后保存界面截图并退出")
    parser.add_argument("--parking", action="store_true", help="以停车监测模式启动")
    parser.add_argument("--training", action="store_true", help="以异常模型训练页面启动")
    parser.add_argument("--video", type=Path, help="启动后直接加载本地测试视频")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("校园非机动车异常监测系统")
    app.setOrganizationName("ProductionPractice")

    window = MainWindow()
    if args.parking:
        window.select_mode("parking")
    if args.training:
        window.show_training()
    if args.video:
        window.load_video(args.video, "parking" if args.parking else "road")
    window.show()
    if not args.video and not args.parking and not args.training:
        QTimer.singleShot(0, window.load_startup_video)

    if args.screenshot:
        output = args.screenshot.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            window.grab().save(str(output))
            app.quit()

        QTimer.singleShot(4200, capture)

    return app.exec()


if __name__ == "__main__":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    raise SystemExit(main())
