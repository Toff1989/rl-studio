"""Genere packaging/icon.ico (et icon.png) : carre arrondi accent + « RL ». Usage : python packaging/make_icon.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen

app = QGuiApplication(sys.argv)
out = Path(__file__).parent


def draw(size):
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#4f7cf0"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(size * .04, size * .04, size * .92, size * .92), size * .22, size * .22)
    p.setPen(QPen(QColor("#ffffff")))
    p.setFont(QFont("Segoe UI", int(size * .36), QFont.Bold))
    p.drawText(QRectF(0, 0, size, size * .98), Qt.AlignCenter, "RL")
    p.end()
    return img


draw(256).save(str(out / "icon.png"))
ok = draw(256).save(str(out / "icon.ico"))  # Qt ecrit un .ico (taille 256, suffisante pour Windows)
print("icon.ico:", ok)
