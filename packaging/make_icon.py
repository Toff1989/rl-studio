"""Genere l'icone de RL Studio : packaging/icon.ico (multi-tailles), packaging/icon.png et docs/logo.png.

    python packaging/make_icon.py            # ecrit les fichiers
    python packaging/make_icon.py --preview apercu.png   # planche de controle (fonds clair et sombre)

Pictogramme : la boucle de l'apprentissage par renforcement (l'agent agit, l'environnement repond) sous forme d'un
anneau a fleche, avec au centre un point vert, la recompense. Chaque taille est dessinee separement (traits plus epais et
pas de reflets en dessous de 48 px) puis reduite depuis un rendu 4x : l'icone reste lisible de 16 a 256 px.
"""
import math
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPolygonF, QRadialGradient)

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SUPERSAMPLE = 4

BLUE_TOP, BLUE_BOTTOM = QColor("#6d96ff"), QColor("#2b46c9")
GREEN = QColor("#3ecf8e")


def _draw_loop(p, cx, cy, u, target, small):
    """Anneau a fleche : la boucle agent <-> environnement."""
    # -- anneau a fleche (la boucle agent <-> environnement) ----------------------------------
    radius = u * (0.265 if small else 0.255)
    stroke = u * (0.125 if small else 0.095)
    ring = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
    start, sweep = 62.0, -292.0  # degres Qt : 0 = 3 h, sens anti-horaire ; sweep negatif = sens horaire
    path = QPainterPath()
    path.arcMoveTo(ring, start)
    path.arcTo(ring, start, sweep)
    pen = QPen(QColor("white"), stroke, Qt.SolidLine, Qt.RoundCap)
    p.setBrush(Qt.NoBrush)
    if target >= 64:  # ombre portee de l'anneau
        shadow = QPen(QColor(10, 20, 80, 70), stroke, Qt.SolidLine, Qt.RoundCap)
        p.save()
        p.translate(0, u * 0.014)
        p.setPen(shadow)
        p.drawPath(path)
        p.restore()
    p.setPen(pen)
    p.drawPath(path)

    # pointe de fleche a l'extremite (sens horaire a l'ecran)
    end_deg = math.radians(start + sweep)
    ex, ey = cx + radius * math.cos(end_deg), cy - radius * math.sin(end_deg)
    dx, dy = math.sin(end_deg), math.cos(end_deg)           # direction de marche, ecran, sens horaire
    nx, ny = -dy, dx
    head_len, head_half = u * (0.17 if small else 0.15), u * (0.135 if small else 0.115)
    base = QPointF(ex - dx * head_len * 0.15, ey - dy * head_len * 0.15)
    tip = QPointF(ex + dx * head_len * 0.95, ey + dy * head_len * 0.95)
    arrow = QPolygonF([tip, QPointF(base.x() + nx * head_half, base.y() + ny * head_half),
                       QPointF(base.x() - nx * head_half, base.y() - ny * head_half)])
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("white"))
    p.drawPolygon(arrow)


def draw(target: int) -> QImage:
    """Icone finale de `target` px de cote."""
    px = target * SUPERSAMPLE
    img = QImage(px, px, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    small = target <= 32
    u = px  # unite : tout est exprime en fraction du cote

    # -- fond : carre arrondi degrade ---------------------------------------------------------
    inset = 0.03 if small else 0.045
    body = QRectF(u * inset, u * inset, u * (1 - 2 * inset), u * (1 - 2 * inset))
    bg = QLinearGradient(body.topLeft(), body.bottomRight())
    bg.setColorAt(0, BLUE_TOP)
    bg.setColorAt(1, BLUE_BOTTOM)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg))
    p.drawRoundedRect(body, u * 0.22, u * 0.22)
    if target >= 48:  # reflet doux en haut
        gloss = QLinearGradient(body.topLeft(), QPointF(body.left(), body.center().y()))
        gloss.setColorAt(0, QColor(255, 255, 255, 70))
        gloss.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(gloss))
        p.drawRoundedRect(QRectF(body.left(), body.top(), body.width(), body.height() * 0.55), u * 0.22, u * 0.22)

    cx = cy = u * 0.5
    _draw_loop(p, cx, cy, u, target, small)

    # -- point central : la recompense --------------------------------------------------------
    dot_r = u * (0.115 if small else 0.095)
    if target >= 48:
        halo = QRadialGradient(QPointF(cx, cy), dot_r * 2.6)
        halo.setColorAt(0, QColor(62, 207, 142, 120))
        halo.setColorAt(1, QColor(62, 207, 142, 0))
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(cx, cy), dot_r * 2.6, dot_r * 2.6)
    p.setBrush(GREEN)
    p.setPen(QPen(QColor(255, 255, 255, 235), u * (0.022 if not small else 0.03)))
    p.drawEllipse(QPointF(cx, cy), dot_r, dot_r)
    p.end()
    return img.scaled(target, target, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: Path, images: dict[int, QImage]) -> None:
    """Ecrit un vrai .ico multi-tailles (entrees PNG, gerees par Windows depuis Vista)."""
    blobs = [(size, png_bytes(img)) for size, img in sorted(images.items())]
    out = bytearray(struct.pack("<HHH", 0, 1, len(blobs)))
    offset = 6 + 16 * len(blobs)
    data = bytearray()
    for size, blob in blobs:
        dim = 0 if size >= 256 else size
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset + len(data))
        data += blob
    path.write_bytes(bytes(out) + bytes(data))


def preview(images: dict[int, QImage], path: Path) -> None:
    """Planche de controle : toutes les tailles sur fond sombre, clair et couleur, plus un agrandissement du 16 px."""
    pad = 22
    row = max(SIZES) + 2 * pad
    width = sum(s + pad for s in SIZES) + pad
    sheet = QImage(width, row * 3, QImage.Format_ARGB32)
    p = QPainter(sheet)
    for i, bg in enumerate(("#13151a", "#f3f4f8", "#d8dce6")):
        p.fillRect(0, i * row, width, row, QColor(bg))
        x = pad
        for size in SIZES:
            p.drawImage(x, i * row + pad + (max(SIZES) - size), images[size])
            x += size + pad
    p.end()
    sheet.save(str(path))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    QGuiApplication(sys.argv[:1])
    images = {s: draw(s) for s in SIZES}
    if "--preview" in argv:
        preview(images, Path(argv[argv.index("--preview") + 1]))
        return
    here = Path(__file__).parent
    write_ico(here / "icon.ico", images)
    images[256].save(str(here / "icon.png"))
    docs = here.parent / "docs"
    docs.mkdir(exist_ok=True)
    draw(512).save(str(docs / "logo.png"))
    print("icon.ico :", ", ".join(f"{s}px" for s in SIZES), "| docs/logo.png : 512px")


if __name__ == "__main__":
    main()
