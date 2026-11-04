"""
Seguimiento de objetos en una secuencia de imágenes con YOLO + OpenCV.

Para cada objetivo y cada imagen:
  1. YOLO detecta todos los objetos de las clases COCO del objetivo (ej. "person", "car").
  2. Inicialización: en el primer frame se elige el candidato por color distintivo
     (fracción de píxeles de ese color en el rectángulo) y/o por cercanía a un punto
     semilla (u, v) dado en OBJETIVOS.
  3. Seguimiento: en los frames siguientes se predice dónde debería estar el rectángulo
     (velocidad constante a partir de las dos últimas detecciones) y se elige el
     candidato con mayor IoU contra esa predicción.
  4. Se dibuja el rectángulo, su centro (u, v) y la trayectoria acumulada.

Si en algún frame no se detecta, el centro se interpola linealmente entre los frames
vecinos y queda marcado como "interpolado" en el CSV.

Nota: la columna t_s sale de la hora del nombre de archivo (img_HH_MM_SS_mmm.jpg).
Esos tiempos tienen mucho jitter (31 a 249 ms entre fotos) y no se correlacionan con
el desplazamiento de los objetos; el movimiento es más regular por número de frame.

Coordenadas: u = columna (eje x, hacia la derecha), v = fila (eje y, hacia abajo),
origen en la esquina superior izquierda de la imagen, en píxeles.

Uso:
    .venv/bin/python seguimiento.py                          # todos los objetivos
    .venv/bin/python seguimiento.py --objetivo senora_roja
    .venv/bin/python seguimiento.py --objetivo auto_negro_blanco auto_gris

Salida en <salida>/<objetivo>/:
    frames/          imágenes anotadas
    centros.csv      rectángulo y centro (u, v) por imagen
    trayectoria.png  trayectoria completa sobre el último frame
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Rangos HSV de OpenCV (H: 0-179, S: 0-255, V: 0-255)
COLORES = {
    "rojo": [((0, 120, 90), (8, 255, 255)), ((165, 120, 90), (179, 255, 255))],
    "amarillo": [((18, 120, 120), (35, 255, 255))],
    "azul": [((100, 120, 50), (130, 255, 255))],
}

# Cada objetivo:
#   clases       clases COCO de YOLO a considerar
#   color        color distintivo (clave de COLORES) o None
#   min_color    fracción mínima de ese color para aceptar un candidato
#   semilla      (u, v) aproximado del centro en el primer frame donde aparece, o None
#   color_dibujo color BGR del rectángulo
OBJETIVOS = {
    "senora_roja": {
        "clases": ["person"],
        "color": "rojo",
        "min_color": 0.05,
        "semilla": None,
        "color_dibujo": (0, 0, 255),
    },
    # Taxi negro con techo amarillo: primer auto que baja por la calle.
    "auto_negro_blanco": {
        "clases": ["car", "truck"],
        "color": None,
        "min_color": 0.0,
        "semilla": (934, 252),
        "color_dibujo": (255, 0, 255),
    },
    # Camioneta pick-up gris oscura: segundo auto, detrás del taxi.
    "auto_gris": {
        "clases": ["car", "truck"],
        "color": None,
        "min_color": 0.0,
        "semilla": (995, 242),
        "color_dibujo": (255, 160, 0),
    },
}

RADIO_SEMILLA = 50  # px: distancia máxima entre la semilla y el centro del candidato
MIN_IOU = 0.1       # IoU mínimo con la caja predicha para aceptar la detección


def tiempo_desde_nombre(nombre):
    """img_10_18_47_351.jpg -> segundos desde medianoche (10:18:47.351)."""
    m = re.search(r"(\d+)_(\d+)_(\d+)_(\d+)", nombre)
    if not m:
        return None
    h, mi, s, ms = map(int, m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def fraccion_color(hsv, caja, rangos):
    x1, y1, x2, y2 = caja
    recorte = hsv[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    mascara = np.zeros(recorte.shape[:2], np.uint8)
    for bajo, alto in rangos:
        mascara |= cv2.inRange(recorte, bajo, alto)
    return float(mascara.mean() / 255)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def caja_de(r):
    return (r["x1"], r["y1"], r["x2"], r["y2"])


def predecir_caja(historial, frame):
    """Desplaza la última caja según la velocidad media (px/frame) de las últimas
    detecciones. Se usa el número de frame y no el tiempo del nombre de archivo
    porque esos tiempos tienen mucho jitter y no se corresponden con el movimiento."""
    ultimo = historial[-1]
    base = historial[max(0, len(historial) - 4)]
    if base["frame"] == ultimo["frame"]:
        return caja_de(ultimo)
    pasos = (frame - ultimo["frame"]) / (ultimo["frame"] - base["frame"])
    du = (ultimo["u"] - base["u"]) * pasos
    dv = (ultimo["v"] - base["v"]) * pasos
    x1, y1, x2, y2 = caja_de(ultimo)
    return (x1 + du, y1 + dv, x2 + du, y2 + dv)


def elegir_objetivo(candidatos, cfg, historial, frame):
    """Devuelve el candidato que corresponde al objetivo en este frame, o None."""
    if cfg["color"]:
        candidatos = [c for c in candidatos if c["color"] >= cfg["min_color"]]
    if not candidatos:
        return None

    if historial:
        pred = predecir_caja(historial, frame)
        ultima = caja_de(historial[-1])
        puntaje = lambda c: max(iou(pred, caja_de(c)), iou(ultima, caja_de(c)))
        mejor = max(candidatos, key=puntaje)
        return mejor if puntaje(mejor) >= MIN_IOU else None

    # Primer frame con el objetivo.
    if cfg["semilla"] is not None:
        su, sv = cfg["semilla"]
        cercano = min(candidatos, key=lambda c: np.hypot(c["u"] - su, c["v"] - sv))
        return cercano if np.hypot(cercano["u"] - su, cercano["v"] - sv) <= RADIO_SEMILLA else None
    if cfg["color"]:
        return max(candidatos, key=lambda c: c["color"])
    return max(candidatos, key=lambda c: c["conf"])


def interpolar_faltantes(registros):
    """Completa por interpolación lineal (según el número de frame) los frames sin
    detección que quedan entre dos frames detectados."""
    detectados = [i for i, r in enumerate(registros) if r["metodo"] == "yolo"]
    for a, b in zip(detectados, detectados[1:]):
        for i in range(a + 1, b):
            r, ra, rb = registros[i], registros[a], registros[b]
            alfa = (i - a) / (b - a)
            for k in ("x1", "y1", "x2", "y2", "u", "v"):
                r[k] = ra[k] + alfa * (rb[k] - ra[k])
            r["metodo"] = "interpolado"


def dibujar(img, r, trayectoria, color):
    for p, q in zip(trayectoria, trayectoria[1:]):
        cv2.line(img, p, q, (0, 255, 255), 2, cv2.LINE_AA)
    for p in trayectoria:
        cv2.circle(img, p, 3, (0, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(img, f"frame {r['frame']}  {r['archivo']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    if r["metodo"] == "no_detectado":
        cv2.putText(img, "objetivo no detectado", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return

    x1, y1, x2, y2 = (int(round(r[k])) for k in ("x1", "y1", "x2", "y2"))
    u, v = int(round(r["u"])), int(round(r["v"]))
    grosor = 2 if r["metodo"] == "yolo" else 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor, cv2.LINE_AA)
    cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.circle(img, (u, v), 4, (0, 255, 0), -1, cv2.LINE_AA)

    texto = f"(u={r['u']:.1f}, v={r['v']:.1f})"
    if r["metodo"] == "interpolado":
        texto += " interp."
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx, ty = x1, max(y1 - 8, th + 4)
    cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, texto, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def guardar_csv(ruta, registros):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "archivo", "t_s", "x1", "y1", "x2", "y2", "u", "v", "confianza", "metodo"])
        for r in registros:
            if r["metodo"] == "no_detectado":
                w.writerow([r["frame"], r["archivo"], f"{r['t']:.3f}", "", "", "", "", "", "", "", r["metodo"]])
                continue
            w.writerow([
                r["frame"], r["archivo"], f"{r['t']:.3f}",
                *(f"{r[k]:.1f}" for k in ("x1", "y1", "x2", "y2", "u", "v")),
                f"{r['conf']:.3f}" if r["conf"] is not None else "",
                r["metodo"],
            ])


def guardar_trayectoria(ruta, img, registros):
    """Trayectoria completa sobre una imagen, numerando cada 5 frames."""
    final = img.copy()
    puntos = [(r["frame"], int(round(r["u"])), int(round(r["v"]))) for r in registros if r["metodo"] != "no_detectado"]
    for (_, u1, v1), (_, u2, v2) in zip(puntos, puntos[1:]):
        cv2.line(final, (u1, v1), (u2, v2), (0, 255, 255), 2, cv2.LINE_AA)
    for n, u, v in puntos:
        cv2.circle(final, (u, v), 4, (0, 255, 0), -1, cv2.LINE_AA)
        if n % 5 == 0:
            cv2.putText(final, str(n), (u - 8, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(ruta), final)


def seguir(nombre, cfg, modelo, rutas, imagenes, args):
    rangos = COLORES[cfg["color"]] if cfg["color"] else None
    ids_clase = [i for i, n in modelo.names.items() if n in cfg["clases"]]
    salida = args.salida / nombre
    (salida / "frames").mkdir(parents=True, exist_ok=True)

    print(f"\n=== {nombre} ===")
    t0 = tiempo_desde_nombre(rutas[0].name)
    registros, historial = [], []

    for i, (ruta, img) in enumerate(zip(rutas, imagenes)):
        res = modelo.predict(img, imgsz=args.imgsz, conf=args.conf, classes=ids_clase, agnostic_nms=True, verbose=False)[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if rangos else None

        candidatos = []
        for caja, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = caja.tolist()
            candidatos.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "u": (x1 + x2) / 2, "v": (y1 + y2) / 2,
                "conf": float(conf),
                "color": fraccion_color(hsv, caja.astype(int), rangos) if rangos else 0.0,
            })

        t = tiempo_desde_nombre(ruta.name)
        t = (t - t0) if t is not None else float(i)
        elegido = elegir_objetivo(candidatos, cfg, historial, i)

        r = {"frame": i, "archivo": ruta.name, "t": t, "conf": None, "metodo": "no_detectado"}
        if elegido:
            r.update({k: elegido[k] for k in ("x1", "y1", "x2", "y2", "u", "v", "conf")})
            r["metodo"] = "yolo"
            historial.append(r)
        registros.append(r)
        print(f"[{i + 1:3d}/{len(rutas)}] {ruta.name}  {r['metodo']:13s}" + (f" u={r['u']:7.1f} v={r['v']:7.1f}" if elegido else ""))

    interpolar_faltantes(registros)

    trayectoria = []
    for r, img in zip(registros, imagenes):
        if r["metodo"] != "no_detectado":
            trayectoria.append((int(round(r["u"])), int(round(r["v"]))))
        anotada = img.copy()
        dibujar(anotada, r, trayectoria, cfg["color_dibujo"])
        cv2.imwrite(str(salida / "frames" / r["archivo"]), anotada)

    guardar_trayectoria(salida / "trayectoria.png", imagenes[-1], registros)
    guardar_csv(salida / "centros.csv", registros)

    n_yolo = sum(r["metodo"] == "yolo" for r in registros)
    n_interp = sum(r["metodo"] == "interpolado" for r in registros)
    print(f"Detectado por YOLO: {n_yolo}/{len(registros)}  interpolados: {n_interp}  -> {salida}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objetivo", nargs="+", default=list(OBJETIVOS), choices=OBJETIVOS.keys())
    ap.add_argument("--imagenes", default="imagenes", type=Path)
    ap.add_argument("--salida", default="resultados", type=Path)
    ap.add_argument("--modelo", default="yolo11m.pt", help="pesos de YOLO (se descargan solos la primera vez)")
    ap.add_argument("--imgsz", default=1280, type=int, help="resolución de inferencia de YOLO")
    ap.add_argument("--conf", default=0.25, type=float, help="confianza mínima de YOLO")
    args = ap.parse_args()

    rutas = sorted(args.imagenes.glob("*.jpg"))
    if not rutas:
        raise SystemExit(f"No hay imágenes .jpg en {args.imagenes}")
    imagenes = [cv2.imread(str(p)) for p in rutas]
    modelo = YOLO(args.modelo)

    for nombre in args.objetivo:
        seguir(nombre, OBJETIVOS[nombre], modelo, rutas, imagenes, args)


if __name__ == "__main__":
    main()
