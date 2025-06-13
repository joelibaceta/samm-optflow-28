import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import imageio

def cargar_frames_por_sujeto(base_path, frame_index=0, max_sujetos=None):
    """
    Carga un solo frame de la primera secuencia de cada sujeto del dataset SAMM.
    
    Args:
        base_path (str): Ruta base del dataset SAMM.
        frame_index (int): Índice del frame a usar (por defecto 0, es decir el primero).
        max_sujetos (int or None): Máximo número de sujetos a procesar (útil para pruebas o grillas limitadas).
    
    Returns:
        List[np.ndarray]: Lista de imágenes (formato RGB).
        List[str]: Lista de nombres de sujetos.
    """
    sujetos = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    imagenes = []
    nombres = []

    for sujeto in sujetos:
        sujeto_path = os.path.join(base_path, sujeto)
        secuencias = sorted([d for d in os.listdir(sujeto_path) if os.path.isdir(os.path.join(sujeto_path, d))])
        if not secuencias:
            continue

        secuencia_path = os.path.join(sujeto_path, secuencias[0])
        frames = sorted([f for f in os.listdir(secuencia_path) if f.endswith('.jpg')])
        if not frames:
            continue

        idx = min(frame_index, len(frames) - 1)
        frame_path = os.path.join(secuencia_path, frames[idx])
        img = cv2.imread(frame_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagenes.append(img_rgb)
        nombres.append(f"Sujeto {sujeto}")

        if max_sujetos and len(imagenes) >= max_sujetos:
            break

    return imagenes, nombres

def presentar_dataset_samm(base_path):
    imagenes, nombres = cargar_frames_por_sujeto(base_path, frame_index=0, max_sujetos=10)

    # Mostrar en grilla
    cols = 5
    rows = (len(imagenes) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle("Vista general del dataset SAMM (1er frame por sujeto)", fontsize=16)

    for i, ax in enumerate(axs.flat):
        if i < len(imagenes):
            ax.imshow(imagenes[i])
            ax.set_title(nombres[i], fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1)
    plt.show()
    return imagenes, nombres 


def mostrar_diferencia_recortes(modelo_mtcnn, a, b,
                                threshold=10, margin=40, output_size=(224,224)):
    """
    Unifica detección, alineación y visualización de diferencias entre dos recortes faciales.
    - Si 'a' y 'b' son paths: a→detect_crop + b→template_matching, luego diff.
    - Si 'a','b' son np.ndarray: diff directo.
    - Si uno es path y otro np.ndarray: usa el ndarray como crop o plantilla según posición.
    """
    def detect_crop(path):
        img = Image.open(path).convert("RGB")
        box,_ = modelo_mtcnn.detect(img)
        if box is None: raise ValueError(f"No se detectó rostro en {path}")
        x1,y1,x2,y2 = map(int, box[0])
        arr = np.array(img)
        crop = arr[y1:y2, x1:x2]
        return crop, (x1,y1,x2,y2)

    def template_align(path, template, prev_box):
        bgr = cv2.imread(path)
        tpl = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
        x1,y1,x2,y2 = prev_box
        h,w = bgr.shape[:2]
        sx1,sy1 = max(0,x1-margin), max(0,y1-margin)
        sx2,sy2 = min(w,x2+margin), min(h,y2+margin)
        region = bgr[sy1:sy2, sx1:sx2]
        if region.shape[0]<tpl.shape[0] or region.shape[1]<tpl.shape[1]:
            raise ValueError("Región de búsqueda demasiado pequeña")
        _,_,_,max_loc = cv2.minMaxLoc(cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED))
        tx,ty = sx1+max_loc[0], sy1+max_loc[1]
        crop = bgr[ty:ty+tpl.shape[0], tx:tx+tpl.shape[1]]
        return cv2.cvtColor(cv2.resize(crop, output_size), cv2.COLOR_BGR2RGB)

    def to_rgb_array(x):
        # convierte PIL.Image o ruta o array BGR a array RGB uint8
        if isinstance(x, str) or isinstance(x, Image.Image):
            img = Image.open(x).convert("RGB") if isinstance(x, str) else x.convert("RGB")
            return np.array(img)
        if isinstance(x, np.ndarray):
            arr = x.astype(np.uint8)
            if arr.ndim==2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if arr.ndim==3 and arr.shape[2]==3:
                # asumimos que viene BGR
                return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            if arr.ndim==3 and arr.shape[2]==1:
                return cv2.cvtColor(arr[:,:,0], cv2.COLOR_GRAY2RGB)
        raise TypeError("No soportado: debe ser ruta, PIL.Image o np.ndarray")

    # 1) Generar crop1_rgb y crop2_rgb
    if isinstance(a, str) and isinstance(b, str):
        crop1, box = detect_crop(a)
        crop2 = template_align(b, crop1, box)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        crop1, crop2 = a, b
    elif isinstance(a, str) and isinstance(b, np.ndarray):
        crop1, box = detect_crop(a)
        crop2 = to_rgb_array(b)
    elif isinstance(a, np.ndarray) and isinstance(b, str):
        crop1 = to_rgb_array(a)
        crop2 = template_align(b, crop1, (0,0,crop1.shape[1],crop1.shape[0]))
    else:
        raise TypeError("Parámetros no válidos")

    # 2) Unificar tamaños
    if crop1.shape != crop2.shape:
        crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))

    # 3) A escala de grises
    g1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY)

    # 4) absdiff + threshold
    diff = cv2.absdiff(g1, g2)
    _, diff_thr = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 5) Plot dinámico: 3 subplots
    titles = ["Crop 1", "Crop 2", f"Dif > {threshold}"]
    imgs   = [g1,      g2,      diff_thr]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, ttl in zip(axs, imgs, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(ttl)
        ax.axis("off")
    plt.tight_layout()
    plt.show()




def mostrar_secuencia_samm(base_dir, subject_id, clip_name, onset, apex, offset):
    """
    Muestra en una grilla las imágenes correspondientes a los frames de Onset, Apex y Offset.

    Parámetros:
    - base_dir: ruta base del dataset (ej. "datasets/SAMM")
    - subject_id: ID del sujeto (ej. 10)
    - clip_name: nombre del clip (ej. "010_2_1")
    - onset, apex, offset: índices de los frames correspondientes
    """
    subject = str(subject_id).zfill(3)
    img_dir = Path(base_dir) / subject / clip_name

    nombres = [onset, apex, offset]
    titulos = ['Onset', 'Apex', 'Offset']
    paths = [img_dir / f"{subject}_{i:04d}.jpg" for i in nombres]

    plt.figure(figsize=(12, 4))
    for idx, path in enumerate(paths):
        if not path.exists():
            print(f"⚠️ Imagen no encontrada: {path}")
            continue
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, idx+1)
        plt.imshow(img)
        plt.title(titulos[idx])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def obtener_marcas_samm(csv_path, subject_id, clip_name):
    """
    Extrae los frames de Onset, Apex y Offset para un sujeto y clip dados del dataset SAMM.

    Parámetros:
        csv_path (str): Ruta al archivo samm.csv
        subject_id (str o int): ID del sujeto, ej. "014"
        clip_name (str): Nombre del clip, ej. "014_1_1"

    Retorna:
        (onset, apex, offset): Tupla de enteros
    """
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    subject_str = str(subject_id).zfill(3)

    fila = df[
        (df["Subject"].astype(str).str.zfill(3) == subject_str) &
        (df["Filename"] == clip_name)
    ]

    if fila.empty:
        raise ValueError(f"No se encontró la fila para sujeto {subject_str}, clip {clip_name}")

    onset = int(fila["Onset Frame"].values[0])
    apex = int(fila["Apex Frame"].values[0])
    offset = int(fila["Offset Frame"].values[0])

    return onset, apex, offset


def generar_gif_microexpresion(base_dir, subject_id, clip_name, onset, apex, offset, output_path="microexp.gif", fps=2):
    """
    Genera un GIF animado mostrando la progresión Onset → Apex → Offset.

    Args:
        base_dir (str): Directorio base donde están las imágenes.
        subject_id (str/int): ID del sujeto, se formatea con ceros (ej. 014).
        clip_name (str): Nombre del clip (ej. "014_1_1").
        onset, apex, offset (int): Índices de frame.
        output_path (str): Ruta donde guardar el GIF.
        fps (int): Fotogramas por segundo (velocidad del gif).
    """
    subject_str = str(subject_id).zfill(3)
    clip_path = Path(base_dir) / subject_str / clip_name

    frame_ids = [onset, apex, offset]  
    images = []

    for fid in frame_ids:
        filename = f"{subject_str}_{str(fid).zfill(4)}.jpg"
        path = clip_path / filename
        if not path.exists():
            print(f"⚠️ Imagen no encontrada: {path}")
            continue
        img = Image.open(path).convert("RGB")
        images.append(img)

    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"✅ GIF guardado en: {output_path}")
    else:
        print("❌ No se pudieron cargar imágenes para el GIF.")