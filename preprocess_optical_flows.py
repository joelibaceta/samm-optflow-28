import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

class SAMMFlowProcessor:
    def __init__(self,
                 csv_path: str,
                 input_base: str,
                 output_base: str,
                 output_size: tuple = (28, 28)):
        """
        Args:
          csv_path: Ruta al samm.csv con columnas Subject, Filename, Onset Frame, Apex Frame, Offset Frame.
          input_base: Directorio de recortes normalizados (e.g. "datasets/SAMM_Normalized_TM").
          output_base: Carpeta donde guardar los mapas de flujo (e.g. "samm_flow_pngs").
          output_size: Tamaño (w, h) de la imagen de salida.
        """
        self.csv_path    = csv_path
        self.input_base  = input_base
        self.output_base = output_base
        self.output_size = output_size
        os.makedirs(self.output_base, exist_ok=True)

    def _find_image(self, subj: str, clip: str, frame: int) -> str:
        """
        Busca de forma robusta el archivo de imagen para un frame dado,
        probando rellenos de 5 y 4 dígitos.
        """
        base_dir = os.path.join(self.input_base, subj, clip)
        for z in (5, 4):
            fname = f"{subj}_{str(frame).zfill(z)}.jpg"
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                return path
        return None

    def _generate_flow(self, img1_path, img2_path, out_path):
        # 1. Leer como gris
        g1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            print(f"❌ Error leyendo imágenes: {img1_path}, {img2_path}")
            return

        # 2. Calcular Farnebäck Optical Flow denso
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 3. Magnitud y ángulo
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # 4. Mapear a HSV
        h, w = g1.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2).astype(np.uint8)  # H: 0–180
        hsv[..., 1] = 255                         # S: 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 5. HSV → BGR y redimensionar
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out = cv2.resize(bgr, self.output_size, interpolation=cv2.INTER_AREA)

        # 6. Guardar
        cv2.imwrite(out_path, out)

    def process(self):
        df = pd.read_csv(self.csv_path, sep=';')
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando SAMM flows"):
            subj = str(row["Subject"]).zfill(3)
            clip = row["Filename"]
            onset = int(row["Onset Frame"])
            apex  = int(row["Apex Frame"])

            # Buscar rutas de onset y apex de forma robusta
            p_onset = self._find_image(subj, clip, onset)
            p_apex  = self._find_image(subj, clip, apex)
            if p_onset is None or p_apex is None:
                print(f"⚠️ No se encontraron imágenes para {subj}/{clip}: onset={onset}, apex={apex}")
                continue

            # Nombre de salida: subXXX_clip.png
            out_name = f"sub{subj}_{clip}.png"
            out_path = os.path.join(self.output_base, out_name)

            self._generate_flow(p_onset, p_apex, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generar mapas de Optical Flow para microexpresiones SAMM'
    )
    parser.add_argument('-c', '--csv', required=True,
                        help='Ruta al samm.csv con anotaciones')
    parser.add_argument('-i', '--input', required=True,
                        help='Directorio de recortes normalizados (SAMM_Normalized_TM)')
    parser.add_argument('-o', '--output', required=True,
                        help='Directorio de salida para guardar mapas de flujo')
    parser.add_argument('--width', type=int, default=28,
                        help='Ancho de la imagen de salida')
    parser.add_argument('--height', type=int, default=28,
                        help='Alto de la imagen de salida')
    args = parser.parse_args()

    processor = SAMMFlowProcessor(
        csv_path=args.csv,
        input_base=args.input,
        output_base=args.output,
        output_size=(args.width, args.height)
    )
    processor.process()
