
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import argparse
import logging


def crop_with_template(
    img: np.ndarray,
    template: np.ndarray,
    prev_box: tuple,
    margin: int = 40,
    output_size: tuple = (224, 224)
) -> np.ndarray:
    """
    Align a face in `img` to `template` using template matching around a fixed `prev_box`.
    The `prev_box` remains the apex bounding box for the entire sequence to avoid
    recursive or chained cropping.

    Args:
        img (np.ndarray): BGR image array.
        template (np.ndarray): RGB template crop from the apex frame.
        prev_box (tuple): (x1, y1, x2, y2) bounding box of the template in the apex frame.
        margin (int): Search margin around prev_box (pixels).
        output_size (tuple): Desired output size of the aligned crop.

    Returns:
        np.ndarray: Aligned BGR crop of `output_size`, or None if matching fails.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = prev_box
    # Define fixed search region once, relative to apex coords
    sx1 = max(0, x1 - margin)
    sy1 = max(0, y1 - margin)
    sx2 = min(w, x2 + margin)
    sy2 = min(h, y2 + margin)
    search_region = img[sy1:sy2, sx1:sx2]

    tpl_h, tpl_w = template.shape[:2]
    if search_region.shape[0] < tpl_h or search_region.shape[1] < tpl_w:
        return None  # Template larger than search region

    # Match template (convert RGB template to BGR for matching)
    res = cv2.matchTemplate(
        search_region,
        template[:, :, ::-1],  # RGB→BGR
        cv2.TM_CCOEFF_NORMED
    )
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # Compute crop coordinates in original image
    top_left = (sx1 + max_loc[0], sy1 + max_loc[1])
    bottom_right = (top_left[0] + tpl_w, top_left[1] + tpl_h)
    crop = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return cv2.resize(crop, output_size)


def process_clip(
    clip_path: str,
    output_clip_dir: str,
    mtcnn: MTCNN
):
    """
    Process a single clip:
    1) Detect the apex frame and extract a fixed template and bounding box.
    2) Save the apex crop.
    3) Align every other frame in the clip to that same template (apex),
       avoiding any recursive or chained cropping.
    """
    os.makedirs(output_clip_dir, exist_ok=True)

    # List all frames in the clip
    jpgs = sorted(
        f for f in os.listdir(clip_path)
        if f.lower().endswith('.jpg')
    )
    if not jpgs:
        logging.warning(f"No images in {clip_path}, skipping.")
        return

    # 1) Choose apex frame (midpoint)
    apex_idx = len(jpgs) // 2
    apex_file = jpgs[apex_idx]
    apex_path = os.path.join(clip_path, apex_file)

    # 2) Detect face in apex and create fixed template
    apex_img = Image.open(apex_path).convert('RGB')
    boxes, _ = mtcnn.detect(apex_img)
    if boxes is None or len(boxes) == 0:
        logging.warning(f"No face detected in apex frame {apex_path}")
        return

    # Use this same box (prev_box) for all subsequent frames
    x1, y1, x2, y2 = map(int, boxes[0])
    apex_np = np.array(apex_img)
    template_face = apex_np[y1:y2, x1:x2]
    apex_crop_bgr = cv2.cvtColor(
        cv2.resize(template_face, (224, 224)),
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(
        os.path.join(output_clip_dir, apex_file),
        apex_crop_bgr
    )

    # 3) Align every other frame to the **same** apex template
    for fname in tqdm(jpgs, desc=f"Clip {os.path.basename(clip_path)}"):
        if fname == apex_file:
            continue

        in_path = os.path.join(clip_path, fname)
        out_path = os.path.join(output_clip_dir, fname)
        try:
            # Read frame as BGR
            frame_bgr = cv2.cvtColor(
                np.array(
                    Image.open(in_path).convert('RGB')
                ),
                cv2.COLOR_RGB2BGR
            )
            aligned = crop_with_template(
                frame_bgr,
                template_face,
                (x1, y1, x2, y2)
            )
            if aligned is not None:
                cv2.imwrite(out_path, aligned)
            else:
                logging.warning(f"Template matching failed for {in_path}")
        except Exception as exc:
            logging.error(f"Error processing {in_path}: {exc}")


def process_dataset(
    input_base: str,
    output_base: str,
    use_gpu: bool = False
):
    """
    Iterate over all subjects and clips in the SAMM dataset,
    normalizing each sequence via fixed-template alignment.
    """
    device = 'cuda' if use_gpu else 'cpu'
    mtcnn = MTCNN(
        image_size=224, margin=20,
        keep_all=False, post_process=False,
        device=device
    )

    for subject in sorted(os.listdir(input_base)):
        subj_path = os.path.join(input_base, subject)
        if not os.path.isdir(subj_path):
            continue
        for clip in sorted(os.listdir(subj_path)):
            clip_path = os.path.join(subj_path, clip)
            if not os.path.isdir(clip_path):
                continue
            output_clip = os.path.join(output_base, subject, clip)
            process_clip(clip_path, output_clip, mtcnn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalize SAMM dataset via fixed-template matching.'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to the original SAMM dataset.'
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Path for the normalized output dataset.'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Enable GPU acceleration for MTCNN.'
    )
    parser.add_argument(
        '--log', default='INFO',
        help='Logging level: DEBUG, INFO, WARNING, ERROR.'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    os.makedirs(args.output, exist_ok=True)
    process_dataset(args.input, args.output, use_gpu=args.gpu)
