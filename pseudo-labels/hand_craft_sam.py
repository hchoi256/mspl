# hand_craft_sam.py
import os
import json
import traceback
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from qwen import Qwen, QwenResult


# -----------------------------
# Initialize Qwen
# -----------------------------

qwen = Qwen()

# -----------------------------
# SAM init / mask filtering
# -----------------------------

def init_sam(gpu_id: int = 0) -> SamAutomaticMaskGenerator:
    torch.cuda.set_device(gpu_id)
    sam_ckpt_path = "../checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt_path).to(f"cuda:{gpu_id}")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    return mask_generator


def filter_keep(keep: torch.Tensor, masks_result) -> list:
    keep = keep.int().cpu().numpy()
    return [masks_result[i] for i in keep]


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            inter = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = inter / union
            iou_matrix[i, j] = iou
            if inter / masks_area[i] < 0.5 and inter / masks_area[j] >= 0.85:
                inner = 1 - (inter / masks_area[j]) * (inter / masks_area[i])
                inner_iou_matrix[i, j] = inner
            if inter / masks_area[i] >= 0.85 and inter / masks_area[j] < 0.5:
                inner = 1 - (inter / masks_area[j]) * (inter / masks_area[i])
                inner_iou_matrix[j, i] = inner

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_max_u, _ = torch.triu(inner_iou_matrix, diagonal=1).max(dim=0)
    inner_max_l, _ = torch.tril(inner_iou_matrix, diagonal=1).max(dim=0)

    keep = (iou_max <= iou_thr) & (scores > score_thr) & \
           (inner_max_u <= 1 - inner_thr) & (inner_max_l <= 1 - inner_thr)

    if keep.sum() == 0:
        keep[scores.topk(3).indices] = True

    return idx[keep]


def masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5):
    if len(masks) == 0:
        return []

    seg_pred = torch.from_numpy(np.stack([m["segmentation"] for m in masks])).to("cuda")
    iou_pred = torch.from_numpy(np.array([m["predicted_iou"] for m in masks])).to("cuda")
    stability = torch.from_numpy(np.array([m["stability_score"] for m in masks])).to("cuda")

    scores = stability * iou_pred
    keep_idx = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)
    return filter_keep(keep_idx, masks)


# -----------------------------
# ROI preprocessing (same behavior as before)
# -----------------------------

def make_emphasized_roi(image_bgr: np.ndarray, segmentation: np.ndarray, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox

    # background blur/desaturate inside bbox
    masked_img = np.zeros_like(image_bgr)

    bbox_crop = image_bgr[y:y+h, x:x+w]
    bbox_gray = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2GRAY)
    bbox_gray_blur = cv2.GaussianBlur(bbox_gray, (31, 31), 0)
    bbox_gray_blur_rgb = cv2.cvtColor(bbox_gray_blur, cv2.COLOR_GRAY2BGR)
    masked_img[y:y+h, x:x+w] = bbox_gray_blur_rgb

    # emphasize saturation/value on segmentation region
    rgb_patch = image_bgr.copy()
    # hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2HSV)
    # hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
    # hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
    # emphasized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    masked_img[segmentation] = rgb_patch[segmentation]

    # draw green bbox
    vis_img = masked_img.copy()
    # cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # crop bbox region
    crop = vis_img[y:y+h, x:x+w]
    roi_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return roi_pil


# -----------------------------
# Main processing
# -----------------------------

def process_single_image(
    img_file: str,
    data_dir: str,
    mask_generator: SamAutomaticMaskGenerator,
    debug: bool = False,
    vis_dir: str = "vis",
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks_raw = mask_generator.generate(image_rgb)
        masks = masks_update(masks_raw)

        img_id = str(int(os.path.splitext(img_file)[0]))

        fg_bboxes, fg_categories, fg_meta = [], [], []
        bg_bboxes, bg_categories, bg_meta = [], [], []

        if debug:
            os.makedirs(vis_dir, exist_ok=True)

        for idx, mask in enumerate(masks):
            segmentation = mask["segmentation"]
            if np.sum(segmentation) == 0:
                continue

            bbox = mask["bbox"]
            x, y, w, h = map(int, bbox)

            if w * h < 10000:
                continue

            bbox = (x, y, w, h)
            roi_pil = make_emphasized_roi(image, segmentation, bbox)

            # Single-pass Qwen inference
            qres = qwen.forward(roi_pil, bbox=bbox)
            if debug:
                roi_path = os.path.join(vis_dir, f"tmp.png")
                roi_pil.save(roi_path)
                print(
                    "[DEBUG] Qwen Output\n"
                    f"  file      : {img_file}\n"
                    f"  bbox      : {bbox}\n"
                    f"  presence  : {qres.presence}\n"
                    f"  class_name  : {qres.class_name}\n"
                    f"  is_foreground     : {qres.is_foreground}\n"
                    f"  description : {qres.description}\n"
                    f"  saved     : {roi_path}\n"
                )
                input()

            # Convert to your downstream FG/BG lists:
            # - If presence == No: ignore this mask (no is_foreground)
            # - Else: use is_foreground Foreground/Background
            if qres.presence == "No":
                continue

            if qres.is_foreground == "Yes":
                fg_bboxes.append(bbox)
                fg_categories.append(qres.class_name)
                fg_meta.append(
                    {"presence": qres.presence, "is_foreground": qres.is_foreground, "description": qres.description}
                )
            elif qres.is_foreground == "No":
                bg_bboxes.append(bbox)
                bg_categories.append(qres.class_name)
                bg_meta.append(
                    {"presence": qres.presence, "is_foreground": qres.is_foreground, "description": qres.description}
                )
            else:
                # If is_foreground empty (should only happen if presence==No, but keep safe)
                continue

        return img_id, {
            "foreground": {"bbox": fg_bboxes, "category": fg_categories, "meta": fg_meta},
            "background": {"bbox": bg_bboxes, "category": bg_categories, "meta": bg_meta},
        }

    except Exception as e:
        print(f"[ERROR] {img_file}: {e}")
        traceback.print_exc()
        return None


def process_dataset(
    data_dir: str,
    output_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    debug: bool = False,
    vis_dir: str = "vis",
):
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    if end_idx is None:
        end_idx = len(image_files)

    image_files = image_files[start_idx:end_idx]
    results: List[Tuple[str, Dict[str, Any]]] = []

    mask_generator = init_sam(gpu_id=0)

    for img_file in tqdm(image_files, desc="Processing"):
        result = process_single_image(img_file, data_dir, mask_generator, debug=debug, vis_dir=vis_dir)
        if result is not None:
            results.append(result)

    annotations = {img_id: bboxes_dict for (img_id, bboxes_dict) in results}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSaved annotations for {len(annotations)} images to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/coco/train2017")
    parser.add_argument("--output_path", type=str, default="0.json")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="vis")
    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        debug=args.debug,
        vis_dir=args.vis_dir,
    )
