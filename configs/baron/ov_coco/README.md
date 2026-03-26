# Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection).

Download the following files:
* [Metadata JSON files](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing)
* [lvis_v1_train_base.json](https://drive.google.com/drive/folders/1d06y8DxfgkitPRGuk3HSp8BwG7ZFRLeu)
* The pre-generated [pseudo-label JSON file](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main)

Place the pseudo-label JSON file as: `data/coco/hchoi/instances_train2017_pseudo_v0_new_caption.json`


The data structure looks like:

```text
checkpoints/
├── clip_vitb16.pth
├── res50_fpn_soco_star_400.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── hchoi
│   │   ├── instances_train2017_base.json
│   │   ├── instances_train2017_pseudo_v0_new_caption.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
├── lvis_v1
│   ├── hchoi
│   │   ├── lvis_v1_train_base.json
│   │   ├── lvis_v1_train_pseudo_v0_new_caption.json
│   │   ├── lvis_v1_train.json
│   │   ├── lvis_v1_val.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

<details>
  <summary>To manually generate these JSON files, run the following scripts (CLICK)</summary>

```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_train2017.json \
      --out_path data/coco/hchoi/instances_train2017_base.json
```
```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/hchoi/instances_val2017_base.json
```
```bash
python tools/pre_processors/keep_coco_novel.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/hchoi/instances_val2017_novel.json
```

Prepare `captions_train2017_tags_allcaps.json` by following the [Detic instructions](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#:~:text=Next%2C%20we%20preprocess%20the%20COCO%20caption%20data%3A).

Then, place it in: `data/coco/hchoi/`

</details>

Lastly, add the pseudo-label categories to the framework.

For pseudo-labels:
* Follow [pseudo-labels](../../../pseudo-labels/README.md) to generate pseudo-labels.
* Or directly download the pre-generated pseudo-labels from [Hugging Face](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main).

## Class Embeddings
### OV-COCO
Following [BARON](https://github.com/wusize/ovdet), we use the output from the last attention layer for classification.

This is because training on COCO tends to converge toward base categories.

Generate the class embeddings by 

```bash
python tools/hand_craft_prompt.py --model_version ViT-B/16 --ann data/coco/annotations/instances_val2017.json \
--out_path data/metadata/coco_clip_hand_craft.npy --dataset coco
```
```bash
python tools/hand_craft_prompt.py --model_version ViT-B/16 --ann data/coco/hchoi/instances_train2017_pseudo_v0_new_caption.json \
--out_path data/metadata/coco_clip_hand_craft_pseudo_v0_new_caption.npy --dataset coco
```
```bash
python tools/hand_craft_prompt.py --model_version ViT-B/16 --out_path coco_clip_hand_craft_background.npy --background
```

The generated files are used for training and testing, respectively.

### OV-LVIS
Follow the instructions in the official [DetPro GitHub repository](https://github.com/dyabel/detpro) to generate the required text embeddings: `lvis_v1_clip_detpro.npy`

**Note:** Generate the embeddings with **CLIP ViT-B/16**, not **ViT-B/32**.

To generate `lvis_v1_clip_detpro_background.npy`:

1. Modify [prompt/classname.py](https://github.com/dyabel/detpro/blob/main/prompt/classname.py): Add your background concepts.

```python
BACKGROUND_CONCEPTS = [
    "sky", "mountain", "grass", "floor", "wall", "road",
    "sidewalk", "building", "water", "tree", "field", "sand"
]
```

2. Modify [prompt/gen_cls_embedding.py](https://github.com/dyabel/detpro/blob/main/prompt/gen_cls_embedding.py): Add the background dataset case.

```python
elif dataset == 'background':
    model = coop_mini.CustomCLIP(BACKGROUND_CONCEPTS, clip_model, False, prompt).to('cuda')
    checkpoint(model, save_name, BACKGROUND_CONCEPTS)
```

Then, generate features following the DetPro instructions and convert them to `.npy` if needed.


## Backbone
If necessary, pre-train larger backbones such as `RN50x4` following the official [SoCo repository](https://github.com/ueoo/SoCo).

Then, replace the backbone checkpoint with:

```bash
# for OV-COCO, baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py
...
load_from = 'checkpoints/<YOUR_BACKBONE_CKPT>.pth'
```

# Testing

The implementation based on MMDet3.x achieves better results compared to the results reported in the paper.

|             | Backbone |  Method  | Supervision  | Novel AP50 |                                        Config                                        |         Download          |
|:-----------:|:--------:|:--------:|:------------:|:----------:|:------------------------------------------------------------------------------------:|:-------------------------:|
|  Paper  | R-50-FPN |  BARON   |     CLIP     |    34.0    |    -     | - |
|  This Repo  | R-50-FPN |  BARON   |     CLIP     |    34.6    |    [config](baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)     | [model](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) |
|  Paper  | R-50-FPN |  CoT-PL   |     CLIP     |    43.4    |    [config](baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)     | [model](https://drive.google.com/drive/folders/1d06y8DxfgkitPRGuk3HSp8BwG7ZFRLeu?usp=sharing) |

To test the models, run
```bash
bash tools/dist_test.sh <CONFIG_FILE> path/to/save/logs/and/checkpoints <NUM_GPUS> <GPU_IDS>
```


# Training
## Contrastive Background Learning for CLIP Knowledge Distillation
Train the detector based on FasterRCNN+ResNet50+FPN with SyncBN and SOCO pre-trained model. Obtain the SOCO pre-trained 
model from [GoogleDrive](https://drive.google.com/file/d/1rIW9IXjWEnFZa4klZuZ5WNSchRYaOC0x/view?usp=sharing) and put it
under `checkpoints`.
```bash
# OV-COCO
CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=<PORT> bash tools/dist_train.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py <NUM_GPUS> <GPU_IDS> <SEED> --work-dir work_dirs/
```

```bash
# OV-LVIS
CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=<PORT> bash tools/dist_train.sh configs/baron/ov_lvis/baron_kd_ens_mask_rcnn_r50_fpn_syncbn_45kx4_lvis.py <NUM_GPUS> <GPU_IDS> <SEED> --work-dir work_dirs/
```