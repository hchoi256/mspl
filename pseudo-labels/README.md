# Pseudo-Label Generation Process
Pseudo-labels for unlabeled COCO images are generated through a two-stage workflow:
1. Pseudo-label generation
2. Pseudo-label post-processing

To apply the same workflow to the OV-LVIS dataset, replace `coco/` with `lvis/` in the instructions.


## Installation

Set up a conda environment:
```bash
conda create -n pbl python=3.8 -y
source activate pbl
```

Install required packages as follows:
```bash
cd pseudo-labels/
pip install transformers
pip install qwen-vl-utils
pip install 'accelerate>=0.26.0'

cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Place the checkpoint `sam_vit_b_01ec64.pth` in the `checkpoints/` directory:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 1) Multi-Step Pseudo-Label Generation
In our setting, pseudo-annotations are generated for a subset of 10,000 training images using a single GPU.

The process can be parallelized across 12 GPUs to handle the full 118,287 images in the COCO training set.

```bash
CUDA_VISIBLE_DEVICES=0 python hand_craft_sam.py --start 0 --end 10000 --output_path result/10000.json
CUDA_VISIBLE_DEVICES=1 python hand_craft_sam.py --start 10000 --end 20000 --output_path result/20000.json
...
CUDA_VISIBLE_DEVICES=11 python hand_craft_sam.py --start 110000 --end 118287 --output_path result/118287.json
```

By default, the proposed pipeline employs the MLLM `Qwen2-VL-7B-Instruct`.
- Optionally, if you want to change a different model, such as InstructBLIP, replace the line `model = Qwen()` with the other model in the `hand_craft_sam.py` file.

## 2) Pseudo-Label Refinement
Place the generated pseudo-label batches in `pseudo_v0/`.

Then, go to `coco/`, where the post-processing scripts such as `pseudo<Number>.py` are located.

Create a `result/` directory to save the processed outputs:


```text
coco/
├── pseudo1.py
├── pseudo2.py
├── pseudo3.py
├── pseudo4.py
├── pseudo5.py
├── pseudo6.py
├── result/
pseudo_v0/
├── 10000.json
├── 20000.json
├── 30000.json
...
├── 118287.json
```

To run the entire pseudo-label generation workflow automatically, use:

```bash
cd ../coco
bash run.sh
```

This script automatically performs the following steps:

<details>
  <summary>CLICK</summary>

```bash
cd coco/
python pseudo1.py # Merge all batch output files
python pseudo2.py # Extract unseen classes
python pseudo3.py # Select pseudo-labels that exceeds a threshold

Expected output:
['dog', 'knife', 'fish', 'stick', 'cup', 'elephant', 'box', 'pole', 'cat', 'lamp', 'airplane', 'house', 'umbrella', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'cell phone', 'cow', 'cake', 'plane', 'shoe', 'skateboard', 'phone', 'bus', 'light', 'wine glass', 'cabinet', 'traffic light', 'cloth', 'keyboard', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'tennis racket', 'computer', 'ship', 'snake', 'fire hydrant', 'sink', 'bread', 'stop sign', 'tomato', 'couch', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign']
```

Copy and paste the output from `pseudo3.py` into the `pseudo_classes` variable in the following files: `pseudo4.py`, `pseudo5.py`, and `pseudo6.py`.

```bash
python pseudo4.py // Match the COCO format
python pseudo5.py // Merge the pseudo-labels with the base dataset
python pseudo6.py // Print the class weights

Expected output:
Extra COCO classes: ['fish', 'stick', 'box', 'lamp', 'house', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'plane', 'shoe', 'phone', 'light', 'cabinet', 'cloth', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'computer', 'ship', 'snake', 'bread', 'tomato', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign'] // the number of classes is 44
Class weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
```

Update the class weights in the configuration file. Set the `num_pls` variable to the number of extra COCO classes, as shown below:


```text
// inside the baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py file
num_pls = 44
class_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1] + [1] * num_pls + [0.7]
```

Copy and paste the extra COCO classes into the `metainfo` section of the `coco.py` file as shown below:


```text
# inside the `../reprod/coco.py` file
METAINFO = {'classes': ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
# pseudo-labels
'fish', 'stick', 'box', 'lamp', 'house', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'plane', 'shoe', 'phone', 'light', 'cabinet', 'cloth', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'computer', 'ship', 'snake', 'bread', 'tomato', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign'),
```

</details>

Move the generated file: `result/instances_train2017_pseudo_v0_new_caption.json`

to the COCO annotation directory: `data/coco/hchoi/`

Then, update `coco.py` in the MMDetection COCO package as follows:

```bash
cd ../reprod/
bash proprocess.sh
```
