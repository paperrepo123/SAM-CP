# 1. Install 
## Install mmdetection and pytorch
see [install](docs/en/get_started.md), the version is:
cuda=11.7
pytorch=2.0.1
torchvision=0.15.2
mmcv=2.1.0
```shell
conda create --name samcp python=3.8 -y
conda activate samcp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
## or ----> pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

```

## Install others
```shell
cd SAM-CP
pip install timm ftfy regex tqdm
pip install segment_anything_main/
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/yinglang/multiprocplus.git
pip install -r requirements.txt
pip install -v -e .
```

# 2. Prepare datasets
1. Download [COCO dataset](https://cocodataset.org/) (with COCO-panoptic annotation) to data/coco
2. Download the SAM patches for training set and val set (from: xxxx), move the patches to pretrained/sam_proposals
3. or, you can generate SAM patches by yourself:

# 3. Generate SAM patches
1. Download SAM weights (from: [Segment anything](https://github.com/facebookresearch/segment-anything)) to pretrained/sam
2. The cmd for training set (dataset_settype='train2017') and val set (dataset_settype='val2017'):

```shell
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0   --master_addr='127.0.0.1' --master_port='29500' segment_anything_main/test/seg_generation_coco.py  --dataset_settype='train2017' --model_type='vit_h'
```
The IoU (Box and Mask) between SAM pathes and GT are automaticly calculated in the SAM patch generation procedure.

# 4. Training for COCO dataset

```shell
# put the COCO dataset to data/coco, then:
./tools/dist_train.sh configs2/SamCP/coco_panoptic/samcp_r50_12e_panoptic_ms_crop_improved_param.py 8
```
