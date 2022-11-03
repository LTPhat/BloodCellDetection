# Blood Cell Detection With YOLOv6


## Introduction
- Demo YOLOv6 model.
- `Task`: Blood Cell Detection on Blood Smear Images.
- Finished time: `03/11/2022`
## About The Dataset

A public BCCD (Blood Cell Count and Detection) dataset is used
for the performance evaluation of our architecture. BCCD Dataset is a small-scale dataset for blood cells detection. It is not uncommon that blood smear images
are in low resolution, and blood cells on them are blurry and overlapping. 

Link dataset: https://github.com/Shenggan/BCCD_Dataset

We have three kind of labels :

- `RBC` (Red Blood Cell)
- `WBC` (White Blood Cell)
- `Platelets `

An example image in the dataset: 

![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/iilustrate_img/example.jpg)

**The `JPEGImages`:** Image Type : `jpeg(JPEG)` ; Width x Height : `640 x 480`

**The `Annotation`** The VOC format .xml file. So we need convert to COCO format for YOLO training.
```sh
<annotation>
	<folder>JPEGImages</folder>
	<filename>BloodImage_00000.jpg</filename>
	<path>/home/pi/detection_dataset/JPEGImages/BloodImage_00000.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>WBC</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>260</xmin>
			<ymin>177</ymin>
			<xmax>491</xmax>
			<ymax>376</ymax>
		</bndbox>
	</object>
    ...
	<object>
		...
	</object>
</annotation>
```
## About YOLOv6
Paper link: https://arxiv.org/abs/2209.02976.

Github repo: https://github.com/meituan/YOLOv6.

## Perfomance Evaluation For Object Detection Task

All details are saved in: https://github.com/LTPhat/BloodCellDetection/blob/main/MOT_Evaluation_Metrics.ipynb

In training with YOLO, we focus on some metrics below:

  ### 1) Intersection over Union (IoU)
  
  Intersection Over Union (IoU) is a number that quantifies the degree of overlap between two boxes. 
  In the case of object detection and segmentation, IoU evaluates the overlap of Ground Truth and Prediction region.
  
  ![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/iilustrate_img/IOU.png)
  ![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/iilustrate_img/IOU_thress.png)
  
  To objectively judge whether the model predicted the box location correctly or not, a threshold is used.
  
  $$\text{Class}= \begin{cases}
  \text{Positive} \rightarrow \text{ IOU} \ge \text{Threshold}\\
  \text{Negative} \rightarrow \text{IOU} < \text{Threshold}
  \end{cases}$$
  
  ### 2) Average Precision (AP)
  
  The average precision (AP) is a way to summarize the precision-recall curve into a single value representing the average of all precisions. 
  The AP is calculated according to the following equation:
  
   $$\text{AP}=  \sum \limits_{k=0}^{k=n-1} \left[\text{Recall(k)}-\text{Recall(k+1)}\right]\times \text{Precision(k)}$$
   
   $$\text{Precision(n)} = 1, \text{Recall(n)}=0, \text{n: Number of thresholds}$$
  
   ### 3) Mean Average Precision (mAP)
   
   mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them.
   
   $$\text{mAP} = \frac{1}{n} \sum \limits_{k=1}^{k=n} AP_{k}$$
   
   $$AP_k: \text{The AP of class k} $$
   
   $$ n: \text{Number of classes in the model}$$
   
## Training YOLO With Custom Dataset

This guidence explains how to train your own custom data with YOLOv6 (take fine-tuning YOLOv6-s model for example).

### 0. Start

Clone this repo and follow README.md to install requirements in a Python3.8 environment.
```shell
$ git clone https://github.com/meituan/YOLOv6.git
```

Install requirement packages in requirement.txt
```shell
!pip install -r/content/drive/MyDrive/BloodcellDetection/BCCD/YOLOv6/requirements.txt
```
### 1. Prepare the dataset

**Step 1**: Prepare your own dataset with images. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme).

**Step 2**: Generate label files in YOLO format.

One image corresponds to one label file, and the label format example is presented as below.

```json
# class_id center_x center_y bbox_width bbox_height
0 0.300926 0.617063 0.601852 0.765873
1 0.575 0.319531 0.4 0.551562
```


- Each row represents one object.
- Class id starts from `0`.
- Boundingbox coordinates must be in normalized `xywh` format (from 0 - 1). If your boxes are in pixels, divide `center_x` and `bbox_width` by image width, and `center_y` and `bbox_height` by image height.

**Step 3**: Organize directories.

Organize your directory of custom dataset as follows:

```shell
custom_dataset
├── images
│   ├── train
│   │   ├── train0.jpg
│   │   └── train1.jpg
│   ├── val
│   │   ├── val0.jpg
│   │   └── val1.jpg
│   └── test
│       ├── test0.jpg
│       └── test1.jpg
└── labels
    ├── train
    │   ├── train0.txt
    │   └── train1.txt
    ├── val
    │   ├── val0.txt
    │   └── val1.txt
    └── test
        ├── test0.txt
        └── test1.txt
```

**Step 4**: Create `dataset.yaml` in `$YOLOv6_DIR/data`.

```yaml
# Please insure that your custom_dataset are put in same parent dir with YOLOv6_DIR
train: ../custom_dataset/images/train # train images
val: ../custom_dataset/images/val # val images
test: ../custom_dataset/images/test # test images (optional)

# whether it is coco dataset, only coco dataset should be set to True.
is_coco: False

# Classes
nc: 20  # number of classes
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names
```
In this repo, custom dataset yaml file are showed below (Saved as bccd/yaml):

```yaml
train: ../BCCD_dataset/images/train # train images
val: ../BCCD_dataset/images/val # val images

# whether it is coco dataset, only coco dataset should be set to True.
is_coco: False

# Classes
nc: 3  # number of classes
names: ['Platelets', 'RBC', 'WBC']  # class names
```

### 2. Create a config file

We use a config file to specify the network structure and training setting, including  optimizer and data augmentation hyperparameters.

If you create a new config file, please put it under the `configs` directory.
Or just use the provided config file in `$YOLOV6_HOME/configs/*_finetune.py`.

```python
#### YOLOv6s Model config file
model = dict(
    type='YOLOv6s',
    pretrained='./weights/yolov6s.pt', # download the pretrained model from YOLOv6 github if you're going to use the pretrained model
    depth_multiple = 0.33,
    width_multiple = 0.50,
    ...
)
solver=dict(
    optim='SGD',
    lr_scheduler='Cosine',
    ...
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    ...
)
```



### 3. Train


```shell
python tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data <URL_to_custom_data_yaml> --device 0
```



### 4. Evaluation

```shell
python tools/eval.py --data <URL_to_custom_data_yaml> --weights output_dir/name/weights/best_ckpt.pt --device 0
```



### 5. Inference

```shell
python tools/infer.py --weights output_dir/name/weights/best_ckpt.pt --source img.jpg --device 0
```

## Result

- Evaluation: 

 ![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/iilustrate_img/evaluation.png)
 
- Test on images:

![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/runs/inference/exp/BloodImage_00015.jpg)


![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/runs/inference/exp/BloodImage_00041.jpg)


![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/runs/inference/exp/BloodImage_00385.jpg)


![alt text](https://github.com/LTPhat/BloodCellDetection/blob/main/runs/inference/exp/BloodImage_00269.jpg)


# Reference

[1] BCCD Dataset is a small-scale dataset for blood cells detection, https://github.com/Shenggan/BCCD_Dataset.

[2] YOLOv6: a single-stage object detection framework dedicated to industrial applications, https://arxiv.org/abs/2209.02976.

[3] YOLOv6, https://github.com/meituan/YOLOv6.

[4] Complete Blood Cell Detection and Counting Based on Deep Neural Networks, https://www.mdpi.com/2076-3417/12/16/8140/pdf.

[5] RepVGG - Sự trở lại của một tượng đài, https://viblo.asia/p/repvgg-su-tro-lai-cua-mot-tuong-dai-GrLZDGenKk0.

[6] Train Custom Data, https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md.

