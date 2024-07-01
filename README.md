## System Requirements

- Linux, Windows
- Python 3.8+
- PyTorch 1.13.1
- CUDA 11.7

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Dataset Preparation

Your data directory tree should look like this:

```

${DATASET_ROOT}
-- annotations (directory name)
    | -- person_keypoints_train2017.json (train annotation file)
    | -- person_keypoints_val2017.json (val annotation file)

-- images (directory name)
    -- train2017 (directory name)
      | -- ~.png
      | -- ~.png
      | -- ~.png
      | -- ... 
    -- val2017 (directory name)
      | -- ~.png
      | -- ~.png
      | -- ~.png
      | -- ... 
```

Please rename the annotation file to `person_keypoints_train2017.json` and `person_keypoints_val2017.json` for training and validation, respectively.

Once the dataset is prepared, please change the `data_root` variable in the `configs/_base_/datasets/coco_keypoint.py` to your own data directory.

## Keypoint Annotation

We use [labelme](https://github.com/wkentaro/labelme) to annotate the keypoints of the infant. You could use your prefered annotation tool to annotate the keypoints of the infant. The annotation file must be in the format of [COCO](https://cocodataset.org/#format-data) Keypoints dataset.

You need to create a json (dictionary-like) file for the annotation. In the annotation file, you need to have three keys: 'image', 'annotation', 'categories'. You can refer to [explore_annotation.ipynb](explore_annotation.ipynb) to see the structure of annotation file. Particularly, the structure shoudld look like:

```
image{
"id": int, 
"width": int, 
"height": int, 
"file_name": str,
}


annotation{
"id": int, 
"image_id": int, 
"category_id": int, 
"area": float, 
"bbox": [x,y,width,height], 
"keypoints": [x1,y1,v1,...],
"num_keypoints" : int,
"iscrowd": 0 or 1,
}


categories[{
"id": int, 
"name": str, 
"supercategory": str,
"keypoints": [str,...],
"skeleton": [edge1, edge2, ...]
}]
```

We use 17 keypoints to annotate the infant. Therefore, the 'keypoints' Keys in the 'cetegories' dictionary in the annotation file should be:
![](/docs/keypoint_format.png)

```
"keypoints": ['nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle'],
```

The 'skeleton' Keys in the 'categories' dictionary in the annotation file should be:

```
"skeleton": [[16, 14],
  [14, 12],
  [17, 15],
  [15, 13],
  [12, 13],
  [6, 12],
  [7, 13],
  [6, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 11],
  [2, 3],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
  [5, 7]]
```

## Fine-tune from model trained on WashU infant dataset

We provide the model fine-tuned on our WashU-NICU newborn pose dataset. Doing this way lets you compare the difference between your infant dataset to WashU NICU infant dataset in terms of model generaliztion.

|  Model  | Backbone | Training Dataset | Fine-tuning Dataset | mAP | AP@50 | AP@75 | AP@M | AP@L |                                            Download                                            |
| :------: | :------: | :--------------: | :-----------------: | :--: | :---: | :---: | :--: | :--: | :---------------------------------------------------------------------------------------------: |
| NICUPose |  Swin-L  |  COCO-train2017  |     WashU NICU     | 86.5 |  100  | 99.6 | 87.5 | 86.0 | [Google Drive](https://drive.google.com/file/d/1tU6d1XcnXJuv5VLeUxNnXAUIDoDRPFpQ/view?usp=sharing) |

The training command is as follows:

```
bash tools/dist_train.sh configs/models/swin-l-p4-w7-224-22kto1k_16x1_100e_coco.py 1 --resume-from nicupose.pth 
```

## Evalution

#### (1) Get the accuracy only :

```
python ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval keypoints
```

A working example :

```
python tools/test.py configs/models/swin-l-p4-w7-224-22kto1k_16x1_100e_coco.py nicupose.pth --eval keypoints
```

#### (2) Get the predicted keypoint in a json file:

```
python ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --eval-options "jsonfile_prefix=./results
```

A working example:

```
python tools/test.py configs/petr/petr_swin-l-p4-w7-224-22kto1k_16x1_100e_coco.py nicupose.pth --format-only --eval-options jsonfile_prefix=./results
```

***Note:*** test_results.bbox.json and test_results.keypoints.json will be generated in the ${ROOT}/ directory. You can further use these two files to label your own dataset.

## Demo

We provide a image demo script to run the model on a single GPU for a single image or a image folder.

The general demo script is as follows:

```
python demo/image_demo.py  ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE}  --out-file ${OUTPUT_PATH} --pose_only
```

***Note:*** The pose-only flag is used to visualize the pose only. If you want to visualize the pose on top of the image, please remove the pose_only flag.

## Credit

[PETR](https://github.com/hikvision-research/opera): End-to-End Multi-Person Pose Estimation with Transformers

[MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab Computer Vision Foundation

## License

This project is released under the [Apache 2.0 License](LICENSE) and includes modifications by [qwe1256 ](https://github.com/qwe1256)as noted in the [NOTICE](NOTICE) file.
