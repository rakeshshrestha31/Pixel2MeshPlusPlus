# Pixel2Mesh++

This is the Pixel2Mesh++ baseline of "A Real World Dataset for Multi-view 3D Reconstruction", ECCV'22 [arXiv](https://arxiv.org/abs/2203.11397).
It is based on [Pixel2Mesh++](https://github.com/walsvid/Pixel2MeshPlusPlus).

## Dependencies

Requirements:

- Python3.6
- numpy
- Tensorflow==1.12.0
- tflearn==0.3.2
- opencv-python

The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 on Ubuntu 16.04.

## Compile CUDA-op

If you use Chamfer Distance for training or evaluation, we have included the cuda implementations of [Fan et. al.](https://github.com/fanhqme/PointSetGeneration) in `external/`.

We recommend readers to follow the official tutorial of Tensorflow for how to compile the CUDA code. Please refer to [official tutorial](https://www.tensorflow.org/guide/extend/op#gpu_support).


## Dataset

Download the preprocessed OCRTOC 3D reconstruction dataset
```
wget https://gruvi-3dv.cs.sfu.ca/ocrtoc-3d-reconstruction/meshmvs_ocrtoc_dataset_full.zip
unzip meshmvs_ocrtoc_dataset_full.zip
```

Edit data_root of [mvp2m_ocrtoc.yaml](cfgs/mvp2m_ocrtoc.yaml) and [p2mpp_ocrtoc.yaml](cfgs/p2mpp_ocrtoc.yaml) to the location you extracted meshmvs_ocrtoc_dataset_full.zip.

## Training

For training, you should first train the coarse shape generation network, then generate intermediate results, and finally train the multi-view deformation network.

#### Step1
For training coarse shape generation, please set your own configuration in `cfgs/mvp2m_ocrtoc.yaml`. Specifically, the meaning of the setting items is as follows. For more details, please refer to [`modules/config_ocrtoc.py`](modules/config_ocrtoc.py).

- `train_file_path`: the path of your own train split file which contains training data name for each instance
- `train_image_path`: input image path
- `train_data_path`: ground-truth model path
- `coarse_result_***`: the configuration items related to the coarse intermediate mesh should be same as the training data

Then execute the script:
```
python train_mvp2m_ocrtoc.py -f cfgs/mvp2m_ocrtoc.yaml
```

#### Step2
Before training multi-view deformation network, you should generated coarse intermediate mesh.

```
python generate_mvp2m_intermediate_ocrtoc.py -f cfgs/mvp2m_ocrtoc.yaml

```

#### Step3
For training multi-view deformation network, please set your own configuration in `cfgs/p2mpp_ocrtoc.yaml`.

The configuration item is similar to Step1. In particular, `train_mesh_root` should be set to the output path of intermediate coarse shape generation.
Then execute the script:

```
python train_p2mpp_ocrtoc.py -f cfgs/p2mpp_ocrtoc.yaml
```

## Evaluation

#### Step 1
Generate coarse shape, you also need to set your own configuration in `cfgs/mvp2m_ocrtoc.yaml` as mentioned previously, then execute the script:
```
python test_mvp2m_ocrtoc.py -f cfgs/mvp2m_ocrtoc.yaml
python test_p2mpp.py -f cfgs/p2mpp.yaml
```

#### Step2
You should set `test_mesh_root` in `cfgs/p2mpp_ocrtoc.yaml` to the output folder in step1 and `test_image_path`,`test_file_path` as it mentioned in Training step.

Then execute the script:
```
python test_p2mpp_ocrtoc.py -f cfgs/p2mpp_ocrtoc.yaml
```

For evaluating F-score and Chamfer distance you can need to execute the script [tools/test_p2mpp.py](https://github.com/rakeshshrestha31/meshmvs_ocrtoc/blob/depth_initialization/tools/test_p2mpp.py) from the `meshmvs_ocrtoc` baseline.
```
python tools/test_p2mpp.py --dataset_root <path_to>/meshmvs_ocrtoc_dataset_full --predict_dir <predict_dir> --p2mpp_dir <p2mpp_dir> --splits_file <path_to>/meshmvs_ocrtoc_dataset_full/ocrtoc_splits_val05.json
```

Set `<predict_dir>` according to your configurations
