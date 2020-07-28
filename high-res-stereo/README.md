# For those who would like a quick start with inference
## Requirements
- tested with python 2.7.15 and 3.6.8
- tested with pytorch 0.4.0, 0.4.1 and 1.0.0
- a few packages need to be installed, for eamxple, texttable, scikit-image

## Prepare your data
* {dataset}
    * {pair_1}
        * im0.png
        * im1.png
    * {pair_2}
        * im0.png
        * im1.png
    * ...

## Execute command
CUDA_VISIBLE_DEVICES=0 python submission.py --datapath {dataset}  --outdir {output} --loadmodel ./final-768px.pth --testres 1 --clean 0.8 --max_disp 512

## Check output
* {output}
    * {pair_1}
        * img_reference.png
        * img_target.png
        * disp.npy
        * uncertainty.npy
        * disp.jpg
        * disp.cbar.jpg
        * disp_min_max.txt
        * disp.mask.jpg
        * uncertainty.jpg
        * time.txt
    * {pair_2}
        * img_reference.png
        * img_target.png
        * disp.npy
        * uncertainty.npy
        * disp.jpg
        * disp.cbar.jpg
        * disp_min_max.txt
        * disp.mask.jpg
        * uncertainty.jpg
        * time.txt
    * ...

---
---
---

# Hierarchical Deep Stereo Matching on High Resolution Images
Architecture:
<img src="./architecture.png" width="800">

Qualitative results on Middlebury (refer to [project webpage](http://www.contrib.andrew.cmu.edu/~gengshay/cvpr19stereo) for more results)
<img src="http://www.contrib.andrew.cmu.edu/~gengshay/wordpress/wp-content/uploads/2019/06/cvpr19-middlebury1-small.gif" width="400">

Performance on Middlebury benchmark (y-axis: the lower the better)
<img src="./middlebury-benchmark.png" width="400">


## Requirements
- tested with python 2.7.15 and 3.6.8
- tested with pytorch 0.4.0, 0.4.1 and 1.0.0
- a few packages need to be installed, for eamxple, texttable

## Weights
[Download](https://drive.google.com/file/d/1BlH7IafX-X0A5kFPd50WkZXqxo0_gtoI/view?usp=sharing)

## Data

### train/val
- [Middlebury (train set and additional images)](https://drive.google.com/file/d/1jJVmGKTDElyKiTXoj6puiK4vUY9Ahya7/view?usp=sharing)
- [High-res-virtual-stereo (HR-VS)](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view?usp=sharing)
- [KITTI-2012&2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### test
[High-res-real-stereo (HR-RS)](): comming soon

## Train
1. Download and extract training data in folder /d/. Training data include Middlebury train set, HR-VS, KITTI-12/15 and SceneFlow.
2. Run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --maxdisp 384 --batchsize 24 --database /d/ --logname log1 --savemodel /somewhere/  --epochs 10
```
3. Evalute on Middlebury additional images and KITTI validation set. After 10 epochs, average error on Middlebury *additional* images with half-res should be around 4.6 (excluding Shopvac).

## Inference
Example:
```
CUDA_VISIBLE_DEVICES=3 python submission.py --datapath ./data-mbtest/   --outdir ./mboutput --loadmodel ./weights/final-768px.pth  --testres 1 --clean 0.8 --max_disp -1
```

Evaluation:
```
CUDA_VISIBLE_DEVICES=3 python submission.py --datapath ./data-HRRS/   --outdir ./output --loadmodel ./weights/final-768px.pth  --testres 0.5
python eval_disp.py --indir ./output --gtdir ./data-HRRS/
```

And use [cvkit](https://github.com/roboception/cvkit) to visualize in 3D.

## Example outputs
<img src="data-mbtest/CrusadeP/im0.png" width="400">
left image
<img src="mboutput/CrusadeP/capture_000.png" width="400">
3D projection
<img src="mboutput/CrusadeP-disp.png" width="400">
disparity map
<img src="mboutput/CrusadeP-ent.png" width="400">
uncertainty map (brighter->higher uncertainty)

## Parameters
- testres: 1 is full resolution, and 0.5 is half resolution, and so on
- max_disp: maximum disparity range to search
- clean: threshold of cleaning. clean=0 means removing all the pixels.

## Citation
```
@InProceedings{yang2019hsm,
author = {Yang, Gengshan and Manela, Joshua and Happold, Michael and Ramanan, Deva},
title = {Hierarchical Deep Stereo Matching on High-Resolution Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Acknowledgement
Part of the code is borrowed from [MiddEval-SDK](http://vision.middlebury.edu/stereo/submit3/), [PSMNet](https://github.com/JiaRenChang/PSMNet), [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) and [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).
Thanks [SorcererX](https://github.com/SorcererX/high-res-stereo) for fixing version compatibility issues.
