# Factorizable Net (F-Net)

This is pytorch implementation of our ECCV-2018 paper: [**Factorizable Net: An Efficient Subgraph-based Framework for Scene Graph Generation**](http://cvboy.com/publication/eccv2018_fnet/). This project is based on our previous work: [**Multi-level Scene Description Network**](https://github.com/yikang-li/MSDN). 

## Progress
- [x] Guide for Project Setup
- [x] Guide for Model Evaluation with pretrained model
- [x] Guide for Model Training
- [x] Uploading pretrained model and format-compatible datasets.
- [x] Update the Model link for VG-DR-Net (We will upload a new model by Aug. 27). 
- [x] Update the Dataset link for VG-DR-Net. 
- [ ] A demonstration of our Factorizable Net 
- [x] Migrate to PyTorch 1.0.1
- [x] Multi-GPU support (beta version): one image per GPU

## Updates
- **Feb 26, 2019: Now we release our beta [Multi-GPU] version of Factorizable Net. Find the stable version at branch [0.3.1](https://github.com/yikang-li/FactorizableNet/tree/0.3.1)**
- Aug 28, 2018: Bug fix for running the evaluation with "--use_gt_boxes". VG-DR-Net has some self-relations, e.g. A-relation-A. Previously, we assumed there is no such relation. This commit may influence the model performance on Scene Graph Generation. 

## Project Settings

1. Install the requirements (you can use pip or [Anaconda](https://www.continuum.io/downloads)):

    ```
    conda install pip pyyaml sympy h5py cython numpy scipy click
    conda install -c menpo opencv3
    conda install pytorch torchvision cudatoolkit=8.0 -c pytorch 
    pip install easydict
    ```

2. Clone the Factorizable Net repository
    ```bash
    git clone git@github.com:yikang-li/FactorizableNet.git
    ```

3. Build the Cython modules for nms, roi pooling,roi align modules
    ```bash
    cd lib
    make all
    cd ..
    ```
5. Download the three datasets [**VG-MSDN**](https://drive.google.com/open?id=1WjetLwwH3CptxACrXnc1NCcccWUVDO76), [**VG-DR-Net**](https://drive.google.com/open?id=1JZecHzzwGj1hxnn77hPOlOvqpjavebcD), [**VRD**](https://drive.google.com/open?id=12oLtVSCEusG7tG4QwxeJEDsVhiE9gb2s) to ```F-Net/data```. And extract the folders with ```tar xzvf ${Dataset}.tgz```. We have converted the original annotations to ```json``` version. 

6. Download [**Visual Genome images**](http://visualgenome.org/api/v0/api_home.html) and [**VRD**](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip) images. 
7. Link the image data folder to 	target folder: ```ln -s /path/to/images F-Net/data/${Dataset}/images```
	- p.s. You can change the default data directory by modifying ```dir``` in ```options/data_xxx.json```.
8. [optional] Download the pretrained RPN for [Visual Genome](https://drive.google.com/open?id=1W7PYyYvkROzC_GZwrgF0XS4fH6r2NyyV) and [VRD](https://drive.google.com/open?id=1OdzZKn5ZBIXFdxeOjCvjqNhFjobWnDS9). Place them into ```output/```.
4. [optional] Download the pretrained Factorizable Net on [VG-MSDN](https://drive.google.com/file/d/1iKgYVLTUHi_VpmWrQJ6o1OMj3aGlrmDC/view), [VG-DR-Net](https://drive.google.com/open?id=1b-RoEeRWju1Mz4EESaagXOIWpUriBm_D) and [VG-VRD](https://drive.google.com/open?id=1n-8d4K7-PywVwuA90x50nnIW1TKyKHU4), and place them to ```output/trained_models/```

## Project Organization
There are several subfolders contained:

- ```lib```: dataset Loader, NMS, ROI-Pooling, evaluation metrics, etc. are listed in the folder.
- ```options```: configurations for ```Data```, ```RPN```, ```F-Net``` and ```hyperparameters```.
- ```models```: model definitions for ```RPN```, ```Factorizable``` and related modules.
- ```data```: containing VG-DR-Net (```svg/```), VG-MSDN (```visual_genome/```) and VRD (```VRD/```).
- ```output```: storing the trained model, checkpoints and loggers.

## Evaluation with our Pretrained Models
Pretrained models on [VG-MSDN](https://drive.google.com/open?id=1iKgYVLTUHi_VpmWrQJ6o1OMj3aGlrmDC), [VG-DR-Net](https://drive.google.com/open?id=1b-RoEeRWju1Mz4EESaagXOIWpUriBm_D) and [VG-VRD](https://drive.google.com/open?id=1n-8d4K7-PywVwuA90x50nnIW1TKyKHU4) are provided. ```--evaluate``` is provided to enable evaluation mode. Additionally, we also provide ```--use_gt_boxes``` to fed the ground-truth object bounding boxes instead of RPN proposals. 

- Evaluation on **VG-MSDN** with pretrained.
Scene Graph Generation results:  Recall@50: ```12.984%```, Recall@100: ```16.506%```.

```
CUDA_VISIBLE_DEVICES=0 python train_FN.py --evaluate --dataset_option=normal \
	--path_opt options/models/VG-MSDN.yaml \
	--pretrained_model output/trained_models/Model-VG-MSDN.h5
```



- Evaluation on **VG-VRD** with pretrained. :  Scene Graph Generation results:  Recall@50: ```19.453%```, Recall@100: ```24.640%```.

```
CUDA_VISIBLE_DEVICES=0 python train_FN.py --evaluate \
	--path_opt options/models/VRD.yaml \
	--pretrained_model output/trained_models/Model-VRD.h5
```

- Evaluation on **VG-DR-Net** with pretrained.
Scene Graph Generation results:  Recall@50: ```19.807%```, Recall@100: ```25.488%```.

```
CUDA_VISIBLE_DEVICES=0 python train_FN.py --evaluate --dataset_option=normal \
	--path_opt options/models/VG-DR-Net.yaml \
	--pretrained_model output/trained_models/Model-VG-DR-Net.h5
```


## Training
- Training Region Proposal Network (RPN). The **shared conv layers** are fixed. We also provide pretrained RPN on [Visual Genome](https://drive.google.com/open?id=1W7PYyYvkROzC_GZwrgF0XS4fH6r2NyyV) and [VRD](https://drive.google.com/open?id=1OdzZKn5ZBIXFdxeOjCvjqNhFjobWnDS9). 
	
	```
	# Train RPN for VG-MSDN and VG-DR-Net
	CUDA_VISIBLE_DEVICES=0 python train_rpn.py --dataset_option=normal 
	
	# Train RPN for VRD
	CUDA_VISIBLE_DEVICES=0 python train_rpn_VRD.py 
	
	```

- Training Factorizable Net: detailed training options are included in ```options/models/```.

	```
	# Train F-Net on VG-MSDN:
	CUDA_VISIBLE_DEVICES=0 python train_FN.py --dataset_option=normal \
		--path_opt options/models/VG-MSDN.yaml --rpn output/RPN.h5
		
	# Train F-Net on VRD:
	CUDA_VISIBLE_DEVICES=0 python train_FN.py  \
		--path_opt options/models/VRD.yaml --rpn output/RPN_VRD.h5
		
	# Train F-Net on VG-DR-Net:
	CUDA_VISIBLE_DEVICES=0 python train_FN.py --dataset_option=normal \
		--path_opt options/models/VG-DR-Net.yaml --rpn output/RPN.h5
	
	```
	
	```--rpn xxx.h5``` can be ignored in end-to-end training from pretrained **VGG16**. Sometime, unexpected and confusing errors appear. *Ignore it and restart to training.*
	
- For better results, we usually re-train the model with additional epochs by resuming the training from the checkpoint with ```--resume ckpt```:

	```
	# Resume F-Net training on VG-MSDN:
	CUDA_VISIBLE_DEVICES=0 python train_FN.py --dataset_option=normal \
		--path_opt options/models/VG-MSDN.yaml --resume ckpt --epochs 30
	```

## Acknowledgement

We thank [longcw](https://github.com/longcw/faster_rcnn_pytorch) for his generous release of the [PyTorch Implementation of Faster R-CNN](https://github.com/longcw/faster_rcnn_pytorch). 


## Reference

If you find our project helpful, your citations are highly appreciated:

@inproceedings{li2018fnet,  
	author={Li, Yikang and Ouyang, Wanli and Bolei, Zhou and Jianping, Shi and Chao, Zhang and Wang, Xiaogang},  
	title={Factorizable Net: An Efficient Subgraph-based Framework for Scene Graph Generation},  
	booktitle = {ECCV},  
	year      = {2018}  
}

We also have two papers regarding to scene graph generation / visual relationship detection:

@inproceedings{li2017msdn,  
	author={Li, Yikang and Ouyang, Wanli and Zhou, Bolei and Wang, Kun and Wang, Xiaogang},  
	title={Scene graph generation from objects, phrases and region captions},  
	booktitle = {ICCV},  
	year      = {2017}  
}

@inproceedings{li2017vip,  
	author={Li, Yikang and Ouyang, Wanli and Zhou, Bolei and Wang, Kun and Wang, Xiaogang},  
	title={ViP-CNN: Visual Phrase Guided Convolutional Neural Network},  
	booktitle = {CVPR},  
	year      = {2017}  
}


## License:

The pre-trained models and the Factorizable Network technique are released for uncommercial use.

Contact [Yikang LI](http://www.cvboy.com/) if you have questions.
