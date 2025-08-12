# AMES: Asymmetric and Memory-Efficient Similarity
***

This repository contains the code for the paper ["AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval"](https://arxiv.org/abs/2408.03282), by the authors Pavel Suma, Giorgos Kordopatis-Zilos, Ahmet Iscen, and Girogos Tolias.
In Proceedings of the European Conference on Computer Vision (ECCV), 2024

## TLDR

Transformer-based model estimating image-to-image similarity that offers a good balance between performance and memory.

***

## Setup
***
This code was implemented using Python 3.12.3 and the following dependencies:

```
torch==2.7.0
hydra-core==1.3.2
numpy==2.1.2
tqdm==4.66.5
h5py==3.12.1
```

You can install them via pip:
```
pip install -r requirements.txt
```


## Trained models
***
We provide AMES trained on GLDv2 in four variants. Available models are trained with full-precision (fp) or binary (dist) local descriptors extracted from either DINOv2 or CVNet backbone: `dinov2_ames`, `dinov2_ames_dist`, `cvnet_ames`, and `cvnet_ames_dist`. 

You can download all models manually from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks) or use torch hub to use the specified model directly:

```
import torch

model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()
```

Example usage of the model:

```python
img_1_feat = torch.randn(1, 100, 768)
img_2_feat = torch.randn(1, 100, 768)
sim = model(src_local=img_1_feat, tgt_local=img_2_feat)
```

## Evaluation
***
In order to evaluate the performance of our models, you need to have the extracted local descriptors of the datasets.
We provide them for ROxford5k, and RParis6k without distractors and GLDv2 test set. For +1M distractors of ROP, please see below how to extract them yourself.
The descriptors along with the extracted global similarities for the query nearest neighbors can be downloaded from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data).

You can also run the following command to download the necessary ROP files into the `data` folder.
```
wget -r -nH --cut-dirs=5 --no-parent --reject="index.html*" --reject-regex ".*/gldv2(-test)?/.*" -P data http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data/
```

A sample command to run the evaluation on these two datasets is as follows:

```
python3 -u src/evaluate.py --multirun \
        descriptors=dinov2 \
        data_root=data \
        model_path=dinov2_ames.pt \
        model.binarized=False \
        dataset@test_dataset=roxford \
        test_dataset.query_desc_num=600 \
        test_dataset.db_desc_num=600 \
        test_dataset.batch_size=300 \
        test_dataset.lamb=[0.55] \
        test_dataset.temp=[0.3] \
        test_dataset.num_rerank=[1600]
```

Hyperparameters used for our best performing AMES setup, tuned on GLDv2 public test split, are as follows:

| Parameter  | DINOv2 (fp) | DINOV2 (dist) | CVNet (fp) | CVNet (dist)  |
|------------|-------------|---------------|------------|---------------|
| `lamb` (λ) | 0.50        | 0.35          | 0.85       | 0.65          |
| `temp` (γ) | 0.20        | 0.10          | 0.80       | 0.20          |

For reference, the provided models, re-ranking SuperGlobal initial ranking, reach the mAP scores below. The full results (for three seeds) are listed in the supplementary material of our ECCV paper.

| Model              | ROxf (M) | ROxf (H) | RPar (M) | RPar (H) |
|--------------------|----------|----------|----------|----------|
| `dinov2_ames`      | 92.7     | 83.7     | 95.2     | 90.6     |
| `dinov2_ames_dist` | 91.1     | 82.2     | 95.2     | 90.2     |
| `cvnet_ames`       | 89.2     | 78.7     | 93.4     | 86.8     |
| `cvnet_ames_dist`  | 88.8     | 77.5     | 93.2     | 86.3     |


## Training
***

To train AMES, you need to have the extracted local descriptors of the training set (GLDv2).
DINOv2 local descriptors (float16) along with their computed global similarities can be downloaded from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data).
You can also run the following command to download them into the `data` folder.

```
wget -r -nH --cut-dirs=7 --no-parent --reject="index.html*" -P data/gldv2 http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data/gldv2/
```

> Note: The training set is large and the download may take a while. You can extract the descriptors yourself by following the instructions below.

A sample command to train AMES is as follows:

```
python3 -u src/train.py --multirun \
        desc_name=dinov2 \
        data_root=${PWD}/data \
        model.binarized=False \
        train_dataset.batch_size=300 \
```

To train AMES with binary local descriptors via distillation, use the following command:

```
python3 -u src/train.py --multirun \
        desc_name=dinov2 \
        data_root=${PWD}/data \
        model.binarized=True \
        train_dataset.batch_size=300 \
        teacher=dinov2_ames.pt
```


## Extracting descriptors
***

The code contains scripts to extract global and local descriptors of GLDv2, ROxford5k, and RParis6k.
Supported backbones are CVNet and DINOv2, however the code can be easily extended to other CNN and ViT backbones.  

Revisited Oxford and Paris (ROP) dataset, along with 1M distractors can be downloaded from the [original site](http://cmp.felk.cvut.cz/revisitop/).
Likewise, GLDv2 train and test can be downloaded from the [official repository](https://github.com/cvdfoundation/google-landmark).

You will need additional dependencies for the extraction of local descriptors:
```
opencv-python-headless==4.10.0.84
```

By default, descriptors are stored in format such as `dinov2_gallery_local.hdf5` in a corresponding dataset folder under `save_path`.
Images are loaded from the `data_path` folder. For each dataset split, a `.txt` file is required to specify the image paths. 
We provide these files for each dataset in the `data` folder.

Extraction of descriptors can be done by running the following command:
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python extract/extract_descriptors.py --dataset [gldv2|roxford5k|rparis6k] \
                              --backbone [cvnet|dinov2] \
                              --save_path data \
                              --data_path [path_to_images] \
                              --split [_gallery|_query|] \
                              --file_name test_gallery.txt \
                              --desc_type "local"
                              --pretrained
```

Weights parameter is only needed for CVNet. Please follow the [original repository](https://github.com/sungonce/CVNet) to download them.
By default, the pretrained detector weights are loaded for the corresponding backbone.
Take a look into the `extract/extract_descriptors.py` file for more argument parameter details.

### Creating the nearest neighbor index

We provide the global-retrieval precomputed nearest neighbor indices for all datasets in files `nn_dinov2.pkl` and `nn_superglobal.pkl` for the two respective backbones. 
To reproduce this index creation using the extracted global descriptors, you can run the following command:
```
python extract/prepare_topk_global.py --dataset [gldv2|gldv2-test|roxford5k|rparis6k] --desc_name dinov2 --data_root ${PWD}/data
```

### Combining multiple hdf5 files

As the number of local features is large for some datasets, it is beneficial to extract the features in parallel chunks and/or store the chunks in individual files. We provide a script that virtually links these chunks into a single hdf5 file for ease of use.
```
python extract/merge_hdf5.py --dataset [gldv2|gldv2-test|roxford5k|rparis6k] --desc_name dinov2 --data_root ${PWD}/data
```


## Citation
***

```
@InProceedings{Suma_2024_ECCV,
    author    = {Suma, Pavel and Kordopatis-Zilos, Giorgos and Iscen, Ahmet and Tolias, Giorgos},
    title     = {AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2024}
}
```

## Acknowledgements

This code is based on the repository of RRT:
[Instance-level Image Retrieval using Reranking Transformers](https://github.com/uvavision/RerankingTransformer).

CVNet extraction code is based on the repository of CVNet:
[Correlation Verification for Image Retrieval](https://github.com/sungonce/CVNet)
