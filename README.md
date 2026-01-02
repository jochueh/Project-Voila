# Project Voilà
This repository contains the code for Project Voilà.



## Dataset
The training process requires a customized combined version of the first two datasets and a filtered version of the third dataset. 
Please contact me for both of them to simplify your training process.

- Large Dataset of Geotagged Images (derived from YFCC100M): https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images
- GSV-Cities: https://www.kaggle.com/datasets/amaralibey/gsv-cities
- ASTER GDEM (topography): https://asterweb.jpl.nasa.gov/gdem.asp

## Requirement

    pip install -r requirements.txt

## Training and Evaluation


# Stage 1: Train Ground CNN
python main.py train

# Build GeoIndex
python main.py geoindex \
  --images-root /path/to/ground_images \
  --meta /path/to/metadata.csv \
  --ckpt /path/to/checkpoints/<RUN_DIR>/weights_epoch30.pth \
  --out-root /path/to/geoindex

# Build DEM Cell Cache
python main.py dem-cache \
  --geo-root /path/to/geoindex \
  --dem-root /path/to/GDEM_filtered \
  --out-root /path/to/geoindex/dem_cache \
  --overwrite

# Stage 2: Train DEM Reasoner (Phase 2)
python main.py phase2 \
  --ckpt /path/to/checkpoints/<RUN_DIR>/weights_epoch30.pth \
  --topk 50

# Eval2: retrieval-only vs reasoned
python main.py eval2 \
  --ckpt-dir /path/to/checkpoints/<RUN_DIR> \
  --geo-root /path/to/geoindex \
  --dem-cache /path/to/geoindex/dem_cache \
  --topk 50 \
  --sample-limit 1000


## Reference
    - https://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Ground-to-Aerial_Image_Geo-Localization_With_a_Hard_Exemplar_Reweighting_Triplet_Loss_ICCV_2019_paper.pdf
    - https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.pdf
    - https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980193.pdf
    - https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf
    - https://arxiv.org/abs/2303.14672
    - https://arxiv.org/abs/2210.10239
    - https://mvrl.cse.wustl.edu/datasets/cvusa/
    - https://www.iqt.org/library/where-in-the-world-part-1-a-new-dataset-for-cross-view-image-geolocalization
    - https://www.iqt.org/library/deep-geolocation-with-satellite-imagery-of-ukraine
    - https://cphoto.fit.vutbr.cz/elevation/
    - http://places2.csail.mit.edu/
    - https://www.researchgate.net/publication/259802674_World-Wide_Scale_Geotagged_Image_Dataset_for_Automatic_Image_Annotation_and_Reverse_Geotagging
    - https://arxiv.org/html/2502.13759v2


