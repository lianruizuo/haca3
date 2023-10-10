# HACA3: A unified approach for multi-site MR image harmonization | [Paper](https://www.sciencedirect.com/science/article/pii/S0895611123001039)

This page provides a gentle introduction to HACA3 inference and training. 
HACA3 is an advanced approach for multi-site MRI harmonization. 

[Zuo et al. HACA3: A unified approach for multi-site MR image harmonization. CMIG, 2023](https://www.sciencedirect.com/science/article/pii/S0895611123001039)

## 1. Introduction and motivation


## 2. Prerequisites 
Standard neuroimage preprocessing steps are needed before running HACA3. These preprocessing steps include:
- inhomogeneity correction
- registration to MNI space (1mm isotropic)
- (optional) super-resolution for 2D acquired scans. This step is optional, but recommended for optimal performance. 
See [SMORE](https://github.com/volcanofly/SMORE-Super-resolution-for-3D-medical-images-MRI).

## 3. Installation

#### 3.1 Option 1: Install from source using `pip`
1. Clone the repository:
    ```bash
    git clone https://gitlab.com/lr_zuo/haca3.git 
    ```
2. Navigate to the directory:
    ```bash
    cd haca3
    ```
3. Install dependencies:
    ```bash
    pip install . 
    ```
Package requirements are automatically handled. To see a list of requirements, see `setup.py` L50-59. 
This installs the `haca3` package and creates two CLI aliases `haca3-train` and `haca3-test`.

#### 3.2 Option 2 (recommended): Run HACA3 through singularity image
1. Download singularity image of HACA3 from [JHU-IACL](https://iacl.ece.jhu.edu/~lianrui/haca3/haca3_main.sif).
2. Pretrained HACA3 model weights can be [downloaded](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
3. Pretrained fusion model weights can be [downloaded](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).

## 4. Usage: Inference
If you use our software, please cite 
   ```bibtex
   @article{ZUO2023102285,
   title = {HACA3: A unified approach for multi-site MR image harmonization},
   journal = {Computerized Medical Imaging and Graphics},
   volume = {109},
   pages = {102285},
   year = {2023},
   issn = {0895-6111},
   doi = {https://doi.org/10.1016/j.compmedimag.2023.102285},
   author = {Lianrui Zuo and Yihao Liu and Yuan Xue and Blake E. Dewey and Samuel W. Remedios and 
   Savannah P. Hays and Murat Bilgel and Ellen M. Mowry and Scott D. Newsome and Peter A. Calabresi and 
   Susan M. Resnick and Jerry L. Prince and Aaron Carass}
   }
   ```
#### Run HACA3 through singularity image (recommended), 
```bash
   singularity exec --nv -e haca3.sif haca3-test \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-1] \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-2, IF THERE ARE MULTIPLE SOURCE IMAGES] \
   --target-image [TARGET-IMAGE] \
   --harmonization-model [PRETRAINED-HACA3-MODEL] \
   --fusion-model [PRETRAINED-FUSION-MODEL] \
   --out-path [PATH-TO-HARMONIZED-IMAGE1] \
   --intermediate-out-dir [DIRECTORY SAVES INTERMEDIATE RESULTS] 
   ```

- ***Example:***
    Suppose the task is to harmonize MR images from `Site A` to match the contrast of a pre-selected T1w image of 
    `Site B`. As a source site, `Site A` has T1w, T2w, and FLAIR images. The files are saved like this:
    ```
    ├──data_directory
        ├──site_A_t1w.nii.gz
        ├──site_A_t2w.nii.gz
        ├──site_A_flair.nii.gz
        └──site_B_t1w.nii.gz
    ```
    In this example, the singularity command to run HACA3 is:
    ```bash
       singularity exec --nv -e haca3.sif haca3-test \
       --in-path data_directory/site_A_t1w.nii.gz \
       --in-path data_directory/site_A_t2w.nii.gz \
       --in-path data_directory/site_A_flair.nii.gz \
       --target-image data_directory/site_B_flair.nii.gz \
       --harmonization-model [PRETRAINED-HACA3-MODEL] \
       --fusion-model [PRETRAINED-FUSION-MODEL] \
       --out-path output_directory/site_A_harmonized_to_site_B_t1w.nii.gz \
       --intermediate-out-dir output_directory
    ```
    The harmonized image and intermediate results will be saved at `output_directory`.

#### All inference phase options:
- ```--in-path```: file path to input source image. Multiple ```--in-path``` may be provided if there are multiple 
source images. See the above example for more details.
- ```--target-image```: file path to target image. HACA3 will match the source images contrast to this target image.
- ```--target-theta```: In [HACA3](https://www.sciencedirect.com/science/article/pii/S0895611123001039), ```theta``` 
is a two-dimensional representation of image contrast. Target image contrast can be directly specified by providing 
a ```theta``` value, e.g., ```--target-theta 0.5 0.5```.
- ```--norm-val```: normalization value. 
- ```--out-path```: file path to harmonized image. 
- ```--harmonization-model```: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP data can 
be [downloaded](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
- ```--fusion-model```: pretrained fusion model weights. HACA3 uses a 3D convolutional network to combine multi-orientation
2D slices into a single 3D volume. Pretrained fusion model can be [downloaded](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
- ```--save-intermediate```: if specified, intermediate results will be saved. Default: ```False```. Action: ```store_true```.
- ```--intermediate-out-dir```: directory to save intermediate results.
- ```--gpu-id```: integer number specifies which GPU to run HACA3.
- ```--num-batches```: During inference, HACA3 takes entire 3D MRI volumes as input. This may cause a considerable amount 
GPU memory. For reduced GPU memory consumption, source images maybe divided into smaller batches. 
However, this may slightly increase the inference time.


## Acknowledgements
The authors thank BLSA participants, as well as colleagues of the Laboratory of Behavioral Neuroscience and 
the Image Analysis and Communications Laboratory. This work was supported in part by the Intramural Research Program 
of the National Institutes of Health, National Institute on Aging and in part by the TREAT-MS study funded by 
the Patient-Centered Outcomes Research Institute (PCORI) grant MS-1610-37115 (Co-PIs: Drs. S.D. Newsome and E.M. Mowry). 
This material is also partially supported by the National Science Foundation Graduate Research Fellowship under 
Grant No. DGE-1746891. The work was also funded in part by the NIH grant (R01NS082347, PI: P. Calabresi), 
National Multiple Sclerosis Society grant (RG-1907-34570, PI: D. Pham), and 
the DOD/Congressionally Directed Medical Research Programs (CDMRP) grant (MS190131, PI: J. Prince).