# HACA3: A unified approach for multi-site MR image harmonization | [Paper](https://www.sciencedirect.com/science/article/pii/S0895611123001039)

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgements](#acknoledgements)

---

## Introduction
This page provides usage guidance of HACA3 training and inference. 

---

## Prerequisites 
Standard neuroimage preprocessing steps are needed before running HACA3. These preprocessing steps include:
inhomogeneity correction, registration to MNI template, and super-resolution for 2D acquired scans (optional, but recommended). 

--- 

## Installation

### Option 1: Install from source using `pip`
1. Clone the repository:
    ```bash
    git clone https://gitlab.com/iacl/haca3.git 
    ```
2. Navigate to the directory:
    ```bash
    cd haca3
    ```
3. Install dependencies:
    ```bash
    pip install . 
    ```
Package requirements are automatically handled. To see a list of requirements, see `setup.py` L50-58. 
This installs the `haca3` package and creates two CLI aliases `haca3-train` and `haca3-test`.

### Option 2: Run HACA3 through Singularity image (recommended)
1. Download Singularity image from [GoogleDrive].
TODO: singularity command will be changed in later versions. Specifying source contrast names will be no longer needed.

---

## Usage
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
#### Run HACA3 through Singularity image (recommended), 
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

#### Example:
Suppose the task is to harmonize MR images from ```Site A``` to match the contrast of a pre-selected T1w image of 
```Site B```. As a source site, ```Site A``` has T1w, T2w, and FLAIR images. The files are saved like this:
```
├──data_directory
        ├──site_A_t1w.nii.gz
        ├──site_A_t2w.nii.gz
        ├──site_A_flair.nii.gz
        └──site_B_t1w.nii.gz
```
The singularity command to run HACA3 is:
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
The harmonized image and intermediate results will be saved at ```output_directory```.


## Acknowledgements
The authors thank BLSA participants, as well as colleagues of the Laboratory of Behavioral Neuroscience and 
the Image Analysis and Communications Laboratory. This work was supported in part by the Intramural Research Program 
of the National Institutes of Health, National Institute on Aging and in part by the TREAT-MS study funded by 
the Patient-Centered Outcomes Research Institute (PCORI) grant MS-1610-37115 (Co-PIs: Drs. S.D. Newsome and E.M. Mowry). 
This material is also partially supported by the National Science Foundation Graduate Research Fellowship under 
Grant No. DGE-1746891. The work was also funded in part by the NIH grant (R01NS082347, PI: P. Calabresi), 
National Multiple Sclerosis Society grant (RG-1907-34570, PI: D. Pham), and 
the DOD/Congressionally Directed Medical Research Programs (CDMRP) grant (MS190131, PI: J. Prince).