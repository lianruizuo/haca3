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

### Option 2: Run HACA3 through singularity image (recommended)
   ```bash
   singularity exec --nv -e -B /iacl haca3.sif haca3-test \
   --t1 [SOURCE-T1W] \
   --t2 [SOURCE-T2W] \
   --pd [SOURCE-PDW] \
   --flair [SOURCE-FLAIR] \
   --target-image [TARGET-IMAGE] \
   --pretrained-harmonization [PRETRAINED-HACA3-MODEL] \
   --pretrained-fusion [PRETRAINED-FUSION-MODEL] \
   --out-dir [OUTPUT-DIRECTORY] \
   --file-name [OUTPUT-FILE-NAME] 
   ```
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
TODO: singularity command will be changed in later versions. Specifying source contrast names will be no longer needed.

## Acknowledgements
The authors thank BLSA participants, as well as colleagues of the Laboratory of Behavioral Neuroscience and 
the Image Analysis and Communications Laboratory. This work was supported in part by the Intramural Research Program 
of the National Institutes of Health, National Institute on Aging and in part by the TREAT-MS study funded by 
the Patient-Centered Outcomes Research Institute (PCORI) grant MS-1610-37115 (Co-PIs: Drs. S.D. Newsome and E.M. Mowry). 
This material is also partially supported by the National Science Foundation Graduate Research Fellowship under 
Grant No. DGE-1746891. The work was also funded in part by the NIH grant (R01NS082347, PI: P. Calabresi), 
National Multiple Sclerosis Society grant (RG-1907-34570, PI: D. Pham), and 
the DOD/Congressionally Directed Medical Research Programs (CDMRP) grant (MS190131, PI: J. Prince).
