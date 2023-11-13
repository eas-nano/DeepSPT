# DeepSPT
## Deep learning assisted Single Particle Tracking for automated correlation between diffusion and function
DeepSPT, a deep learning framework to interpret the diffusional 2D or 3D temporal behavior of objects in a rapid and efficient manner, agnostically. DeepSPT is a deep learning framework, encompassing three sequentially connected modules: A temporal behavior segmentation module; a diffusional fingerprinting module; and a task-specific downstream classifier module (Fig. 1a). The first two modules are universal, applicable directly to any trajectory dataset characterized by x, y, (z) and t coordinates across diverse biological systems. The final module capitalizes on experimental data to learn a task that is specific to the system under investigation.

![image](_Images/Figure_1.png)
### Citing
TBA
### Dependencies
  - python==3.8
  - pandas==1.5.3
  - matplotlib==3.7.1
  - scipy==1.10.1
  - hdbscan==0.8.29
  - opencv==4.6.0
  - scikit-learn==1.2.2

### Usage
#### Installation
DeepSPT's installation guide utilize conda environment setup, therefore either miniconda or anaconda is required to follow the bellow installation guide.
 - Anaconda install guide: [here](https://www.anaconda.com/download)
 - Mini conda install guide: [here](https://docs.conda.io/en/latest/miniconda.html)

DeepSPT is most easily setup in a new conda environment with dependecies and channels found in dependency.yml - Open Terminal / Commando promt at whished location of SEMORE and run the bash commands below, which creates the environemnt, downloades and installs packages, in less than 5 minutes. Run time is expected below 1 second per track.

```bash
git clone https://github.com/hatzakislab/SEMORE
cd SEMORE
conda env create -f dependency.yml
conda activate SEMORE
```
DeepSPT modules and additional/helpful functions are contained in the `Scripts` folder.
DeepSPT modules are imported as:
```python
from DeepSPT import *

```
Three test python scripts are provided:
  - `Data_sim_test.py` - test data generation.
  - `Segmentation_test.py` - test the clustering module on simulated data.
  - `Fingerprint_test.py` - test the fingerprint modules on the resulting data from Segmentation_test.py.
### Own data
`SEMORE_clustering.find_clust` accepts 2-D localizations containing a temporal element [x,y,t] while `SEMORE_fingerprint.Morphology_fingerprint` accepts localizations [x,y] both static and temporal resolved. The output of the fingerprintg can then freely be used for further analysis.

### For demostration
For demostration regarding presented data contained in the manuscript, please refer to the `_For_puplicaiton` folder where you will find the required information and scripts.

### Contact

Jacob Kæstel-hansen, PhD fellow\
Department of Chemistry\
jkh@chem.ku.dk

or commit an issue to this github. 
