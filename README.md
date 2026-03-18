# HOGBEN-optimising-PNR

Jupyter notebook workflow for (i) fitting polarised neutron reflectometry (PNR) data using **refnx**
and (ii) Fisher-information-based experimental design/optimisation using **HOGBEN**.

## Contents
- `optimisation_Fe_MRL_SiO2_cap.ipynb` -- main analysis notebook (refnx fitting + HOGBEN optimisation)
- `functions/` -- helper modules used by the notebook
- `figures/` -- generated figures akin to the ones in the article
- `environment.yml` -- Conda environment specification

## Data
The **complete** PNR dataset associated with this is available from the ISIS Neutron and Muon Source data repository:
**doi:10.5286/ISIS.E.RB2220651-1**

This repository includes a **minimal subset** in `POLREF_data/` sufficient to reproduce the figures/tables in the article.

## Setup

### Prerequisites
- Download and unzip the ZIP of this GitHub repository.
- Install Conda (Anaconda/Miniconda).

### Create and activate the environment
From a terminal **inside the repository folder** (extracted `HOGBEN-optimising-PNR-main` folder):

```bash
cd HOGBEN-optimising-PNR*
conda env create -f environment.yml
conda activate refnx
pip install hogben==3.1.1
python -c "import hogben, refnx; print('hogben', hogben.__version__, 'refnx', refnx.__version__)"
```

## Running
With the `refnx` environment activated, start Jupyter **from the repository folder**:

```bash
jupyter lab
```
(or the classic interface)
```bash
jupyter notebook
```

Then open `optimisation_Fe_MRL_SiO2_cap.ipynb` and run all cells.

## Troubleshooting
- If Jupyter opens but imports fail, select the right kernel: Kernel → Change Kernel → refnx
- If `conda` is not found, Conda isn't installed or isn't on PATH.
- If `jupyter` is not found, run `conda activate refnx` again.

## Citation
- Software archive (Zenodo): DOI will be added after the v1.0 release.
- Dataset (ISIS): **doi:10.5286/ISIS.E.RB2220651-1**

## Contact
Ivan Yakymenko - ivan.yakymenko@liu.se \
Jos Cooper - jos.cooper@ess.eu

## Acknowledgements
We thank Alessandra Luchini for her assistance in fitting the substrate assembly and lipid bilayer data sets.

## License
Distributed under the BSD 3-Clause License. See [license](/LICENSE) for more information.