# HOGBEN-optimising-PNR

Jupyter notebook workflow for (i) fitting polarised neutron reflectometry (PNR) data using **refnx**
and (ii) Fisher-information-based experimental design/optimisation using **HOGBEN**.

## Contents
- `optimisation_Fe_MRL_SiO2_cap.ipynb` -- main analysis notebook (refnx fitting + HOGBEN optimisation)
- `functions/` -- helper modules used by the notebook
- `figures/` -- generated figures used consistent with the article”
- `environment.yml` -- Conda environment specification

## Data
The **complete** PNR dataset associated with this is available from the ISIS Neutron and Muon Source data repository:
**doi:10.5286/ISIS.E.RB2220651-1**

This repository includes a **minimal subset** in `POLREF_data/` sufficient to reproduce the figures/tables in the article.

## Setup

### Prerequisites
- Download and unzip the ZIP of this GitHub repository.
- Install Anaconda (Anaconda Navigator or Anaconda Prompt).

### Create and activate the environment
From a terminal **inside the repository folder** (extracted `HOGBEN-optimising-PNR-main` folder):

```bash
cd HOGBEN-optimising-PNR*
conda env create -f environment.yml
conda activate hogben-optimising-pnr
python -m pip install hogben==3.1.1
python -c "import hogben, refnx; print('hogben', hogben.__version__, 'refnx', refnx.__version__)"
```

## Running
With the `hogben-optimising-pnr` environment activated, start Jupyter **from the repository folder**:

```bash
jupyter lab
```

### (Optional) VS Code
You can also run the notebook in Visual Studio Code for a more integrated workflow (editor, Git integration, and notebook UI). Open the repository folder in VS Code and select the `hogben-optimising-pnr` Python environment/kernel when prompted.

If the `code` command is available, you can launch VS Code from the repository folder with:

```bash
code .
```

### (Optional) Register the kernel (recommended for classic Notebook users)
Run this once before using classic Notebook:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name hogben-optimising-pnr --display-name "hogben-optimising-pnr"
```
Then start classic Notebook:
```bash
jupyter notebook
```
Then open `optimisation_Fe_MRL_SiO2_cap.ipynb` and run all cells.

## Troubleshooting
- If Jupyter opens but imports fail, select the right kernel: Kernel → Change Kernel → hogben-optimising-pnr
- If `conda` is not found, Anaconda isn't installed or isn't on PATH.
- If `jupyter` is not found, run `conda activate hogben-optimising-pnr` again.

## Compatibility
Tested on Windows 11, macOS, and Ubuntu 24.04 LTS using Python 3.11 via Conda (`environment.yml`). The workflow was verified with JupyterLab; classic Notebook was also tested after registering the `hogben-optimising-pnr` kernel.

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