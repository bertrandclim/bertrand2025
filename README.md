# bertrand et al. 2025
[![DOI](https://zenodo.org/badge/944806702.svg)](https://doi.org/10.5281/zenodo.15786066)

Figure plotting code and data for the paper "Increasing wintertime cloud opacity increases surface longwave radiation at a long-term Arctic observatory". 

The following `.ipynb` files generate figures from the paper based on the netCDF files stored in this repository:
1. `figure1_2025-06-29.ipynb`
2. `figure2_2025-06-29.ipynb`
3. `figure3_2025-06-29.ipynb`
4. `figure4_2025-06-29.ipynb`

See main text of paper for descriptions of data sources, methodology, and figure descriptions.

# setup
1. `git clone https://github.com/bertrandclim/bertrand2025.git`
2. `cd bertrand2025`
3. `pip install requirements.txt` (should take <15 minutes on a typical computer)
4. `jupyter-lab`

Then launch one of the notebooks and click `kernel -> run all cells` (should take <5 minutes on a typical computer).

Tested with `matplotlib==3.8.0`, `xarray==2023.9.0`, `pandas==2.1.1`, `numpy==1.24.4`, `statsmodels==0.14.0`, `metpy==1.5.1`, `scipy==1.11.3`.
