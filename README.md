# carbonpie
This repository contains the code related to the following paper:
> Breton, Charles, Pierre Blanchet, Ben Amor, and Francesco Pomponi. « A novel method to calculate SSP-consistent remaining carbon budgets for the building sector: A case study of Canada ». Building and Environment 269: 112474. https://doi.org/10.1016/j.buildenv.2024.112474.

## Description
The [script](./src/budget_carbone.py) in this repository can calculate country-specific carbon budgets using open-source datasets, for different scenarios and allocation methods.

Notably, the script relies on methods and data from the following papers: 
> Gütschow, Johannes, M. Louise Jeffery, Annika Günther, and Malte Meinshausen. « Country-Resolved Combined Emission and Socio-Economic Pathways Based on the Representative Concentration Pathway (RCP) and Shared Socio-Economic Pathway (SSP) Scenarios ». Earth System Science Data 13, nᵒ 3: 1005‑40. https://doi.org/10.5194/essd-13-1005-2021.

> Van Den Berg, Nicole J., Heleen L. Van Soest, Andries F. Hof, Michel G. J. Den Elzen, Detlef P. Van Vuuren, Wenying Chen, Laurent Drouet, et al. « Implications of Various Effort-Sharing Approaches for National Carbon Budgets and Emission Pathways ». Climatic Change 162, nᵒ 4: 1805‑22. https://doi.org/10.1007/s10584-019-02368-y.

The PMSSPBIE files (Version 1.0) are available [here on Zenodo](https://doi.org/10.5281/zenodo.3638137), and the allocation methods and parameters are described in the [Electronic supplementary material (ESM1)](https://doi.org/10.1007/s10584-019-02368-y) from Van Den Berg et al (2020).

## Status
**Work in Progress**

The code for the first part of the paper (Section 2.1, 2.2 and Figure 5) is already available, as discussed in the Methods section. The remaining data, scripts, and notebooks are in the process of being cleaned, reorganized, and uploaded. Please check back soon for the complete implementation.

## Usage
[Speficic instructions](./docs/usage.md) on how to run the code are available. These will be revised once the full upload is complete.

## License
The code is provided under the [MIT License](./LICENSE).

## Requirements
This code was run on Python 3.8.10. Although this was not tested, we expect it to work with Python 3.6+. The main [requirements](./requirements.txt) are : toml, pandas, matplotlib, seaborn, plotly and kaleido. 

## Contact
For any questions or issues, please open an issue on the repository.