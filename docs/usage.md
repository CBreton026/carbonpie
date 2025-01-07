# Instructions

## 1. Create and activate the environment
First, create the Python environment:
```
python3 -m venv .venv
source .venv/bin/activate
```
Then, install the requirements:
```
pip install -r requirements.txt
```

## 2. Prepare the necessary data
The 'Data' directory should already have the "UNSD-Methodology-Updated.csv" file, adapted from [UNStats](https://unstats.un.org/unsd/methodology/m49/overview/).

In the 'data' directory, create a 'large_files' directory. The dataset from [Gütschow et al.](https://doi.org/10.5281/zenodo.3638137) (in our case, PMSSPBIE_05Feb20.csv) must be saved there.

## 3. Create the config file
In the main directory, create the following text file:
results/config/config.toml

An example config file can be [found here](./config.toml). Please make sure to change all paths under **[paths]** and **[database]** to your own. Specifically, the 'fig_dir' path is where you want the script to save the resulting figures.

## 4. Create the database
Run [database.py](../src/carbonpie/database.py) to initialize the database. This initial setup consumes ~5Go RAM and takes about 15 minutes on a Microsoft Surface Book 2 (Intel® Core™, i7–8650U CPU @ 1.90GHz, 16.0 Go RAM). This could likely be improved with better optimization and programming knowledge.

## 5. Run the main script
Run [budget_carbone.py](../src/budget_carbone.py) to output the results and figures. The main resulting figure is "newplot_small_alt.png"; other exploratory figures are also created, although these were not used for the paper.