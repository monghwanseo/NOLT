Replication code for "Level-Transition Asymmetry in the Cross-Section of Option Linearity Residuals."

## Structure

```
code/
├── src/                  library (NOLT, baselines, training, metrics, synthetic)
├── 01_data/              Bloomberg panel build, quality, freeze, PC1, BSM theorem
└── 02_experiments/       paper experiments

data_raw/
├── Bloomberg/            SPX index option Excel (spx_1.xlsx, spx_2.xlsx)
└── synthetic/            Heston / Bates / BSM panels 
```

## Data

`data_raw/` is hosted as a release asset. Download and unzip into the repo root:

```
curl -L -o data_raw.zip https://github.com/monghwanseo/NOLT/releases/download/v1.0/data_raw.zip
unzip data_raw.zip
```

## Run

```
bash run_all.sh
```

## Data and license note

The MIT License applies only to the source code. Raw Bloomberg option data remain subject to Bloomberg L.P.'s data licensing policies and are included for replication only.

## Contact

monghwanseo@yonsei.ac.kr
