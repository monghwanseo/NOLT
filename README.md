Replication code for "Level-Transition Asymmetry in the Cross-Section of Option Linearity Residuals."

## Structure

```
code/
├── src/                  library (NOLT, baselines, training, metrics, synthetic)
├── 01_data/              Bloomberg panel build, quality, freeze, PC1, BSM theorem
└── 02_experiments/       paper experiments (Sec. 3-5, Appendix)

data_raw/
├── Bloomberg/            SPX index option Excel (spx_1.xlsx, spx_2.xlsx)
└── synthetic/            Heston / Bates / BSM panels used in the paper
```

## Run

```
bash run_all.sh
```

## Data and license note

The MIT License applies only to the source code. Raw Bloomberg option data remain subject to Bloomberg L.P.'s data licensing policies and are included for replication only.

## Contact

monghwanseo@yonsei.ac.kr
