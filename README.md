# PMCGNN

## Dataset

### JARVIS Dataset
For JARVIS Dataset, we follow PotNet and use the same training, validation, and test set. We evaluate our PMCGNN on five important crystal property tasks, including formation energy, bandgap(OPT), Total energy, Bandgap(MBJ), and Ehull. The training, validation, and test set contains 44578, 5572, and 5572 crystals for tasks of Formation Energy, Total Energy, and Bandgap(OPT). The numbers are 44296, 5537, 5537 for Ehull, and 14537, 1817, 1817 for Bandgap(MBJ). The used metric is test MAE. The baseline results are taken form PotNet.

### The Materials Project Dataset
For The Materials Project Dataset, we follow PotNet and use the same training, validation, and test set. We evaluate our PMCGNN on four important crystal property tasks, including Formation Energy, Band Gap, Bulk Moduli and Shear Moduli. The training, validation, and test set contains 60000, 5000, and 4239 crystals for tasks of formation energy and band gap. The numbers are 4664, 393, 393 for Bulk Moduli and Shear Moduli.The used metric is test MAE. The baseline results are taken form PotNet.


## Availability and implementation
The code and data will then be available for download after this study is published.

## Acknowledgement

This repo is built upon the previous work PotNet's [[codebase]](https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet). Thank you very much for the excellent codebase. 
