# PAC Bayes Generalization Certificates for Inductive Conformal Prediction

This repository contains source code to reproduce the results from the paper:

- Apoorva Sharma, Sushant Veer, Asher Hancock, Heng Yang, Marco Pavone, Anirudha Majumdar, "PAC-Bayes Generalization Certificates for Inductive Conformal Prediction." In *Conf. on  Neural Information Processing Systems*, volume 36, 2023 (in press)

The source code is split into two python modules:
- `confpred`, which contains generic utilities to turn pytorch modules into set-predictors via ICP, including using the PAC-Bayes algorithm in this paper.
- `confpred_eval` which contains code specific to the evaluation in this paper, e.g. defining datasets, etc.

### Installation
To install the package and dependencies, run the following from the root of the repository, ideally after setting up a virtual environment:
```bash
pip install -e .
```

### Running experiments
After installing the package, enter the `experiments` directory to run the experiments.
Applying ICP involves three steps, which are implemented with separate scripts.
1. Training a base predictor for a task:
```bash
python 0_train_model.py experiment=<EXPT>
```
2. Tuning and calibrating a model on calibration data:
```bash
python 1_calibrate_model.py experiment=<EXPT> calibrate=<METHOD> calibrate.alpha=<ALPHA> calibrate.delta=<DELTA> calibrate.alpha_hat=<ALPHA_HAT>
```
3. Evaluating the calibrated set-valued predictor on test data:
```bash
python 2_eval_model.py experiment=<EXPT> calibrate=<METHOD> calibrate.alpha=<ALPHA> calibrate.delta=<DELTA> calibrate.alpha_hat=<ALPHA_HAT>
```
The eval script looks for a predictor calibrated using the same command line arguments -- ensure that these match those used when running `1_calibrate_model.py`

`<EXPT>` specifies the experiment to run:
- `toy` runs the 1d regression task
- `mnist` runs the corrupted mnist classification task 

`<METHOD>` can take three values:
- `confpred` for standard, non-optimized ICP; 
- `learned`, which follows [Stutz et al, 2021](https://arxiv.org/abs/2110.09192) to optimize predictors for efficiency on a portion of the data
- `pacbayes` which implements our method

The `.vscode/launch.json` file has commands to automatically run the parameter sweeps used to generate results in the paper. The command line arguments specify overrides to the `hydra` configuration specified in `experiments/conf/`.

### Visualizing results
The Jupyter notebooks `experiments/mnist_results.ipynb` and `experiments/regression_results.ipynb` contain analysis and plotting code that was used to generate all the figures in the experiments. To reproduce results, ensure that all the commands in the vscode launch file for the corresponding to the experiment have been run.

### Visualizing theory
The `experiments/theory_figs.ipynb` contains code to visualize the KL bound derived from our theory.
