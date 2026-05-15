# Case Study on Continual Learning

This repository provides a modular pipeline for Continual Learning experiments.
You can run everything locally but some experiments could take longer than others.

## Demo

A demo notebook demonstrating our code and key results is available in `notebooks/demo.ipynb`.

## What is Continual Learning?

In Continual Learning, a model learns tasks sequentially instead of seeing all data at once.
The key challenge is **catastrophic forgetting**: after learning a new task, performance on earlier tasks can drop.

This project helps you compare mitigation strategies such as:

- **Naive** (baseline, no explicit forgetting mitigation)
- **EWC** (regularization-based)
- **GEM** (gradient constraints with memory)

## Quick Start (Local)

### 1. Create environment

```bash
conda env create -f environment.yaml
conda activate catforget
```

### 2. Run one experiment

```bash
python src/train.py experiment=mnist_naive
```

Hydra writes outputs to:

```text
outputs/train/<timestamp>/
```

### 3. Try another strategy or dataset

```bash
python src/train.py experiment=mnist_ewc
python src/train.py experiment=cifar10_naive
python src/train.py experiment=mnist_permutations_ewc
```

## How to work with configs

The pipeline is controlled by Hydra configs in `configs/`.

### Main idea

- `configs/config.yaml` selects one experiment config by default
- each file in `configs/experiment/` composes dataset, model, strategy, optimizer, and trainer
- component details live in dedicated folders (`configs/dataset/`, `configs/model/`, `configs/strategy/`, ...)

Example (`configs/experiment/mnist_naive.yaml`):

```yaml
defaults:
	- /dataset: mnist
	- /model: mlp_mnist
	- /strategy: naive
	- /optimizer: adam
	- /trainer: sequential

experiment_name: mnist_naive
seed: 42
task_order: ["A", "B", "C"]
```

### Typical overrides from the command line

You usually do not need to touch Python code unless you want to design a whole new experiment. We did not provide exhaustive experiment or model configs because of the number of combinations. Either write your own new configs or use Hydra overrides:

```bash
python src/train.py experiment=mnist_ewc experiment.seed=7
python src/train.py experiment=mnist_naive experiment.trainer.max_epochs_per_task=5
python src/train.py experiment=cifar10_naive experiment.dataset.batch_size=256
```

### If you want a new experiment preset

1. Create a new file in `configs/experiment/`.
2. Reuse existing components via `defaults`.
3. Set `seed` and `task_order`.
4. Run with `python src/train.py experiment=<your_file_name_without_yaml>`.

For normal usage, this is enough. No additional pipeline steps are required.

## Reproducibility

The pipeline is designed to be reproducible:

- runs are fully defined by config files and CLI overrides
- each experiment has a configurable random seed (`experiment.seed`)
- outputs are stored in timestamped folders so runs are easy to track

Practical note: on different hardware (especially GPU), very small numerical differences can still appear.

## Strategy Pattern (Implementation)

The trainer uses strategy hooks with a shared interface:

- `before_task()`
- `after_task()`
- `after_backward()`
- `compute_loss()`

This keeps the training loop stable while allowing different Continual Learning methods.

## Metrics: Understanding Forgetting

The project tracks **Matrix R** (Lopez-Paz et al., 2017):

$$R[i,j] = \text{accuracy on task } j \text{ after training on task } i$$

Interpretation:

- **Diagonal** ($R[i][i]$): learning quality on each task
- **Off-diagonal** ($R[i][j]$ with $j < i$): retention of earlier tasks
- **Backward Transfer (BWT)**: summary of ability to retain old tasks (negative values indicate forgetting)

## Project Structure

```text
continual_learning/
|- configs/
|  |- config.yaml
|  |- experiment/      # experiment presets (dataset/model/strategy/... composition)
|  |- dataset/         # data module configs and task splits
|  |- model/           # model architecture configs
|  |- strategy/        # continual learning strategy configs (naive, ewc, gem)
|  |- trainer/         # trainer behavior (epochs, device, logging)
|  |- optimizer/       # optimizer configs
|- src/
|  |- train.py         # main entrypoint
|  |- data/            # data modules
|  |- models/          # model implementations
|  |- strategies/      # strategy implementations
|  |- training/        # trainer implementation
|  |- metrics/         # continual learning metrics
|- outputs/train/      # Hydra run outputs
|- notebooks/          # optional analysis notebooks
|- environment.yaml    # reproducible environment
```

## Minimal workflow summary

1. Activate the environment.
2. Choose an experiment config.
3. Run `python src/train.py experiment=<name>`.
4. Check results in `outputs/train/<timestamp>/`.

This is the complete local pipeline for users.


# References

[1] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009

[2] MNIST handwritten digit database, LeCun, Y., Cortes, C., \& Burges, C. J., 2010

[3] Overcoming catastrophic forgetting in neural networks, Kirkpatrick, James et al., 2017

[4] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, Han Xiao and Kashif Rasul and Roland Vollgraf, 2017

[5] Gradient Episodic Memory for Continual Learning, Lopez-Paz, D., Ranzato, M., 2017

[6] A Comprehensive Survey of Continual Learning: Theory, Method and Application, Wang et al., 2023

[7] Continual Learning – A Deep Dive Into Elastic Weight Consolidation Loss, Alexey Kravets, 2024 

[8] Deep Residual Learning for Image Recognition, He et al., 2015