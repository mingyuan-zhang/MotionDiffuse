# Text-driven Motion Generation

<!-- TOC -->

- [Installation](#installation)
- [Training](#prepare-environment)
- [Acknowledgement](#acknowledgement)

<!-- TOC -->

## Installation

Please refer to [install.md](install.md) for detailed installation.

## Training

Due to the requirement of a large batchsize, we highly recommend you to use DDP training. A slurm-based script is as below:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} -n8 --gres=gpu:8 -u \
    python -u tools/train.py \
    --name t2m_sample \
    --batch_size 128 \
    --times 200 \
    --num_epochs 50 \
    --dataset_name t2m \
    --distributed
```

Otherwise, you can run the training code on a single GPU like:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py \
    --name t2m_sample \
    --batch_size 128 \
    --times 200 \
    --num_epochs 50 \
    --dataset_name t2m
```

## Evaluation

```shell
# GPU_ID indicates which gpu you want to use
python -u tools/evaluation.py checkpoints/kit/kit_motiondiffuse/opt.txt GPU_ID
# Or you can omit this option and use cpu for evaluation
python -u tools/evaluation.py checkpoints/kit/kit_motiondiffuse/opt.txt
```

## Acknowledgement

This code is developed on top of [Generating Diverse and Natural 3D Human Motions from Text](https://github.com/EricGuo5513/text-to-motion)