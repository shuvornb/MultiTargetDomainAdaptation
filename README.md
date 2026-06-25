# Multi-Target Domain Adaptation for Deepfake Detection

Official implementation of our IJCNN 2026 paper on **single-source multi-target domain adaptation for deepfake detection**.

This repository provides code for:

* dataset split generation
* training Xception-based single-source multi-target domain adaptation models
* evaluating source, target, and combined-target performance
* generating t-SNE feature visualization plots
* running main experiments, ablation studies, labeled-target studies, and unlabeled-target studies

---

## Repository Structure

```text
multi-target-da-deepfake/

README.md
requirements.txt

src/
    gen_ssmt_dataset.py
    train_ssmt_xception.py
    test_ssmt_xception.py
    tsne_ssmt_xception.py

scripts/
    common.sh
    run_main_results.sh
    run_ablation.sh
    run_label_size.sh
    run_unlabeled_size.sh

results/
    figures/
```

---

## Environment Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The code was tested with:

```text
Python 3.11.5
PyTorch 2.5.1
CUDA 12.3
Ubuntu 22.04.3 LTS
```

---

## Raw Data Format

The dataset generator expects the raw images to be organized as:

```text
RAW_DIR/
    DF/
    FS/
    F2F/
    NT/
    Pristine/
```

where:

* `DF`, `FS`, `F2F`, `NT` contain fake images from different manipulation domains
* `Pristine` contains real images

---

## Dataset Generation

Example: generate a dataset for `DF → FS, F2F`:

```bash
python src/gen_ssmt_dataset.py \
    --raw_dir /path/to/raw/images \
    --out_dir datasets/dataset_ssmt_DF__FS_F2F_seed123 \
    --source DF \
    --targets FS F2F \
    --seed 123
```

Default target-domain split per target:

```text
labeled   : 500 fake + 500 real
unlabeled : 4500 fake + 4500 real
test      : 2500 fake + 2500 real
```

The default source-domain split is:

```text
train : 20000 fake + 20000 real
test  : 5000 fake + 5000 real
```

---

## Training

```bash
python src/train_ssmt_xception.py \
    --dataset_dir datasets/dataset_ssmt_DF__FS_F2F_seed123 \
    --out_dir runs/ssmt_DF__FS_F2F_seed123 \
    --source DF \
    --targets FS F2F \
    --seed 123 \
    --epochs 10 \
    --steps_per_epoch 500 \
    --img_size 128
```

The training objective contains:

* supervised binary cross-entropy loss
* moment matching loss
* unsupervised class-alignment loss

Loss weights can be controlled using:

```bash
--lambda_mm 1.0
--lambda_ca 0.1
```

---

## Evaluation

```bash
python src/test_ssmt_xception.py \
    --dataset_dir datasets/dataset_ssmt_DF__FS_F2F_seed123 \
    --ckpt runs/ssmt_DF__FS_F2F_seed123/checkpoint_last.pt \
    --source DF \
    --targets FS F2F \
    --img_size 128
```

The script reports performance on:

* source test set
* each target test set
* combined target test set

The results are saved as:

```text
test_results.json
```

inside the corresponding run directory.

---

## t-SNE Visualization

```bash
python src/tsne_ssmt_xception.py \
    --dataset_dir datasets/dataset_ssmt_DF__FS_F2F_seed123 \
    --ckpt runs/ssmt_DF__FS_F2F_seed123/checkpoint_last.pt \
    --source DF \
    --targets FS F2F \
    --img_size 128 \
    --out_png results/figures/tsne_DF_to_FS_F2F.png
```

The plot visualizes feature alignment between:

* source fake
* source real
* target fake
* target real

---

## Reproducing Experiments

Before running the scripts, update `RAW_DIR` in `scripts/common.sh` or export it from the terminal:

```bash
export RAW_DIR=/path/to/raw/images
```

Then run:

### Main Results

```bash
bash scripts/run_main_results.sh
```

### Ablation Study

```bash
bash scripts/run_ablation.sh
```

### Target Labeled-Size Study

```bash
bash scripts/run_label_size.sh
```

This runs target labeled sizes:

```text
0, 800, 1600, 2400
```

### Target Unlabeled-Size Study

```bash
bash scripts/run_unlabeled_size.sh
```

This runs target unlabeled sizes:

```text
4000, 8000, 12000
```

---

## Notes

* The scripts use iteration-based training. One epoch consists of a fixed number of training steps, controlled by `--steps_per_epoch`.
* Target domains are sampled independently at every training step.
* The unlabeled target folder contains mixed fake and real images without label subfolders.
* The test folders contain `fake/` and `real/` subfolders for evaluation.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{seraj2026multitargetda,
  title     = {Multi-target Deep Domain Adaptation for Deepfake Detection},
  author    = {SM Seraj, S Chakraborty},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

---

## License

Please see the `LICENSE` file for details.
