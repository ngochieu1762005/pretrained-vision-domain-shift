# Pretrained Vision Models Across Image Domains

This project studies how ImageNet-pretrained computer vision models behave when the visual style of the input images changes. We train each model only on real-world photographs from Office-Home, then evaluate the same trained model on photographs, product images, artistic images, and clipart.

The experiment is a custom source-only domain generalization study with held-out test sets. It is not the standard Office-Home domain adaptation benchmark.

## Live Demo

[Open the Streamlit demo](https://pretrained-vision-domain-shift.streamlit.app/).

The interactive demo runs the trained ResNet18 and ResNet50 configurations, compares linear probing with partial fine-tuning, and shows how predictions change across Office-Home image domains. Checkpoints are downloaded lazily from the supplementary archive and verified before inference; they are not stored in Git.

## Project Idea

We use `Real-World` as the source domain for training and model selection. The selected model is evaluated on four domains:

- Real-World
- Product
- Art
- Clipart

We compare two ImageNet-pretrained architectures:

- ResNet18
- ResNet50

Each architecture is trained in two ways:

- **Linear probing:** only the new classification head is trained.
- **Fine-tuning:** the final residual block (`layer4`) and classification head are trained.

This setup lets us compare same-domain performance with performance after a change in image domain.

## Dataset

We use the [Office-Home dataset](https://www.hemanthdv.org/officeHomeDataset.html), which contains 65 shared object classes across Art, Clipart, Product, and Real-World. The notebook downloads the data from the public [Flower Labs mirror](https://huggingface.co/datasets/flwrlabs/office-home), pinned to revision `2a083645e3177afbd91ba4fa2651238f8994d335`.

Validation found 15,588 images and no corrupted images. After removing 412 byte-identical duplicates, 15,176 images remained. Each domain and class was split approximately 70/15/15 using split seed 2026.

| Domain | Train | Validation | Test |
|---|---:|---:|---:|
| Art | 1,617 | 349 | 349 |
| Clipart | 2,960 | 646 | 646 |
| Product | 2,985 | 646 | 646 |
| Real-World | 3,030 | 651 | 651 |

The same class mapping and data partitions are used for all four training configurations. Only the Real-World training and validation subsets are used to train models and select checkpoints; target-domain training and validation subsets are not used.

See [`DATA.md`](DATA.md) for the official dataset URL, exact version, partition procedure, preprocessing, and notebook sections required to reproduce the experiment data.

## Data Preparation

The preparation pipeline:

1. validates the downloaded images;
2. removes byte-identical duplicates using SHA-256 hashes;
3. creates class-stratified train, validation, and test partitions within each domain;
4. resizes and crops images for the pretrained ResNet input size; and
5. normalizes images with ImageNet statistics.

Training images use a random resized crop to `224 × 224` with scale `0.7–1.0` and random horizontal flipping. Validation and test images are resized so the shorter edge is 256 pixels, then center-cropped to `224 × 224`.

## Models

The experiment uses torchvision's ResNet18 and ResNet50 with `IMAGENET1K_V1` pretrained weights. The original classification layer is replaced with a new linear layer for the 65 Office-Home classes.

BatchNorm running statistics remain fixed in both training strategies.

## Training

For **linear probing**, the pretrained backbone is frozen and only the classification head is updated.

For **fine-tuning**, `layer4` and the classification head are updated while the earlier backbone layers remain frozen.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Head learning rate | 0.001 |
| `layer4` learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Scheduler | Cosine annealing |
| Gradient clipping | Maximum norm 1.0 |
| Batch size | 32 |
| Maximum epochs | 15 for linear probing; 20 for fine-tuning |
| Early stopping | 5 epochs without validation macro-F1 improvement |
| Training seed | 42 |
| Precision | FP16 mixed precision |

The checkpoint with the highest Real-World validation macro-F1 is used for final evaluation.

## Evaluation

We evaluate each selected checkpoint using:

- **Accuracy:** the proportion of correctly classified test images.
- **Macro-F1:** the average of the 65 class-level F1 scores, giving each class equal weight.

The difference between Real-World accuracy and each target-domain accuracy is reported in percentage points to show the observed source-to-target performance change.

## Results

### Accuracy

| Model | Training strategy | Real-World | Product | Art | Clipart |
|---|---|---:|---:|---:|---:|
| ResNet18 | Linear probe | 79.42% | 67.34% | 55.59% | 36.22% |
| ResNet18 | Fine-tune last block | 74.81% | 60.68% | 49.00% | 33.75% |
| ResNet50 | Linear probe | 82.03% | 72.91% | 62.75% | 37.15% |
| ResNet50 | Fine-tune last block | 82.64% | 71.83% | 61.60% | 40.87% |

### Macro-F1

| Model | Training strategy | Real-World | Product | Art | Clipart |
|---|---|---:|---:|---:|---:|
| ResNet18 | Linear probe | 0.7671 | 0.6550 | 0.5182 | 0.3345 |
| ResNet18 | Fine-tune last block | 0.7230 | 0.5828 | 0.4423 | 0.3160 |
| ResNet50 | Linear probe | 0.7954 | 0.7150 | 0.5898 | 0.3837 |
| ResNet50 | Fine-tune last block | 0.8109 | 0.7042 | 0.5781 | 0.4372 |

These tables report one training seed on fixed test partitions. Full-precision metrics, selected epochs, losses, sample counts, and domain differences are available in [`results/results_all.csv`](results/results_all.csv).

![Accuracy by domain](figures/accuracy_Real_World.png)

![Source-to-target accuracy differences](figures/domain_gap_Real_World.png)

## What We Observed

- Performance drops when the models are evaluated outside the Real-World source domain.
- Clipart is the most difficult domain in this experiment.
- ResNet50 performs better than ResNet18 in the corresponding comparisons across all four domains.
- Fine-tuning does not always improve cross-domain performance.
- ResNet50 fine-tuning improves Clipart accuracy by 3.72 percentage points, while ResNet18 fine-tuning performs worse than its linear probe on every domain in this experiment.

These observations apply to this experiment, fixed split, and single training seed. They should not be treated as universal conclusions about the architectures or domains.

## Repository Structure

```text
pretrained-vision-domain-shift/
├── .streamlit/
│   └── config.toml
├── OfficeHome_DomainShift_Colab_T4.ipynb
├── README.md
├── DATA.md
├── PROVENANCE.md
├── requirements.txt
├── demo/
│   ├── app.py
│   ├── inference.py
│   ├── checkpoints.json
│   ├── sample_manifest.json
│   └── requirements.txt
├── figures/
│   ├── accuracy_Real_World.png
│   ├── domain_gap_Real_World.png
│   └── ...
├── results/
│   ├── config.json
│   ├── results_all.csv
│   ├── split_manifest.csv
│   └── ...
├── report/
│   ├── figures/
│   ├── report.pdf
│   └── report.tex
└── scripts/
    └── verify_results.py
```

## Running the Project

The completed notebook is designed for Google Colab:

1. Open [`OfficeHome_DomainShift_Colab_T4.ipynb`](OfficeHome_DomainShift_Colab_T4.ipynb) in Colab using **File → Upload notebook** or Colab's GitHub import.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order, or choose **Runtime → Run all**.

The notebook installs its additional dependencies, downloads and validates Office-Home, trains the four configurations, evaluates all four domains, and exports a compact result archive. A new run trains the models unless matching recovery checkpoints already exist in the Colab runtime.

Runtime files are stored under `/content/OfficeHome_DomainShift_T4/`. The exported result archive excludes checkpoints, and Colab-local files disappear when the runtime is deleted. The repository contains the completed metrics and predictions but does not include the dataset or model checkpoints.

If a DataLoader worker-cleanup warning appears but execution continues, it does not by itself indicate a failed run. For a fresh run, `NUM_WORKERS` can be set to `0` before later cells are executed. The archived run used two workers; changing this setting produces a different configuration fingerprint.

Supplementary checkpoints and large artifacts, when available, are stored in the [project Google Drive folder](https://drive.google.com/drive/folders/1BwltTDh_qRiF26FS5loaZozat-fKJAqY?usp=sharing). In a checkpoint archive, `best.pt` is intended for evaluation and `last.pt` contains the state used to resume training.

For a separate local environment, `requirements.txt` records the main packages and the PyTorch/torchvision versions observed in the completed Colab run. The recorded build used CUDA 13.0; choose the PyTorch wheel appropriate for the local system. Colab users should use the notebook's installation cell.

## Verify Results

The archived predictions can be checked without a GPU or model checkpoints. Install `numpy`, `pandas`, and `scikit-learn`, then run from the repository root:

```bash
python scripts/verify_results.py
```

The script recomputes accuracy and macro-F1 from all 16 prediction files and checks the class mapping, test membership, domain differences, selected checkpoint epochs, and saved data partitions. It does not retrain the models or download the dataset.

## Project Report

The English project report is available as a [PDF](report/report.pdf), with its [LaTeX source](report/report.tex) and figure files in `report/figures/`.

Compile from the repository root using a LaTeX installation with the required packages:

```bash
cd report
pdflatex report.tex
pdflatex report.tex
```

The second pass resolves citations and cross-references.

## Team Members

We worked on this project together, but each member focused more on a different part of the pipeline.

| Member | Student ID | GitHub | Contribution |
|---|---|---|---|
| Nguyễn Tiến Dũng | 23BA14068 | [Dung092005](https://github.com/Dung092005) | Implemented the ResNet models, training configurations, linear probing, and partial fine-tuning procedures. |
| Nguyễn Ngọc Hiếu | 23BA14109 | [ngochieu1762005](https://github.com/ngochieu1762005) | Established dataset provenance and version tracking; implemented reproducibility checks and organized experiment artifacts. |
| Vũ Minh Châu | 23BA14028 | [minmiwn](https://github.com/minmiwn) | Implemented image preprocessing, training augmentation, and the data preparation pipeline. |
| Nguyễn Minh Hiếu | 23BA14105 | [MinhHieu1601](https://github.com/MinhHieu1601) | Prepared and edited the report; synthesized the related work and organized citations and references. |
| Lê Đức Anh | 23BA14005 | [leducanh21122003](https://github.com/leducanh21122003) | Conducted model evaluation, compared domain-level results, and analyzed prediction errors. |
| Hoàng Lê Anh Đức | 23BA14057 | [duchla2005](https://github.com/duchla2005) | Designed the experiments and comparison protocol; integrated the project components and final deliverables. |

## Limitations

- The experiment uses one source domain: Real-World.
- It uses one fixed data split and one training seed.
- Domain difficulty and image composition may also affect accuracy, so the differences do not isolate domain shift completely.
- Exact-byte deduplication does not detect near-duplicate images.
- Only the final residual block is fine-tuned; full-backbone fine-tuning is not evaluated.
- Possible overlap between Office-Home images and ImageNet pretraining data cannot be ruled out.
- The protocol is not directly comparable with studies using the standard Office-Home domain adaptation setup.

## References

- Venkateswara, H., Eusebio, J., Chakraborty, S., & Panchanathan, S. (2017). *Deep Hashing Network for Unsupervised Domain Adaptation*. CVPR, 5018–5027. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.html)
- [Office-Home dataset and fair-use notice](https://www.hemanthdv.org/officeHomeDataset.html)
- [Flower Labs Office-Home mirror](https://huggingface.co/datasets/flwrlabs/office-home)
- [Torchvision ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Torchvision ResNet50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

Office-Home is provided under the dataset authors' terms for noncommercial research and education. Third-party data and pretrained model weights retain their respective terms.

