# RICO-Screen2Words Cross-Modal Retrieval

**COMP 1030 Multimodal Data Analytics, Project 2.2: bidirectional retrieval between Android screenshots and text summaries**
**Student Name - Acvin Gonsalves**
**Student ID - a2996845**

This project builds and compares models that retrieve written summaries from a screenshot (screen to text) and screenshots from a written summary (text to screen) on the RICO-Screen2Words dataset. It covers data quality checks, a leakage-aware split, four models (ridge baseline, early fusion, contrastive dual encoder and a frozen CLIP reference), statistical comparison, efficiency, overfitting diagnosis, subgroup and failure analysis, and ethics. Everything is run from one notebook, `Multimodal_Project_2_2_RICO_Retrieval_a2996845.ipynb`.

## Contents

1. [Data flow and models](#1-data-flow-and-models)
2. [Files in this project](#2-files-in-this-project)
3. [Requirements and dependencies](#3-requirements-and-dependencies)
4. [Folder layout and dataset placement](#4-folder-layout-and-dataset-placement)
5. [One-time setup and how to run](#5-one-time-setup-and-how-to-run)
6. [Output files](#6-output-files)
7. [Configuration](#7-configuration)
8. [Method summary and assumptions](#8-method-summary-and-assumptions)
9. [Output files to inspect for the report](#9-output-files-to-inspect-for-the-report)

---

## 1. Data flow and models

```text
Hugging Face dataset  rootsautomation/RICO-Screen2Words   (downloaded automatically)
        |
        v
Section 2   quality checks, cleaning, privacy review
        |
        v
Frozen features (cached):  ResNet-50 screen vectors, BERT caption vectors, CLIP vectors
        |
        v
Leakage audit and grouped split  (train / validation / test)
        |
        +--> E0  ridge regression baseline
        +--> E1  early fusion MLP
        +--> E2  contrastive dual encoder (late fusion)
        +--> E3  frozen CLIP, reference only (not trained)
        v
Section 4   Recall@K, paired bootstrap intervals, efficiency, overfitting diagnosis,
            category recall, failure analysis
        |
        v
outputs/   (figures, tables, configuration record)
```

| Model | Role | Trained here |
|---|---|---|
| E0 ridge regression | Simple baseline | Yes |
| E1 early fusion | Joins image and text first and scores every pair | Yes, three seeds |
| E2 contrastive dual encoder | Encodes each side separately, compares at the end | Yes, three seeds |
| E3 frozen CLIP ViT-B/32 | Reference point only | No |

The re-ranker (E4) from the proposal is not part of this build and is listed as future work in the report.

The notebook reads its data only from the Hugging Face dataset and from its own feature cache. No other data file is needed.

---

## 2. Files in this project

| File | Purpose |
|---|---|
| `Multimodal_Project_2_2_RICO_Retrieval_a2996845.ipynb` | The full pipeline with saved outputs: data, features, models, evaluation, ethics |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

Where each part is implemented in the notebook:

| Part | Notebook section |
|---|---|
| Setup, seeds, configuration dictionary `CFG`, helper functions | 2.1 |
| Dataset loading, schema and licence | 2.2 |
| Text quality, image quality, manual privacy review | 2.3 |
| Cleaning and token limit | 2.4 |
| Frozen feature extraction and cache | 2.5 |
| Leakage audit and grouped split | 2.6 |
| Retrieval metrics with multiple positives | 3.1 |
| E0 ridge baseline and BERT pooling | 3.2 |
| E1 early fusion and E2 contrastive model | 3.3 |
| Hyper-parameter search and final training runs | 3.4 and 3.5 |
| CLIP reference and main results | 4.1 |
| Recall at K and paired bootstrap comparison | 4.2 and 4.3 |
| Efficiency and scalability | 4.4 |
| Overfitting and leakage diagnosis | 4.5 |
| Recall by app category and failure analysis | 4.6 and 4.7 |
| Ethics, reflection, limitations, conclusion, references | 5 and 6 |

---

## 3. Requirements and dependencies

| Item | Requirement |
|---|---|
| Python | 3.10 or newer (the recorded run used 3.13) |
| GPU | An NVIDIA GPU is strongly recommended. The recorded run used a Tesla T4 with 15.6 GB memory on Google Colab. A CPU run works but is much slower |
| Internet | Needed on the first run to download the dataset and the pretrained ResNet-50, BERT and CLIP weights |
| Disk space | About 10 GB free for the dataset, model weights and the feature cache |
| Packages | `torch`, `torchvision`, `transformers`, `datasets`, `huggingface_hub`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `pillow`, `tqdm`, `ipython`, `ipykernel`, `jupyter`, `nbconvert` (listed in `requirements.txt`) |

> **Note:** The recorded run used torch 2.11.0, transformers 5.20.2, datasets 4.8.5 and torchvision 0.26.0. Newer or older versions usually work, but exact numbers can differ slightly. For a GPU on Windows, install the CUDA build of PyTorch from pytorch.org before Step 2.

---

## 4. Folder layout and dataset placement

The project folder is `C:\Multimodal_assign`. It can be placed elsewhere, because no path in the notebook is hard coded: the notebook uses the folder it is started from.

The dataset is **not** placed in the folder by hand. It is downloaded from Hugging Face automatically the first time the notebook runs.

Before running, the folder must look like this:

```text
Multimodal_assign\
|-- Multimodal_Project_2_2_RICO_Retrieval_a2996845.ipynb
|-- requirements.txt
|-- README.md
|
|-- outputs\                  created automatically when the notebook runs
|     |-- figures\
|     |-- tables\
|     |-- configs\
|
|-- cache\                    created automatically (decoded screenshots and feature files)
```

- The notebook, `requirements.txt` and `README.md` go directly in `Multimodal_assign\`.
- `outputs\` and `cache\` are created by the notebook. Do not create them yourself.
- Start Jupyter from the `Multimodal_assign` folder, so that `outputs\` and `cache\` are created inside it.
- `outputs\` and `cache\` are not part of the submission.
- When the notebook runs in Google Colab instead, the same files are written to `/content/cache` and `MyDrive/rico_2_2_outputs` (see Option B in section 5).

---

## 5. One-time setup and how to run

All commands are written for the Windows Command Prompt (`cmd`). Follow the steps in order.

### Step 1: Open Command Prompt in the project folder

1. Open the `Multimodal_assign` project folder in File Explorer.
2. Click the address bar at the top of File Explorer.
3. Type `cmd` and press Enter.

Command Prompt opens directly in the project folder and shows, for example:

```cmd
C:\Multimodal_assign>
```

> **Important:** All commands in this guide must be run from this folder unless stated otherwise.

### Step 2: Create a virtual environment and install the packages

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Wait until all packages have finished installing. If you have an NVIDIA GPU, install the CUDA build of PyTorch first, using the command from pytorch.org.

### Step 3: Verify the installation

Run:

```cmd
python -c "import torch, torchvision, transformers, datasets, sklearn, scipy, pandas, numpy, matplotlib, seaborn, PIL, pyarrow; print('All packages OK'); print('GPU available:', torch.cuda.is_available())"
```

If everything is installed correctly, the output is:

```text
All packages OK
GPU available: True
```

`GPU available: False` means the notebook will run on the CPU, which is much slower.

### Step 4: Choose how to run the notebook

Use **one** of the following.

**Option A: run on your computer (this folder).** Run the notebook from the command line:

```cmd
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Multimodal_Project_2_2_RICO_Retrieval_a2996845.ipynb
```

Or open it in Jupyter and choose Run All:

```cmd
python -m jupyter lab
```

**Option B: run in Google Colab.** Upload the notebook to Colab, choose a T4 GPU runtime and select Run all. When asked, allow access to Google Drive. Google Drive is optional and only keeps the outputs between sessions. Large caches stay on the local disk of the Colab session.

> **Important:** The notebook contains one Google Drive mount cell. It is wrapped in a `try` block, so on your own computer it is skipped automatically and the message `Running in Colab: False` is printed. Nothing needs to be changed.

### Step 5: Wait for the run to finish

The first run downloads the dataset and the model weights, extracts the frozen features and trains the models with three seeds. Feature files are cached, so a rerun loads them in seconds. Do not close the window while the notebook is running.

For a quick test of the whole pipeline, set `max_screens` in the configuration cell (section 2.1) to a small number such as `2000`, run the notebook, then set it back to `None` for the full run.

### Step 6: Check the results

A successful run executes every cell without an error and ends with the conclusion and references sections. All generated figures and tables are saved in `outputs\` (section 6).

> **Execution summary:** Run the notebook once from top to bottom. No other script is needed. The notebook keeps its outputs, so the results can also be read without running it again.

### Other options

| Option | Meaning |
|---|---|
| `max_screens` | Number of screens to use. `None` uses the full dataset of 22,417 screens |
| `force_recompute` | Set to `True` to rebuild the cache instead of loading it |
| `run_clip` | Set to `False` to skip the frozen CLIP reference |
| `seeds` | Training seeds for the trained models (default 0, 1 and 2) |

---

## 6. Output files

All outputs are written under `outputs\`.

**Figures and tables**

| File | Content |
|---|---|
| `figures\figNN_name.png` | Every figure of the notebook, numbered in order of appearance |
| `tables\tNN_name.csv` | Every table of the notebook, numbered in order of appearance |
| `configs\` | The configuration record of the run, the split manifest and the library and GPU versions |

**Cache**

| File | Content |
|---|---|
| `cache\imgs_*.npy` | Decoded and resized screenshots |
| `cache\imgstats_*.parquet` | Image quality statistics |
| `cache\features_*.npz` | ResNet-50, BERT and CLIP feature files, named with a hash of the data manifest |

The cache can be deleted at any time. The notebook rebuilds it on the next run.

---

## 7. Configuration

All parameters are in the dictionary `CFG` in the setup cell of section 2.1.

| Group | Parameter | Value | Meaning |
|---|---|---|---|
| Data | `dataset_id` | `rootsautomation/RICO-Screen2Words` | Hugging Face dataset |
| Data | `max_screens` | `None` | Full dataset. A small number gives a dry run |
| Data | `img_size` | 224 | Screenshot size for the encoders |
| Seeds | `seed` | 42 | Data split and sampling |
| Seeds | `seeds` | 0, 1, 2 | Training seeds for every trained model |
| Split | `val_frac`, `test_frac` | 0.10, 0.10 | Validation and test share |
| Split | `near_dup_thr` | 0.95 | Cosine threshold for near-duplicate screens |
| Split | `near_dup_k` | 10 | Neighbours checked per screen |
| Split | `max_component_share` | 0.03 | Largest allowed share of one group |
| Training | `epochs`, `patience` | 60, 10 | Maximum epochs and early stopping |
| Training | `lr`, `wd`, `dropout` | 1e-3, 1e-2, 0.3 | Default learning rate, weight decay, dropout |
| Training | `hid`, `emb`, `batch` | 512, 256, 256 | Hidden size, embedding size, batch size |
| Search | `search_trials` | 8 | Hyper-parameter trials, tuned on validation data only |
| Evaluation | `bootstrap` | 1000 | Paired bootstrap resamples |
| Reference | `run_clip` | `True` | Frozen CLIP, reference only |

Output and cache paths are derived automatically: `outputs\` and `cache\` in the folder the notebook is started from, or the Colab paths when run in Colab.

---

## 8. Method summary and assumptions

### Data and cleaning

- The dataset has 22,417 screens, 112,085 summaries and 6,269 apps in 28 categories. Within-screen repeated summaries are removed, which leaves 111,106 summaries.
- Screens in the Hugging Face dataset keep their original split label, which is used only for the secondary comparison with the official split.

### Leakage-aware split

- Near-duplicate screens (cosine similarity above 0.95 on frozen features) and screens of the same app are placed in the same group. Whole groups go to training, validation or test, giving 18,080, 2,161 and 2,176 screens.
- About 22.7 percent of the grouped test screens still have a close neighbour. This limit is stated in the report.

### Models and training

- Features are frozen: ResNet-50 for screens, BERT-base for text (mean pooling over valid tokens) and CLIP for the reference.
- Standardisation statistics are fitted on the training split only.
- The contrastive model uses a symmetric InfoNCE loss with one sampled summary per screen per step and masks false negatives. Early fusion scores every screen and summary pair.
- Hyper-parameters are tuned on validation data only. The test split is used once for every final model.

### Metrics

- Recall@K, MRR and median rank with several correct answers per screen, reported in both directions.
- Mean Recall@10 is the main metric. Differences use a paired bootstrap with 1000 resamples and group-level intervals.

### Overfitting and leakage diagnosis

- The training and validation gap is checked with a shuffled-pairs control, a smaller model head and a comparison of the official split with the grouped split.

### Repeated runs

- Seeds are fixed, and the feature cache is keyed to a hash of the data manifest, so a rerun gives the same features.
- Exact numbers can differ slightly between GPU types and library versions.

---

## 9. Output files to inspect for the report

Open these files in `outputs\` after a successful run to check the numbers and figures used in the report. The notebook also prints every number as evidence in its executed cells.

### Data and quality

| Notebook section | What to check |
|---|---|
| 2.2 | Schema, dataset size (22,417 screens, 112,085 summaries, 6,269 apps) and licence notes |
| 2.3 | Text quality, normalisation speed, image quality and the manual privacy review (80 screens) |
| 2.4 | Cleaning impact (979 repeated summaries removed, 111,106 left) and the token limit |
| 2.6 | Leakage audit, grouped split sizes (18,080 / 2,161 / 2,176) and close-neighbour share (22.7 percent) |

### Models and main results

| Notebook section | What to check |
|---|---|
| 3.2 | Ridge baseline and BERT pooling choice |
| 3.4 and 3.5 | Hyper-parameter search, final runs and training curves |
| 4.1 | Main results table: mean Recall@10 of random 0.50, ridge 7.87, early fusion 9.68, contrastive 11.51 and CLIP 27.36 |
| 4.2 | Recall at K curves |
| 4.3 | Paired bootstrap comparison: gain over ridge of +2.63 (screen to text) and +4.66 (text to screen), against the target of +5 |

### Evaluation and analysis

| Notebook section | What to check |
|---|---|
| 4.4 | Query time at 100,000 candidates (early fusion 4.29 ms, contrastive 0.54 ms), training time and GPU memory |
| 4.5 | Train 60.27 against validation 11.29, shuffled-pairs control (56.74 train, 0.78 validation, chance 0.42) |
| 4.6 | Recall by app category (0 to 29.2 percent) |
| 4.7 | Success and failure examples and the failure breakdown |
| 5.1 | Ethics evidence: sensitive wording (13.03 percent of screens flagged, 7.9 points lower), representation, privacy and licence |
| 5.2 to 5.4 | Reflection, recommended model for three situations, limitations and conclusion |

### Where each report part comes from

| Report part | Source |
|---|---|
| Dataset and quality results | Notebook sections 2.2 to 2.4 |
| Split and leakage results | Notebook section 2.6 |
| Methods and training setup | Notebook sections 3.1 to 3.5 and the configuration record in `outputs\configs\` |
| Main results and comparison | Notebook sections 4.1 to 4.3 |
| Efficiency | Notebook section 4.4 |
| Overfitting and leakage diagnosis | Notebook section 4.5 |
| Category and failure analysis | Notebook sections 4.6 and 4.7 |
| Ethics and reflection | Notebook sections 5.1 to 5.4 |
