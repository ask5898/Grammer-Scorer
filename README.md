# Spoken-English Grammar Scoring

Solution for the **SHL Hiring Assessment 2026** Kaggle competition: predict a grammar score (0–5) for short clips of spoken English.

## Overview

I treated this as two problems: getting good transcripts, and then scoring their grammar.

Each clip is transcribed with **Whisper** (`large-v3-turbo`) in verbatim mode, so filler words, repetitions and false starts are kept. These say a lot about how someone speaks, so I didn't want the transcriber to clean them up. A **DeBERTa-v3-large** model, fine-tuned with **LoRA** adapters, then predicts the grammar score from each transcript.

With only around 600 labelled clips, I didn't trust a single validation split, so I used **5-fold cross-validation**. That gave me a more reliable score to compare ideas against, and five models whose averaged predictions make up the final submission.

## Pipeline

```
audio (.wav)
   │
   ▼
Whisper large-v3-turbo ──► verbatim transcript
   │
   ▼
Prompt template  ──►  "Transcript of a spoken English answer:\n<text>\n\nGrammar score:"
   │
   ▼
DeBERTa-v3-large + LoRA + regression head     (one model per CV fold, 5 folds)
   │
   ▼
de-normalise → clip to [0, 5] → average across folds → round to 1 decimal
   │
   ▼
submission.csv
```

## Results

| Metric | Score |
|---|---|
| 5-fold CV RMSE (out-of-fold) | _fill in from notebook_ |
| 5-fold CV Pearson | _fill in from notebook_ |
| Public leaderboard | _fill in from Kaggle_ |

The CV scores are computed on out-of-fold predictions covering every training clip (see `oof_deberta.csv`). The notebook prints them at the end of the training section.

## Key design decisions

- **Verbatim transcription.** Disfluencies such as "um", "uh", repeated words and abandoned sentences are kept, because they are part of the grammar signal.
- **DeBERTa-v3-large.** A strong encoder for text-quality and essay-scoring tasks. LoRA (rank 16) on the attention and dense projections keeps training within a single 16 GB T4 GPU, while the classifier and pooler are trained in full.
- **Standardised targets.** Labels are z-scored using each fold's *training* mean and standard deviation, so nothing leaks from that fold's validation clips.
- **Stratified 5-fold CV.** Folds are stratified on the rounded score so that each one has a similar label distribution. Averaging the five fold models reduces variance on the test set.
- **Mixed-precision stability.** The MSE loss is computed in fp32 so that fp16 logits don't clash with fp32 labels.
- **Resumable training.** Each fold saves its adapter, label statistics and validation predictions, so an interrupted Kaggle session continues from the next unfinished fold.

## Repository structure

```
.
├── shl_grammar_scoring.ipynb   # end-to-end pipeline: transcription → CV training → inference → submission
├── requirements.txt
└── README.md
```

The notebook is organised into sections:

1. Environment setup
2. Configuration: a single `Config` dataclass holds every path and hyperparameter
3. Speech-to-text (`AudioTranscriber`)
4. Data preparation and fold assignment
5. Model (DeBERTa-v3-large + LoRA)
6. Training with 5-fold cross-validation
7. Inference (fold ensemble)
8. Submission

## How to run

### On Kaggle (recommended)
1. Open the notebook on Kaggle and attach the competition dataset (`shl-hiring-assessment-2026`).
2. Set the accelerator to **GPU T4** and enable internet access (to install Whisper and download model weights).
3. Click **Run All**. All outputs are written to `/kaggle/working`.

### Locally
```bash
pip install -r requirements.txt
```
Download the competition data, then change `data_path` and `work_dir` in the `Config` cell to point at your local folders. A CUDA GPU with at least 16 GB of memory is recommended.

## Generated files

| File | Contents |
|---|---|
| `train_labels.csv`, `test_labels.csv` | Whisper transcripts paired with their labels |
| `deberta_lora_adapter_fold{k}/` | Fold *k* LoRA adapter, regression head and `norm.json` (label mean and std) |
| `oof_fold{k}.npy` | Fold *k* out-of-fold predictions |
| `oof_deberta.csv` | Out-of-fold predictions for all training clips |
| `submission.csv` | Final test predictions (`filename,label`) |

The competition data and the generated files are not included in this repository (see `.gitignore`).

## Limitations and next steps

- **Transcription errors become grammar errors.** Whenever Whisper mishears a word, the model sees a mistake the speaker never made. Comparing Whisper models, or adding ASR confidence as a feature, could reduce this.
- **Small dataset.** About 600 labelled clips limits how much a large model can learn. Ensembling models with different architectures, or adding handcrafted features (LanguageTool error counts, disfluency rate, speaking rate) through a gradient-boosted stacker, are natural next steps.
- **Label noise.** Human grammar ratings vary between raters, which limits the best RMSE any model can reach.
