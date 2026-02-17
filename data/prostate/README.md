# Prostate z-scores dataset (Ignatiadis & Wager / Efron)

This repository's *real-data* Gaussian empirical Bayes experiment (the "prostate" example)
uses the classic microarray prostate cancer dataset consisting of **6033** z-scores.

## What is in this folder?

- `prostz.txt` (optional): the vector of 6033 z-values.
  - If you place this file here, the scripts will use it directly.
  - If the file is missing, the scripts can **auto-download** it (see below).

## Auto-download

The scripts will try to download `prostz.txt` from Trevor Hastie's CASI data page if it is
not present locally.

If your compute environment has no outbound internet access (common on some clusters),
download `prostz.txt` once on a machine with internet and copy it into:

```
data/prostate/prostz.txt
```

## Provenance

The z-scores are described in Ignatiadis & Wager's empirical Bayes paper as:

> For each gene, a t-statistic is computed and z-scores are calculated as
> \(Z_i = \Phi^{-1}(F_{100}(T_i))\).

and they are the same 6033 z-values used in Efron/Hastie's *Computer Age Statistical
Inference* materials ("prostz").
