# 🔍 Credit Card Fraud Detection — Unsupervised Deep Learning

> Detecting credit card fraud using **Restricted Boltzmann Machines (RBM)** and **Autoencoders** on a heavily imbalanced dataset (0.17% fraud rate), without any labelled training signal.

---

## 📌 Overview

This project explores two unsupervised anomaly detection approaches for identifying fraudulent credit card transactions. Since fraud is rare and labels are expensive, the models are trained entirely on unlabelled data and use reconstruction error / free energy as a fraud signal.

| Model | Approach | Anomaly Signal | AUC |
|-------|----------|----------------|-----|
| Autoencoder | Reconstruction error (MSE) | High MSE = anomaly | **~0.95** |
| RBM | Free energy scoring | High FE = anomaly | Evaluated via ROC |

---

## 📂 Repository Structure

```
├── autoencoder_demo.ipynb   # Autoencoder-based fraud detection
├── rbm_demo.ipynb           # RBM-based fraud detection
├── rbm.py                   # Custom RBM implementation
├── creditcard.csv           # Dataset (from Kaggle, not included)
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud)
- **Size:** 284,807 transactions spanning ~2 days
- **Fraud rate:** 0.17% (highly imbalanced)
- **Features:** 28 PCA-transformed features (`V1`–`V28`) + `Time` + `Amount`

---

## 🧠 Models

### 1. Autoencoder (`autoencoder_demo.ipynb`)

An autoencoder is trained on all transactions (unsupervised). The intuition is that the model learns to reconstruct normal transactions well, but struggles with fraudulent ones — resulting in a higher reconstruction loss (MSE) for anomalies.

**Architecture:**
```
Input (28) → Encoder (15, tanh) → Decoder (28, tanh) → Output
```

**Key details:**
- Optimizer: RMSProp (`lr = 0.001`)
- Batch size: 256, Epochs: 10
- Train/test split: 75% / 25% (time-ordered)
- Normalisation: Z-score standardisation
- Anomaly signal: per-sample batch MSE

**Results:**
- Autoencoder (unsupervised) AUC: **~0.95**
- With autoencoder embeddings + FC classifier: **~0.967**
- Precision boost at threshold=7: from **0.132% → ~7.86%** (~60× lift)

---

### 2. Restricted Boltzmann Machine (`rbm_demo.ipynb`)

An RBM with Gaussian visible units is trained using contrastive divergence. Free energy is used as the anomaly score — fraudulent transactions yield atypically high free energy values.

**Key details:**
- Hidden units: 10
- Visible unit type: Gaussian
- Gibbs sampling steps: 4
- Optimizer: SGD with momentum (`lr = 0.001`, momentum = 0.95)
- Batch size: 512, Epochs: 10
- Train/test split: 50% / 50% (time-ordered)
- Normalisation: Z-score standardisation
- Anomaly signal: free energy score

**Evaluation:**
- ROC-AUC curve on validation set
- Free energy distribution plots (fraud vs. non-fraud)
- Precision-Recall curve with threshold sweep over FE range (0–200)
- Top-500 transaction analysis to assess model lift

---

## 🔁 Methodology

```
Raw Transactions
       │
       ▼
  Z-score Normalisation
       │
       ▼
Unsupervised Training (no labels used)
       │
  ┌────┴────┐
  ▼         ▼
 RBM     Autoencoder
  │         │
Free     Reconstruction
Energy      MSE
  │         │
  └────┬────┘
       ▼
Anomaly Score → Threshold → Fraud Flag
       │
       ▼
  Evaluate: AUC, Precision-Recall, Lift
```

---

## 📈 Key Results

- Autoencoder achieves **AUC ~0.95** using reconstruction error alone — with no fraud labels during training.
- Using autoencoder embeddings as features for a downstream FC classifier improves AUC to **~0.967**.
- A supervised FC baseline trained from scratch achieves comparable AUC, validating the embedding quality.
- RBM free energy separates fraud and non-fraud distributions visually, with threshold-tunable precision.
- At a detection threshold of 7 (MSE), precision improves **~60× over the base rate** (0.132% → ~7.86%).

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

> ⚠️ Built with **TensorFlow 1.x**. For TF2, wrap sessions in `tf.compat.v1`.

### Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud) and place it in the root directory.

### Run

```bash
jupyter notebook autoencoder_demo.ipynb
jupyter notebook rbm_demo.ipynb
```

---

## 💡 Key Takeaways

- Unsupervised anomaly detection is viable even on extreme class imbalance (0.17% fraud).
- Autoencoders outperform RBMs on this task due to better representation learning via backpropagation.
- Threshold tuning on the Precision-Recall curve is critical for real-world deployment where false positives carry a cost.
- Autoencoder-learned embeddings are transferable — they improve downstream supervised classifiers.

---

## 📜 License

MIT License. Dataset subject to [Kaggle's terms of use](https://www.kaggle.com/dalpozz/creditcardfraud).
