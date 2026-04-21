# Audiobooks Customer Retention — Binary Classification with TensorFlow

A machine learning project that predicts whether an audiobook customer will make a repeat purchase, helping the business focus retention efforts on high-probability buyers.

## Business Problem

An audiobook platform wants to reduce marketing spend by identifying which customers are likely to buy again. Instead of targeting all users, the model flags customers worth re-engaging — turning a broad campaign into a targeted one.

## Dataset

`Audiobooks_data.csv` — each row represents a customer. Features include engagement metrics such as:

- Book length (overall and average per purchase)
- Price paid (overall and average)
- Review score and whether the customer left a review
- Total minutes listened
- Completion rate
- Support requests
- Last interaction (days since last visit)

**Target:** `1` = customer returned to buy again, `0` = did not return.

The dataset is imbalanced (~68% negative class), so it is balanced before training by removing excess negative samples.

## Approach

| Step | Notebook |
|---|---|
| Data preprocessing | [BusinessCase_preprocess.ipynb](BusinessCase_preprocess.ipynb) |
| Model training & evaluation | [BusinessCase_model.ipynb](BusinessCase_model.ipynb) |

**Preprocessing pipeline:**
1. Balance classes (equal number of `0` and `1` targets)
2. Standardize inputs with `sklearn.preprocessing.scale`
3. Shuffle and split into train / validation / test (80 / 10 / 10)

**Model architecture (TensorFlow / Keras):**
```
Input (10 features)
    → Dense(50, activation='relu')
    → Dense(50, activation='softmax')
    → Dense(2,  activation='softmax')
```
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Early stopping on validation loss (patience = 2)

## Results

| Set | Accuracy |
|---|---|
| Validation | ~83% |
| **Test** | **81.70%** |

## Setup

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:
1. `BusinessCase_preprocess.ipynb` — generates the `.npz` split files
2. `BusinessCase_model.ipynb` — trains and evaluates the model

## Requirements

See [requirements.txt](requirements.txt).
