# ⚙️ Ship Engine Performance Prediction using FastAI

This project leverages **FastAI** and **deep learning** techniques to analyze and predict **ship engine performance metrics** based on sensor and operational data. The goal is to use machine learning to improve energy efficiency and maintenance scheduling by predicting key engine behaviors.

---

## 📘 Overview

This repository contains a Jupyter Notebook (`project-ship-engine-fastai.ipynb`) that demonstrates how to preprocess ship engine data, train a neural network using **FastAI**, and evaluate the model's performance.  

The workflow includes data ingestion, cleaning, normalization, training, and evaluation using a **tabular learner** from FastAI’s API.

---

## 🧩 Features

- Preprocessing of tabular sensor data
- Automated feature normalization and missing value handling
- Neural network training using **FastAI Tabular Learner**
- Performance evaluation and loss tracking
- Model experimentation with different learning rates and epochs

---

## 🧠 Workflow Summary

### 1. Data Preparation
- The dataset is loaded from a CSV file (e.g., `Ship_Engine_Data.csv`).
- Data is explored using **pandas** and visualized with **matplotlib**.
- FastAI preprocessing functions like `Categorify`, `FillMissing`, and `Normalize` are applied.

```python
from fastai.tabular.all import *

dls = TabularDataLoaders.from_csv(
    'Ship_Engine_Data.csv',
    y_names='Target_Variable',
    cont_names=['Speed', 'Torque', 'Fuel_Rate', 'Engine_Temp'],
    procs=[Categorify, FillMissing, Normalize],
    bs=64
)
```

### 2. Model Training

A **FastAI Tabular Learner** is created and trained with cyclical learning rates:

```python
learn = tabular_learner(dls, metrics=rmse)
learn.lr_find()
learn.fit_one_cycle(50, lr_max=0.5)
```

### 3. Evaluation

Model results and validation performance are visualized:

```python
learn.show_results()
```

---

## 📊 Results

| Metric | Description | Result |
|---------|--------------|--------|
| `train_loss` | Decreased over multiple cycles | ✅ |
| `valid_loss` | Stabilized with low variance | ✅ |
| Model | FastAI Tabular Neural Net | ✅ Trained successfully |

---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install fastai pandas matplotlib
```

Or use **Google Colab** (recommended):

```python
from fastai.tabular.all import *
```

---

## 📁 Repository Structure

```
├── project-ship-engine-fastai.ipynb
├── Ship_Engine_Data.csv
├── README.md
```

---

## 🚀 Future Enhancements

- Add feature importance analysis
- Implement hyperparameter tuning with Optuna
- Test deployment via FastAPI or Streamlit
- Compare model with other regression algorithms (XGBoost, LightGBM)

---

## 🧑‍💻 Author

**Morteza Naseri**  
Developed as part of a Machine Learning research project using **FastAI**.

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to fork and adapt for your own experiments.

---
