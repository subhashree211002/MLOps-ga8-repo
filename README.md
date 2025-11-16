# **Iris Data Poisoning & MLflow Experimentation**

This project demonstrates the impact of **data poisoning** on the Iris classification dataset by intentionally corrupting feature values at different levels — **0%, 5%, 10%, and 50%** — and tracking all training outcomes using **MLflow**.

## **What This Project Includes**

* A poisoning module that injects random noise into Iris features.
* ML model training using Scikit-learn.
* Automated logging of:

  * metrics (accuracy, precision, recall, F1)
  * confusion matrices
  * classification reports
  * trained model artifacts
* All experiments executed on **Google Cloud Vertex AI Workbench (Jupyter Notebook)**.

## **How to Run**

```bash
pip install -r requirements.txt
python src/train_poison_mlflow.py --csv data/iris.csv --poison-levels 0.0,0.05,0.1,0.5
mlflow ui
```

## **Key Findings**

* Even small poisoning levels significantly degrade model performance.
* MLflow makes comparisons between multiple poisoned datasets easy and visual.
* Data quality has a direct and strong impact on required data quantity.
