# Student Placement Prediction — Classification Algorithms Comparison

A simple machine learning project that compares different classification algorithms using a real-world dataset and provides an interactive user interface to predict:

**Will a student get placed after graduation or not?**

## Dataset

**Kaggle:** [Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

Download `Placement_Data_Full_Class.csv` and place it inside the `data/` folder with the following name:

```text
data/placement.csv
```

The dataset contains 215 students, and the target column is `status` (`Placed` / `Not Placed`).

- The `salary` column is removed because it is only available for placed students, which would cause **data leakage**.
- The `mba_p` and `specialisation` columns are removed because this project is designed for undergraduate Computer Science students who have not pursued a master's degree.

## Project Structure

```text
placement_project/
├── data/
│   └── placement.csv
├── models/                  # Best model and results table
│                             # Generated after running train.py
├── app.py                   # Streamlit user interface
├── dataprocessing.py        # Data cleaning and preprocessing
├── train.py                 # Model training and comparison
├── requirements.txt
└── readme.MD
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train and compare all models:

```bash
python train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## Classification Algorithms

The project compares the following algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Note on Dataset Size

The dataset is relatively small, containing only 215 rows. Therefore, evaluation results can vary significantly depending on the train-test split.

To address this limitation, the results table includes **CV F1** (the average F1-score across 5-fold cross-validation) alongside the standard F1-score.

This provides a more reliable view of each model's performance across different data splits.