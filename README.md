# ML Assignment 1: The ML Process

Two supervised learning problems, each taken through the full ML process: problem definition, data collection, EDA, cleaning, feature engineering, splitting, model selection, training, tuning, under/overfitting check, final testing, deployment and a monitoring plan.

**Author:** Nephtalem Woldeyohanes · Machine Learning, AAU

**Live app:** [ml-assignment-1-nephtalem.streamlit.app](https://ml-assignment-1-nephtalem.streamlit.app/) (one tab per model)

| Problem | Dataset | Final model | Test result |
| --- | --- | --- | --- |
| Regression: forecast tomorrow's PM2.5 in Addis Ababa | [Air Quality Dataset for Addis Ababa](https://www.kaggle.com/datasets/nitirajkulkarni/addis-ababa-et-344979) (Kaggle) | Ridge regression (α = 30), 13 features | RMSE 3.93 µg/m³ vs 4.19 for the persistence baseline |
| Classification: predict stroke (yes / no) | Healthcare stroke prediction dataset (Kaggle) | Logistic Regression (balanced, C = 0.01) | Recall 0.82 (41 of 50 strokes), ROC-AUC 0.835 |

## Repository structure

```
regression/
  data/raw, data/processed     Addis Ababa air-quality data (raw and cleaned)
  notebooks/01_understand_data.ipynb      EDA and cleaning
  notebooks/02_regression_model.ipynb     features, split, models, tuning, test, saved model
  reports/figures/             figures used in the presentation
  models/ridge_pm25.joblib     final model (pipeline)
classification/
  data/raw, data/processed     stroke data (raw and cleaned)
  notebooks/01_understand_data.ipynb      EDA and cleaning
  notebooks/02_classification_model.ipynb pipeline, models, tuning, test, saved model
  reports/figures/
  models/stroke_model.joblib   final model (pipeline)
app/
  app.py                       Streamlit app with one tab per model
  models/                      copies of both saved models
  requirements.txt             pinned versions for deployment
report/
  ML_Assignment1_Presentation.pptx
```

## How to run

Each project has its own virtual environment and `requirements.txt` (Python 3.12):

```bash
cd regression            # or: cd classification
python -m venv .venv
.venv\Scripts\activate   # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then run notebook `01` and then notebook `02` (Restart & Run All).

To run the app locally:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

Both datasets are public Kaggle datasets and are included here for reproducibility; credit belongs to their original authors. The Addis Ababa data is model-based (Open-Meteo CAMS), not ground-sensor data.
