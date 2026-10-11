# 🩺 Diabetes Risk Predictor

An interactive Streamlit web app that estimates whether a person is likely to have diabetes, using a Logistic Regression model trained on 100,000 patient records.

> ⚕️ **Educational project only.** This is not a medical tool and must not be used for diagnosis. Please consult a healthcare professional for any health concern.

---

## ✨ Features

- **Interactive prediction form** for gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level and blood glucose level
- **Probability gauge** showing the model's diabetic risk score
- **Explanation chart** showing which inputs pushed the prediction toward or away from "diabetic" (coefficient × scaled value)
- **Model page** with the pipeline steps, test-set metrics and learned coefficients

---

## 🧠 How the model works

| Step | Detail |
|------|--------|
| Data | [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset), 100,000 rows, 8 features + target |
| Encoding | `LabelEncoder` on `gender` and `smoking_history` |
| Split | 80% train / 20% test (`random_state=42`) |
| Scaling | `StandardScaler`, fit on the training set only |
| Model | `LogisticRegression(class_weight="balanced")` |
| Decision rule | Probability ≥ 0.5 → labelled diabetic |

### Test-set results

| Accuracy | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| 0.89 | 0.43 | 0.88 | 0.57 |

**Reading these numbers:** `class_weight="balanced"` pushes the model to catch most diabetic patients (high recall), which suits a screening scenario, but it also flags many healthy people (low precision). It also inflates the predicted probabilities, so they are best read as a **risk score**, not a literal chance of having diabetes.

---

## 📁 Project structure

```
diabetes_app/
├── app.py                              # Streamlit UI
├── model_utils.py                      # Loads the saved model, or rebuilds it from the CSV
├── logistic_reg.pkl                    # Trained Logistic Regression model
├── scaler.pkl                          # Fitted StandardScaler (needed for correct predictions)
├── diabetes_prediction_dataset.csv     # Dataset (used only if the .pkl files are missing)
├── metrics.json                        # Test-set metrics shown in the app
├── requirements.txt
└── README.md
```

The notebook (`logistic_reg.ipynb`) contains the full training workflow: EDA, preprocessing, training and evaluation.

---

## 🚀 Run it locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

If `logistic_reg.pkl` or `scaler.pkl` is missing, the app retrains the model from the CSV automatically using the same steps as the notebook.

---

## ⚠️ Limitations

- **Low precision** (0.43): many "diabetic" predictions are false alarms.
- **Linear model:** Logistic Regression can't capture non-linear relationships between features.
- **Dataset quality:** the dataset is large but likely synthetic or heavily processed, so the results may not reflect real clinical populations.
- **Not validated clinically:** no external validation has been done.

## 🔭 Possible improvements

- Compare against Random Forest and XGBoost
- Tune the decision threshold using a precision–recall curve
- Try SMOTE or other imbalance handling instead of class weights
- Tune hyperparameters with `GridSearchCV`
- Replace the label encoding with a proper `ColumnTransformer` pipeline

---

## 🛠️ Tech stack

Python · pandas · scikit-learn · Plotly · Streamlit · joblib

---

## 👤 Author

**Shreyash Singh Rawat**, B.Tech CSE (AI & ML)
Built as a learning project while strengthening Python and data analytics skills.
