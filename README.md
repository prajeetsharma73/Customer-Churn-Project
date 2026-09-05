# Customer Churn Predictor 📉

A Streamlit web app that predicts whether a telecom customer is likely to
churn, using a Random Forest model trained on the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**🔗 Live demo:** _add your streamlit.app link here after deploying_

## What it does
Enter a customer's details (contract type, tenure, monthly charges, services
subscribed, etc.) and the app returns a churn prediction with a probability
score.

## How it was built
- **EDA & cleaning:** handled missing `TotalCharges`, dropped non-predictive
  ID column, examined churn distribution across contract type and tenure
- **Preprocessing:** one-hot encoding for categorical features, standard
  scaling for numeric features
- **Class imbalance:** SMOTE oversampling on the training set only
- **Model:** Random Forest Classifier (100 trees, max depth 10)
- **Deployment:** trained model + preprocessing objects serialized with
  `joblib`, served through a Streamlit form

## Tech stack
Python · pandas · scikit-learn · imbalanced-learn · Streamlit

## Run it locally
```bash
pip install -r requirements.txt
python train_and_save_model.py   # trains model, needs the CSV in this folder
streamlit run app.py
```
See `SETUP.md` for detailed step-by-step instructions (including Windows
PATH troubleshooting).

## Project structure
```
├── app.py                     # Streamlit UI
├── train_and_save_model.py    # training pipeline
├── requirements.txt
├── rf_model.pkl                  # trained model (generated)
├── scaler.pkl                    # fitted scaler (generated)
├── model_columns.pkl             # feature column order (generated)
└── total_charges_median.pkl      # imputation value (generated)
```

## Results
Evaluated on a held-out 20% test split (see console output when running
`train_and_save_model.py` for exact numbers on your run).

## License
MIT — see [LICENSE](LICENSE).
