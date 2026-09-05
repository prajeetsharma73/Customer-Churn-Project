# Customer Churn Predictor — Deployment Guide

## 1. Local setup
Put these 4 files in one folder, plus your dataset CSV
(`WA_Fn-UseC_-Telco-Customer-Churn.csv`):

```
churn-app/
├── train_and_save_model.py
├── app.py
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Install dependencies:
```
pip install -r requirements.txt
```

## 2. Train and save the model (run once)
```
python train_and_save_model.py
```
This creates 4 files: `rf_model.pkl`, `scaler.pkl`, `model_columns.pkl`,
`total_charges_median.pkl`. Keep these — the app loads them.

## 3. Test it locally
```
streamlit run app.py
```
Opens at `http://localhost:8501`. Fill in the form and click **Predict Churn**.

## 4. Deploy for free (Streamlit Community Cloud)
1. Create a GitHub repo and push all files above **including** the 4 `.pkl`
   files (small enough to commit directly).
2. Go to https://share.streamlit.io → "New app" → connect your GitHub repo.
3. Set the main file to `app.py` → Deploy.
4. You'll get a public URL like `https://your-app.streamlit.app` — put this
   in your resume/portfolio.

## Notes
- The CSV itself does NOT need to be uploaded to GitHub for the deployed
  app to work — only the 4 `.pkl` files are needed at runtime, since the
  model is already trained.
- If you retrain later, just rerun step 2 and re-push the updated `.pkl` files.
