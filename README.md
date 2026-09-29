# Customer Churn Predictor 📉

A Streamlit web app that predicts whether a telecom customer is likely to
churn, using a Random Forest model trained on the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
plus a Power BI report that quantifies the revenue at stake.

**🔗 Live demo:** https://prajeet-sharma-customer-churn-project.streamlit.app/
*(Free Streamlit apps sleep after a period of inactivity. If you see a "wake up" button, click it and wait a few seconds.)*

## Screenshots

**Churn Overview**
![Churn Overview](images/churn%20overview.png)

**Behavioral Analysis**
![Behavioral Analysis](images/behavioral%20analysis.png)

**Operational Retention Action Center**
![Operational Retention Action Center](images/at-risk%20action%20centre.png)

## What it does
Enter a customer's details (contract type, tenure, monthly charges, services
subscribed, etc.) and the app returns a churn prediction with a probability
score.

## Key findings
- **26.5% of customers churned**, taking about **$139K of monthly recurring revenue (30.5% of the total)** with them.
- Churned customers pay more per month than retained ones (about **$74 vs $61**), so the revenue loss is larger than the customer loss.
- **Month-to-month contracts account for roughly 87% of the lost revenue.** Their churn rate is 42.7%, versus 11.3% for one-year and 2.8% for two-year contracts.
- **Fiber-optic customers churn at 41.9%**, compared with 19.0% for DSL and 7.4% for customers with no internet service.
- The highest-risk group is active month-to-month customers with 12 months or less of tenure: 970 customers, about $47.8K in monthly revenue.

## How it was built
- **EDA & cleaning:** handled missing `TotalCharges`, dropped non-predictive
  ID column, examined churn distribution across contract type and tenure
- **Preprocessing:** one-hot encoding for categorical features, standard
  scaling for numeric features
- **Class imbalance:** SMOTE oversampling on the training set only
- **Model:** Random Forest Classifier (100 trees, max depth 10)
- **Deployment:** trained model + preprocessing objects serialized with
  `joblib`, served through a Streamlit form

## Results
Evaluated on a held-out 20% test split:

| Metric | Value |
|---|---|
| Accuracy | 75.6% |
| Churn recall | 0.75 |
| Churn precision | 0.53 |
| Churn F1 | 0.62 |

The model favors catching churners (recall) over maximizing accuracy. A model that predicts "nobody churns" would already score about 73.5% accuracy, so recall is the more useful number here.

## Power BI report
A multi-page report on the same dataset with DAX measures for churn rate, lost monthly revenue, ARPU, active high-risk customers and monthly revenue at risk, plus a rule-based **Risk Level** (High / Medium / Low) and a **Recommended Action** per customer. Files are in the `powerbi/` folder. The report reads the CSV from a local path, so after cloning, point the data source in Power Query to your own copy of the file.

## Tech stack
Python · pandas · scikit-learn · imbalanced-learn · Streamlit · Power BI · DAX

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
├── app.py                        # Streamlit UI
├── train_and_save_model.py       # training pipeline
├── Customer_Churn.ipynb          # EDA, modeling and evaluation
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── rf_model.pkl                  # trained model (generated)
├── scaler.pkl                    # fitted scaler (generated)
├── model_columns.pkl             # feature column order (generated)
├── total_charges_median.pkl      # imputation value (generated)
├── powerbi/                      # Power BI project (.pbip, .Report, .SemanticModel)
└── images/                       # dashboard screenshots
```

## Limitations and next steps
- Results come from a single train/test split. Cross-validation and a logistic regression baseline would make the comparison more reliable.
- The default 0.5 decision threshold was used. Tuning it to the cost of a missed churner versus a wasted retention offer would add business value.
- The retention offers in the Power BI report (discount, free support, credit) are illustrative assumptions, not values derived from the data.

## Author
**Prajeet Sharma**
LinkedIn: https://www.linkedin.com/in/prajeet-sharma/ | GitHub: https://github.com/prajeetsharma73/

## License
MIT — see [LICENSE](LICENSE).
