# 🧠 Kreditriskmodeller med maskininlärning

Detta projekt syftar till att bygga, analysera och jämföra maskininlärningsmodeller för att **förutsäga kreditrisk** – närmare bestämt sannolikheten att en kund som beviljats lån går i default (betalningsinställelse). Arbetet är utfört på **syntetiskt genererad data**, vilket möjliggör full kontroll och insyn i variabler och beroenden.



## Datamängd

Projektet använder en syntetisk datamängd: [`synthetic_credit_data.csv`](synthetic_credit_data.csv), skapad specifikt för denna studie. Datat innehåller fiktiva men realistiskt konstruerade kundprofiler, inklusive:

- Demografiska faktorer: ålder, inkomst, anställningsstatus, boendeform  
- Finansiella faktorer: kreditpoäng, lånebelopp, DTI (debt-to-income)  
- Historik: antal missade betalningar, kundrelationens längd  
- Målvariabel: `default` (1 = betalningsinställelse, 0 = betalar tillbaka)

Endast kunder som **beviljats lån (`has_loan == 1`)** ingår i modelleringen.



## Projektstruktur

Projektet är organiserat i flera Jupyter Notebooks som representerar olika faser i analysen:

1. [`EDA.ipynb`](EDA.ipynb)  
   → **Exploratory Data Analysis** (EDA) för att förstå data, identifiera mönster, utvärdera datakvalitet och förbereda inför modellering.

2. [`multicollinearity_analysis.ipynb`](multicollinearity_analysis.ipynb)  
   → Undersöker **multikollinearitet** med hjälp av Variance Inflation Factor (VIF), för att säkerställa robusthet i regressionsmodellen.

3. [`Kreditriskmodell_ML.ipynb`](Kreditriskmodell_ML.ipynb)  
   → Tränar och utvärderar flera modeller (Logistic Regression, Random Forest, XGBoost).  
   → Cross-validation, kalibrering, threshold-tuning och utvärdering på testdata ingår.

4. [`Logreg_model_kreditrisk.ipynb`](Logreg_model_kreditrisk.ipynb)  
   → Fokus på den **förbättrade logistiska regressionsmodellen**, inklusive:  
     - Feature engineering  
     - ElasticNet regularisering  
     - Kalibrering (Isotonic)  
     - Tolkning av koefficienter  
     - Slutlig utvärdering på testset

---

## 📈 Resultatöversikt

| Modell         | ROC AUC | Recall | Precision | F1-score | Brier |
|----------------|---------|--------|-----------|----------|--------|
| Baseline       | 0.692   | 26.8%  | 53.6%     | 0.357    | 0.173  |
| Förbättrad LogReg | 0.943   | 56.4%  | 33.7%     | 0.422    | 0.039  |

Den förbättrade modellen visade **mycket högre rangordningsförmåga**, bättre kalibrerade sannolikheter, samt en bättre balans mellan recall och precision.

---

## 📁 Övriga filer

- `models/`: Tränade och kalibrerade modeller sparade som `.joblib`
- `reports/`: JSON-filer med utvärderingsresultat från testdata
- `requirements.txt`: Pythonpaket som krävs för att återskapa miljön

---

##  Slutsats

Logistisk regression, trots sin enkelhet, visade sig vara en **mycket stark modell** för kreditrisk – särskilt i kombination med:

- Tydlig feature engineering
- Kalibrering av sannolikheter
- Threshold-tuning utifrån verksamhetsmål

Genom att jämföra modeller och fokusera på tolkning, noggrannhet och återkallning har vi uppnått en modell som är både **praktiskt användbar och tekniskt robust**.



---


*Detta projekt är en del av kursen Fördjupad Python – Data Science vid EC Utbildning, 2025.*


