import numpy as np                     
import pandas as pd                    
from scipy.special import expit        # sigmoiden (1 / (1 + e^-x))

rng = np.random.default_rng(42)        
n = 5000                               

# 1. Grundvariabler
age = rng.integers(18, 91, n)          
max_years = np.clip(age - 18, 0, 40)   
years_with_bank = rng.integers(0, max_years + 1)                                     

# 2. Sysselsättning beroende på ålder

is_student = (age < 23).astype(int)     
is_retired = (age >= 65).astype(int)    
is_employed = ((age >= 23) & (age < 65)).astype(int)

employment_status = np.select(
    [is_student == 1,
     is_employed == 1,
     is_retired == 1],
    ["student", "employed", "retired"],
    default="unknown"
)

# 3. Inkomst (kr/mån) som blandning av normalfördelningar per status
income = (                            
    is_student  * rng.normal(12000, 3000, n) +
    is_employed * rng.normal(40000, 12000, n) +
    is_retired  * rng.normal(22000, 6000, n)
)
income = np.clip(income, 5000, 200000)  

# 4. Kreditpoäng – korrelerad med inkomst och bankrelation + brus
credit_score = (                        
    550
    + 0.0008 * (income - 30000)         
    + 0.6 * years_with_bank             
    + rng.normal(0, 40, n)              
)

credit_score = np.clip(credit_score,     # kreditpoäng begränsas till typiskt intervall
                       300, 850)


# 5. Lånebeviljande (has_loan) – åldersregel + kredit/inkomst
# åldersregel = kunder >= 74 får inte beviljat lån!

eligible_by_age = (age < 74).astype(int)  
approval_score = (                      
    -1.0
    + 1.6 * ((credit_score - 600) / 200)   
    + 0.6 * ((income - 30000) / 30000)     
    - 0.5 * np.maximum(age - 60, 0) / 20   
    + 0.2 * (years_with_bank / 10)         
)
approval_prob = expit(approval_score) * eligible_by_age              
has_loan = rng.binomial(1, approval_prob,  
                        n).astype(int)

# 6. Lånebelopp 
loan_amount = np.zeros(n)                  
mask = has_loan == 1                       
loan_amount[mask] = rng.normal(            # belopp ≈ 30% av inkomst, med 15% sd
    0.3 * income[mask],
    0.15 * income[mask]
)
loan_amount = np.clip(loan_amount, 0, 400000)  

# 7. Missade betalningar – påverkas av DTI & score
dti = np.zeros(n)                         
dti[mask] = (loan_amount[mask] /           
             np.maximum(income[mask], 1e-6)  
            ).clip(0, 5)                   

# Lambda för Poisson ökar med DTI och sjunker med score
miss_lambda = np.zeros(n)                  
miss_lambda[mask] = (                      # intensitet av missar: högre DTI och lägre score
    0.3 + 0.8 * dti[mask] + 1.2 * (1 - credit_score[mask] / 850)
)
miss_lambda = np.clip(miss_lambda, 0.05, 6.0)  
missed_payments = rng.poisson(miss_lambda)   
missed_payments[has_loan == 0] = 0             

# 8. Default-sannolikhet – bara för låntagare

risk_score = (                             
    -3.0
    + 2.6 * dti                            # högre DTI -> högre risk
    + 2.2 * (1 - credit_score / 850)       # lägre score -> högre risk
    + 0.6 * np.minimum(missed_payments, 5) # fler missar -> högre risk (mättas vid 5)
    - 0.6 * (years_with_bank / 30)         # längre relation -> lägre risk
    + 0.4 * (age < 23).astype(int)         # väldigt ung -> något högre risk
    + 0.3 * (age > 68).astype(int)         # hög ålder -> något högre risk
)
prob_default = expit(risk_score) *  has_loan    
                
# liten säkerhetsmarginal för numerik
eps = 1e-4                                
prob_default = (prob_default * (1 - 2 * eps) +  # "drar in" mot (0,1) intervallet
                eps * has_loan)

default = rng.binomial(1, prob_default)    # sampel: 1 om default inträffar, annars 0

# 9. Boendeform beroende på sysselsättning
# Sannorlikhet per status (hyresrätt, bostadsrätt, villa, inneboende)
p_student =  np.array([0.60, 0.10, 0.05, 0.25])
p_employed = np.array([0.40, 0.35, 0.20, 0.05])
p_retired =  np.array([0.35, 0.40, 0.20, 0.05])

P = (
    (is_student[:, None]  * p_student) +
    (is_employed[:, None] * p_employed) +
    (is_retired[:, None]  * p_retired)
)
#säkra upp normalisering)
P = P / P.sum(axis=1, keepdims=True)

u = rng.random(n)
cum = np.cumsum(P, axis=1) 
idx = (u[:, None] <= cum).argmax(axis=1)

housing_labels = np.array(["hyresrätt", "bostadsrätt", "villa", "inneboende"])
housing = housing_labels[idx]

# 10. Samla ihop allt i en DataFrame 
data = pd.DataFrame({                     
    "age": age,
    "years_with_bank": years_with_bank,
    "income": np.round(income, 2),         
    "credit_score": np.round(credit_score, 0).astype(int),
    "has_loan": has_loan,
    "loan_amount": np.round(loan_amount, 2),
    "dti": np.round(dti, 3),
    "missed_payments": missed_payments,
    "prob_default": np.round(prob_default, 4),
    "default": default,
    "employment_status": employment_status,
    "housing": housing,
    
})

# Validera åldersregeln
assert (data.loc[data.age >= 74, "has_loan"] == 0).all(), "Åldersregel bruten: någon >=74 har lån."
assert (data["credit_score"].between(300, 850)).all()
assert (data["years_with_bank"] <= (data["age"] - 18).clip(lower=0)).all()
assert (data["years_with_bank"] <= 40).all()
assert (data.loc[data.has_loan == 0, "missed_payments"].eq(0)).all()
assert set(data["employment_status"].unique()) <= {"student", "employed", "retired", "unknown"}
assert set(data["housing"].unique()) <= set(housing_labels)

if __name__ == "__main__":
    data.to_csv("synthetic_credit_data.csv", index=False)
    print(f"Antal rader: {len(data)}")
    print(f"Antal låntagare: {data.has_loan.sum()} ({100*data.has_loan.mean():.2f}%)")
    if data.has_loan.sum() > 0:
        print(f"Default rate (låntagare): {100*data.loc[data.has_loan==1,'default'].mean():.2f}%")
    print("CSV sparad som synthetic_credit_data.csv")