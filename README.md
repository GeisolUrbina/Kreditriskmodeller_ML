
# Kreditriskmodeller med maskininlärning
Detta projekt utforskar konstruktionen och utvärderingen av maskininlärningsmodeller för att bedöma kreditrisk. Syftet är att skapa en modell som kan förutsäga sannolikheten för en låntagares "default" baserat på en uppsättning syntetiska kreditdata.

## Projektets struktur
Projektet är organiserat i flera Jupyter Notebooks för att ge en logisk och stegvis genomgång av analysen.

1. [**Exploratory_Data_Analysis.ipynb**](Exploratory_Data_Analysis.ipynb): En detaljerad Exploratory Data Analysis (EDA) som undersöker datamängdens egenskaper, identifierar trender och förbereder data för modellering.

2. Feature_Engineering_and_Modeling.ipynb: Här skapas nya funktioner från rådata och flera maskininlärningsmodeller tränas och utvärderas. Fokus ligger på att jämföra olika modellers prestanda.

3. Model_Evaluation_and_Interpretation.ipynb: Denna notebook går djupare in på utvärderingen av den valda modellen, inklusive tolkning av dess beslut och prestanda mot olika utvärderingsmått.

multicollinearity_analysis.ipynb: Ett specifikt dokument som behandlar frågan om multikollinearitet i de syntetiska data. Det inkluderar en implementering av Variance Inflation Factor (VIF) för att identifiera korrelerade variabler.

Datamängd
Projektet använder en syntetisk datamängd, synthetic_credit_data.csv, som genererats för detta specifika projekt. Datamängden innehåller fiktiva kunddata som ålder, inkomst, kredithistorik och andra relevanta variabler som används för att förutsäga kreditrisk.
