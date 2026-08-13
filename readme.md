# Body Fat Percentage Predictor

A machine learning project that predicts body fat percentage from simple body circumference measurements, with a local web app interface. Trained separately for male and female subjects using real density-based body fat measurements as ground truth.

---

## Overview

Instead of using the US Navy circumference formula directly, a **Linear Regression model** is trained on actual body density measurements (converted to body fat % via Siri's equation — the same method used as a reference standard in clinical settings), giving a data-driven prediction calibrated on real measurements.

Two separate models are trained — one for males, one for females — because body fat distribution differs significantly between sexes, and a combined model washes out within-group signal (Simpson's paradox observed during correlation analysis).

A **US Navy formula** column is also computed and included for comparison to assess how well the formula tracks the measured values (consistent overestimation bias observed, worse for females than males).

---

## Dataset

**Source:** [Body Fat Extended Dataset — simonezappatini (Kaggle)](https://www.kaggle.com/datasets/simonezappatini/body-fat-extended-dataset)

- 436 subjects (252 male, 184 female)
- Body fat % derived from underwater weighing (hydrostatic weighing) via Siri's equation
- Measurements: Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist

> `Density` is excluded from features — it is a 1:1 transform of the target via Siri's equation (pure leakage). `NavyBodyFat` is also excluded from features — it is derived from the same input measurements (circular). Both columns are retained in the engineered CSV for reference only.

---

## Project Structure

```text
bodyfat-analytics/
├── data/
│   ├── raw/                        # Raw downloaded Kaggle dataset
│   └── processed/                  # Cleaned datasets split by sex (anomalies removed)
├── notebooks/
│   └── EDA_&_FE.ipynb  # Full EDA, IQR outlier removal, and correlation analysis
├── src/
│   ├── data_ingestion.py           # Kaggle API download script
│   └── train_models.py             # Model evaluation and pipeline export script
├── models/
│   ├── model_male.pkl              # Final trained pipeline for males
│   └── model_female.pkl            # Final trained pipeline for females
├── web_app/
│   ├── app.py                      # Flask web application backend
│   └── templates/
│       └── index.html              # Frontend UI
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Methodology

### Preprocessing (`notebooks/01_eda_and_preprocessing.ipynb`)

- Downloads dataset via `src/data_ingestion.py` (uses Kaggle API, saves to `data/raw/`)
- Drops the "in original dataset" indicator column (no predictive value)
- Fixes unit inconsistency: `Height` was stored in meters, converted to cm for consistency with circumference columns
- Applies Interquartile Range (IQR) filtering to remove extreme physical measurement anomalies.
- Adds `NavyBodyFat` using the US Navy circumference formula:
  - **Male:** `86.010 × log10(Abdomen − Neck) − 70.041 × log10(Height) + 36.76`
  - **Female:** `163.205 × log10(Waist + Hip − Neck) − 97.684 × log10(Height) − 78.387`
  - All measurements converted to inches before applying the formula
- Splits and exports cleaned subsets to `data/processed/`.

### Feature Selection

Pearson correlation with `BodyFat` was computed overall and split by sex. Key findings:

- All-rows correlations were misleadingly weak (Abdomen: +0.36 overall vs +0.81 male-only, +0.74 female-only) — confirming sex-stratified modelling is necessary
- `Age` showed the largest sex difference: +0.29 for males, −0.06 for females
- `Height` was essentially useless for both sexes (r < 0.1)

**Final feature sets — 6 measurements each, practical to take at home with a tape measure and scale:**

| Sex    | Features |
|--------|----------|
| Male   | Abdomen, Chest, Hip, Weight, Thigh, Neck |
| Female | Abdomen, Hip, Weight, Biceps, Thigh, Forearm |

Reducing from 13 to 6 features cost less than 0.002 R² — effectively zero accuracy loss.

### Model Selection

Multiple algorithms (Linear Regression, Random Forest Regressor, SVR) were evaluated to mitigate the risk of overfitting on a small dataset. 

**→ Plain LinearRegression was selected as the best performing model for both cohorts.**

**Final performance (5-fold CV):**

| Sex    | R²     | MAE    | RMSE   |
|--------|--------|--------|--------|
| Male   | 0.6977 | 3.5681 | 4.3619 |
| Female | 0.5493 | 2.9189 | 3.7108 |

### Overfitting / Underfitting Check

Learning curves were generated (training R² vs CV R² across training sizes):

- **Male:** Curves converge cleanly → good fit
- **Female:** CV curve slightly still rising at dataset limit → mild underfitting from limited data, not overfitting

Neither model overfits. Female model variance is higher due to smaller sample size and more complex fat distribution patterns.

---

## Running the App

```bash
pip install -r "requirements.txt"

# Step 1: download dataset (requires Kaggle API token in ~/.kaggle/kaggle.json)
python src/data_ingestion.py

#run preprocessing notebook to generate processed data
#open notebooks/EDA_&_FE.ipynb 

#evaluate models and save .pkl files
python src/train_models.py

#launch web app (opens browser automatically)
python web_app/app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### App Features

- Male / Female toggle — swaps input fields and loads the correct model
- 6 number inputs per sex with sensible defaults
- Predicts body fat % with a reference category (Essential / Athletic / Fitness / Average / Above average)
- **Lean mass & goal calculator** — appears after prediction, shows fat mass and lean mass, lets you type a target body fat % and calculates exactly how many kg of fat to gain or lose:

```text
lean_mass   = weight × (1 − bf% / 100)
goal_weight = lean_mass / (1 − target_bf% / 100)
fat_change  = goal_weight − current_weight
```

---

## Dependencies

```text
pandas
numpy
scikit-learn
flask
matplotlib
kagglehub
```

## Disclaimer

Personal ML project. Predictions are estimates based on circumference measurements and should not be used as a substitute for clinical body composition assessment.