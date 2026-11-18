_# Smart Energy Consumption Forecasting Using Machine Learning

A machine learning project that forecasts household electricity consumption for the **next hour** and the **next day** using historical usage data.

## Dataset

- **Source:** [Individual Household Electric Power Consumption (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Content:** One-minute readings from a single household, December 2006 to November 2010 (about 2.07 million records)
- **Target variable:** `Global_active_power` (kW), aggregated to hourly values
- The raw dataset is not stored in this repository because of its size. The notebook downloads it automatically.

## Project Workflow

1. Data collection and understanding
2. Data cleaning and preprocessing (datetime conversion, chronological sorting, missing-value handling, hourly resampling)
3. Outlier detection (IQR method; outliers were kept because they represent genuine peak demand)
4. Exploratory data analysis (daily, weekly, monthly and weekday/weekend patterns)
5. Feature engineering (time features, lag features, rolling averages)
6. Model training with a chronological train/test split (train: 2007-2009, test: 2010)
7. Model evaluation and comparison
8. Next-hour and next-day forecasting
9. Insights and recommendations

## Feature Engineering

- **Time features:** Hour, Day, Day of week, Month, Year, Week of year, Weekend indicator
- **Lag features:** previous hour, previous day, previous week
- **Rolling averages:** 24-hour and 168-hour (7-day) averages
- **Aggregates:** previous day average, previous month average
- All lag and rolling features use only past data to avoid data leakage.

## Model Comparison (Next-Hour Forecast, Test Year 2010)

| Model | MAE | RMSE | R2 | MAPE (%) |
|---|---|---|---|---|
| **Gradient Boosting** | **0.3461** | **0.4946** | **0.5937** | 44.27 |
| Random Forest | 0.3503 | 0.5048 | 0.5769 | 45.18 |
| Decision Tree | 0.3601 | 0.5235 | 0.5449 | 45.24 |
| Linear Regression | 0.3793 | 0.5330 | 0.5282 | 49.44 |
| Baseline (previous hour) | 0.4057 | 0.6168 | 0.3682 | 45.19 |

Gradient Boosting performed best and reduced RMSE by about 20% compared with the baseline. MAPE is high for all models because very low night-time consumption values inflate percentage errors, so MAE and RMSE are the more reliable metrics here.

## Next-Day Forecast

A separate day-ahead model uses only information available before the forecast day starts (lags of 24 hours or more).

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline (same hour yesterday) | 0.5575 | 0.8241 | -0.1300 |
| **Gradient Boosting (day-ahead)** | **0.4480** | **0.6126** | **0.3755** |

The average daily consumption is predicted with a MAE of 0.18 kW.

## Key Findings

- Consumption peaks in the evening around 8 PM, with a smaller morning peak around 7-8 AM.
- Weekend consumption (1.233 kW) is about 19% higher than weekday consumption (1.036 kW). Saturday is the highest-usage day.
- December is the highest-usage month and August the lowest, showing a clear winter-high, summer-low seasonal pattern.
- The previous hour's consumption is the most important predictor (77% importance), followed by hour of day and consumption one week and one day earlier.

## Recommendations

- **Households and businesses:** shift flexible loads (washing machine, dishwasher, water heater) away from the 7-9 PM peak; reduce winter heating load through insulation and thermostat scheduling.
- **Electricity providers:** use hour-ahead and day-ahead forecasts for generation planning; consider time-of-use pricing to flatten evening peaks.

## Visualizations

| File | Description |
|---|---|
| `boxplot_outliers.png` | Outlier detection |
| `monthly_trend.png` | Monthly average consumption, 2006-2010 |
| `hourly_weekday_weekend.png` | Hourly pattern, weekday vs weekend |
| `histogram_distribution.png` | Distribution of hourly consumption |
| `barchart_day_of_week.png` | Average consumption by day of week |
| `barchart_monthly_average.png` | Average consumption by month |
| `heatmap_hour_day.png` | Day of week vs hour heatmap |
| `actual_vs_predicted.png` | Actual vs predicted consumption |
| `feature_importance.png` | Feature importance of the best model |
| `next_day_forecast.png` | Next-day forecast vs actual |

## Repository Contents

| File | Description |
|---|---|
| `Energy_forecasting.ipynb` | Complete notebook: cleaning, EDA, feature engineering, modelling, forecasting |
| `best_model_gradient_boosting.pkl` | Trained next-hour model |
| `feature_list.pkl` | Feature names expected by the next-hour model |
| `day_ahead_model.pkl` | Trained next-day model |
| `day_ahead_features.pkl` | Feature names expected by the next-day model |
| `hourly_clean.csv` | Cleaned hourly data |
| `features.csv` | Engineered feature table |
| `model_comparison.csv` | Model comparison results |

## How to Run

1. Open `Energy_forecasting.ipynb` in Google Colab or Jupyter Notebook.
2. Install the requirements:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Run all cells from top to bottom. The first cell downloads the dataset from UCI.

## How to Use the Trained Model

```python
import joblib
import pandas as pd

model = joblib.load("best_model_gradient_boosting.pkl")
features = joblib.load("feature_list.pkl")

# One row containing every feature listed in feature_list.pkl
# (see predict_next_hour() in the notebook for how each value is built)
X = pd.DataFrame([{
    "Hour": 20, "Day": 25, "DayOfWeek": 3, "Month": 11, "WeekOfYear": 47,
    "Is_Weekend": 0, "Lag_1h": 2.1, "Lag_24h": 2.3, "Lag_168h": 2.0,
    "Rolling_Mean_24h": 1.2, "Rolling_Mean_168h": 1.1,
    "Prev_Day_Avg": 1.2, "Prev_Month_Avg": 1.0,
}])[features]

print("Predicted consumption (kW):", model.predict(X)[0])
```

The notebook also contains ready-to-use functions `predict_next_hour(timestamp)` and `predict_next_day(date)`.

The model files were created with scikit-learn. Load them with the same scikit-learn version used for training, otherwise you may see version warnings.

## Limitations and Future Improvements

- Data comes from a single household in France, so results may not generalise to other homes.
- Weather data (temperature, humidity) was not used.
- Next-day accuracy (R2 0.38) is lower than next-hour accuracy (R2 0.59), because the most recent hours are unknown.
- Possible improvements: SARIMA/Prophet, LSTM networks, weather features, and a Streamlit dashboard.

## Author

 Asma kausar - Machine Learning Internship Project
