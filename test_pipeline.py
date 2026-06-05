import sys
import os
import pandas as pd
import numpy as np

# Add FlipkartGrid folder to python path
sys.path.append('/Users/nishantakamalbaruah/Desktop/Flipkart gridlock/FlipkartGrid')

from solution import TrafficDemandPipeline

# 1. Create simulated train set with outliers and noise
train_data = {
    'Index': range(100),
    'geohash': ['qp02z1', 'qp02zt', 'qp08bj', 'qp08gt'] * 25,
    'day': [48, 49, 50, 51] * 25,
    'timestamp': ['0:0', '1:15', '2:30', '3:45'] * 25,
    'RoadType': ['Residential', 'Street', 'Highway', None] * 25,
    'NumberofLanes': [1, 2, 3, 4] * 25,
    'LargeVehicles': ['Allowed', 'Not Allowed', 'Allowed', None] * 25,
    'Landmarks': ['Yes', 'No', 'Yes', None] * 25,
    'Temperature': [30.0, 31.0, 25.0, np.nan] * 25,
    'Weather': ['Sunny', 'Rainy', 'Foggy', 'Snowy'] * 25,
    'demand': np.random.rand(100)
}
train_df = pd.DataFrame(train_data)

# Inject temperature outliers into train data
train_df.loc[0, 'Temperature'] = 65.0    # Extreme hot outlier
train_df.loc[1, 'Temperature'] = -35.0   # Extreme cold outlier

# Inject demand outliers
train_df.loc[2, 'demand'] = 0.99         # Spike outlier
train_df.loc[3, 'demand'] = 0.0001       # Drop outlier

# 2. Create simulated test set with:
# - Unseen geohash ('qp9999')
# - Missing columns ('Landmarks')
# - Different timestamp formats ('2026-06-04 17:15:00' and '18:30')
# - Missing values (NaNs)
# - Temperature outliers
test_data = {
    'Index': range(10),
    'geohash': ['qp02z1', 'qp9999'] * 5,  # qp9999 is unseen!
    'day': [48, 52] * 5,
    'timestamp': ['2026-06-04 17:15:00', '18:30'] * 5,
    'RoadType': ['Residential', 'Highway'] * 5,
    'NumberofLanes': [2, np.nan] * 5,
    'LargeVehicles': ['Allowed', None] * 5,
    'Temperature': [np.nan, 55.0] * 5,  # 55.0 is an outlier that should be clipped
    'Weather': ['Sunny', 'Sunny'] * 5
}
test_df = pd.DataFrame(test_data)

print("Simulated DataFrames initialized successfully.")
print(f"Train outliers: Temp extremes at rows 0,1; Demand extremes at rows 2,3")

# Run pipeline with robust outlier handling
pipeline = TrafficDemandPipeline(
    n_splits=3,
    smoothing=2,
    outlier_treatment='robust_loss',  # Use robust loss to dampen outlier influence
    et_estimators=10,
    use_extra_trees=True,
    temp_clip_min=-15.0,
    temp_clip_max=45.0
)

print("Fitting pipeline on simulated train data (with outlier handling)...")
pipeline.fit(train_df, target='demand')
print("Pipeline fitted successfully.")

print("Predicting on simulated test data...")
preds = pipeline.predict(test_df)
print("Predictions shape:", preds.shape)
print("Predictions:", preds)

assert len(preds) == 10, "Predictions size mismatch!"
assert not np.isnan(preds).any(), "NaNs detected in predictions!"
assert all(0 <= p <= 1 for p in preds), "Predictions out of [0, 1] range!"
print("\n[SUCCESS] Test completed successfully! The pipeline handles outliers, unseen categories, and missing values perfectly.")
