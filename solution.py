"""
Traffic Demand Prediction - Hackathon Solution
Refactored, Leakage-Free, and Configurable Pipeline
With Robust Outlier Handling and Overfitting Prevention
Evaluation: score = max(0, 100 * r2_score(actual, predicted))
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 0. PIPELINE CONFIGURATION (defaults for original small dataset)
# ─────────────────────────────────────────────
N_FOLDS = 5
SMOOTHING_M = 10            # m-estimate smoothing parameter for target encoding
OUTLIER_TREATMENT = 'robust_loss'  # 'robust_loss' (absolute_error loss, dampens outlier influence), 'clip' (caps targets), 'none' (squared error)
ET_ESTIMATORS = 500         # Number of trees in ExtraTreesRegressor (reduce to 100 for faster dev)
TEMP_CLIP_MIN = -15.0       # Clip temperature outliers below this value
TEMP_CLIP_MAX = 45.0        # Clip temperature outliers above this value
USE_EXTRA_TREES = True      # Set to False to speed up training drastically


# ─────────────────────────────────────────────
# 1. TRAFFIC DEMAND PIPELINE CLASS
# ─────────────────────────────────────────────
class TrafficDemandPipeline:
    """
    A robust, leakage-free pipeline for traffic demand prediction.
    Features:
    - K-Fold Out-Of-Fold (OOF) Target Encoding with Bayesian/m-estimate smoothing.
    - Handling of unseen categories and locations on general test sets.
    - Robust timestamp parsing and temperature/categorical imputation.
    - Flexible outlier robustness control (none, robust loss, target clipping).
    - Ensembling via weighted OOF R² blending.
    """
    def __init__(self, n_splits=5, smoothing=10, outlier_treatment='robust_loss', 
                 outlier_pct=99.0, et_estimators=500, use_extra_trees=True,
                 temp_clip_min=-15.0, temp_clip_max=45.0, random_state=42):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.outlier_treatment = outlier_treatment
        self.outlier_pct = outlier_pct
        self.et_estimators = et_estimators
        self.use_extra_trees = use_extra_trees
        self.temp_clip_min = temp_clip_min
        self.temp_clip_max = temp_clip_max
        self.random_state = random_state
        
        self.target_encodings_stats = {}
        self.global_target_mean = 0.0
        self.global_target_std = 0.1
        self.global_target_median = 0.0
        self.temperature_global_median = 25.0
        
        self.encoders = {}
        self.feature_cols = []
        self.models_a = []
        self.models_b = []
        self.models_c = []
        self.blend_weights = []
        self.is_fitted = False

    def _get_target_encoding_features(self):
        """Specifies the spatial and spatiotemporal target encodings."""
        return [
            ('geo6', 'geo6'),
            ('geo5', 'geo5'),
            ('geo4', 'geo4'),
            ('geo3', 'geo3'),
            (['geo6', 'time_slot'], 'geo6_slot'),
            (['geo6', 'hour'], 'geo6_hour'),
            (['geo6', 'day'], 'geo6_day'),
            (['geo5', 'time_slot'], 'geo5_slot'),
            (['geo4', 'time_slot'], 'geo4_slot'),
            (['geo4', 'hour'], 'geo4_hour'),
            (['geo4', 'day'], 'geo4_day'),
            (['geo3', 'time_slot'], 'geo3_slot'),
            ('time_slot', 'slot'),
            ('hour', 'hour'),
            ('day', 'day'),
            (['RoadType_enc', 'time_slot'], 'road_slot'),
            (['RoadType_enc', 'hour'], 'road_hour'),
            (['Weather_enc', 'time_slot'], 'wx_slot'),
            (['NumberofLanes', 'time_slot'], 'lanes_slot'),
            (['geo4', 'RoadType_enc'], 'geo4_road'),
            (['geo4', 'NumberofLanes'], 'geo4_lanes')
        ]

    def engineer_features(self, df, is_train=True):
        """
        Parses timestamps, builds geohash prefixes, encodes categoricals,
        and handles missing values robustly for any train or test dataset.
        """
        df = df.copy()

        # 1. Robust Timestamp parsing
        if 'timestamp' in df.columns:
            import re
            dt = pd.to_datetime(df['timestamp'], errors='coerce')
            
            nan_mask = dt.isna()
            if nan_mask.any():
                # Fallback to regex extraction of HH:MM
                fallback_hours = []
                fallback_minutes = []
                for val in df.loc[nan_mask, 'timestamp'].astype(str):
                    match = re.search(r'(\d+):(\d+)', val)
                    if match:
                        fallback_hours.append(int(match.group(1)))
                        fallback_minutes.append(int(match.group(2)))
                    else:
                        fallback_hours.append(0)
                        fallback_minutes.append(0)
                
                parsed_hours = dt.dt.hour.values.copy()
                parsed_minutes = dt.dt.minute.values.copy()
                
                idx = 0
                for i, is_nan in enumerate(nan_mask):
                    if is_nan:
                        parsed_hours[i] = fallback_hours[idx]
                        parsed_minutes[i] = fallback_minutes[idx]
                        idx += 1
                
                df['hour'] = pd.Series(parsed_hours).fillna(0).astype(int).values
                df['minute'] = pd.Series(parsed_minutes).fillna(0).astype(int).values
            else:
                df['hour'] = dt.dt.hour.fillna(0).astype(int)
                df['minute'] = dt.dt.minute.fillna(0).astype(int)
        else:
            df['hour']   = 0
            df['minute'] = 0
            
        df['time_slot'] = df['hour'] * 4 + df['minute'] // 15    # 0–95

        # Cyclical time encoding
        df['hour_sin']  = np.sin(2 * np.pi * df['hour']      / 24)
        df['hour_cos']  = np.cos(2 * np.pi * df['hour']      / 24)
        df['slot_sin']  = np.sin(2 * np.pi * df['time_slot'] / 96)
        df['slot_cos']  = np.cos(2 * np.pi * df['time_slot'] / 96)

        # Day features
        if 'day' in df.columns:
            df['day_mod7']   = df['day'] % 7
            df['is_weekend'] = (df['day_mod7'] >= 5).astype(int)
        else:
            df['day']        = 0
            df['day_mod7']   = 0
            df['is_weekend'] = 0
            
        df['is_morning_peak']  = ((df['hour'] >= 7)  & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak']  = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_night']         = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_noon']          = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 5)  & (df['hour'] <= 7)).astype(int)

        # Geohash prefix hierarchy
        if 'geohash' in df.columns:
            df['geo3'] = df['geohash'].str[:3]
            df['geo4'] = df['geohash'].str[:4]
            df['geo5'] = df['geohash'].str[:5]
            df['geo6'] = df['geohash']
        else:
            for g in ['geo3', 'geo4', 'geo5', 'geo6', 'geohash']:
                df[g] = 'unknown'

        # Binary/indicator encodings
        if 'LargeVehicles' in df.columns:
            df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(float)
        else:
            df['LargeVehicles_enc'] = 0.0

        if 'Landmarks' in df.columns:
            df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(float)
        else:
            df['Landmarks_enc']     = 0.0

        # Road type ordinal
        if 'RoadType' in df.columns:
            road_map = {'Residential': 1.0, 'Street': 2.0, 'Highway': 3.0}
            df['RoadType_enc'] = df['RoadType'].map(road_map)
        else:
            df['RoadType_enc'] = np.nan
        df['RoadType_enc'] = df['RoadType_enc'].fillna(-1.0)

        # Weather ordinal
        if 'Weather' in df.columns:
            weather_map = {'Sunny': 1.0, 'Rainy': 2.0, 'Foggy': 3.0, 'Snowy': 4.0}
            df['Weather_enc'] = df['Weather'].map(weather_map)
        else:
            df['Weather_enc'] = np.nan
        df['Weather_enc'] = df['Weather_enc'].fillna(-1.0)

        # Temperature imputation and outlier clipping
        if 'Temperature' in df.columns:
            # Clip extreme temperature outliers (sensor faults, anomalies)
            df['Temperature'] = df['Temperature'].clip(lower=self.temp_clip_min, upper=self.temp_clip_max)
            if 'Weather' in df.columns and is_train:
                self.temperature_global_median = df['Temperature'].median()
            if 'Weather' in df.columns:
                df['Temperature'] = df.groupby('Weather')['Temperature'].transform(
                    lambda x: x.fillna(x.median())
                )
            df['Temperature'] = df['Temperature'].fillna(self.temperature_global_median)
        else:
            df['Temperature'] = self.temperature_global_median

        if 'NumberofLanes' in df.columns:
            df['NumberofLanes'] = df['NumberofLanes'].fillna(-1.0)
        else:
            df['NumberofLanes'] = -1.0

        # String columns Label/Ordinal Encoding
        str_cols = ['geo3', 'geo4', 'geo5', 'geo6', 'geohash']
        for col in str_cols:
            if col in df.columns:
                if is_train:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    df[col] = oe.fit_transform(df[[col]].astype(str))
                    self.encoders[col] = oe
                else:
                    oe = self.encoders.get(col)
                    if oe is not None:
                        df[col] = oe.transform(df[[col]].astype(str))
                    else:
                        df[col] = -1.0

        return df

    def fit_transform_target_encoding(self, df, target='demand'):
        """
        Computes target encodings using K-Fold Out-Of-Fold splits to prevent leakage.
        Applies Bayesian m-estimate smoothing to prevent overfitting on rare categories.
        Optimized for large datasets using numpy array pre-allocation.
        """
        df = df.copy()
        self.global_target_mean   = df[target].mean()
        self.global_target_std    = df[target].std()
        self.global_target_median = df[target].median()
        
        te_features = self._get_target_encoding_features()
        
        # Pre-allocate numpy arrays for all target encoding columns (FAST)
        n = len(df)
        te_arrays = {}
        for _, prefix in te_features:
            for stat in ['mean', 'std', 'median', 'count']:
                te_arrays[f'{prefix}_{stat}'] = np.full(n, np.nan, dtype=np.float64)
                
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(df)):
            print(f"    Target encoding fold {fold_num + 1}/{self.n_splits}...")
            df_train_fold = df.iloc[train_idx]
            df_val_fold   = df.iloc[val_idx]
            
            fold_global_mean = df_train_fold[target].mean()
            fold_global_std = df_train_fold[target].std()
            fold_global_median = df_train_fold[target].median()
            
            for group_cols, prefix in te_features:
                cols = group_cols if isinstance(group_cols, list) else [group_cols]
                
                # Verify group columns exist in training split
                if any(c not in df_train_fold.columns for c in cols):
                    continue
                
                # Compute stats on the training fold
                stats = df_train_fold.groupby(group_cols)[target].agg(['sum', 'count', 'std', 'median']).reset_index()
                
                # Compute Bayesian m-estimate mean
                stats['mean'] = (stats['sum'] + self.smoothing * fold_global_mean) / (stats['count'] + self.smoothing)
                stats = stats.drop(columns=['sum'])
                
                # Rename statistics columns
                stats.columns = cols + [f'{prefix}_count', f'{prefix}_std', f'{prefix}_median', f'{prefix}_mean']
                
                # Merge stats to the validation fold
                merged = df_val_fold[cols].merge(stats, on=group_cols, how='left')
                
                # Fill missing target encoding statistics with training fold priors
                merged[f'{prefix}_mean']   = merged[f'{prefix}_mean'].fillna(fold_global_mean)
                merged[f'{prefix}_std']    = merged[f'{prefix}_std'].fillna(fold_global_std)
                merged[f'{prefix}_median'] = merged[f'{prefix}_median'].fillna(fold_global_median)
                merged[f'{prefix}_count']  = merged[f'{prefix}_count'].fillna(0.0)
                
                # Assign to pre-allocated numpy arrays (FAST — avoids df.iloc bottleneck)
                for stat in ['mean', 'std', 'median', 'count']:
                    te_arrays[f'{prefix}_{stat}'][val_idx] = merged[f'{prefix}_{stat}'].values

        # Assign all target encoding columns to the dataframe at once
        for col_name, arr in te_arrays.items():
            df[col_name] = arr

        # Compute and save global stats using the entire training set for testing/transform time
        self.target_encodings_stats = {}
        for group_cols, prefix in te_features:
            cols = group_cols if isinstance(group_cols, list) else [group_cols]
            if any(c not in df.columns for c in cols):
                continue
                
            stats = df.groupby(group_cols)[target].agg(['sum', 'count', 'std', 'median']).reset_index()
            stats['mean'] = (stats['sum'] + self.smoothing * self.global_target_mean) / (stats['count'] + self.smoothing)
            stats = stats.drop(columns=['sum'])
            
            stats.columns = cols + [f'{prefix}_count', f'{prefix}_std', f'{prefix}_median', f'{prefix}_mean']
            
            self.target_encodings_stats[prefix] = {
                'group_cols': group_cols,
                'stats': stats,
                'global_std': self.global_target_std,
                'global_median': self.global_target_median
            }
            
        return df

    def transform_target_encoding(self, df):
        """
        Applies pre-computed global target encodings to the test/unseen dataset.
        Unseen categories fall back gracefully to training-set global averages.
        """
        df = df.copy()
        te_features = self._get_target_encoding_features()
        
        for _, prefix in te_features:
            if prefix not in self.target_encodings_stats:
                # Fallback if specific target encoding features are entirely missing
                for stat in ['mean', 'std', 'median', 'count']:
                    if stat == 'count':
                        df[f'{prefix}_{stat}'] = 0.0
                    elif stat == 'mean':
                        df[f'{prefix}_{stat}'] = self.global_target_mean
                    elif stat == 'std':
                        df[f'{prefix}_{stat}'] = self.global_target_std
                    else:
                        df[f'{prefix}_{stat}'] = self.global_target_median
                continue
                
            entry = self.target_encodings_stats[prefix]
            group_cols = entry['group_cols']
            stats = entry['stats']
            global_std = entry['global_std']
            global_median = entry['global_median']
            
            cols = group_cols if isinstance(group_cols, list) else [group_cols]
            
            # Merge stats to target dataframe
            merged = df[cols].merge(stats, on=group_cols, how='left')
            
            # Fill missing/unseen values with global priors
            merged[f'{prefix}_mean']   = merged[f'{prefix}_mean'].fillna(self.global_target_mean)
            merged[f'{prefix}_std']    = merged[f'{prefix}_std'].fillna(global_std)
            merged[f'{prefix}_median'] = merged[f'{prefix}_median'].fillna(global_median)
            merged[f'{prefix}_count']  = merged[f'{prefix}_count'].fillna(0.0)
            
            for stat in ['mean', 'std', 'median', 'count']:
                df[f'{prefix}_{stat}'] = merged[f'{prefix}_{stat}'].values
                
        return df

    def fit(self, df_train, target='demand'):
        """
        Preprocesses, target encodes, and trains the ensemble model with CV.
        Calculates blend weights based on leakage-free out-of-fold R² scores.
        Includes robust outlier handling and overfitting prevention.
        """
        print("Preprocessing training features...")
        df_train = self.engineer_features(df_train, is_train=True)
        
        print("Computing out-of-fold target encodings...")
        df_train = self.fit_transform_target_encoding(df_train, target=target)
        
        # Define features
        drop_cols = ['Index', 'demand', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'timestamp']
        self.feature_cols = [c for c in df_train.columns if c not in drop_cols and not c.startswith('demand')]
        print(f"Total features: {len(self.feature_cols)}")
        
        X_train = df_train[self.feature_cols].values.astype(np.float64)
        y_train = df_train[target].values.astype(np.float64)
        
        # Adaptive regularization: scale min_samples_leaf with dataset size
        # to prevent overfitting on large datasets
        n_samples = len(X_train)
        adaptive_min_leaf_a = max(20, int(n_samples * 0.0003))   # ~0.03% of data
        adaptive_min_leaf_b = max(30, int(n_samples * 0.0005))   # ~0.05% of data
        adaptive_min_leaf_c = max(5,  int(n_samples * 0.0001))   # ~0.01% of data
        print(f"  Adaptive min_samples_leaf: HGB-A={adaptive_min_leaf_a}, HGB-B={adaptive_min_leaf_b}, ExtraT={adaptive_min_leaf_c}")
        
        # Outlier handling on training targets
        y_fit = y_train.copy()
        if self.outlier_treatment == 'clip':
            upper_limit = np.percentile(y_train, self.outlier_pct)
            print(f"  Outlier treatment 'clip': capping target at {self.outlier_pct}th percentile ({upper_limit:.5f})")
            y_fit = np.clip(y_fit, 0.0, upper_limit)
        elif self.outlier_treatment == 'log_transform':
            print("  Outlier treatment 'log_transform': converting target to log1p(y)")
            y_fit = np.log1p(y_fit)
            
        hgb_loss = 'squared_error'
        if self.outlier_treatment == 'robust_loss':
            print("  Outlier treatment 'robust_loss': training HGB with 'absolute_error' loss")
            hgb_loss = 'absolute_error'
        else:
            print("  Outlier treatment 'none' / 'reduced_robustness': training with 'squared_error' loss")
            
        # Out-of-fold arrays
        oof_a = np.zeros(len(X_train))
        oof_b = np.zeros(len(X_train))
        oof_c = np.zeros(len(X_train))
        
        self.models_a = []
        self.models_b = []
        self.models_c = []
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        print(f"Training ensemble over {self.n_splits} folds...")
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr   = X_train[tr_idx], y_fit[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            # 1. Model A: HGBR (learning_rate=0.03, stronger regularization)
            m_a = HistGradientBoostingRegressor(
                loss=hgb_loss,
                learning_rate=0.03,
                max_iter=2000,
                max_leaf_nodes=127,
                min_samples_leaf=adaptive_min_leaf_a,
                l2_regularization=0.5,
                max_depth=8,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=42,
                verbose=0,
            )
            m_a.fit(X_tr, y_tr)
            pred_val_a = m_a.predict(X_val)
            if self.outlier_treatment == 'log_transform':
                pred_val_a = np.expm1(pred_val_a)
            oof_a[val_idx] = pred_val_a
            self.models_a.append(m_a)
            
            # 2. Model B: HGBR (learning_rate=0.01, deeper regularization)
            m_b = HistGradientBoostingRegressor(
                loss=hgb_loss,
                learning_rate=0.01,
                max_iter=5000,
                max_leaf_nodes=63,
                min_samples_leaf=adaptive_min_leaf_b,
                l2_regularization=1.0,
                max_depth=6,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=75,
                random_state=123,
                verbose=0,
            )
            m_b.fit(X_tr, y_tr)
            pred_val_b = m_b.predict(X_val)
            if self.outlier_treatment == 'log_transform':
                pred_val_b = np.expm1(pred_val_b)
            oof_b[val_idx] = pred_val_b
            self.models_b.append(m_b)
            
            # 3. Model C: ExtraTrees (imputed NaNs because ExtraTrees lacks native NaN support)
            if self.use_extra_trees:
                # Check for NaNs and fill for ExtraTrees safety
                X_tr_c  = np.nan_to_num(X_tr, nan=-1.0)
                X_val_c = np.nan_to_num(X_val, nan=-1.0)
                
                m_c = ExtraTreesRegressor(
                    n_estimators=self.et_estimators,
                    max_features=0.5,
                    min_samples_leaf=adaptive_min_leaf_c,
                    max_depth=20,
                    n_jobs=-1,
                    random_state=42,
                )
                m_c.fit(X_tr_c, y_tr)
                pred_val_c = m_c.predict(X_val_c)
                if self.outlier_treatment == 'log_transform':
                    pred_val_c = np.expm1(pred_val_c)
                oof_c[val_idx] = pred_val_c
                self.models_c.append(m_c)
                
            r2_a_fold = r2_score(y_val, oof_a[val_idx])
            r2_b_fold = r2_score(y_val, oof_b[val_idx])
            fold_msg = f"  Fold {fold+1} R²: HGB-A={r2_a_fold:.5f}, HGB-B={r2_b_fold:.5f}"
            if self.use_extra_trees:
                r2_c_fold = r2_score(y_val, oof_c[val_idx])
                fold_msg += f", ExtraT={r2_c_fold:.5f}"
            print(fold_msg)

        # Compute global OOF scores and blend weights
        r2_a = r2_score(y_train, oof_a)
        r2_b = r2_score(y_train, oof_b)
        
        models_perf = [
            (r2_a, 'HGB-A', oof_a),
            (r2_b, 'HGB-B', oof_b)
        ]
        if self.use_extra_trees:
            r2_c = r2_score(y_train, oof_c)
            models_perf.append((r2_c, 'ExtraT', oof_c))
            
        # Calculate blend weights based on positive R² scores
        total_r2 = sum(r for r, _, _ in models_perf if r > 0)
        if total_r2 > 0:
            self.blend_weights = [(r / total_r2 if r > 0 else 0.0) for r, _, _ in models_perf]
        else:
            self.blend_weights = [1.0 / len(models_perf)] * len(models_perf)
            
        print("\n--- Out-of-Fold R² Performance (Leakage-Free) ---")
        for (r2, name, _), w in zip(models_perf, self.blend_weights):
            print(f"  {name}: OOF R² = {r2:.5f} (Weight = {w:.3f})")
            
        oof_blend = np.zeros(len(y_train))
        for (_, _, oof), w in zip(models_perf, self.blend_weights):
            oof_blend += w * oof
            
        blend_r2 = r2_score(y_train, oof_blend)
        print(f"Blended OOF R² = {blend_r2:.5f}  ->  {max(0, 100 * blend_r2):.2f} / 100")
        
        self.is_fitted = True
        return blend_r2

    def predict(self, df_test):
        """
        Applies preprocessing and target encoding to testing set, and
        computes ensembled predictions using OOF blend weights.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before running prediction.")
            
        df_test = self.engineer_features(df_test, is_train=False)
        df_test = self.transform_target_encoding(df_test)
        
        X_test = df_test[self.feature_cols].values.astype(np.float64)
        
        pred_blend = np.zeros(len(X_test))
        
        models_list = [
            (self.models_a, 'HGB-A'),
            (self.models_b, 'HGB-B')
        ]
        if self.use_extra_trees:
            models_list.append((self.models_c, 'ExtraT'))
            
        for (models, name), w in zip(models_list, self.blend_weights):
            pred_model = np.zeros(len(X_test))
            # Cleanly impute NaNs for ExtraTrees if any exist
            X_test_model = np.nan_to_num(X_test, nan=-1.0) if name == 'ExtraT' else X_test
            
            for m in models:
                pred_fold = m.predict(X_test_model)
                if self.outlier_treatment == 'log_transform':
                    pred_fold = np.expm1(pred_fold)
                pred_model += pred_fold / self.n_splits
            pred_blend += w * pred_model
            
        # Target variable demand is strictly bounded between [0.0, 1.0]
        pred_blend = np.clip(pred_blend, 0.0, 1.0)
        return pred_blend


# ─────────────────────────────────────────────
# 2. MAIN CLI SCRIPT RUNNER
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traffic Demand Prediction Pipeline")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Directory containing train.csv and test.csv")
    parser.add_argument('--folds', type=int, default=N_FOLDS,
                        help=f"Number of CV folds (default: {N_FOLDS})")
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help="Fraction of training data to use (default: 1.0 = all)")
    parser.add_argument('--no-extra-trees', action='store_true',
                        help="Disable ExtraTreesRegressor (saves memory on large datasets)")
    parser.add_argument('--outlier-treatment', type=str, default=OUTLIER_TREATMENT,
                        choices=['none', 'robust_loss', 'clip', 'log_transform'],
                        help=f"Outlier treatment strategy (default: {OUTLIER_TREATMENT})")
    parser.add_argument('--smoothing', type=int, default=SMOOTHING_M,
                        help=f"Target encoding smoothing parameter (default: {SMOOTHING_M})")
    parser.add_argument('--et-estimators', type=int, default=ET_ESTIMATORS,
                        help=f"Number of ExtraTrees estimators (default: {ET_ESTIMATORS})")
    args = parser.parse_args()
    
    # Auto-detect data directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.data_dir:
        DATA_DIR = args.data_dir
    else:
        CANDIDATE_DIRS = [
            os.path.join(SCRIPT_DIR, 'dataset'),
            os.path.join(SCRIPT_DIR, '..', 'dataset'),
            SCRIPT_DIR,
            os.getcwd(),
        ]
        DATA_DIR = None
        for candidate in CANDIDATE_DIRS:
            if os.path.exists(os.path.join(candidate, 'train.csv')):
                DATA_DIR = candidate
                break
            
    if DATA_DIR is None or not os.path.exists(os.path.join(DATA_DIR, 'train.csv')):
        raise FileNotFoundError(
            "Could not locate train.csv. Use --data-dir to specify the dataset directory."
        )
        
    print(f"Loading datasets from: {os.path.abspath(DATA_DIR)}")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    print(f"Train size: {train.shape}, Test size: {test.shape}")
    
    # Subsample training data if requested (for faster dev iterations)
    if args.sample_rate < 1.0:
        n_sample = int(len(train) * args.sample_rate)
        train = train.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"Subsampled training data to {len(train):,} rows ({args.sample_rate:.0%})")
    
    use_et = USE_EXTRA_TREES and (not args.no_extra_trees)
    
    # Initialize and train pipeline
    pipeline = TrafficDemandPipeline(
        n_splits=args.folds,
        smoothing=args.smoothing,
        outlier_treatment=args.outlier_treatment,
        et_estimators=args.et_estimators,
        use_extra_trees=use_et,
        temp_clip_min=TEMP_CLIP_MIN,
        temp_clip_max=TEMP_CLIP_MAX
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  Outlier Treatment: {args.outlier_treatment}")
    print(f"  Temperature Clipping: [{TEMP_CLIP_MIN}, {TEMP_CLIP_MAX}]°C")
    print(f"  CV Folds: {args.folds}")
    print(f"  ExtraTrees: {'Enabled' if use_et else 'Disabled'}")
    print(f"  Smoothing: {args.smoothing}")
    
    print("\nFitting Traffic Demand Pipeline...")
    blend_r2 = pipeline.fit(train, target='demand')
    
    print("\nGenerating predictions for test set...")
    predictions = pipeline.predict(test)
    
    # Save predictions
    submission = pd.DataFrame({'Index': test['Index'].values, 'demand': predictions})
    out_path = os.path.join(SCRIPT_DIR, 'submission.csv')
    submission.to_csv(out_path, index=False)
    
    print(f"\n[SUCCESS] Submission saved -> {out_path}")
    print(submission.head(10).to_string())
    print(f"\n[SCORE] Predicted hackathon score = {max(0, 100 * blend_r2):.2f} / 100")
