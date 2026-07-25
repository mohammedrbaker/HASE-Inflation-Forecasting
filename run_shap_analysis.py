# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from models import HASE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

XAI_DIR = 'results/4_XAI'
os.makedirs(XAI_DIR, exist_ok=True)

MODELS_TO_EXPLAIN = ['HASE', 'ElasticNet', 'RandomForest', 'XGBoost', 'LightGBM']


def generate_shap_for_model(model, model_name, X_train, X_test, horizon, output_dir):
    try:
        print(f"      -> {model_name}...", end='', flush=True)
        if any(t in model_name for t in ['XGBoost', 'LightGBM', 'RandomForest']):
            shap_vals = shap.TreeExplainer(model).shap_values(X_test)
        elif model_name == 'ElasticNet':
            shap_vals = shap.LinearExplainer(model, X_train).shap_values(X_test)
        elif model_name == 'HASE':
            if hasattr(model, 'base_models_') and 'EN' in model.base_models_:
                shap_vals = shap.LinearExplainer(
                    model.base_models_['EN'], X_train).shap_values(X_test)
            else:
                print(" (skipped)"); return
        else:
            print(" (skipped)"); return

        model_dir = os.path.join(output_dir, f'H{horizon}')
        os.makedirs(model_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, show=False)
        plt.title(f'SHAP Summary — {model_name} (H={horizon})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{model_name}_summary.png'),
                    dpi=300, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, plot_type='bar', show=False)
        plt.title(f'Feature Importance — {model_name} (H={horizon})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{model_name}_bar.png'),
                    dpi=300, bbox_inches='tight'); plt.close()

        pd.DataFrame({'Feature': X_test.columns,
                      'Mean_AbsSHAP': np.abs(shap_vals).mean(axis=0)
                      }).sort_values('Mean_AbsSHAP', ascending=False
                      ).to_csv(os.path.join(model_dir, f'{model_name}_importance.csv'),
                               index=False)
        print(" [OK]")
    except Exception as e:
        print(f" [FAIL] {e}")


def run_shap_analysis():
    df_raw   = load_macro_dataset(use_realtime=True, publication_delays=True)
    df_feats = create_features(df_raw)

    for H in [1, 3, 6, 12]:
        print(f"\n  H={H}")
        X, y        = create_supervised_dataset(df_feats, 'Inflation_Rate', horizon=H)
        X_train     = X[X.index <  '2013-01-01']
        y_train     = y[y.index <  '2013-01-01']
        X_test      = X[X.index >= '2013-01-01']
        selector    = MutualInfoFeatureSelector(k=25)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel  = selector.transform(X_test)

        for model_name in MODELS_TO_EXPLAIN:
            if model_name == 'HASE':
                model = HASE(horizon=H, random_state=42)
            elif model_name == 'ElasticNet':
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elif model_name == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                              random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                     random_state=42, n_jobs=-1, verbosity=0)
            elif model_name == 'LightGBM':
                model = LGBMRegressor(n_estimators=100, max_depth=-1, learning_rate=0.1,
                                      random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train_sel, y_train)
            generate_shap_for_model(model, model_name, X_train_sel, X_test_sel, H, XAI_DIR)

    print(f"\nDone. Output saved to: {XAI_DIR}/")


if __name__ == "__main__":
    run_shap_analysis()
