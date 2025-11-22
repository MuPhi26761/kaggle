import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_ids = test['Id']

# Delete outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

y_train = np.log1p(train['SalePrice'])
train.drop(['SalePrice', 'Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)

"""# Feature Engineering + Cleaning"""

# Fill NA Values
cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in cols_none:
  all_data[col] = all_data[col].fillna('None')


# We replace by median or average depending on the case
all_data['LotFrontage'] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

categorical_cols = all_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# New features

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data = pd.get_dummies(all_data)

# Separation of train/test

X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

"""# XGBoost + Lasso"""

model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    random_state=7,
    n_jobs=-1
)

lasso = make_pipeline(
    SimpleImputer(strategy='mean'), # Remplace les trous par la moyenne
    RobustScaler(),                 # Gère les outliers
    Lasso(alpha=0.0005, random_state=1)
)

"""# Cross-validation"""

def rmsle_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

score = rmsle_cv(model_xgb)
print(f"\nScore XGBoost (CV RMSLE) : {score.mean():.4f} (std: {score.std():.4f})")

# Cross-validation of Lasso 
score_lasso = rmsle_cv(lasso)
print(f"\nScore Lasso (CV RMSLE) : {score_lasso.mean():.4f} (std: {score_lasso.std():.4f})")

print("Lasso training")
lasso.fit(X_train, y_train)

print("XGBoost training")
model_xgb.fit(X_train, y_train)

# Predictions
lasso_pred = lasso.predict(X_test)
xgb_pred = model_xgb.predict(X_test)


# FINAL RESULT
final_pred = (0.6 * lasso_pred) + (0.4 * xgb_pred)

"""# Submission file"""

submission = pd.DataFrame()
submission['Id'] = test_ids
submission['SalePrice'] = final_pred
submission.to_csv('submission_ensemble_Lasso_XGB.cs', index=False)

print("File 'submission_ensemble_Lasso_XGB.csv' generated")