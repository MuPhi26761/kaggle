import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc # to free the memory
from sklearn.metrics import roc_auc_score

# ==========================================
# LOAD THE DATA
# ==========================================
train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')

# IDs
train_ids = train['SK_ID_CURR']
test_ids = test['SK_ID_CURR']
y = train['TARGET'] 

# Delete target and IDs for training
train = train.drop(columns=['TARGET', 'SK_ID_CURR'])
test = test.drop(columns=['SK_ID_CURR'])

# ==========================================
# FEATURE ENGINEERING
# ==========================================

for df in [train,test]:
    # Pourcentage du crédit par rapport aux revenus
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # Pourcentage de l'annuité (mensualité) par rapport aux revenus
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # Durée du crédit en années (approximatif)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # Prix des biens vs Montant du crédit (Est-ce qu'on finance 100% ?)
    df['GOODS_LOAN_PERCENT'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    # Nettoyage basique des anomalies connues dans ce dataset
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)


# ==========================================
# ENCODING OF CATEGORIES
# ==========================================

cat_cols = train.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# LIGHTGBM + CROSS-VALIDATION
# ==========================================

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

# Matrix that stores predictions on the test set (average of 5 folds)
predictions_test = np.zeros(test.shape[0]) 

# Matrix that stores prediction on the train set (to calculate OOF score)
oof_preds = np.zeros(train.shape[0])

# Hyperparameters
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 10000, # early_stopping will stop before
    'learning_rate': 0.02,
    'num_leaves': 34,
    'colsample_bytree': 0.9497,
    'subsample': 0.8716,
    'max_depth': 8,
    'reg_alpha': 0.0415,
    'reg_lambda': 0.0735,
    'min_split_gain': 0.0222,
    'min_child_weight': 39.3259,
    'silent': -1,
    'verbose': -1,
    'metric': 'auc',
    'n_jobs': -1
}

# Training for on the 5 folds

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train, y)):
    train_x, train_y = train.iloc[train_idx], y.iloc[train_idx]
    valid_x, valid_y = train.iloc[valid_idx], y.iloc[valid_idx]

    # LightGBM datasets
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    clf = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )

    # Predictions on validation fold
    oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    # Predictions on the real test set
    predictions_test += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

    # Memory cleaning
    del train_x, train_y, valid_x, valid_y, dtrain, dvalid
    gc.collect()

    print(f"\nScore AUC Global (Out-Of-Fold) : {roc_auc_score(y, oof_preds):.5f}")


# ==========================================
# SUBMISSION
# ==========================================

submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_ids
submission['TARGET'] = predictions_test

submission.to_csv('submission_lightgbm_homecredit.csv', index=False)
print("File 'submission_lightgbm_homecredit.csv' generated !")