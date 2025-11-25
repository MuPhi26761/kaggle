import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates = ['Date'])
    store = pd.read_csv(STORE_PATH)
    
    # Merge avec la table "store.csv" qui contient des infos supplémentaires
    df = pd.merge(train,store, on = 'Store', how = 'left')
    
    # On retire les jours fermés + les jours où il y a 0 ventes
    df = df[(df['Open'] != 0) & (df['Sales'] > 0)]
    
    # Log tranform de la target
    df['log_sales'] = np.log1p(df['Sales'])
    return df

def time_features(df):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['week_of_the_year'] = df['Date'].dt.isocalendar().week
    dt['day_of_the_week'] = df['Date'].dt.dayofweek
    
    # Durée depuis le début des temps
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
    return df