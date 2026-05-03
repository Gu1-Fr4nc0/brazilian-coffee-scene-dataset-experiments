import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

def train_ml_traditional():
    df_folds = pd.read_csv('/content/crop-image-classification/data/folds_coffee_dataset.csv')
    df_feat = pd.read_csv('/content/crop-image-classification/data/coffee_meta_features.csv')
    
    all_performances = []
    
    for seed in range(0, 30):
        print(f" * Running Random Forest for seed = {seed}")
        df_seed = df_folds[df_folds['Seed'] == seed]
        df_train = df_seed[df_seed["Fold"] == "Training"]
        df_test = df_seed[df_seed["Fold"] == "Test"]
        
        def get_features_labels(df_subset):
            feats, labels = [], []
            for _, row in df_subset.iterrows():
                nome_arquivo = os.path.basename(row["im_path"])
                feat_row = df_feat[df_feat['Name_file'] == nome_arquivo]
                if not feat_row.empty:
                    feats.append(feat_row.drop(columns=['Name_file']).values[0])
                    labels.append(row["Y"])
            return np.nan_to_num(np.array(feats), nan=0.0), np.array(labels)

        X_train, y_train = get_features_labels(df_train)
        X_test, y_test = get_features_labels(df_test)
        
        if len(X_train) == 0: continue

        # Classificador Tradicional
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        all_performances.append([accuracy_score(y_test, preds), f1_score(y_test, preds), seed])
        
    return pd.DataFrame(all_performances, columns=["acc", "f1s", "seed"])

if __name__ == "__main__":
    results = train_ml_traditional()
    results.to_csv("/content/crop-image-classification/results/performances_ml.csv", index=False)
    print("Treino Machine Learning Clássico finalizado e salvo!")