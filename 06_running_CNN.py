import os 
import pandas as pd
import numpy as np
import keras
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score

def get_custom_cnn(input_shape=(64,64,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPool2D((2,2)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def read_images(df, seed):
    df_seed = df.loc[df['Seed'] == seed]
    df_training = df_seed[df_seed["Fold"] == "Training"]
    df_testing = df_seed[df_seed["Fold"] == "Test"]
    
    def load_imgs(df_subset):
        images, labels = [], []
        base_path = '/content/crop-image-classification/dataset-brazilian_coffee_scenes/images'
        for _, row in df_subset.iterrows():
            nome_arquivo = os.path.basename(row["im_path"])
            caminho_absoluto = os.path.join(base_path, nome_arquivo)
            if os.path.exists(caminho_absoluto):
                images.append(np.array(Image.open(caminho_absoluto).resize((64, 64))) / 255.0)
                labels.append(row["Y"])
        return np.array(images), np.array(labels)

    t_img, t_lab = load_imgs(df_training)
    v_img, v_lab = load_imgs(df_testing)
    return t_img, t_lab, v_img, v_lab

def train_cnn(df):
    all_performances = []
    for seed in range(0, 30):
        print(f" * Running Custom CNN for seed = {seed}")
        t_img, t_lab, v_img, v_lab = read_images(df, seed)
        
        if len(t_img) == 0: continue

        model = get_custom_cnn((64,64,3))
        model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
        
        model.fit(t_img, t_lab, epochs=50, batch_size=16, verbose=0)
        
        preds = np.round(model.predict(v_img, verbose=0))
        all_performances.append([accuracy_score(v_lab, preds), f1_score(v_lab, preds), seed])
        
    return pd.DataFrame(all_performances, columns=["acc", "f1s", "seed"])

if __name__ == "__main__":
    df = pd.read_csv('/content/crop-image-classification/data/folds_coffee_dataset.csv')
    results = train_cnn(df)
    results.to_csv("/content/crop-image-classification/results/performances_cnn.csv", index=False)
    print("Treino CNN customizada finalizado e salvo!")