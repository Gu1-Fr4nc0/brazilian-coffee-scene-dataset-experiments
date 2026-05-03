import os 
import pandas as pd
import numpy as np
import keras
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score

def get_pure_vgg_model(input_shape=(64,64,3)):
    model = Sequential()
    # Carrega a VGG16 base sem a camada final
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Congela os pesos
    
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def read_pure_images(df, seed):
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
                img = np.array(Image.open(caminho_absoluto).resize((64, 64))) / 255.0
                images.append(img)
                labels.append(row["Y"])
                
        return np.array(images), np.array(labels)

    t_img, t_lab = load_imgs(df_training)
    v_img, v_lab = load_imgs(df_testing)
    return t_img, t_lab, v_img, v_lab

def train_pure_vgg(df):
    all_performances = []
    for seed in range(0, 30):
        print(f" * Running Pure VGG16 for seed = {seed}")
        t_img, t_lab, v_img, v_lab = read_pure_images(df, seed)
        
        if len(t_img) == 0: continue

        model = get_pure_vgg_model((64,64,3))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])
        
        model.fit(t_img, t_lab, epochs=50, batch_size=16, verbose=0)
        
        preds = np.round(model.predict(v_img, verbose=0))
        all_performances.append([accuracy_score(v_lab, preds), f1_score(v_lab, preds), seed])
        
    return pd.DataFrame(all_performances, columns=["acc", "f1s", "seed"])

if __name__ == "__main__":
    df = pd.read_csv('/content/crop-image-classification/data/folds_coffee_dataset.csv')
    results = train_pure_vgg(df)
    
    # Salva com o nome exato que o teste estatístico vai procurar
    results.to_csv("/content/crop-image-classification/results/performances_vgg16.csv", index=False)
    print("Treino VGG16 original finalizado e salvo!")
