import os 
import pandas as pd
import numpy as np
import keras
from PIL import Image
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def get_hybrid_model(input_shape_img, input_shape_features):
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_img)
    vgg_base.trainable = False 
    
    img_input = Input(shape=input_shape_img)
    x_img = vgg_base(img_input)
    x_img = Flatten()(x_img)
    
    feat_input = Input(shape=(input_shape_features,))
    x_feat = BatchNormalization()(feat_input)
    x_feat = Dense(64, activation='relu')(x_feat)
    
    combined = Concatenate()([x_img, x_feat])
    z = Dense(128, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    
    return Model(inputs=[img_input, feat_input], outputs=output)

def read_data_hybrid(df, df_features, seed):
    df_seed = df.loc[df['Seed'] == seed]
    df_training = df_seed[df_seed["Fold"] == "Training"]
    df_testing = df_seed[df_seed["Fold"] == "Test"]
    
    def load_images_and_feats(df_subset):
        images, feats, labels = [], [], []
        base_path = '/content/crop-image-classification/dataset-brazilian_coffee_scenes/images'
        
        for _, row in df_subset.iterrows():
            nome_arquivo = os.path.basename(row["im_path"])
            caminho_absoluto = os.path.join(base_path, nome_arquivo)
            
            if os.path.exists(caminho_absoluto):
                img = np.array(Image.open(caminho_absoluto).resize((64, 64))) / 255.0
                feat_row = df_features[df_features['Name_file'] == nome_arquivo]
                if not feat_row.empty:
                    images.append(img)
                    feats.append(feat_row.drop(columns=['Name_file']).values[0])
                    labels.append(row["Y"])
        
        return np.array(images), np.nan_to_num(np.array(feats), nan=0.0), np.array(labels)

    train_img, train_feat, train_lab = load_images_and_feats(df_training)
    test_img, test_feat, test_lab = load_images_and_feats(df_testing)
    return train_img, train_feat, train_lab, test_img, test_feat, test_lab

def train_hybrid_vgg(df, df_features):
    all_performances = []
    for seed in range(0, 30):
        print(f" * Running Hybrid VGG for seed = {seed}")
        t_img, t_feat, t_lab, v_img, v_feat, v_lab = read_data_hybrid(df, df_features, seed)
        
        if len(t_img) == 0: continue

        model = get_hybrid_model((64,64,3), t_feat.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])
        
        model.fit(x=[t_img, t_feat], y=t_lab, epochs=50, batch_size=16, verbose=0)
        
        preds = np.round(model.predict([v_img, v_feat], verbose=0))
        all_performances.append([accuracy_score(v_lab, preds), f1_score(v_lab, preds), seed])
        
    return pd.DataFrame(all_performances, columns=["acc", "f1s", "seed"])

if __name__ == "__main__":
    df = pd.read_csv('/content/crop-image-classification/data/folds_coffee_dataset.csv')
    df_feat = pd.read_csv('/content/crop-image-classification/data/coffee_meta_features.csv')
    results = train_hybrid_vgg(df, df_feat)
    results.to_csv("/content/crop-image-classification/results/performances_hybrid.csv", index=False)
    print("Treino Híbrido finalizado e salvo!")