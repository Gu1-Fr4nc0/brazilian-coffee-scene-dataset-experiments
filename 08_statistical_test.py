import pandas as pd
from scipy.stats import wilcoxon

def run_statistical_test():
    try:
        df_vgg = pd.read_csv('/content/crop-image-classification/results/performances_vgg16.csv')
        df_hybrid = pd.read_csv('/content/crop-image-classification/results/performances_hybrid.csv')
        df_cnn = pd.read_csv('/content/crop-image-classification/results/performances_cnn.csv')
        df_ml = pd.read_csv('/content/crop-image-classification/results/performances_ml.csv')
        
        print("--- MÉDIAS DE ACURÁCIA ---")
        print(f"ML Clássico (Random Forest): {df_ml['acc'].mean():.4f}")
        print(f"CNN Customizada:             {df_cnn['acc'].mean():.4f}")
        print(f"VGG16 (Baseline DL):         {df_vgg['acc'].mean():.4f}")
        print(f"Modelo Híbrido:              {df_hybrid['acc'].mean():.4f}\n")
        
        print("--- TESTES DE WILCOXON (P-VALORES) ---")
        # Testa se Híbrido é melhor que VGG
        stat, p_vgg_hib = wilcoxon(df_vgg['acc'], df_hybrid['acc'])
        print(f"Híbrido vs VGG16:        p = {p_vgg_hib:.10f}")
        
        # Testa se Híbrido é melhor que ML Tradicional
        stat, p_ml_hib = wilcoxon(df_ml['acc'], df_hybrid['acc'])
        print(f"Híbrido vs ML Clássico:  p = {p_ml_hib:.10f}")

        # Testa se Híbrido é melhor que CNN Customizada
        stat, p_cnn_hib = wilcoxon(df_cnn['acc'], df_hybrid['acc'])
        print(f"Híbrido vs CNN:          p = {p_cnn_hib:.10f}")
        
        print("\nConclusão:")
        if p_vgg_hib < 0.05 and p_ml_hib < 0.05:
            print("O modelo Híbrido é estatisticamente superior às baselines isoladas (p < 0.05).")
        else:
            print("Não foi possível comprovar superioridade estatística do modelo híbrido em todas as frentes.")
            
    except FileNotFoundError as e:
        print(f"Erro: Arquivo de resultados não encontrado. Certifique-se de rodar todos os treinos antes. Detalhe: {e}")

if __name__ == "__main__":
    run_statistical_test()