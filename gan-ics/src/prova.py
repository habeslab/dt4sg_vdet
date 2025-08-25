#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# Percorso al CSV unificato
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(BASE_DIR, "processed", "gan_training_data.csv")

def main():
    # 1. Carica i dati
    df = pd.read_csv(DATA_CSV)
    
    # 2. Visualizza un campione
    print("\n=== Sample delle prime 5 righe ===")
    print(df.head())

    # 3. Distribuzione delle classi
    print("\n=== Distribuzione delle classi ===")
    class_counts = df['label'].value_counts()
    print(class_counts)

    # 4. Statistiche descrittive
    print("\n=== Statistiche descrittive ===")
    print(df.describe().T)

    # 5. Istogrammi delle feature chiave
    features = ['duration', 'packet_count', 'bytes_per_packet', 'packets_per_sec', 'iat_mean']
    for feat in features:
        plt.figure()
        df[feat].hist(bins=50)
        plt.title(f"Istogramma di {feat}")
        plt.xlabel(feat)
        plt.ylabel("Frequenza")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
