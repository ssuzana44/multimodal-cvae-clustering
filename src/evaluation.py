import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# GLOBAL CONFIG
METRICS_FILE = "../results/clustering_metrics.csv"

def log_to_csv(task, method, metrics):
    """
    Saves a row of metrics to the CSV file.
    Handles different columns (e.g. Medium has DB, Hard has Purity) gracefully.
    """
    data = {"Task": task, "Method": method, **metrics}
    new_row = pd.DataFrame([data])
    
    if os.path.exists(METRICS_FILE):
        existing_df = pd.read_csv(METRICS_FILE)
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        updated_df.to_csv(METRICS_FILE, index=False)
    else:
        new_row.to_csv(METRICS_FILE, index=False)
    print(f"Saved metrics for '{method}' to {METRICS_FILE}")

  
# EASY TASK EVALUATION
  
def evaluate_easy(z_mean_preds, y, save_path):
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(z_mean_preds)
    
    sil = silhouette_score(z_mean_preds, clusters)
    ari = adjusted_rand_score(y, clusters)
    
    print(f"\nResults:")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"ARI Score: {ari:.4f}")
    
    log_to_csv("Easy", "Simple VAE", {"Silhouette": sil, "ARI": ari})
    
    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z_mean_preds)
    
    plt.figure(figsize=(10, 4))
    
    # Plot 1: True Labels
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=y, palette='bright')
    plt.title("True Language Labels (0=Eng, 1=Ban)")
    
    # Plot 2: VAE Clusters
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=clusters, palette='viridis')
    plt.title(f"VAE Clusters (ARI: {ari:.2f})")
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

  
# MEDIUM TASK EVALUATION
  
def evaluate_medium(z_mean_preds, y_labels, save_path):
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(z_mean_preds)
    # DBSCAN
    dbscan = DBSCAN(eps=3.0, min_samples=5)
    labels_dbscan = dbscan.fit_predict(z_mean_preds)
    # Agglomerative
    agglo = AgglomerativeClustering(n_clusters=2)
    labels_agglo = agglo.fit_predict(z_mean_preds)
    
    def eval_and_log(labels, method_name):
        if len(set(labels)) < 2:
            print(f"--- {method_name} ---\nFailed: Found only 1 cluster.")
            return
        
        sil = silhouette_score(z_mean_preds, labels)
        ari = adjusted_rand_score(y_labels, labels)
        db = davies_bouldin_score(z_mean_preds, labels)
        
        print(f"--- {method_name} ---\nSilhouette: {sil:.4f}\nARI: {ari:.4f}\nDavies-Bouldin: {db:.4f}\n")
   
        log_to_csv("Medium", method_name, {"Silhouette": sil, "ARI": ari, "Davies-Bouldin": db})

    eval_and_log(labels_kmeans, "K-Means")
    eval_and_log(labels_agglo, "Agglomerative Clustering")
    eval_and_log(labels_dbscan, "DBSCAN")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z_mean_preds)
    
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=y_labels, palette='coolwarm')
    plt.title("True Language")
    plt.subplot(1, 4, 2)
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels_kmeans, palette='viridis')
    plt.title("K-Means")
    plt.subplot(1, 4, 3)
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels_agglo, palette='viridis')
    plt.title("Agglomerative")
    plt.subplot(1, 4, 4)
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels_dbscan, palette='viridis')
    plt.title("DBSCAN")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

  
# HARD TASK EVALUATION (Baselines + Plots)
  
def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def evaluate_method_hard(X_features, y_true, method_name):
    if "Spectral" in method_name:
        clusterer = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
        labels = clusterer.fit_predict(X_features)
    else:
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X_features)
        
    sil = silhouette_score(X_features, labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    pur = purity_score(y_true, labels)
    
    print(f"--- {method_name} ---\nSilhouette: {sil:.4f}\nNMI: {nmi:.4f}\nARI: {ari:.4f}\nPurity: {pur:.4f}\n")
 
    log_to_csv("Hard", method_name, {"Silhouette": sil, "NMI": nmi, "ARI": ari, "Purity": pur})
    
    return labels

def evaluate_hard(model, X_aud, X_txt, conditions, y_true, vis_dir):
    # Flatten Audio for baselines
    X_aud_flat = X_aud.reshape(X_aud.shape[0], -1)
    X_combined = np.hstack([X_aud_flat, X_txt])

    print("Running Baselines...")
    # PCA
    pca = PCA(n_components=32, random_state=42)
    X_pca = pca.fit_transform(X_combined)
    labels_pca = evaluate_method_hard(X_pca, y_true, "Baseline: PCA + K-Means")

    # Autoencoder
    inp_ae = keras.Input(shape=(X_combined.shape[1],))
    enc_ae = layers.Dense(32, activation="relu")(inp_ae)
    dec_ae = layers.Dense(X_combined.shape[1], activation="sigmoid")(enc_ae)
    ae = keras.Model(inp_ae, dec_ae)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_combined, X_combined, epochs=20, batch_size=32, verbose=0)
    encoder_ae = keras.Model(inp_ae, enc_ae)
    X_ae = encoder_ae.predict(X_combined)
    labels_ae = evaluate_method_hard(X_ae, y_true, "Baseline: Autoencoder + K-Means")

    # Spectral
    labels_spectral = evaluate_method_hard(X_pca, y_true, "Baseline: Spectral Clustering")

    # CVAE
    _, _, z_cvae = model.encoder.predict([X_aud, X_txt, conditions])
    labels_cvae = evaluate_method_hard(z_cvae, y_true, "Method: Conditional VAE (CVAE)")

    # Plot Comparison
    print("Generating Comparison Plot...")
    tsne = TSNE(n_components=2, random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods = [
        (X_pca, labels_pca, "PCA + K-Means"),
        (X_ae, labels_ae, "Autoencoder + K-Means"),
        (X_pca, labels_spectral, "Spectral Clustering"),
        (z_cvae, labels_cvae, "CVAE Latent Space")
    ]
    for i, (features, labels, title) in enumerate(methods):
        ax = axes[i//2, i%2]
        reduced = tsne.fit_transform(features)
        sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=y_true, palette='coolwarm', ax=ax, alpha=0.6)
        ax.set_title(title)
        ax.legend(title="True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "hard_task_comparison.png"))
    plt.close()

    # Reconstruction
    print("Generating Reconstruction Plot...")
    sample_aud = X_aud[0:1]
    sample_txt = X_txt[0:1]
    sample_cond = conditions[0:1]
    _, _, z_sample = model.encoder.predict([sample_aud, sample_txt, sample_cond])
    recon_aud, _ = model.decoder.predict([z_sample, sample_cond])
    
    plt.figure(figsize=(10, 4))
    plt.plot(sample_aud.flatten(), label="Original Audio Input", color='blue', linewidth=2)
    plt.plot(recon_aud.flatten(), label="CVAE Reconstruction", color='red', linestyle='--', linewidth=2)
    plt.title("CVAE Audio Reconstruction Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "reconstruction_example.png"))
    plt.close()