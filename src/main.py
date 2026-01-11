import os
import tensorflow as tf
from tensorflow import keras
import dataset
import vae
import evaluation
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Seeds
tf.random.set_seed(42)

RESULT_DIR = "../results"
VIS_DIR = os.path.join(RESULT_DIR, "latent_visualization")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

def main():
   
    if not (os.path.exists("bangla_embeddings.pt") or os.path.exists("../data/lyrics/bangla_embeddings.pt")):
        print("Embeddings missing. Generating...")
        dataset.generate_embeddings()
    else:
        print("Embeddings found. Skipping generation.")

def run_easy(): 
    # TASK 1: EASY
    print("\nTASK 1: EASY (Simple VAE)")
    X_scaled, y_true = dataset.load_easy_data()
    
    enc, dec = vae.build_simple_vae(X_scaled.shape[1])
    model = vae.VAE(enc, dec)
    model.compile(optimizer=keras.optimizers.Adam())
    model.fit(X_scaled, epochs=50, batch_size=32, verbose=0)
    
    z_mean_preds, _, _ = model.encoder.predict(X_scaled)
    evaluation.evaluate_easy(z_mean_preds, y_true, os.path.join(VIS_DIR, "VAE_Clusters.png"))

def run_medium():  
    # TASK 2: MEDIUM
    print("\n TASK 2: MEDIUM (Hybrid VAE) ")
    
    X_aud, X_txt, y_genres = dataset.load_hybrid_data()
    
    if X_aud is None:
        print("Data load failed. Exiting.")
        return

    
    enc, dec = vae.build_hybrid_vae()
    model = vae.HybridVAE(enc, dec)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model.fit([X_aud, X_txt], epochs=50, batch_size=32, verbose=0)
    
    z_mean_preds, _, _ = model.encoder.predict([X_aud, X_txt])
    evaluation.evaluate_medium(z_mean_preds, y_genres, os.path.join(VIS_DIR, "hybrid_cnn_vae_results.png"))

def run_hard(): 
    # TASK 3: HARD
    print("\nTASK 3: HARD (CVAE & Baselines)")
    X_aud, X_txt, y_genres = dataset.load_hybrid_data()
    
    if X_aud is None:
        print("Data load failed. Exiting.")
        return
    # 1. One-Hot Encode Genres
    conditions = keras.utils.to_categorical(y_genres, num_classes=5)
    
    # 2. Build CVAE with 5 conditioning dimensions
    enc, dec = vae.build_cvae(cond_dim=5)
    model = vae.CVAE(enc, dec)
    model.compile(optimizer='adam')
    
    print("Training CVAE on 5 Genres...")
    model.fit([X_aud, X_txt, conditions], epochs=60, batch_size=32, verbose=0)
    
    # 3. Evaluate
    evaluation.evaluate_hard(model, X_aud, X_txt, conditions, y_genres, VIS_DIR)


def run_beta_vae_bangla():
    print("\n--- EXPERIMENT: Simple Beta-VAE (Bangla Only) ---")
    
    # 1. Load Data
    X, labels_true = dataset.load_torch_simple_bangla()
    
    # 2. Config (Exact matches)
    input_dim = X.shape[1]
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    BETA = 4.0
    
    # 3. Model
    model = vae.SimpleBetaVAE_Torch(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training Simple Beta-VAE on {len(X)} songs for {EPOCHS} epochs...")
    
    # 4. Train
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            loss = vae.beta_vae_loss_torch(recon, x_batch, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss / len(X):.4f}")

    # 5. Visualize (Exact Function)
    evaluation.evaluate_beta_bangla(model, X, labels_true, VIS_DIR)


def run_beta_vae_merged():
    print("\n--- EXPERIMENT: Merged Beta-VAE (Bangla + English) ---")
    
    # 1. Load Merged Data
    X, labels_raw = dataset.load_torch_merged()

    input_dim = X.shape[1]
    BATCH_SIZE = 64
    EPOCHS = 80         
    LR = 1e-3
    BETA = 4.0

    # 3. Model 
    model = vae.MergedBetaVAE_Torch(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training on {len(X)} songs (Bangla + English)...")

    # 4. Train
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for batch in loader:
            x_batch = batch[0]
            
            # Skip single-sample batches (BatchNorm error)
            if x_batch.shape[0] < 2: 
                continue 

            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            
            # Reusing the shared loss function from vae.py
            loss = vae.beta_vae_loss_torch(recon, x_batch, mu, logvar, beta=BETA)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss / len(X):.4f}")

    # 5. Evaluate & Visualize
    evaluation.evaluate_beta_merged(model, X, labels_raw, VIS_DIR)

# ... (Previous code)

def run_cvae_bangla():
    print("\n--- EXPERIMENT: Conditional VAE (Bangla Genre Transfer) ---")
    
    # 1. Load Data
    data = dataset.load_cvae_data_exact()
    audio, text, genres_oh, genre_names, genres_raw, AUD_DIM, COND_DIM = data
    
    # 2. Config (Exact)
    LATENT_DIM = 16       
    TEXT_DIM = 128        
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    BETA = 2.0 

    # 3. Model
    model = vae.CVAE_Exact(audio_dim=AUD_DIM, text_dim=TEXT_DIM, cond_dim=COND_DIM, 
                           latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    dataset_torch = TensorDataset(audio, text, genres_oh)
    loader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training CVAE on Single Domain (Bangla)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            a, t, c = batch
            optimizer.zero_grad()
            recon, mu, logvar = model(a, t, c)
            loss = vae.cvae_loss_exact(recon, a, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(audio):.4f}")

    # 4. Evaluate & Visualize
    evaluation.evaluate_cvae_bangla(model, audio, text, genres_oh, genres_raw, VIS_DIR)
    evaluation.plot_style_transfer(model, audio, text, genres_oh, genre_names, VIS_DIR)


def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run VAE Tasks")
    parser.add_argument(
        "--task", 
        type=str, 
        default="all", 
        choices=["easy", "medium", "hard", "all", "beta_bangla", "beta_merged", "cvae_bangla"],
        help="Choose task: 'easy', 'medium', 'hard', 'all', or specific models like 'beta_bangla'"
    )
    
    args = parser.parse_args()

    # --- Pre-check for Embeddings (Only needed for Hard task or CVAE) ---
    if args.task in ["hard", "all", "cvae_bangla"]:
        if not (os.path.exists("bangla_embeddings.pt") or os.path.exists("../data/lyrics/bangla_embeddings.pt")):
            print("Embeddings missing. Generating...")
            dataset.generate_embeddings()
        else:
            print("Embeddings found. Skipping generation.")


    # 1. RUN EVERYTHING
    if args.task == "all":
        run_easy()
        run_medium()
        run_hard()
        run_beta_vae_bangla()
        run_beta_vae_merged()
        run_cvae_bangla()


    elif args.task == "easy":
        run_easy()

    elif args.task == "medium":
        run_medium()

    elif args.task == "hard":
        run_hard()

    elif args.task == "beta_bangla":
        run_beta_vae_bangla()

    elif args.task == "beta_merged":
        run_beta_vae_merged()

    elif args.task == "cvae_bangla":
        run_cvae_bangla()
    print("\nExecution Completed.")

if __name__ == "__main__":
    main()