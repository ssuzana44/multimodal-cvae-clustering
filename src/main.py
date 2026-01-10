import os
import tensorflow as tf
from tensorflow import keras
import dataset
import vae
import evaluation
import argparse
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

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run VAE Tasks")
    parser.add_argument(
        "--task", 
        type=str, 
        default="all", 
        choices=["easy", "medium", "hard", "all"],
        help="Choose which task to run: 'easy', 'medium', 'hard', or 'all'"
    )
    
    args = parser.parse_args()

    # Execute based on selection
    if args.task == "easy" or args.task == "all":
        run_easy()
        
    if args.task == "medium" or args.task == "all":
        run_medium()
        
    if args.task == "hard" or args.task == "all":
        run_hard()

    print("\nExecution Completed.")

if __name__ == "__main__":
    main()