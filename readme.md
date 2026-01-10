Multimodal Conditional VAE (CVAE) for Audio-Lyric Clustering

This project implements a **Conditional Variational Autoencoder (CVAE)** to perform multimodal clustering and generation. It fuses **Audio Features (MFCCs)** and **Lyrics (Text)** to learn a disentangled latent representation, aiming to separate songs based on Language (Bangla vs. English) or Genre.

## Project Overview

This system tackles the "Hard Task" of multimodal learning:
1.  **Input:** Raw Audio (MFCC features) + Lyrics (Tokenized Text) + Condition (Language Label).
2.  **Model:** A Conditional VAE that forces the model to learn content (lyrics/melody) separately from the category (language).
3.  **Goal:** * Condition the latent space on **5 Universal Genres** (Rock, Pop, Hip-Hop, Folk, Classical).
    * Evaluate how well the model separates these genres compared to baselines.