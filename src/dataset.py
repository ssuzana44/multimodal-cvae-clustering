import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# PATHS
AUDIO_BANGLA = "../data/audio/dataset.csv"
AUDIO_ENGLISH = "../data/audio/features_30_sec.csv"
LYRICS_BANGLA = "../data/lyrics/BanglaSongLyrics.csv"
LYRICS_ENGLISH = "../data/lyrics/songs.csv"         
ARTIST_GENRES = "../data/lyrics/artists-data.csv"   

EMB_BANGLA = "bangla_embeddings.pt"
EMB_ENGLISH = "english_embeddings.pt"


# GENRE MAPPING LOGIC
def map_genre_english(genre_str):
    """Maps complex English/Portuguese genres to 5 Universal Classes."""
    if not isinstance(genre_str, str): return None
    g = genre_str.lower()
    
    # Priority matching
    if any(x in g for x in ['rock', 'metal', 'punk', 'grunge', 'indie']):
        return 0 # Rock/Band
    elif any(x in g for x in ['pop', 'dance', 'disco', 'romântico']):
        return 1 # Pop/Modern
    elif any(x in g for x in ['hip hop', 'rap', 'r&b', 'black']):
        return 2 # Hip-Hop
    elif any(x in g for x in ['country', 'folk', 'blues', 'sertanejo']):
        return 3 # Folk/Country
    elif any(x in g for x in ['jazz', 'classical', 'instrumental', 'piano']):
        return 4 # Classical/Trad
    return None

def map_genre_bangla(category_str):
    """Maps Bangla categories to the same 5 Universal Classes."""
    if not isinstance(category_str, str): return None
    c = category_str.lower()
    
    if 'ব্যান্ড' in c or 'band' in c:
        return 0 # Rock/Band
    elif 'আধুনিক' in c or 'adhunik' in c or 'pop' in c:
        return 1 # Pop/Modern
    elif 'hiphop' in c or 'র\u200d্যাপ' in c: # Unicode for Rap
        return 2 # Hip-Hop
    elif 'বাউল' in c or 'পল্লীগীতি' in c or 'folk' in c or 'palligeeti' in c:
        return 3 # Folk/Country
    elif 'রবীন্দ্র' in c or 'নজরুল' in c or 'rabindra' in c or 'nazrul' in c:
        return 4 # Classical/Trad
    return None


# EASY TASK DATA 
def load_easy_data():
    print("Loading Easy Task Data...")
    df_bangla = pd.read_csv(AUDIO_BANGLA)
    df_english = pd.read_csv(AUDIO_ENGLISH)

    # Align columns
    bangla_cols = ['chroma_frequency', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo'] + [f'mfcc{i}' for i in range(20)]
    english_cols = ['chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean', 'tempo'] + [f'mfcc{i}_mean' for i in range(1, 21)]
    generic_cols = ['chroma', 'rms', 'spec_cent', 'spec_bw', 'rolloff', 'tempo'] + [f'mfcc{i}' for i in range(1, 21)]

    df_b_sub = df_bangla[bangla_cols].copy()
    df_b_sub.columns = generic_cols
    df_b_sub['language'] = 1  

    df_e_sub = df_english[english_cols].copy()
    df_e_sub.columns = generic_cols
    df_e_sub['language'] = 0 

    df_final = pd.concat([
        df_b_sub.sample(n=500, random_state=42),
        df_e_sub.sample(n=500, random_state=42)
    ], axis=0).reset_index(drop=True)

    X = df_final.drop(['language'], axis=1).values
    y = df_final['language'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

  
# EMBEDDING GENERATION 
def generate_embeddings():
    print("Loading Datasets for Embedding...")
    df_b = pd.read_csv(LYRICS_BANGLA, engine="python", on_bad_lines="skip")
    
    # Merge English Lyrics with Genres
    df_songs = pd.read_csv(LYRICS_ENGLISH, engine="python", on_bad_lines="skip")
    df_artists = pd.read_csv(ARTIST_GENRES, engine="python", on_bad_lines="skip")
    
    # Normalize for merge
    df_songs['Artist_lower'] = df_songs['Artist'].str.lower().str.strip()
    df_artists['Artist_lower'] = df_artists['Artist'].str.lower().str.strip()
    
    print("Merging English Songs with Genres...")
    df_e = df_songs.merge(df_artists[['Artist_lower', 'Genres']], on='Artist_lower', how='inner')
    print(f"Matched {len(df_e)} English songs with Genre info.")
    
    # Filter for valid genres ONLY
    print("Filtering Data by Genre...")
    df_b['universal_genre'] = df_b['category'].apply(map_genre_bangla)
    df_e['universal_genre'] = df_e['Genres'].apply(map_genre_english)
    
    df_b = df_b.dropna(subset=['universal_genre'])
    df_e = df_e.dropna(subset=['universal_genre'])
    
    # Balance the dataset (take 500 max total if possible, split evenly)
    n = min(len(df_b), len(df_e), 500)
    df_b = df_b.sample(n, random_state=42)
    df_e = df_e.sample(n, random_state=42)
    
    print(f"Final Training Set: {len(df_b)} Bangla songs, {len(df_e)} English songs.")

    # Generate Embeddings
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

    def embed_lyrics(text_list):
        embeddings = []
        batch_size = 16
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_emb)
        return torch.cat(embeddings, dim=0)

    print("Embedding Bangla...")
    txt_col_b = 'lyrics' if 'lyrics' in df_b.columns else 'Lyrics'
    emb_bangla = embed_lyrics(df_b[txt_col_b].astype(str).tolist())
    
    print("Embedding English...")
    txt_col_e = 'Lyrics' if 'Lyrics' in df_e.columns else 'lyrics'
    emb_english = embed_lyrics(df_e[txt_col_e].astype(str).tolist())

    torch.save(emb_bangla, EMB_BANGLA)
    torch.save(emb_english, EMB_ENGLISH)
    
    
    np.save("bangla_genres.npy", df_b['universal_genre'].values)
    np.save("english_genres.npy", df_e['universal_genre'].values)
    
    print("Embeddings and Genre Labels Saved.")

  
# HARD TASK DATA (Multi-Modal with Genre)
  
def load_hybrid_data():
    # Audio (Standardized to 500 samples)
    df_bangla = pd.read_csv(AUDIO_BANGLA)
    df_english = pd.read_csv(AUDIO_ENGLISH)
    
    # Load Embeddings & Genres
    try:
        X_txt_b = torch.load(EMB_BANGLA).numpy()
        X_txt_e = torch.load(EMB_ENGLISH).numpy()
        y_genre_b = np.load("bangla_genres.npy")
        y_genre_e = np.load("english_genres.npy")
    except FileNotFoundError:
        print("Embeddings/Genres not found. Run generation step first!")
        return None, None, None

    n_samples = min(len(X_txt_b), len(X_txt_e))
    
    # Prepare Audio (Slice to match n_samples)
    bangla_cols = ['chroma_frequency', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo'] + [f'mfcc{i}' for i in range(20)]
    english_cols = ['chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean', 'tempo'] + [f'mfcc{i}_mean' for i in range(1, 21)]
    
    df_b = df_bangla[bangla_cols].sample(n=n_samples, random_state=42).reset_index(drop=True)
    df_e = df_english[english_cols].sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    scaler = MinMaxScaler()
    X_audio_b = scaler.fit_transform(df_b.values).reshape(n_samples, 26, 1)
    X_audio_e = scaler.fit_transform(df_e.values).reshape(n_samples, 26, 1)
    
    # Stack Everything
    X_audio_train = np.vstack([X_audio_b, X_audio_e])
    X_text_train = np.vstack([X_txt_b[:n_samples], X_txt_e[:n_samples]])
    y_genres = np.hstack([y_genre_b[:n_samples], y_genre_e[:n_samples]]) # Labels 0-4
    
    return X_audio_train, X_text_train, y_genres



def load_torch_simple_bangla():
    """Loads Bangla Audio Features for Beta-VAE"""
    df = pd.read_csv(AUDIO_BANGLA)
    
    ignore_cols = ['file_name', 'label', 'audio_path', 'lyrics', 'genre', 'id', 'title']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Log what we found to avoid confusion
    print(f"Found {len(feature_cols)} feature columns: {feature_cols[:3]}...")
    
    data = df[feature_cols].values.astype(np.float32)
    
    # Normalize
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return torch.tensor(data), df['label'].values
    

def load_torch_merged():
    print("Loading and merging datasets...")

    # 1. Load the raw CSVs (Using global constants for paths)
    df_bangla = pd.read_csv(AUDIO_BANGLA)
    df_english = pd.read_csv(AUDIO_ENGLISH)

    
    df_english = df_english.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Define the Columns (Manual Alignment)
    # Bangla Mapping
    b_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'tempo']
    b_cols += [f'mfcc{i}' for i in range(20)]

    # English Mapping
    e_cols = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean', 'tempo']
    e_cols += [f'mfcc{i}_mean' for i in range(1, 21)]

    # 3. Extract and Rename
    X_b = df_bangla[b_cols].copy()
    X_b.columns = [f'feat_{i}' for i in range(len(b_cols))]
    y_b = "Bangla_" + df_bangla['label'].astype(str)

    X_e = df_english[e_cols].copy()
    X_e.columns = [f'feat_{i}' for i in range(len(e_cols))]
    y_e = "English_" + df_english['label'].astype(str)

    # 4. Concatenate
    X_merged = pd.concat([X_b, X_e], axis=0)
    y_merged = pd.concat([y_b, y_e], axis=0)

    print(f"Merged shapes: {X_merged.shape}")
    print(f"Unique English Genres found: {df_english['label'].unique()}") 

    # 5. Normalize (StandardScaler)
    print("Applying StandardScaler to bridge the Domain Gap...")
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_merged.values)

    return torch.tensor(X_final, dtype=torch.float32), y_merged.values

def load_cvae_data_exact():
    print("Loading Bangla Dataset (CVAE Exact)...")
    
    # 1. Load separately (Using global path constants)
    df_audio = pd.read_csv(AUDIO_BANGLA) 
    df_text = pd.read_csv(LYRICS_BANGLA) 
    
    # 2. Align (Truncate to minimum)
    min_len = min(len(df_audio), len(df_text))
    df_audio = df_audio.iloc[:min_len]
    df_text = df_text.iloc[:min_len]
    
    # 3. Inject Lyrics
    df_audio['lyrics'] = df_text['lyrics'].fillna("").values
    
    # 4. Audio Features
    ignore_cols = ['file_name', 'label', 'audio_path', 'lyrics', 'genre', 'id', 'title', 'category']
    feat_cols = [c for c in df_audio.columns if c not in ignore_cols]
    
    audio_data = df_audio[feat_cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    audio_data = scaler.fit_transform(audio_data)
    
    # 5. Lyrics Features
    print("Vectorizing Lyrics...")
    tfidf = TfidfVectorizer(max_features=128)
    lyrics_data = tfidf.fit_transform(df_audio['lyrics']).toarray()
    
    # 6. Labels (Genre)
    le = LabelEncoder()
    genres_raw = df_audio['label'].values
    genre_indices = le.fit_transform(genres_raw)
    num_genres = len(le.classes_)
    
    # One-Hot Encoding
    genre_onehot = np.zeros((len(genres_raw), num_genres))
    genre_onehot[np.arange(len(genres_raw)), genre_indices] = 1
    
    return (torch.tensor(audio_data, dtype=torch.float32), 
            torch.tensor(lyrics_data, dtype=torch.float32), 
            torch.tensor(genre_onehot, dtype=torch.float32),
            le.classes_,
            genres_raw,  
            audio_data.shape[1],
            num_genres)