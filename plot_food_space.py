import gensim.downloader as api
import numpy as np
import umap
import sys
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Configuration
MODEL_NAME = "word2vec-google-news-300"
FOOD_FILE_PATH = 'food_list.txt'
OUTPUT_HTML_FILE = 'food_embeddings_interactive.html'
CLUSTER_REPORT_FILE = 'cluster_contents.txt'

# We lock this to 5 to match the patterns you identified
NUM_CLUSTERS = 5

# The labels you derived from the report
CLUSTER_LABELS = {
    0: "Hortifruti, Ervas & Grãos",
    1: "Padaria, Laticínios & Doces",
    2: "Bebidas",
    3: "Carnes Cruas & Frutos do Mar",
    4: "Refeições & Pratos Preparados"
}

def load_model_from_api(model_name):
    """Downloads (if necessary) and loads the model via Gensim API."""
    print(f"Attempting to load '{model_name}' via Gensim Downloader...")
    try:
        model = api.load(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_food_embeddings(model, file_path):
    """Reads food names and generates embeddings."""
    embeddings = []
    labels = []
    
    print(f"Reading food names from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            food_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    print("Processing food items...")
    for food in tqdm(food_names, desc="Encoding words", unit="word"):
        token = food
        if token not in model:
            token = food.replace(' ', '_')
            
        if token in model:
            embeddings.append(model[token])
            labels.append(food) 
            
    return np.array(embeddings), labels

def perform_clustering(embeddings, n_clusters=5):
    """
    Clusters data into a fixed number of groups using K-Means.
    We use random_state=42 to ensure Cluster 0 is always Cluster 0.
    """
    print(f"Clustering data into {n_clusters} named groups...")
    
    # Normalize for cosine similarity behavior
    norm_embeddings = normalize(embeddings, norm='l2')
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(norm_embeddings)
    return clusters

def save_cluster_report(labels, clusters, filename):
    """Saves a text file listing the words in each cluster."""
    print(f"Generating cluster report at '{filename}'...")
    
    df = pd.DataFrame({'food': labels, 'cluster': clusters})
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("CLUSTER REPORT\n")
        f.write("========================================\n\n")
        
        unique_clusters = sorted(df['cluster'].unique())
        
        for c in unique_clusters:
            foods = df[df['cluster'] == c]['food'].tolist()
            cluster_name = CLUSTER_LABELS.get(c, f"Cluster {c}")
            
            f.write(f"{cluster_name} (ID: {c}, {len(foods)} items)\n")
            f.write("-" * 40 + "\n")
            f.write(", ".join(foods))
            f.write("\n\n")
            
    print("Report saved.")

def reduce_dimensions(embeddings):
    """Reduces dimensions from 300 to 2 using UMAP."""
    print("Reducing dimensions using UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42,
        verbose=True
    )
    return reducer.fit_transform(embeddings)

def plot_and_save_plotly(reduced_embeddings, labels, clusters, output_file):
    """Plots using Plotly with descriptive legend labels."""
    print("Generating interactive plot...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'food': labels,
        'cluster_id': clusters
    })
    
    # Map the numeric cluster ID to the descriptive string
    df['category'] = df['cluster_id'].map(CLUSTER_LABELS)
    
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        text='food',
        color='category',  # Use the new descriptive column for colors/legend
        hover_name='food',
        title=f'Mapa de Comidas ({len(labels)} itens)',
        template='plotly_white',
        opacity=0.7,
        width=1400,
        height=1000
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(
            size=12,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        textfont=dict(
            size=11,
            color='black'
        )
    )
    
    fig.update_layout(
        xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
        legend_title_text='Categoria de Comida'
    )
    
    print(f"Saving interactive plot to '{output_file}'...")
    
    config = {
        'scrollZoom': True,
        'displaylogo': False, 
        'responsive': True
    }
    
    fig.write_html(output_file, config=config)
    print("Done! Open the HTML file in your browser to view.")

if __name__ == "__main__":
    # 1. Load Model
    w2v_model = load_model_from_api(MODEL_NAME)
    
    # 2. Get Embeddings
    vectors, food_labels = get_food_embeddings(w2v_model, FOOD_FILE_PATH)
    
    if len(vectors) > 0:
        # 3. Perform Clustering (Fixed to 5 to match our labels)
        cluster_labels = perform_clustering(vectors, n_clusters=NUM_CLUSTERS)
        
        # 4. Save the Cluster Content Report (now with names)
        save_cluster_report(food_labels, cluster_labels, CLUSTER_REPORT_FILE)
        
        # 5. Reduce Dimensions
        vectors_2d = reduce_dimensions(vectors)
        
        # 6. Plot and Save (HTML)
        plot_and_save_plotly(vectors_2d, food_labels, cluster_labels, OUTPUT_HTML_FILE)
    else:
        print("No valid food embeddings were found.")