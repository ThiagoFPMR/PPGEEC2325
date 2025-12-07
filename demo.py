import numpy as np
import streamlit as st
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering

# --- 1. Load the Authentic Word2Vec Model ---
# We use 'word2vec-google-news-300'. This is the standard pre-trained Word2Vec model.
# WARNING: This file is ~1.6GB. It will take time to download on the first run.
@st.cache_resource
def load_word2vec(model_path="mini_wiki_word2vec.model"):
    # return api.load("word2vec-google-news-300")
    model = Word2Vec.load(model_path)
    return model

# --- 2. Clustering Logic (Same logic, better vectors) ---
def get_clusters(word_list, model, threshold):
    # Filter valid words (check if word exists in the 3 million word vocabulary)
    valid_words = [w for w in word_list if w in model.wv]
    missing = [w for w in word_list if w not in model.wv]
    
    if not valid_words:
        return None, missing

    # Get vectors (300 dimensions per word)
    vectors = np.array([model.wv[w] for w in valid_words])

    # Cluster using Cosine Distance
    # 'distance_threshold' is the cut-off for how different words can be
    # before they are forced into separate groups.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    clustering.fit(vectors)
    
    # Organize results into a dictionary
    groups = {}
    for word, label in zip(valid_words, clustering.labels_):
        if label not in groups:
            groups[label] = []
        groups[label].append(word)
        
    return groups, missing

# --- 3. Streamlit Interface ---
st.title("Demonstração de Agrupamento de Palavras com Word2Vec do Google")
st.write("Usa o modelo `word2vec-google-news-300` para agrupar palavras. O modelo foi treinado em notícias do Google em inglês.")

# Load model with a loading spinner
with st.spinner("Carregando Google Word2Vec Model (1.6GB)... isso pode levar um tempo"):
    try:
        model = load_word2vec()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

# Input Area
raw_text = st.text_area("Digite palavras em inglês (separadas por vírgula)", 
                        "orange, banana, pineapple, ferrari, porsche, dog, cat, bird")

# Slider for Sensitivity
# Word2Vec vectors are normalized differently than GloVe, so the threshold sweet spot
# might differ. 0.6 is usually a good starting point for cosine distance.
threshold = st.slider(
    "Limiar (Limiar de Distância de Cosseno)", 
    min_value=0.1, 
    max_value=1., 
    value=0.5, 
    step=0.05,
    help="Menor = Mais rigoroso (mais grupos). Maior = Mais flexível (menos grupos)."
)

if st.button("Agrupar Palavras"):
    word_list = [w.strip().lower() for w in raw_text.replace(" ", "").split(",") if w.strip()]
    
    if word_list:
        groups, missing = get_clusters(word_list, model, threshold)
        
        # Warning for missing words
        if missing:
            st.warning(f"Palavras ignoradas (não estão no dicionário Word2Vec): {', '.join(missing)}")
        
        if groups:
            st.success(f"Agrupado com sucesso em {len(groups)} clusters!")
            
            # Display groups in columns
            cols = st.columns(min(len(groups), 3))
            group_ids = list(groups.keys())
            
            # distinctive colors for headers
            colors = ["blue", "green", "orange", "red", "violet"]
            
            for i, group_id in enumerate(group_ids):
                col = cols[i % 3]
                color = colors[i % len(colors)]
                
                with col:
                    # Use a colored header to visually distinguish groups
                    st.markdown(f":{color}[Grupo {i+1}]")
                    for word in groups[group_id]:
                        st.code(word, language="text")
        else:
            st.error("Nenhuma palavra válida encontrada para agrupar.")
    else:
        st.warning("Por favor, insira algumas palavras.")