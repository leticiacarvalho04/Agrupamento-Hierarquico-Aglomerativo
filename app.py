from flask import Flask, render_template, request
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------
# Pré-processamento
# -----------------------------------------
def limpar_texto(t):
    return t.lower().strip()


# -----------------------------------------
# TF-IDF
# -----------------------------------------
def gerar_tfidf(docs):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs).toarray()
    features = vectorizer.get_feature_names_out()
    return matrix, features


# -----------------------------------------
# Word2Vec
# -----------------------------------------
def gerar_word2vec(docs):
    tokenizado = [t.split() for t in docs]

    model = Word2Vec(
        sentences=tokenizado,
        vector_size=100,
        window=5,
        min_count=1,
        workers=2
    )

    vectors = []
    for palavras in tokenizado:
        if palavras:
            vet = np.mean([model.wv[p] for p in palavras], axis=0)
            vectors.append(vet)
        else:
            vectors.append(np.zeros(100))

    return np.array(vectors)


# -----------------------------------------
# Transformers
# -----------------------------------------
def gerar_transformer(docs):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(docs)
    return embeddings


# -----------------------------------------
# Geração de gráficos
# -----------------------------------------
def gerar_graficos(matrix, nome):
    dist = cosine_distances(matrix)

    # Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(dist, cmap="viridis", aspect="auto")
    plt.colorbar()
    heatmap_path = f"{OUTPUT_DIR}/{nome}_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

    # Dendrograma
    Z = linkage(dist, "ward")
    plt.figure(figsize=(6, 5))
    dendrogram(Z)
    dendro_path = f"{OUTPUT_DIR}/{nome}_dendrograma.png"
    plt.savefig(dendro_path)
    plt.close()

    return {
        "heatmap": f"outputs/{nome}_heatmap.png",
        "dendrograma": f"outputs/{nome}_dendrograma.png"
    }


# -----------------------------------------
# Insights (só TF-IDF)
# -----------------------------------------
def gerar_insights(matrix, feature_names=None):
    if feature_names is None:
        return {"note": "Insights disponíveis apenas para TF-IDF."}

    top_terms = []
    for row in matrix:
        idx = row.argsort()[-5:][::-1]
        termos = [feature_names[i] for i in idx]
        top_terms.append(termos)

    return {
        "note": "Análise concluída automaticamente.",
        "top_terms_per_doc": top_terms
    }

def gerar_insights_tfidf(matrix, feature_names):
    top_terms = []
    for row in matrix:
        idx = row.argsort()[-5:][::-1]
        termos = [feature_names[i] for i in idx]
        top_terms.append(termos)

    return {
        "tipo": "tfidf",
        "note": "Análise TF-IDF concluída.",
        "top_terms_per_doc": top_terms
    }


# -----------------------------------------
# Insights para Word2Vec / Transformers
# -----------------------------------------
def gerar_insights_embedding(matrix):
    dist = cosine_distances(matrix)

    # textos mais parecidos
    similar = []
    for i in range(len(dist)):
        idx = dist[i].argsort()[1]  # o mais próximo, ignorando ele mesmo
        similar.append((i, idx, dist[i][idx]))

    # par mais distante
    max_pair = np.unravel_index(dist.argmax(), dist.shape)
    max_dist = dist[max_pair]

    return {
        "tipo": "embedding",
        "note": "Análise baseada em similaridade entre embeddings.",
        "mais_parecidos": similar,
        "par_mais_distante": {
            "textos": max_pair,
            "distancia": float(max_dist)
        }
    }


# -----------------------------------------
# Página inicial
# -----------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        example_literatura="Dom Casmurro é um romance clássico.\nCapitu tinha olhos de ressaca.",
        example_reviews="Produto ótimo!\nChegou atrasado."
    )


# -----------------------------------------
# Execução da pipeline
# -----------------------------------------
@app.route("/run", methods=["POST"])
def run_pipeline():
    tipo = request.form["metodo"]

    # Entrada bruta
    literatura_raw = request.form["literatura_text"].split("\n")
    reviews_raw = request.form["reviews_text"].split("\n")

    # Para exibição
    literatura_display = [t.strip() for t in literatura_raw if t.strip()]
    reviews_display = [t.strip() for t in reviews_raw if t.strip()]

    # Para processamento
    literatura_clean = [limpar_texto(t) for t in literatura_raw if t.strip()]
    reviews_clean = [limpar_texto(t) for t in reviews_raw if t.strip()]

    # Textos unificados
    textos = literatura_clean + reviews_clean

    # ---------- Seletor de método ----------
    if tipo == "tfidf":
        matrix, features = gerar_tfidf(textos)
        insights = gerar_insights(matrix, features)

    elif tipo == "word2vec":
        matrix = gerar_word2vec(textos)
        insights = {"note": "Insights disponíveis apenas para TF-IDF."}
        features = []

    elif tipo == "transformer":
        matrix = gerar_transformer(textos)
        insights = {"note": "Insights disponíveis apenas para TF-IDF."}
        features = []

    else:
        return "Método inválido", 400

    # Gera gráficos
    outputs = gerar_graficos(matrix, tipo)

    return render_template(
        "resultados.html",
        literatura_clean=literatura_display,
        reviews_clean=reviews_display,
        textos=textos,
        metodo=tipo.upper(),
        outputs=outputs,
        insights=insights
    )


# -----------------------------------------
# Run
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
