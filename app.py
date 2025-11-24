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


# ----------------------
# Pré-processamento
# ----------------------
def limpar_texto(t):
    return t.lower().strip()


# ----------------------
# TF-IDF
# ----------------------
def gerar_tfidf(docs):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs).toarray()
    features = vectorizer.get_feature_names_out()
    return matrix, features


# ----------------------
# Word2Vec
# ----------------------
def gerar_word2vec(docs):
    tokenizado = [t.split() for t in docs]

    model = Word2Vec(sentences=tokenizado, vector_size=100, window=5, min_count=1, workers=2)
    vectors = []

    for palavras in tokenizado:
        if palavras:
            vet = np.mean([model.wv[p] for p in palavras], axis=0)
            vectors.append(vet)
        else:
            vectors.append(np.zeros(100))
    return np.array(vectors)


# ----------------------
# Transformers (sem PyTorch)
# ----------------------
def gerar_transformer(docs):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(docs)
    return embeddings


# ----------------------
# Geração dos gráficos
# ----------------------
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

def gerar_insights(matrix):
    return {
        "note": "Análise concluída automaticamente."
    }


# ----------------------
# Rota principal
# ----------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------
# Execução da pipeline
# ----------------------
@app.route("/run", methods=["POST"])
def run_pipeline():
    tipo = request.form["metodo"]
    literatura_raw = request.form["literatura_text"].split("\n")
    reviews_raw = request.form["reviews_text"].split("\n")

    textos = [limpar_texto(t) for t in literatura_raw + reviews_raw if t.strip()]

    # Seleção da técnica
    if tipo == "tfidf":
        matrix, features = gerar_tfidf(textos)
    elif tipo == "word2vec":
        matrix = gerar_word2vec(textos)
        features = []
    elif tipo == "transformer":
        matrix = gerar_transformer(textos)
        features = []
    else:
        return "Método inválido", 400

    # Gera gráficos
    outputs = gerar_graficos(matrix, tipo)
    insights = gerar_insights(matrix)

    return render_template(
        "resultados.html",
        textos=textos,
        metodo=tipo.upper(),
        outputs=outputs,
        insights=insights
    )


if __name__ == "__main__":
    app.run(debug=True)