import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import zscore

df = pd.read_csv("assets/europe.csv")

countries = df["Country"]
X = df.drop("Country", axis=1)
X_scaled = zscore(X, axis=0)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_pca[:, 0], X_pca[:, 1], color="blue")
for i, country in enumerate(countries):
    ax.text(X_pca[i, 0], X_pca[i, 1], country, fontsize=8, color="black")

for i, var in enumerate(X.columns):
    ax.arrow(0, 0, pca.components_[0, i] * 2, pca.components_[1, i] * 2, color="red", alpha=0.8, head_width=0.05)
    ax.text(pca.components_[0, i] * 2.2, pca.components_[1, i] * 2.2, var, color="black", fontsize=10)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Valores de las Componentes Principales 1 y 2")
ax.grid(True)
plt.tight_layout()
plt.savefig("biplot.png")
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(countries, X_pca[:, 0])
ax.set_xlabel("País")
ax.set_ylabel("PC1")
ax.set_title("Valor de PC1 por país")
ax.set_xticklabels(countries, rotation=90)
plt.tight_layout()
plt.savefig("pc1_by_country.png")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
loadings = pd.Series(pca.components_[0], index=X.columns)
loadings.sort_values(ascending=False).plot(kind="bar", ax=ax, width=0.8)
ax.set_xlabel("Variable")
ax.set_ylabel("Contribución a PC1")
ax.set_title("Contribución de cada variable al Primer Componente Principal")
plt.tight_layout()
plt.savefig("pc1_loadings.png")
plt.show()
