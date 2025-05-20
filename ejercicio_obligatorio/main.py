import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the data
df = pd.read_csv('ejercicio_obligatorio/europe.csv')

# 2. Preprocess: Remove 'Country', standardize the data
countries = df['Country']
X = df.drop('Country', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. Biplot: countries and variable vectors
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='purple')
for i, country in enumerate(countries):
    plt.text(X_pca[i, 0], X_pca[i, 1], country, fontsize=8, color='purple')

# Draw arrows for each variable
for i, var in enumerate(X.columns):
    plt.arrow(0, 0, pca.components_[0, i]*2, pca.components_[1, i]*2, color='cyan', alpha=0.8, head_width=0.05)
    plt.text(pca.components_[0, i]*2.2, pca.components_[1, i]*2.2, var, color='cyan', fontsize=10)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Valores de las Componentes Principales 1 y 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar plot of PC1 per country
plt.figure(figsize=(14, 6))
plt.bar(countries, X_pca[:, 0])
plt.xlabel('País')
plt.ylabel('PC1')
plt.title('Valor de PC1 por país')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 5. Interpret PC1: Print loadings
loadings = pd.Series(pca.components_[0], index=X.columns)
print('PC1 Loadings:')
print(loadings.sort_values(ascending=False))

# Theoretical interpretation:
print('\nInterpretación teórica de PC1:')
print('Las cargas (loadings) de PC1 muestran qué variables contribuyen más a la primera componente principal. ')
print('Valores altos (positivos o negativos) indican mayor influencia. Analiza las variables con mayor valor absoluto para interpretar el eje principal de variación entre los países.')
