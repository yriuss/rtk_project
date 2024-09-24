import pandas as pd
import matplotlib.pyplot as plt
import utm
import matplotlib.image as mpimg

# Carregar o CSV com os dados de latitude e longitude
df = pd.read_csv('gnss_positions.csv')

# Converter latitude e longitude para UTM
utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(df['Latitude'], df['Longitude'])]
utm_x = [coord[0] for coord in utm_coords]
utm_y = [-coord[1] for coord in utm_coords]  # Usar coordenada Y invertida para compatibilidade com o sistema gráfico

# Carregar a imagem PNG
img = mpimg.imread('pista_cooper.png')

# Definir o ponto de referência (por exemplo, o primeiro ponto)
ref_x = utm_x[0]
ref_y = utm_y[0]

# Ajustar os pontos para a imagem
adjusted_x = [(x - ref_x) for x in utm_x]
adjusted_y = [(y - ref_y) for y in utm_y]

# Plot 1: Plotar a imagem de fundo e os pontos UTM ajustados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotar a imagem de fundo
ax1.imshow(img, extent=[min(adjusted_x), max(adjusted_x), min(adjusted_y), max(adjusted_y)])

# Plotar os pontos GNSS sobre o mapa
ax1.scatter(adjusted_x, adjusted_y, color='red', s=10, label='GNSS Points')

# Configurações do gráfico
ax1.set_title("Mapa com Pontos GNSS (UTM)")
ax1.set_xlabel("UTM X")
ax1.set_ylabel("UTM Y")
ax1.legend()

# Plot 2: Plotar as longitudes e latitudes cruas no scatter plot
# Definir o primeiro ponto como origem
origin_lat = df['Latitude'][0]
origin_lon = df['Longitude'][0]

# Ajustar as coordenadas relativas à origem
relative_lat = df['Latitude'] - origin_lat
relative_lon = df['Longitude'] - origin_lon

# Scatter plot para latitudes e longitudes cruas
ax2.scatter(relative_lon, relative_lat, color='blue', s=10, label='Longitude/Latitude Raw')

# Configurações do gráfico
ax2.set_title("Pontos de Longitude e Latitude Relativos à Origem")
ax2.set_xlabel("Longitude Relativa")
ax2.set_ylabel("Latitude Relativa")
ax2.legend()

# Exibir os gráficos
plt.tight_layout()
plt.show()
