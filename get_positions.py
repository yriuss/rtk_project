import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Função para calcular a distância em metros entre duas coordenadas
def calcular_distancia(coord1, coord2):
    return geodesic(coord1, coord2).meters

# Ler o CSV
df = pd.read_csv('gps_data_test3.csv')

# Coordenada inicial como referência
ref_coord = (df['Latitude'][0], df['Longitude'][0])

# Calcular as distâncias em metros em relação à coordenada inicial
df['Distancia_Latitude'] = df['Latitude'].apply(lambda lat: calcular_distancia(ref_coord, (lat, ref_coord[1])))
df['Distancia_Longitude'] = df['Longitude'].apply(lambda lon: calcular_distancia(ref_coord, (ref_coord[0], lon)))

# Corrigir sinais das distâncias (N/S e E/W)
df['Distancia_Latitude'] = df['Distancia_Latitude'] * df['Latitude'].apply(lambda lat: 1 if lat >= ref_coord[0] else -1)
df['Distancia_Longitude'] = df['Distancia_Longitude'] * df['Longitude'].apply(lambda lon: 1 if lon >= ref_coord[1] else -1)

# Calcular média, desvio padrão e a máxima diferença em metros
media_metros_lat = df['Distancia_Latitude'].mean()
media_metros_lon = df['Distancia_Longitude'].mean()
desvio_metros_lat = df['Distancia_Latitude'].std()
desvio_metros_lon = df['Distancia_Longitude'].std()
max_diferenca_lat = df['Distancia_Latitude'].max() - df['Distancia_Latitude'].min()
max_diferenca_lon = df['Distancia_Longitude'].max() - df['Distancia_Longitude'].min()

# Exibir resultados
print(f'Média Latitude em Metros: {media_metros_lat:.3f} m')
print(f'Média Longitude em Metros: {media_metros_lon:.3f} m')
print(f'Desvio Padrão Latitude em Metros: {desvio_metros_lat:.3f} m')
print(f'Desvio Padrão Longitude em Metros: {desvio_metros_lon:.3f} m')
print(f'Máxima Diferença Latitude em Metros: {max_diferenca_lat:.3f} m')
print(f'Máxima Diferença Longitude em Metros: {max_diferenca_lon:.3f} m')

# Plotar o gráfico com as distâncias em metros
plt.figure(figsize=(10, 6))
plt.plot(df['Distancia_Longitude'], df['Distancia_Latitude'], marker='o', linestyle='-')
plt.title('Trajetória em Metros')
plt.xlabel('Distância Longitude (m)')
plt.ylabel('Distância Latitude (m)')
plt.grid(True)
plt.show()
