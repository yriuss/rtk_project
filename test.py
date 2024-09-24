import csv
from math import radians, sin, cos, sqrt, atan2

# Função para calcular a distância Haversine entre dois pontos
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Raio da Terra em metros
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

# Ponto de referência (latitude, longitude)
lat_ref, lon_ref = -8.0500649,-34.9498112

# Lista para armazenar as distâncias
distances = []

# Caminho para o arquivo CSV
csv_file = 'gnss_positions_test7.csv'

# Ler o arquivo CSV e calcular as distâncias
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        # Calcular a distância do ponto de referência
        distance = haversine(lat_ref, lon_ref, lat, lon)
        distances.append(distance)

# Calcular distância mínima, média e máxima
min_distance = min(distances)
max_distance = max(distances)
avg_distance = sum(distances) / len(distances)

# Exibir os resultados
print(f"Distância mínima: {min_distance:.2f} metros")
print(f"Distância média: {avg_distance:.2f} metros")
print(f"Distância máxima: {max_distance:.2f} metros")
