import pandas as pd
import utm
import time
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv('gnss_positions_test13.csv')

# Função para converter latitude e longitude para UTM
def latlon_to_utm(lat, lon):
    return utm.from_latlon(lat, lon)

# Converter todas as coordenadas para UTM
df['UTM'] = df.apply(lambda row: latlon_to_utm(row['Latitude'], row['Longitude']), axis=1)

# Pegar a primeira coordenada como origem
utm_origem = df['UTM'].iloc[0]
x_origem, y_origem = utm_origem[0], utm_origem[1]

# Inicializar listas para coordenadas relativas
x_rel = []
y_rel = []
timestamps = []

# Configurar o gráfico
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("X (metros)")
ax.set_ylabel("Y (metros)")
ax.set_title("Posição UTM relativa ao longo do tempo")
scat, = ax.plot([], [], 'ro')  # Usando plot ao invés de scatter

# Mostrar uma coordenada de cada vez com o timestamp, usando a origem como referência
for i, row in df.iterrows():
    utm_coord = row['UTM']
    timestamp = row['Timestamp']
    
    # Calcular as coordenadas relativas subtraindo a origem
    x_relativo = utm_coord[0] - x_origem
    y_relativo = utm_coord[1] - y_origem
    
    # Adicionar as coordenadas relativas e timestamp à lista
    x_rel.append(x_relativo)
    y_rel.append(y_relativo)
    timestamps.append(timestamp)
    
    # Atualizar o gráfico com as novas coordenadas
    scat.set_data(x_rel, y_rel)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(1)  # Pausar 1 segundo antes de mostrar a próxima coordenada
    
    # Imprimir as coordenadas relativas
    print(f"Timestamp: {timestamp}, Coordenada UTM Relativa: (X: {x_relativo}, Y: {y_relativo})")

plt.ioff()  # Desativar o modo interativo
plt.show()
