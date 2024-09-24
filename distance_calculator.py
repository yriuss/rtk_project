from geopy.distance import geodesic

def calcular_distancia_vincenty(lat1, lon1, lat2, lon2):
    # Define os pontos
    ponto1 = (lat1, lon1)
    ponto2 = (lat2, lon2)
    
    # Calcula a distância usando o método geodesic, que usa a fórmula de Vincenty
    distancia = geodesic(ponto1, ponto2).meters  # A distância em metros
    
    return distancia

# Exemplo de uso
lat1, lon1 = -8.0306521, -34.5720393 
lat2, lon2 = -8.0306535, -34.5720457 

distancia = calcular_distancia_vincenty(lat1, lon1, lat2, lon2)
print(f"A distância entre os pontos é de {distancia:.4f} metros.")
distancia_cm = distancia * 100
print(f"A distância entre os pontos é de {distancia_cm:.2f} centímetros.")


#0803.06521,S,03457.20393

#0803.06535,S,03457.20474,W,131200.00,A,