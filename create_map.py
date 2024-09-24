import pandas as pd
import pygame
import utm

# Inicialização do Pygame
pygame.init()

# Carregar o CSV com os dados de latitude e longitude
df = pd.read_csv('gnss_positions_test12.csv')

# Converter latitude e longitude para UTM
utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(df['Latitude'], df['Longitude'])]
utm_x = [coord[0] for coord in utm_coords]
utm_y = [-coord[1] for coord in utm_coords]

# Carregar a imagem PNG
img = pygame.image.load('pista_cooper.png')
img_width, img_height = img.get_size()

# Inicializar variáveis para movimento e escala
x_offset, y_offset = 0, 0
scale_factor = 1.0

# Definir ponto de referência (por exemplo, o primeiro ponto)
ref_x = utm_x[0]
ref_y = utm_y[0]

# Configuração da tela com o tamanho exato da imagem
screen = pygame.display.set_mode((img_width, img_height), pygame.RESIZABLE)
pygame.display.set_caption('Mapa com Pygame')

# Função para desenhar a imagem e os pontos
def draw_map():
    # Limpa a tela
    screen.fill((211, 211, 211))
    
    # Desenha a imagem como fundo
    screen.blit(img, (0, 0))

    # Desenha os pontos do mapa
    for x, y in zip(utm_x, utm_y):
        adjusted_x = int((x - ref_x) * scale_factor) + x_offset
        adjusted_y = int((y - ref_y) * scale_factor) + y_offset
        pygame.draw.circle(screen, (211, 0, 0), (adjusted_x, adjusted_y), 3)  # Ponto vermelho maior

    pygame.display.flip()

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:  # Pressione 'q' para sair
        running = False
    if keys[pygame.K_i]:  # Aumentar escala
        scale_factor *= 1.01
    if keys[pygame.K_u]:  # Diminuir escala
        scale_factor *= 0.99
    if keys[pygame.K_LEFT]:  # Tecla esquerda
        x_offset -= 0.5
    if keys[pygame.K_RIGHT]:  # Tecla direita
        x_offset += 0.5
    if keys[pygame.K_UP]:  # Tecla cima
        y_offset -= 0.5
    if keys[pygame.K_DOWN]:  # Tecla baixo
        y_offset += 0.5

    draw_map()

# Finaliza o Pygame
pygame.quit()
