import pygame
from location.tiledata import TileData


class Location:
    # Carrega o arquivo .csv como um nivel
    def __init__(self):
        self.tile_data = TileData()
        self.tile_retangulos = []
        self.tile_coords = []
        self._leveldata = None

        self.loadnivel('crowd\location.csv')

    def loadnivel(self, nomearquivo):
        file_reader = []
        with open(nomearquivo, 'r') as txt:
            for linha in txt.read().splitlines():
                file_reader.append(linha.split(','))

        self.leveldata = file_reader

    def blitnivel(self, screen, scroll):
        tile_retangulos = []
        tile_size = self.tile_data.tile_size

        y = 0
        for row in self.leveldata:
            x = 0
            for tile in row:
                if tile != '0':
                    tilepos = x * tile_size, y * tile_size, tile_size, tile_size
                    renderpos = x * tile_size - scroll[0], y * tile_size - scroll[1]

                    tile_retangulos.append({
                            "body": pygame.Rect(tilepos),
                            "id": tile,
                            "colide": self.tile_data.tiles[tile].colide
                    })

                    self.tile_data.tiles[tile].draw(
                        screen, renderpos[0], renderpos[1])
                x += 1
            y += 1

        self.tile_retangulos = tile_retangulos
