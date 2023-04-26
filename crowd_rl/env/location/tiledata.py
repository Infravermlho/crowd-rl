import pygame


class TileData:
    def __init__(self):
        self.tiles = {}
        self.tile_size = 16
        self._load_tiles()

    def _load_tiles(self):
        # Data Tile 1
        self.tiles["1"] = Tile()
        self.tiles["1"].name = "Wall"
        self.tiles["1"].colide = True
        self.tiles["1"].draw = lambda screen, x, y: pygame.draw.rect(
            screen, (0, 0, 0), (x, y, 15, 15), 15)

        # Data Tile 2
        self.tiles["2"] = Tile()
        self.tiles["2"].body = True
        self.tiles["2"].colide = True
        self.tiles["2"].draw = lambda screen, x, y: pygame.draw.rect(
            screen, (255, 255, 255), (x, y, 15, 15), 5)

        # Data Tile 3
        self.tiles["3"] = Tile()
        self.tiles["3"].body = True
        self.tiles["3"].colide = False
        self.tiles["3"].draw = lambda screen, x, y: pygame.draw.circle(
            screen, (255, 160, 122), (x + 7.5, y + 7.5), 15/2, 5)


class Tile:
    def __init__(self):
        self.name = ''
        self.colide = True
        self.draw = None
