import pygame
import random


def checkcolision(rect, tiles):
    hit_list = []
    for tile in tiles:
        if rect.colliderect(tile["body"]):
            hit_list.append(tile)
    return hit_list


class Actor(pygame.sprite.Sprite):
    def __init__(self, x, y):
        self._rect = pygame.Rect(350, 350, 5, 5)
        self._color = (random.randrange(
            255), random.randrange(255), random.randrange(255))
        self._rect.x = x
        self._rect.y = y

        # Movimento
        self._changeX = 0
        self._changeY = 0
        self._baseSpeed = 1

    def draw(self, screen, scroll):
        imaginary_rect = self._rect
        imaginary_rect.x -= scroll[0]
        imaginary_rect.y -= scroll[1]

        pygame.draw.rect(screen, self._color, self._rect)

        # TENHO QUE ARRUMAR ISSO
        imaginary_rect.x += scroll[0]
        imaginary_rect.y += scroll[1]

    def update(self, tile_list):
        self._rect.x += self._changeX
        lista_colisao = checkcolision(self._rect, tile_list)

        for tile in lista_colisao:
            print("[on_update] Colisão com bloco: " + tile["id"])

            if (tile["colide"]):
                if self._changeX > 0:
                    self._rect.right = tile["body"].left
                elif self._changeX < 0:
                    self._rect.left = tile["body"].right

        self._rect.y += self._changeY
        lista_colisao = checkcolision(self._rect, tile_list)

        for tile in lista_colisao:
            print("[on_update] Colisão com bloco: " + tile["id"])

            if (tile["colide"]):
                if self._changeY > 0:
                    self._rect.bottom = tile["body"].top
                elif self._changeY < 0:
                    self._rect.top = tile["body"].bottom

    def goUp(self):
        self._changeY = -self._baseSpeed

    def goDown(self):
        self._changeY = self._baseSpeed

    def goRight(self):
        self._changeX = self._baseSpeed

    def goLeft(self):
        self._changeX = -self._baseSpeed

    def stopX(self):
        self._changeX = 0

    def stopY(self):
        self._changeY = 0
