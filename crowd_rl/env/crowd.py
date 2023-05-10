import pygame
from crowd_rl.env.actor import Actor
from crowd_rl.env.location.location import Location

clock = pygame.time.Clock()


class Crowd:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.weight, self.height = 1200, 800
        self.display = self.weight, self.height = 600, 400
        self.scrollvalue = [0, 0]

        self.actors = []
        self.selected_actor = 0

    def on_execute(self):
        self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            clock.tick(60)
        self.on_cleanup

    def on_init(self):
        pygame.init()
        self._screen = pygame.display.set_mode(
            self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._display_surf = pygame.Surface((300, 200))

        pygame.display.set_caption("Debugging CrowdSim")

        self.location = Location()
        self.actors.append(Actor(x=120, y=120))

        self._running = True

    def on_loop(self):
        for actor in self.actors:
            actor.update(self.location.tile_retangulos)

    def on_render(self):
        self._display_surf.fill((105, 155, 125))
        for actor in self.actors:
            actor.draw(self._display_surf, self.scrollvalue)
        self.location.blitnivel(self._display_surf, self.scrollvalue)

        displaysurf = pygame.transform.scale(self._display_surf, self.size)
        self._screen.blit(displaysurf, (0, 0))
        pygame.display.update()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.actors[self.selected_actor].goLeft()
            elif event.key == pygame.K_RIGHT:
                self.actors[self.selected_actor].goRight()
            elif event.key == pygame.K_UP:
                self.actors[self.selected_actor].goUp()
            elif event.key == pygame.K_DOWN:
                self.actors[self.selected_actor].goDown()
            elif event.key == pygame.K_x:
                self.actors[self.selected_actor].stopY()
                self.actors[self.selected_actor].stopX()
                self.selected_actor = (
                    self.selected_actor + 1) % len(self.actors)
                print("[on_event] Ator ativo: " + str(self.selected_actor))
            elif event.key == pygame.K_c:
                self.actors.append(Actor(x=120, y=120))

            elif event.key == pygame.K_k:
                print("[on_event] scrollvalue X: " +
                      str(self.scrollvalue[0]) + "Y: " + str(self.scrollvalue[1]))
                self.scrollvalue[1] += 1
            elif event.key == pygame.K_i:
                print("[on_event] scrollvalue X: " +
                      str(self.scrollvalue[0]) + "Y: " + str(self.scrollvalue[1]))
                self.scrollvalue[1] += -1
            elif event.key == pygame.K_l:
                print("[on_event] scrollvalue X: " +
                      str(self.scrollvalue[0]) + "Y: " + str(self.scrollvalue[1]))
                self.scrollvalue[0] += 1
            elif event.key == pygame.K_j:
                print("[on_event] scrollvalue X: " +
                      str(self.scrollvalue[0]) + "Y: " + str(self.scrollvalue[1]))
                self.scrollvalue[0] += -1

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT and self.actors[self.selected_actor]._changeX < 0:
                self.actors[self.selected_actor].stopX()
            elif event.key == pygame.K_RIGHT and self.actors[self.selected_actor]._changeX > 0:
                self.actors[self.selected_actor].stopX()
            elif event.key == pygame.K_UP and self.actors[self.selected_actor]._changeY < 0:
                self.actors[self.selected_actor].stopY()
            elif event.key == pygame.K_DOWN and self.actors[self.selected_actor]._changeY > 0:
                self.actors[self.selected_actor].stopY()

    def on_cleanup(self):
        pass

