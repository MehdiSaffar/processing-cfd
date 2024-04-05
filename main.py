import numpy as np
import pygame
import pygame.gfxdraw
import itertools as it

np.random.seed(0)
np.seterr("raise")


class K:
    window_width = 800
    window_height = 600
    g = 9.81
    # g = 0
    # g = g / 4
    bounce_coeff = 0.9
    # scale = window_height / 50
    scale = 1
    scene_width = window_width / scale
    scene_height = window_height / scale
    radius = 4
    sr = radius * 20
    target_density = 0.5
    pressure_multiplier = 1
    mass = 1
    gap = 4


cell_offsets = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


class SpatialLookup:
    def get_cell_coords(self, x, y):
        i = int(x // K.sr)
        j = int(y // K.sr)
        return i, j

    def get_cell_hash(self, i, j):
        return i * 15823 + j * 9737333

    def get_cell_key(self, hash):
        return hash % len(self.positions)

    def update(self, positions):
        self.positions = positions
        N = len(self.positions)
        self.spatial_lookup = [None] * N
        self.start_indices = [None] * N

        for i in range(N):
            x, y = positions[i]
            ci, cj = self.get_cell_coords(x, y)
            cell_hash = self.get_cell_hash(ci, cj)
            cell_key = self.get_cell_key(cell_hash)
            self.spatial_lookup[i] = (cell_key, i)

        self.spatial_lookup.sort()

        for i in range(N):
            key = self.spatial_lookup[i][0]
            prev_key = self.spatial_lookup[i - 1][0] if i else None
            if not i or key != prev_key:
                self.start_indices[key] = i

    def iter_close_cells(self, i, j):
        for di, dj in cell_offsets:
            yield i + di, j + dj

    def iter_points_in_cell(self, i, j):
        key = self.get_cell_key(self.get_cell_hash(i, j))
        start_index = self.start_indices[key]
        if start_index == None:
            return

        for next_key, point_index in it.islice(self.spatial_lookup, start_index, None):
            if next_key == key:
                yield point_index

    def close_points(self, ai):
        a = self.positions[ai]
        i, j = self.get_cell_coords(a[0], a[1])

        for ni, nj in self.iter_close_cells(i, j):
            for bi in self.iter_points_in_cell(ni, nj):
                b = self.positions[bi]
                if np.linalg.norm(a - b) <= K.sr:
                    yield bi


class App:
    def init(self):
        pygame.init()
        self.screen = pygame.display.set_mode([K.window_width, K.window_height])
        self.dt = 0
        self.clock = pygame.time.Clock()
        self.running = False
        self.side = 20
        self.N = self.side**2
        self.positions = np.random.uniform(
            [0, 0], [K.scene_width, K.scene_height], size=(self.N, 2)
        )
        # self.positions = self.generate_grid_positions(self.side, K.radius, K.gap)
        # self.positions += np.random.rand(*self.positions.shape)*0.7
        self.velocities = np.zeros((self.N, 2))
        self.densities = np.zeros(self.N)
        self.pressure_forces = np.zeros((self.N, 2))
        self.colors = np.random.randint(1 / 4 * 255, 255, size=(self.N, 3))
        self.spatial_lookup = SpatialLookup()

    def generate_grid_positions(self, side, radius, gap):
        x_range = np.arange(side, dtype=float) * (2 * radius + gap) + radius + gap + 40
        y_range = np.arange(side, dtype=float) * (2 * radius + gap) + radius + gap + 10
        X, Y = np.meshgrid(x_range, y_range)
        return np.dstack([X, Y]).reshape((-1, 2))

    def resolve_collisions(self):
        ps, vs = self.positions, self.velocities

        top_condition = (ps[:, 1] + K.radius) >= K.scene_height
        bottom_condition = (ps[:, 1] - K.radius) <= 0
        left_condition = (ps[:, 0] - K.radius) <= 0
        right_condition = (ps[:, 0] + K.radius) >= K.scene_width

        vs[left_condition, 0] = np.abs(vs[left_condition, 0]) * K.bounce_coeff
        vs[right_condition, 0] = -np.abs(vs[right_condition, 0]) * K.bounce_coeff

        vs[top_condition, 1] = -np.abs(vs[top_condition, 1]) * K.bounce_coeff
        vs[bottom_condition, 1] = np.abs(vs[bottom_condition, 1]) * K.bounce_coeff

        # Update positions to ensure they stay within bounds
        ps[:, 0] = np.clip(ps[:, 0], K.radius, K.scene_width - K.radius)
        ps[:, 1] = np.clip(ps[:, 1], K.radius, K.scene_height - K.radius)

    def density_to_pressure(self, density):
        error = density - K.target_density
        return error * K.pressure_multiplier

    ############## VECTORIZED

    # def calculate_pressure_force(self, sample_points):
    #     # calculate direction vectors for all positions and sample points
    #     direction = self.positions[:, np.newaxis] - sample_points[np.newaxis, :]

    #     # Calculate distances for all positions and sample points
    #     distance = np.linalg.norm(direction, axis=2)

    #     # Normalize direction vectors
    #     direction /= distance[:, :, np.newaxis]

    #     # Calculate slope for all distances
    #     slope = self.smoothing_kernel_derivative(distance)

    #     # Calculate pressure for all positions
    #     pressure = self.density_to_pressure(self.densities)
    #     pressure = (pressure[:, np.newaxis] + pressure[np.newaxis, :]) / 2

    #     # Calculate force components
    #     x = -pressure[:, :, np.newaxis] * direction * slope[:, :, np.newaxis] * K.mass
    #     y = self.densities[:, np.newaxis]
    #     z = x / y
    #     np.nan_to_num(z, 0)

    #     # Sum forces along the sample points axis
    #     total_force = np.sum(z, axis=0)

    #     return total_force

    # def calculate_pressure_forces(self):
    #     self.pressure_forces = self.calculate_pressure_force(self.positions)

    # def calculate_densities(self):
    #     self.densities = self.calculate_density(self.positions)

    # def calculate_density(self, points):
    #     distance = np.linalg.norm(self.positions[:, np.newaxis] - points[np.newaxis, :], axis=2)
    #     influence = self.smoothing_kernel(distance)
    #     return np.sum(K.mass * influence, axis=0)

    ############ NORMAL

    def calculate_densities(self, positions):
        for i in range(self.N):
            density = 0
            for j in self.spatial_lookup.close_points(i):
                distance = np.linalg.norm(positions[j] - positions[i])
                influence = self.smoothing_kernel(distance)
                density += influence * K.mass
            assert density > 0
            self.densities[i] = density

    def calculate_pressure_forces(self):
        for i in range(self.N):
            force = 0
            for j in self.spatial_lookup.close_points(i):
                direction = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(direction)
                if distance == 0:
                    continue
                direction /= distance
                slope = self.smoothing_kernel_derivative(distance)

                pi = self.density_to_pressure(self.densities[i])
                pj = self.density_to_pressure(self.densities[j])
                sp = (pi + pj) / 2

                force += (-sp * direction * slope * K.mass) / self.densities[j]

            self.pressure_forces[i] = force

    ############### KERNEL

    def smoothing_kernel(self, dst):
        if dst >= K.sr:
            return 0
        volume = (np.pi * K.sr**4) / 6
        value = (K.sr - dst) ** 2 / volume
        return value

    def smoothing_kernel_derivative(self, dst):
        if dst >= K.sr:
            return 0
        scale = 12 / (np.pi * K.sr**4)
        return (dst - K.sr) * scale

    ############### MAIN

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def simulate(self):
        self.resolve_collisions()
        self.spatial_lookup.update(self.positions)
        self.calculate_densities(self.positions)
        self.calculate_pressure_forces()
        self.velocities += (
            self.pressure_forces / self.densities[:, np.newaxis] * self.dt
        ) + np.array([0, K.g]) * self.dt
        self.positions += self.velocities * self.dt

    def map_density_to_color(self, density):
        # Example colormap: blue to red gradient
        blue = (0, 0, 255)
        red = (255, 0, 0)
        # Map density values to colors
        min_density = np.min(density)
        max_density = np.max(density)
        normalized_density = (density - min_density) / (
            (max_density - min_density) if max_density > min_density else 1
        )
        colors = []
        for d in normalized_density:
            r = int((1 - d) * blue[0] + d * red[0])
            g = int((1 - d) * blue[1] + d * red[1])
            b = int((1 - d) * blue[2] + d * red[2])
            colors.append((r, g, b))
        return colors

    def map_cell_to_color(self):
        colors = []
        for i in range(self.N):
            x, y = self.positions[i]
            ci, cj = self.spatial_lookup.get_cell_coords(x, y)
            key = self.spatial_lookup.get_cell_key(
                self.spatial_lookup.get_cell_hash(ci, cj)
            )
            c = (int(key * 255 / self.N), 0, 0)
            # print(c)
            colors.append(c)
        return colors

    def draw(self):
        self.screen.fill((255, 255, 255))
        pos = self.positions.copy()

        self.colors = self.map_density_to_color(self.densities)
        # self.colors = self.map_cell_to_color()

        for i in range(self.N):
            x, y = self.positions[i]
            pygame.draw.circle(
                self.screen,
                self.colors[i],
                pos[i],
                K.radius,
            )

            # pygame.draw.circle(
            #     self.screen,
            #     self.colors[i],
            #     pos[i],
            #     K.sr,
            #     width=1,
            # )

            # pressure_force = self.pressure_forces[i] / self.densities[i]

            # endpoint = (x + pressure_force[0], y + pressure_force[1])
            # pygame.draw.line(
            #     self.screen,
            #     self.colors[i],  # Arrow color (black)
            #     (x, y),
            #     endpoint,
            #     2,  # Line thickness
            # )

        pygame.display.flip()

    def run(self):
        self.running = True
        while self.running:
            self.process_events()
            self.simulate()
            self.draw()

            self.dt = self.clock.tick() / 1000

        pygame.quit()


# Run until the user asks to quit
app = App()
app.init()
app.run()
