"""
Neuro Bonsai V3
A bonsai whose branches are GROWN BY A NEURAL NETWORK, and whose brain
EVOLVES from generation to generation (Nature of Code: neural networks +
neuroevolution + noise + particles + recursion).

How it works
  - The trunk is a bonsai-style S-curve with tiered side branches.
  - Every branch tip asks a small neural network (6 inputs -> 6 hidden -> 3
    outputs) what to do:  turn angle, length factor, split or not.
  - Each tree is scored (canopy coverage, balance, height, staying in frame).
  - Reproduction = mutate the best brain so far ("1+1 evolution strategy").
  - Pulses of "signal" travel through the branches like through axons.
  - Seasons, wind noise, falling leaves and snow keep it alive.

Controls
  SPACE pause | 1 2 3 speed | N next generation | E evolve 60 generations
  R new lineage | T change bonsai style | M show/hide the network | ESC quit
"""

import math
import random
import sys

import pygame

pygame.init()

# ----------------------------------------------------------------------
# Window / pixel-art canvas
# ----------------------------------------------------------------------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640

SCALE = 4
WIDTH = WINDOW_WIDTH // SCALE
HEIGHT = WINDOW_HEIGHT // SCALE

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neuro Bonsai V3")
canvas = pygame.Surface((WIDTH, HEIGHT))

font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 16)
small_font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 13)
tiny_font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 12)

# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
BACKGROUND = (21, 26, 29)
GROUND = (31, 38, 37)
GROUND_LINE = (65, 75, 65)

POT = (122, 70, 58)
POT_DARK = (92, 52, 45)
SOIL = (44, 34, 30)

TRUNK_DARK = (66, 45, 34)
TRUNK_LIGHT = (116, 88, 62)

LEAF_PINK = (232, 170, 190)
LEAF_SPRING = (126, 190, 98)
LEAF_SUMMER = (62, 140, 78)
LEAF_AUTUMN_1 = (222, 168, 52)
LEAF_AUTUMN_2 = (200, 72, 44)
SNOW = (225, 232, 240)

PULSE = (110, 235, 255)

# ----------------------------------------------------------------------
# World layout
# ----------------------------------------------------------------------
GROUND_Y = HEIGHT - 35
CX = WIDTH // 2
POT_TOP = GROUND_Y - 10
SOIL_Y = POT_TOP - 3

# ----------------------------------------------------------------------
# Simulation settings
# ----------------------------------------------------------------------
YEAR_SECONDS = 18        # real seconds per year at speed x1
LIFESPAN = 14            # years a tree lives before it "reproduces"
FIRST_BIRTH = 0.15       # growth-years before the trunk sprouts
GROW_STEP = 0.32         # delay between a branch and its children
GROW_TIME = 0.55         # growth-years a single branch needs to grow
MAX_SUB = 5              # how many times a side branch may keep branching
MAX_NODES = 260
LEAVES_PER_TIP = 13
MAX_FALLING = 1000
PULSE_PERIOD = 130.0

STYLE_NAMES = ["Informal upright", "Slanting", "Windswept"]

# ----------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------
N_IN, N_HID, N_OUT = 6, 6, 3
IN_LABELS = ["sun", "height", "depth", "crowd", "heading", "lateral"]
OUT_LABELS = ["turn", "length", "split"]

_W1 = 0
_B1 = N_IN * N_HID
_W2 = _B1 + N_HID
_B2 = _W2 + N_HID * N_OUT
N_WEIGHTS = _B2 + N_OUT

SUN = (-0.45, -0.893)    # light comes from the upper left


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp(t)
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )


def shade(color, amount):
    return tuple(int(clamp(c + amount, 0, 255)) for c in color)


def smooth(t):
    return t * t * (3 - 2 * t)


def _hash01(n):
    n = (n << 13) ^ n
    v = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF
    return v / 0x7FFFFFFF


def noise1d(x):
    """Smooth 1D value noise, returns 0..1."""
    i = int(math.floor(x))
    f = x - i
    f = f * f * (3 - 2 * f)
    return lerp(_hash01(i), _hash01(i + 1), f)


class Genome:
    """The weights of the growth network (this is what evolves)."""

    def __init__(self, weights=None):
        if weights is None:
            weights = [random.gauss(0, 0.9) for _ in range(N_WEIGHTS)]
        self.w = weights

    def mutated(self, rate=0.3, sigma=0.4):
        return Genome(
            [x + random.gauss(0, sigma) if random.random() < rate else x for x in self.w]
        )

    def forward(self, x):
        w = self.w
        hidden = []
        for h in range(N_HID):
            s = w[_B1 + h]
            for i in range(N_IN):
                s += w[_W1 + h * N_IN + i] * x[i]
            hidden.append(math.tanh(s))
        out = []
        for o in range(N_OUT):
            s = w[_B2 + o]
            for h in range(N_HID):
                s += w[_W2 + o * N_HID + h] * hidden[h]
            out.append(math.tanh(s))
        return hidden, out


# ----------------------------------------------------------------------
# Seasons
# ----------------------------------------------------------------------
class Season:
    def __init__(self, year_fraction):
        t = year_fraction

        if t < 0.25:
            self.name = "Spring"
        elif t < 0.5:
            self.name = "Summer"
        elif t < 0.75:
            self.name = "Autumn"
        else:
            self.name = "Winter"

        if t < 0.5:
            self.growth_rate = 1.0
        elif t < 0.75:
            self.growth_rate = 0.25
        else:
            self.growth_rate = 0.0

        if t < 0.2:
            self.leaf_grow = t / 0.2
        elif t < 0.75:
            self.leaf_grow = 1.0
        else:
            self.leaf_grow = 0.0

        if t < 0.5:
            self.autumn = 0.0
        elif t < 0.75:
            self.autumn = (t - 0.5) / 0.25
        else:
            self.autumn = 1.0

        if t < 0.15:
            self.leaf_color = lerp_color(LEAF_PINK, LEAF_SPRING, t / 0.15)
        elif t < 0.5:
            self.leaf_color = lerp_color(LEAF_SPRING, LEAF_SUMMER, (t - 0.15) / 0.35)
        elif t < 0.75:
            p = self.autumn
            if p < 0.5:
                self.leaf_color = lerp_color(LEAF_SUMMER, LEAF_AUTUMN_1, p * 2)
            else:
                self.leaf_color = lerp_color(LEAF_AUTUMN_1, LEAF_AUTUMN_2, (p - 0.5) * 2)
        else:
            self.leaf_color = LEAF_AUTUMN_2

        self.snowing = t >= 0.78


# ----------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------
class Leaf:
    def __init__(self, rng):
        # sample a flat elliptical foliage pad (bonsai "cloud")
        r = math.sqrt(rng.random())
        th = rng.random() * math.tau
        self.dx = math.cos(th) * r * 6.5
        self.dy = math.sin(th) * r * 3.2 - 1.5
        self.grow_at = rng.uniform(0.0, 0.9)
        self.drop_at = rng.uniform(0.0, 0.95)
        self.shade = rng.randint(-10, 10) - int(self.dy * 3)  # lighter on top
        self.fallen = False


class Branch:
    def __init__(self, depth, angle, length, birth, dist, trunk=False):
        self.depth = depth
        self.angle = angle            # relative to parent direction (0 = up)
        self.length = length
        self.birth = birth
        self.dist = dist              # path length from the root (for pulses)
        self.trunk = trunk
        self.flex = (depth + 1) * 0.004 * (1.0 if trunk else 1.4)
        self.children = []
        self.leaves = []
        self.ex = 0.0                 # static end position (for scoring)
        self.ey = 0.0
        self.tips = 1
        self.width = 1.0
        self.w_end = 1.0


class Tree:
    def __init__(self, genome, style, seed):
        self.genome = genome
        self.style = style
        self.rng = random.Random(seed)
        self.growth = 0.0

        self.grid = {}
        self.count = 0
        self.decisions = []   # (inputs, hidden, outputs) of every growth decision
        self.terminals = []

        self.root = self._build()
        self.fitness = self._score()

    # ---------------- construction ----------------
    def _register(self, x, y):
        key = (int(x // 7), int(y // 7))
        self.grid[key] = self.grid.get(key, 0) + 1
        self.count += 1

    def _crowd(self, x, y):
        cx, cy = int(x // 7), int(y // 7)
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n += self.grid.get((cx + dx, cy + dy), 0)
        return n

    def _build(self):
        rng = self.rng
        n = 8

        if self.style == 0:        # informal upright: S-curve
            lean, amp = rng.uniform(-0.1, 0.1), 0.42
        elif self.style == 1:      # slanting
            lean, amp = rng.choice((-1, 1)) * 0.4, 0.28
        else:                      # windswept
            lean, amp = rng.choice((-1, 1)) * 0.5, 0.2

        phase = rng.uniform(0, math.tau)
        x, y = CX, SOIL_Y
        prev = 0.0
        dist = 0.0
        parent = None
        root = None

        for i in range(n):
            theta = lean + amp * math.sin(i * 0.9 + phase)
            length = 12 - i * 0.85
            b = Branch(i, theta - prev, length, FIRST_BIRTH + i * GROW_STEP, dist, True)
            ex = x + math.sin(theta) * length
            ey = y - math.cos(theta) * length
            b.ex, b.ey = ex, ey
            self._register(ex, ey)

            if parent is None:
                root = b
            else:
                parent.children.append(b)

            # tiered side branch (alternating, or all one way if windswept)
            if 1 <= i <= n - 2:
                if self.style == 2:
                    side = 1 if lean > 0 else -1
                else:
                    side = 1 if i % 2 == 0 else -1
                rel = side * rng.uniform(1.2, 1.55)
                sb = Branch(i + 1, rel, 17 - i * 1.9, b.birth + GROW_STEP + 0.2, dist + length)
                b.children.append(sb)
                self._grow(sb, ex, ey, theta + rel, 0)

            parent = b
            prev = theta
            x, y = ex, ey
            dist += length

        self._finish(root)
        return root

    def _grow(self, b, x, y, a, sub):
        """Grow branch b (which starts at x, y heading a) and ask the NN what next."""
        ex = x + math.sin(a) * b.length
        ey = y - math.cos(a) * b.length
        b.ex, b.ey = ex, ey
        self._register(ex, ey)

        if sub >= MAX_SUB or b.length < 3.2 or self.count >= MAX_NODES:
            return

        hx, hy = math.sin(a), -math.cos(a)
        inputs = [
            hx * SUN[0] + hy * SUN[1],                    # facing the light?
            clamp((SOIL_Y - ey) / 90.0) * 2 - 1,          # height
            sub / MAX_SUB * 2 - 1,                        # how deep in the branch
            clamp(self._crowd(ex, ey) / 8.0) * 2 - 1,     # crowding
            hx,                                           # horizontal heading
            clamp((ex - CX) / 40.0, -1, 1),               # lateral position
        ]
        hidden, out = self.genome.forward(inputs)
        self.decisions.append((inputs, hidden, out))

        turn = out[0] * 0.9
        length_factor = 0.55 + 0.35 * (out[1] * 0.5 + 0.5)
        n_children = 2 if out[2] > -0.25 else 1

        for k in range(n_children):
            if n_children == 1:
                rel = turn
            else:
                rel = turn + (0.5 if k == 0 else -0.5)
            rel += self.rng.uniform(-0.1, 0.1)
            length = b.length * length_factor * (1.0 if k == 0 else 0.85)
            child = Branch(b.depth + 1, rel, length, b.birth + GROW_STEP, b.dist + b.length)
            b.children.append(child)
            self._grow(child, ex, ey, a + rel, sub + 1)

    def _finish(self, b):
        """Pipe model: thickness ~ sqrt(number of tips). Add leaf pads to tips."""
        if not b.children:
            b.tips = 1
            for _ in range(LEAVES_PER_TIP):
                b.leaves.append(Leaf(self.rng))
            self.terminals.append(b)
        else:
            for c in b.children:
                self._finish(c)
            b.tips = sum(c.tips for c in b.children)

        b.width = 0.62 * math.sqrt(b.tips) + 0.5
        if b.children:
            b.w_end = max(c.width for c in b.children)
        else:
            b.w_end = max(1.0, b.width * 0.8)

    def _score(self):
        """Fitness: canopy coverage, balance, sensible height, stay in frame."""
        cells = set()
        out = 0
        left = right = 0
        min_y = SOIL_Y
        for t in self.terminals:
            x, y = t.ex, t.ey
            if abs(x - CX) > 46 or y < 12 or y > SOIL_Y - 12:
                out += 1
            min_y = min(min_y, y)
            if x < CX:
                left += 1
            else:
                right += 1
            cx, cy = int(x // 5), int(y // 5)
            for dx in (-1, 0, 1):
                cells.add((cx + dx, cy))
            cells.add((cx, cy - 1))

        fit = len(cells) - 4 * out - 0.1 * abs((SOIL_Y - min_y) - 70)
        if self.style != 2:
            fit -= 0.6 * abs(left - right)
        return fit

    # ---------------- life ----------------
    def reset_leaves(self):
        for b in self.terminals:
            for leaf in b.leaves:
                leaf.fallen = False

    def update(self, year_dt, season):
        self.growth += year_dt * season.growth_rate

    def stage(self, age):
        g = self.growth
        if g < FIRST_BIRTH:
            return "Seed"
        if age > LIFESPAN - 3:
            return "Ancient"
        if g < 1.0:
            return "Sprout"
        if g < 2.5:
            return "Sapling"
        if g < 4.0:
            return "Young tree"
        return "Mature tree"

    # ---------------- drawing ----------------
    def draw(self, surface, env):
        env.pending = []
        if self.growth < FIRST_BIRTH:
            pygame.draw.rect(surface, TRUNK_LIGHT, (CX - 1, SOIL_Y - 1, 2, 2))
            return

        maturity = clamp(self.growth / 5.0)
        wscale = 0.35 + 0.65 * maturity
        self.wscale = wscale

        # root flare (nebari)
        w = self.root.width * wscale
        pygame.draw.polygon(
            surface,
            TRUNK_DARK,
            [
                (CX - w / 2 - 6, SOIL_Y + 1),
                (CX - w / 2 - 1, SOIL_Y - 2),
                (CX + w / 2 + 1, SOIL_Y - 2),
                (CX + w / 2 + 6, SOIL_Y + 1),
            ],
        )

        self._draw_branch(surface, self.root, CX, SOIL_Y, 0.0, env)
        self._draw_leaves(surface, env)

    def _draw_branch(self, surface, b, x, y, parent_angle, env):
        g = (self.growth - b.birth) / GROW_TIME
        if g <= 0:
            return
        g = smooth(clamp(g))

        sway = b.flex * (env.wind + 0.5 * math.sin(env.time * 1.7 + b.depth * 0.6))
        angle = parent_angle + b.angle + sway
        length = b.length * g
        x2 = x + math.sin(angle) * length
        y2 = y - math.cos(angle) * length

        bark = lerp_color(TRUNK_DARK, TRUNK_LIGHT, b.depth / 12.0)
        bright = pulse_brightness(b.dist + length * 0.5, env.pulse)
        if bright > 0.02:
            bark = lerp_color(bark, PULSE, bright * 0.5)

        draw_segment(surface, bark, x, y, x2, y2, b.width * self.wscale, b.w_end * self.wscale)

        if b.leaves and g >= 1.0:
            env.pending.append((b, x2, y2, pulse_brightness(b.dist + b.length, env.pulse)))

        for child in b.children:
            self._draw_branch(surface, child, x2, y2, angle, env)

    def _draw_leaves(self, surface, env):
        season = env.season
        for b, x2, y2, bright in env.pending:
            for leaf in b.leaves:
                if leaf.fallen:
                    continue
                lx = x2 + leaf.dx
                ly = y2 + leaf.dy
                visible = season.leaf_grow > leaf.grow_at

                if season.autumn > leaf.drop_at:
                    leaf.fallen = True
                    if visible and len(env.falling) < MAX_FALLING:
                        env.falling.append(
                            FallingLeaf(lx, ly, shade(season.leaf_color, leaf.shade))
                        )
                    continue

                if visible:
                    c = shade(season.leaf_color, leaf.shade)
                    if bright > 0.05:
                        c = lerp_color(c, PULSE, bright * 0.55)
                    surface.fill(c, (int(lx), int(ly), 2, 2))


def pulse_brightness(d, front):
    """Two comet-like signals travelling outward from the roots."""
    best = 0.0
    for off in (0.0, PULSE_PERIOD / 2):
        u = (front + off - d) % PULSE_PERIOD
        if u < 26:
            best = max(best, math.exp(-u / 5.0))
    return best


def draw_segment(surface, color, x, y, x2, y2, w0, w1):
    if w0 < 1.5 and w1 < 1.5:
        pygame.draw.line(surface, color, (x, y), (x2, y2), 1)
        return
    dx, dy = x2 - x, y2 - y
    length = math.hypot(dx, dy)
    if length < 0.01:
        return
    nx, ny = -dy / length, dx / length
    pts = [
        (x + nx * w0 / 2, y + ny * w0 / 2),
        (x2 + nx * w1 / 2, y2 + ny * w1 / 2),
        (x2 - nx * w1 / 2, y2 - ny * w1 / 2),
        (x - nx * w0 / 2, y - ny * w0 / 2),
    ]
    pygame.draw.polygon(surface, color, pts)
    if w1 >= 2:
        pygame.draw.circle(surface, color, (round(x2), round(y2)), max(1, int(w1 / 2)))


# ----------------------------------------------------------------------
# Evolution lab
# ----------------------------------------------------------------------
class Lab:
    """(1+1) evolution strategy: mutate the best brain, keep it if it's not worse."""

    def __init__(self, style=0):
        self.style = style
        self.seed = random.randrange(1_000_000)
        self.generation = 1
        self.best_genome = Genome()
        self.tree = Tree(self.best_genome, self.style, self.seed)
        self.best_fit = self.tree.fitness

    def plant(self):
        child = self.best_genome.mutated()
        self.tree = Tree(child, self.style, self.seed)
        self.generation += 1
        if self.tree.fitness >= self.best_fit:
            self.best_genome, self.best_fit = child, self.tree.fitness
        return self.tree

    def evolve(self, gens=60):
        for _ in range(gens):
            child = self.best_genome.mutated()
            t = Tree(child, self.style, self.seed)
            self.generation += 1
            if t.fitness >= self.best_fit:
                self.best_genome, self.best_fit = child, t.fitness
        self.tree = Tree(self.best_genome, self.style, self.seed)
        return self.tree

    def restyle(self, style):
        self.style = style
        self.tree = Tree(self.best_genome, self.style, self.seed)
        self.best_fit = self.tree.fitness
        return self.tree


# ----------------------------------------------------------------------
# Particles
# ----------------------------------------------------------------------
class FallingLeaf:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.1, 0.1)
        self.vy = 0.0
        self.color = color
        self.phase = random.uniform(0, math.tau)
        self.land_y = GROUND_Y + random.randint(2, 16)
        self.resting = False
        self.life = 0.25

    def update(self, env, year_dt):
        if self.resting:
            self.life -= year_dt
            return
        self.vy = min(self.vy + 0.015, 0.45)
        self.vx += env.wind * 0.004 + math.sin(env.time * 3 + self.phase) * 0.01
        self.vx *= 0.97
        self.x += self.vx
        self.y += self.vy
        floor = SOIL_Y if abs(self.x - CX) < 17 else self.land_y
        if self.y >= floor:
            self.y = floor
            self.resting = True

    @property
    def alive(self):
        return self.life > 0 and -5 < self.x < WIDTH + 5

    def draw(self, surface):
        surface.fill(self.color, (int(self.x), int(self.y), 1, 1 if self.resting else 2))


class Snowflake:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = -2
        self.vy = random.uniform(0.15, 0.4)
        self.phase = random.uniform(0, math.tau)
        self.floor = GROUND_Y + random.randint(0, 20)

    def update(self, env):
        self.y += self.vy
        self.x += math.sin(env.time * 2 + self.phase) * 0.15 + env.wind * 0.08

    @property
    def alive(self):
        return self.y < self.floor and -5 < self.x < WIDTH + 5

    def draw(self, surface):
        surface.fill(SNOW, (int(self.x), int(self.y), 1, 1))


class Env:
    def __init__(self):
        self.time = 0.0
        self.wind = 0.0
        self.pulse = 0.0
        self.season = Season(0.0)
        self.falling = []
        self.snow = []
        self.pending = []


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
_star_rng = random.Random(7)
STARS = [(_star_rng.randrange(WIDTH), _star_rng.randrange(GROUND_Y - 45)) for _ in range(34)]


def draw_background(surface, env):
    surface.fill(BACKGROUND)
    for i, (sx, sy) in enumerate(STARS):
        b = 55 + int(50 * noise1d(env.time * 0.6 + i * 7.3))
        surface.fill((b, b + 6, b + 12), (sx, sy, 1, 1))
    pygame.draw.rect(surface, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.line(surface, GROUND_LINE, (0, GROUND_Y), (WIDTH, GROUND_Y))


def draw_pot(surface):
    pygame.draw.polygon(
        surface,
        POT,
        [(CX - 18, POT_TOP), (CX + 18, POT_TOP), (CX + 13, GROUND_Y + 4), (CX - 13, GROUND_Y + 4)],
    )
    pygame.draw.polygon(
        surface,
        POT_DARK,
        [(CX + 8, POT_TOP), (CX + 18, POT_TOP), (CX + 13, GROUND_Y + 4), (CX + 6, GROUND_Y + 4)],
    )
    pygame.draw.rect(surface, POT, (CX - 20, POT_TOP - 2, 40, 3))
    pygame.draw.rect(surface, SOIL, (CX - 17, POT_TOP - 4, 34, 2))


# ----------------------------------------------------------------------
# Neural network panel (drawn at full resolution)
# ----------------------------------------------------------------------
PANEL_BG = (14, 20, 22)
EDGE_POS = (80, 170, 255)
EDGE_NEG = (255, 110, 90)


def _node(surface, pos, a):
    base = (30, 38, 42)
    target = (110, 230, 255) if a >= 0 else (255, 120, 200)
    pygame.draw.circle(surface, lerp_color(base, target, abs(a)), pos, 5)
    pygame.draw.circle(surface, (70, 85, 90), pos, 5, 1)


def draw_network(surface, genome, decision, title):
    px, py, pw, ph = WINDOW_WIDTH - 252, 40, 244, 152
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    pygame.draw.rect(panel, (8, 12, 14, 190), panel.get_rect(), border_radius=8)
    surface.blit(panel, (px, py))
    surface.blit(tiny_font.render(title, True, (140, 160, 160)), (px + 10, py + 8))

    inputs, hidden, outs = decision
    in_xy = [(px + 70, py + 32 + i * 21) for i in range(N_IN)]
    hid_xy = [(px + 134, py + 32 + i * 21) for i in range(N_HID)]
    out_xy = [(px + 196, py + 54 + k * 36) for k in range(N_OUT)]
    w = genome.w

    for h in range(N_HID):
        for i in range(N_IN):
            wt = w[_W1 + h * N_IN + i]
            s = clamp(abs(wt * inputs[i]) * 0.9)
            col = lerp_color(PANEL_BG, EDGE_POS if wt > 0 else EDGE_NEG, s)
            pygame.draw.line(surface, col, in_xy[i], hid_xy[h], 1)
    for o in range(N_OUT):
        for h in range(N_HID):
            wt = w[_W2 + o * N_HID + h]
            s = clamp(abs(wt * hidden[h]) * 0.9)
            col = lerp_color(PANEL_BG, EDGE_POS if wt > 0 else EDGE_NEG, s)
            pygame.draw.line(surface, col, hid_xy[h], out_xy[o], 1)

    for i, pos in enumerate(in_xy):
        _node(surface, pos, inputs[i])
        label = tiny_font.render(IN_LABELS[i], True, (130, 145, 145))
        surface.blit(label, (pos[0] - 9 - label.get_width(), pos[1] - 6))
    for h, pos in enumerate(hid_xy):
        _node(surface, pos, hidden[h])
    for o, pos in enumerate(out_xy):
        _node(surface, pos, outs[o])
        surface.blit(tiny_font.render(OUT_LABELS[o], True, (130, 145, 145)), (pos[0] + 10, pos[1] - 6))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    clock = pygame.time.Clock()

    lab = Lab(style=0)
    tree = lab.tree
    env = Env()
    calendar = 0.0
    birth_cal = 0.0
    last_year = 0
    speed = 1
    paused = False
    show_net = True

    def replant(new_tree):
        nonlocal tree, birth_cal
        tree = new_tree
        birth_cal = calendar

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # ---------- events ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_1:
                    speed = 1
                elif k == pygame.K_2:
                    speed = 4
                elif k == pygame.K_3:
                    speed = 16
                elif k == pygame.K_m:
                    show_net = not show_net
                elif k == pygame.K_n:
                    replant(lab.plant())
                elif k == pygame.K_e:
                    replant(lab.evolve(60))
                elif k == pygame.K_t:
                    replant(lab.restyle((lab.style + 1) % len(STYLE_NAMES)))
                elif k == pygame.K_r:
                    lab = Lab(style=lab.style)
                    env = Env()
                    calendar = 0.0
                    last_year = 0
                    replant(lab.tree)

        # ---------- update ----------
        year_dt = 0.0 if paused else dt * speed / YEAR_SECONDS
        calendar += year_dt

        if int(calendar) != last_year:
            last_year = int(calendar)
            tree.reset_leaves()
            if calendar - birth_cal >= LIFESPAN:      # old tree -> next generation
                replant(lab.plant())

        if not paused:
            env.time += dt
        env.wind = (noise1d(env.time * 0.3) - 0.5) * 2.0
        env.season = Season(calendar % 1.0)
        if not paused:
            env.pulse = (env.pulse + dt * (5 + 22 * env.season.growth_rate)) % PULSE_PERIOD

        tree.update(year_dt, env.season)

        for p in env.falling:
            p.update(env, year_dt)
        env.falling = [p for p in env.falling if p.alive]

        if env.season.snowing and not paused and random.random() < 0.5:
            env.snow.append(Snowflake())
        for s in env.snow:
            s.update(env)
        env.snow = [s for s in env.snow if s.alive]

        # ---------- draw ----------
        draw_background(canvas, env)
        draw_pot(canvas)
        tree.draw(canvas, env)
        for p in env.falling:
            p.draw(canvas)
        for s in env.snow:
            s.draw(canvas)

        screen.blit(pygame.transform.scale(canvas, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))

        age = calendar - birth_cal
        hud1 = f"Year {age:5.1f}   {env.season.name}   {tree.stage(age)}   x{speed}"
        if paused:
            hud1 += "   [paused]"
        hud2 = (
            f"Gen {lab.generation}   fitness {tree.fitness:5.1f}   "
            f"best {lab.best_fit:5.1f}   style: {STYLE_NAMES[lab.style]}"
        )
        screen.blit(font.render(hud1, True, (200, 210, 200)), (12, 10))
        screen.blit(small_font.render(hud2, True, (150, 175, 160)), (12, 32))
        screen.blit(
            small_font.render(
                "SPACE pause | 1/2/3 speed | N next gen | E evolve | R reset | T style | M net | ESC",
                True,
                (100, 115, 105),
            ),
            (12, WINDOW_HEIGHT - 24),
        )

        if show_net and tree.decisions and tree.growth >= FIRST_BIRTH:
            idx = int(env.time * 4) % len(tree.decisions)
            draw_network(screen, lab.best_genome if tree.genome is lab.best_genome else tree.genome,
                         tree.decisions[idx], f"growth brain  (decision {idx + 1}/{len(tree.decisions)})")

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
