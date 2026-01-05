"""
Tic-Tac-Toe with Pygame UI and Neural Network visualization
- Human (X) vs AI (O)
- AI modes: Minimax (perfect) or NeuralNet (trained quickly against Minimax)
- NeuralNet implemented in numpy with optional PyTorch fallback
- Real-time visualization of NN forward pass (activations & predicted move scores)

Requirements:
  - pygame
  - numpy
  - (optional) torch

Run:
  python tic_tac_toe_gui.py

Controls:
  - Click a cell to play (you are X, go first)
  - R to reset
  - M to use Minimax AI, N to use NeuralNet AI
  - T to (re)train the NeuralNet quickly against Minimax

"""

import time
import random

try:
    import pygame
except Exception as e:
    print('pygame is required. Install with: pip install pygame')
    raise

try:
    import numpy as np
except Exception as e:
    print('numpy is required. Install with: pip install numpy')
    raise

# Try to import torch optionally
USE_TORCH = False
try:
    import torch
    USE_TORCH = True
except Exception:
    USE_TORCH = False

# Game constants (limit window to 720p and make NN viz compact)
WIDTH, HEIGHT = 1280, 720
BOARD_SIZE = 360
MARGIN = 12
CELL_SIZE = BOARD_SIZE // 3
FPS = 60

HUMAN = 'X'
AI = 'O'

WIN_COMBOS = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

# Colors
BG = (14, 21, 33)
PANEL = (10, 14, 20)
ACCENT = (59,130,246)
SUCCESS = (16,185,129)
MUTED = (148,163,184)
WHITE = (230,238,248)
HIGHLIGHT = (80,200,180)

# ----------------------------
# Minimax (same logic as console)
# ----------------------------

def available_moves(board):
    return [i for i, v in enumerate(board) if v == ' ']


def check_winner(board):
    for a,b,c in WIN_COMBOS:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a], (a,b,c)
    if all(cell != ' ' for cell in board):
        return 'draw', None
    return None, None


def minimax(board, depth, is_maximizing):
    winner, _ = check_winner(board)
    if winner == AI:
        return 10 - depth
    if winner == HUMAN:
        return depth - 10
    if winner == 'draw':
        return 0

    if is_maximizing:
        best = -9999
        for i in available_moves(board):
            board[i] = AI
            score = minimax(board, depth+1, False)
            board[i] = ' '
            if score > best:
                best = score
        return best
    else:
        best = 9999
        for i in available_moves(board):
            board[i] = HUMAN
            score = minimax(board, depth+1, True)
            board[i] = ' '
            if score < best:
                best = score
        return best


def minimax_best_move(board):
    if all(c == ' ' for c in board):
        return 4
    best_score = -9999
    move = None
    for i in available_moves(board):
        board[i] = AI
        score = minimax(board, 0, False)
        board[i] = ' '
        if score > best_score:
            best_score = score
            move = i
    return move

# ----------------------------
# Simple Neural Network (numpy)
# ----------------------------
class NumpyNet:
    def __init__(self, layer_sizes=(9, 36, 18, 9), seed=42):
        np.random.seed(seed)
        self.sizes = layer_sizes
        self.weights = [np.random.randn(self.sizes[i], self.sizes[i-1]) * np.sqrt(2/self.sizes[i-1])
                        for i in range(1, len(self.sizes))]
        self.biases = [np.zeros((s,1)) for s in self.sizes[1:]]

    def forward(self, x):
        # x shape: (9, 1)
        activations = [x]
        a = x
        for i, (w,b) in enumerate(zip(self.weights, self.biases)):
            z = w.dot(a) + b
            if i < len(self.weights)-1:
                a = np.maximum(0, z)  # ReLU
            else:
                # output layer: raw scores
                a = z
            activations.append(a)
        return activations

    def predict(self, x):
        x = np.array(x).reshape(9,1).astype(np.float32)
        activations = self.forward(x)
        out = activations[-1].reshape(-1)
        # Mask illegal moves by setting very low score
        return out, activations

    def train_supervised(self, X, Y, epochs=100, lr=0.01, batch=32):
        # X: (N,9), Y: (N,9) one-hot
        N = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch):
                batch_idx = idx[start:start+batch]
                grad_w = [np.zeros_like(w) for w in self.weights]
                grad_b = [np.zeros_like(b) for b in self.biases]
                for i in batch_idx:
                    x = X[i].reshape(9,1)
                    y = Y[i].reshape(-1,1)
                    acts = self.forward(x)
                    logits = acts[-1]
                    # softmax and cross-entropy gradient
                    exps = np.exp(logits - np.max(logits))
                    probs = exps / np.sum(exps)
                    delta = probs - y  # shape (9,1)
                    # Backprop through linear layers and ReLU
                    for l in reversed(range(len(self.weights))):
                        a_prev = acts[l]
                        grad_w[l] += delta.dot(a_prev.T)
                        grad_b[l] += delta
                        if l > 0:
                            w = self.weights[l]
                            delta = w.T.dot(delta)
                            z_prev = self.weights[l-1].dot(acts[l-1]) + self.biases[l-1] if l-1 >= 0 else None
                            # apply ReLU mask on delta for hidden layers
                            delta = np.where(acts[l] > 0, delta, 0)
                # SGD update
                for l in range(len(self.weights)):
                    self.weights[l] -= lr * (grad_w[l] / len(batch_idx))
                    self.biases[l] -= lr * (grad_b[l] / len(batch_idx))
            # simple verbose
            if (epoch+1) % max(1, epochs//5) == 0:
                print(f'Train epoch {epoch+1}/{epochs}')


# ----------------------------
# PyTorch backend (optional)
# ----------------------------
if USE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class PyTorchNet:
        def __init__(self, layer_sizes=(9,36,18,9), seed=42, device='cpu'):
            self.sizes = layer_sizes
            self.device = torch.device(device)
            torch.manual_seed(seed)
            # build layers
            layers = []
            for i in range(1, len(self.sizes)):
                layers.append(nn.Linear(self.sizes[i-1], self.sizes[i]))
                if i < len(self.sizes)-1:
                    layers.append(nn.ReLU())
            # store linear layers separately for activations tracing
            self.linears = nn.ModuleList([m for m in layers if isinstance(m, nn.Linear)])
            self.model = nn.Sequential(*layers).to(self.device)

            # init weights similar to kaiming
            for m in self.model:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            # Accept x as numpy array shape (9,) or (9,1)
            with torch.no_grad():
                tx = torch.tensor(np.array(x).reshape(1,9).astype(np.float32), device=self.device)
                activations = [tx.squeeze(0).unsqueeze(1).cpu().numpy()]  # (9,1)
                a = tx  # shape (1,9)
                for module in self.model:
                    a = module(a)  # keep shape (1, features)
                    activations.append(a.squeeze(0).unsqueeze(1).cpu().numpy())
                return activations

        def predict(self, x):
            acts = self.forward(np.array(x))
            logits = acts[-1].reshape(-1)
            return logits, acts

        def train_supervised(self, X, Y, epochs=100, lr=0.01, batch=32):
            # X: (N,9) numpy, Y: (N,9) one-hot numpy
            X_t = torch.tensor(X.astype(np.float32), device=self.device)
            Y_idx = torch.tensor(np.argmax(Y, axis=1), dtype=torch.long, device=self.device)
            dataset = torch.utils.data.TensorDataset(X_t, Y_idx)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
            for epoch in range(epochs):
                running_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    outputs = self.model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                if (epoch+1) % max(1, epochs//5) == 0:
                    print(f'PyTorch Train epoch {epoch+1}/{epochs}  loss={running_loss/len(loader):.4f}')


# ----------------------------
# Helper: generate dataset by querying Minimax
# ----------------------------
def board_to_input(board):
    # represent X=1, O=-1, empty=0
    arr = np.array([1 if c==HUMAN else (-1 if c==AI else 0) for c in board], dtype=np.float32)
    return arr


def best_move_minimax_label(board):
    # return one-hot label for best move (Minimax)
    mv = minimax_best_move(board)
    label = np.zeros(9, dtype=np.float32)
    if mv is not None:
        label[mv] = 1.0
    else:
        # pick random legal move
        legal = available_moves(board)
        if legal:
            label[random.choice(legal)] = 1.0
    return label


def generate_dataset(n_samples=300):
    X = []
    Y = []
    for _ in range(n_samples):
        # generate random board by playing random moves (but keep legal positions)
        board = [' '] * 9
        turns = random.randint(0,6)
        player = HUMAN
        for t in range(turns):
            moves = available_moves(board)
            if not moves: break
            mv = random.choice(moves)
            board[mv] = player
            w, _ = check_winner(board)
            if w: break
            player = AI if player == HUMAN else HUMAN
        # label using minimax (supervisor)
        X.append(board_to_input(board))
        Y.append(best_move_minimax_label(board))
    return np.stack(X), np.stack(Y)

# ----------------------------
# Pygame UI and Visualization
# ----------------------------
class GameUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Tic-Tac-Toe AI (Minimax + NN Viz)')
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        # fonts tuned smaller so UI fits in 720p
        self.font = pygame.font.SysFont('Arial', 14)
        # very small font for compact labels/titles
        self.small_font = pygame.font.SysFont('Arial', 10)
        # large font scaled down so X/O are not overwhelming
        self.large_font = pygame.font.SysFont('Arial', max(36, int(CELL_SIZE * 0.45)), bold=True)

        self.board = [' '] * 9
        self.current = HUMAN
        self.is_game_over = False
        self.winner = None
        self.winning_combo = None

        # AI mode: 'minimax' or 'nn'
        self.ai_mode = 'minimax'
        # Select backend: prefer torch if available, otherwise numpy
        self.net_backend = 'torch' if USE_TORCH else 'numpy'
        if self.net_backend == 'torch':
            try:
                self.net = PyTorchNet()
            except Exception as e:
                print('Could not initialize PyTorch backend, falling back to NumpyNet:', e)
                self.net_backend = 'numpy'
                self.net = NumpyNet()
        else:
            self.net = NumpyNet()
        self.training = False
        self.last_activations = None
        self.last_logits = None
        # Visualization animation state
        self.prev_activations = None
        self.viz_anim_start = 0.0
        self.viz_anim_duration = 0.4  # seconds

    def switch_backend(self, backend):
        """Switch net backend to 'numpy' or 'torch' (if available)."""
        backend = backend.lower()
        if backend == self.net_backend:
            print(f'Backend already set to {backend}')
            return
        if backend == 'torch':
            if not USE_TORCH:
                print('PyTorch is not installed. Install torch to use this backend.')
                return
            try:
                self.net = PyTorchNet()
                self.net_backend = 'torch'
                print('Switched backend to PyTorch')
            except Exception as e:
                print('Failed to initialize PyTorch backend:', e)
        elif backend == 'numpy':
            self.net = NumpyNet()
            self.net_backend = 'numpy'
            print('Switched backend to Numpy')
        else:
            print('Unknown backend:', backend)

    def reset(self):
        self.board = [' '] * 9
        self.current = HUMAN
        self.is_game_over = False
        self.winner = None
        self.winning_combo = None
        self.last_activations = None
        self.prev_activations = None
        self.last_logits = None
        self.viz_anim_start = 0.0

    def _draw_panel_header(self, panel_rect):
        # Status
        status = 'Tour: Vous (X)' if self.current == HUMAN and not self.is_game_over else ('IA pense...' if self.current==AI and not self.is_game_over else 'Fin')
        if self.is_game_over:
            if self.winner == 'draw': status = 'Match nul'
            elif self.winner == HUMAN: status = 'Vous avez gagné'
            else: status = 'IA a gagné'
        title = self.font.render('Tic-Tac-Toe AI', True, WHITE)
        self.screen.blit(title, (panel_rect.x + 16, panel_rect.y + 12))
        text = self.font.render(status, True, WHITE)
        self.screen.blit(text, (panel_rect.x + 16, panel_rect.y + 36))
        # Mode & backend rows as chips
        chip_h = 24
        chip_pad = 8
        def draw_chip(text_s, x, y, color=(45,60,88)):
            surf = self.font.render(text_s, True, WHITE)
            rect = pygame.Rect(x, y, surf.get_width()+16, chip_h)
            pygame.draw.rect(self.screen, color, rect, border_radius=12)
            self.screen.blit(surf, (rect.x+8, rect.y+chip_h/2 - surf.get_height()/2))
            return rect.right
        right = draw_chip(f'Mode: {self.ai_mode} (M/N)', panel_rect.x + 16, panel_rect.y + 64)
        draw_chip(f'Backend: {self.net_backend} (P)', right + chip_pad, panel_rect.y + 64)

    def _ease(self, t):
        # smooth cubic ease-in-out
        t = max(0.0, min(1.0, t))
        return 3*t*t - 2*t*t*t

    def _interp_layers(self, prev, cur, alpha):
        if prev is None or cur is None: return cur
        out = []
        for p, c in zip(prev, cur):
            # ensure same shape
            pv = np.array(p).reshape(-1)
            cv = np.array(c).reshape(-1)
            if pv.shape != cv.shape:
                out.append(cv)
            else:
                out.append(pv*(1-alpha) + cv*alpha)
        return [np.array(a) for a in out]

    def _draw_probabilities(self, panel_rect):
        if self.last_logits is None: return
        logits = np.array(self.last_logits).reshape(-1)
        exps = np.exp(logits - np.max(logits))
        probs = (exps / np.sum(exps))
        # Draw as grouped rows with bars (compact for 720p)
        base_x = panel_rect.x + 10
        # Position probabilities block starting below its title area (keep responsive)
        base_y = panel_rect.y + 34
        bar_w = panel_rect.width - 32
        bar_h = 12
        gap = 5
        for i in range(9):
            r = i // 3
            c = i % 3
            # use small font for probability labels to reduce clutter
            label = self.small_font.render(f'Cell {r+1},{c+1}', True, MUTED)
            y = base_y + i*(bar_h+gap)
            self.screen.blit(label, (base_x, y))
            # bar background
            bar_rect = pygame.Rect(base_x + 90, y, bar_w - 120, bar_h)
            pygame.draw.rect(self.screen, (32,48,72), bar_rect, border_radius=6)
            # bar fill with gradient-like two-tone
            p = float(probs[i])
            fill_w = int((bar_rect.width) * p)
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_h)
            color = (59,130,246) if self.board[i] == ' ' else (90,90,90)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=6)
            val = self.small_font.render(f'{p:.2f}', True, WHITE)
            self.screen.blit(val, (bar_rect.right - val.get_width() - 6, y))

    def draw_panel(self):
        panel_rect = pygame.Rect(BOARD_SIZE + 2*MARGIN, MARGIN, WIDTH - BOARD_SIZE - 3*MARGIN, HEIGHT - 2*MARGIN)
        # panel background layers for subtle depth
        pygame.draw.rect(self.screen, (8,12,18), panel_rect, border_radius=12)
        inner = panel_rect.inflate(-8, -8)
        pygame.draw.rect(self.screen, PANEL, inner, border_radius=12)

        self._draw_panel_header(inner)

        # Allocate more height to neural network activations and less to probabilities.
        # Keep sensible clamps so the UI remains stable at 720p.
        probs_h = max(90, int(inner.height * 0.22))
        block_h = max(160, inner.height - probs_h - 36)
        block = pygame.Rect(inner.x + 12, inner.y + 60, inner.width - 24, block_h)
        pygame.draw.rect(self.screen, (18,26,40), block, border_radius=10)
        title = self.small_font.render('Neural Network Activations', True, WHITE)
        self.screen.blit(title, (block.x + 12, block.y + 10))

        # Draw layers as columns with nodes and faint connections
        if self.last_activations is not None:
            # animation alpha
            elapsed = time.time() - self.viz_anim_start
            alpha = self._ease(min(1.0, elapsed / self.viz_anim_duration))
            layers = self._interp_layers(self.prev_activations, self.last_activations, alpha)

            # layout (even more compact)
            col_count = len(layers)
            col_spacing = max(48, int((block.width - 40) / max(1, col_count-1)))
            start_x = block.x + 14
            # start_y remonté pour afficher les activations un peu plus haut dans le bloc
            start_y = block.y + 24
            # very small node radius for compact display
            node_radius = max(3, int(CELL_SIZE * 0.03))
            max_nodes = max(len(np.array(a).reshape(-1)) for a in layers)
            # tighter vertical spacing
            vertical_spacing = max(10, int((block.height - 50) / max(1, max_nodes)))

            # precompute positions
            positions = []
            for li, layer in enumerate(layers):
                arr = np.array(layer).reshape(-1)
                n = len(arr)
                lx = start_x + li*col_spacing
                ly0 = start_y
                # center vertically
                total_h = (n-1)*vertical_spacing
                ly_start = ly0 + (max_nodes*vertical_spacing - total_h)/2
                pos_layer = []
                for ni in range(n):
                    ly = int(ly_start + ni*vertical_spacing)
                    pos_layer.append((int(lx), ly))
                positions.append(pos_layer)

            # connections
            for li in range(col_count-1):
                left = positions[li]
                right = positions[li+1]
                # faint bundle lines
                for p in left:
                    for q in right:
                        pygame.draw.line(self.screen, (26,38,56), p, q, 1)

            # nodes colored by activation
            for li, layer in enumerate(layers):
                arr = np.array(layer).reshape(-1)
                for ni, val in enumerate(arr):
                    v = float(val)
                    v = max(min(v, 5.0), -5.0)
                    # map -5..5 to hue-like green/blue
                    intensity = (v + 5.0) / 10.0
                    r = int(40 + 20*(1-intensity))
                    g = int(160 + 80*intensity)
                    b = int(220 - 80*intensity)
                    color = (r, g, b)
                    pygame.draw.circle(self.screen, color, positions[li][ni], node_radius)
                    pygame.draw.circle(self.screen, (12,18,28), positions[li][ni], node_radius, 2)

        # Probabilities block (smaller, placed under the activations block)
        probs_block = pygame.Rect(inner.x + 12, block.bottom + 12, inner.width - 24, probs_h - 6)
        pygame.draw.rect(self.screen, (18,26,40), probs_block, border_radius=10)
        ptitle = self.small_font.render('Predicted Move Probabilities', True, WHITE)
        self.screen.blit(ptitle, (probs_block.x + 12, probs_block.y + 8))
        self._draw_probabilities(probs_block)

    def draw_board(self):
        base_x = MARGIN
        base_y = MARGIN
        # board bg
        board_rect = pygame.Rect(base_x, base_y, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, BG, board_rect, border_radius=8)
        # grid lines
        for i in range(1,3):
            x = base_x + i*CELL_SIZE
            pygame.draw.line(self.screen, MUTED, (x, base_y+8), (x, base_y+BOARD_SIZE-8), 2)
            y = base_y + i*CELL_SIZE
            pygame.draw.line(self.screen, MUTED, (base_x+8, y), (base_x+BOARD_SIZE-8, y), 2)
        # cells
        for i in range(9):
            r = i // 3
            c = i % 3
            cx = base_x + c*CELL_SIZE
            cy = base_y + r*CELL_SIZE
            rect = pygame.Rect(cx+6, cy+6, CELL_SIZE-12, CELL_SIZE-12)
            # highlight winning combo
            if self.winning_combo and i in self.winning_combo:
                pygame.draw.rect(self.screen, HIGHLIGHT, rect, border_radius=8)
            else:
                pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            # draw X/O
            if self.board[i] == HUMAN:
                txt = self.large_font.render('X', True, ACCENT)
                self.screen.blit(txt, (cx + CELL_SIZE/2 - txt.get_width()/2, cy + CELL_SIZE/2 - txt.get_height()/2))
            elif self.board[i] == AI:
                txt = self.large_font.render('O', True, SUCCESS)
                self.screen.blit(txt, (cx + CELL_SIZE/2 - txt.get_width()/2, cy + CELL_SIZE/2 - txt.get_height()/2))

        # Draw control hints under the board (bottom-left) so they are not hidden by the panel
        hints = 'R: reset   T: train NN   Q: quit   M: Minimax   N: NN   P: toggle backend'
        hint_surf = self.small_font.render(hints, True, MUTED)
        hint_x = MARGIN
        # prefer to place directly below the board, but clamp so it stays inside the window
        preferred_y = MARGIN + BOARD_SIZE + 12
        hint_y = min(preferred_y, HEIGHT - MARGIN - hint_surf.get_height() - 6)
        self.screen.blit(hint_surf, (hint_x, hint_y))

    def human_move_at(self, pos):
        if self.is_game_over: return
        if self.board[pos] != ' ': return
        self.board[pos] = HUMAN
        winner, combo = check_winner(self.board)
        if winner:
            self.is_game_over = True
            self.winner = winner
            self.winning_combo = combo
            return
        self.current = AI

    def ai_move(self):
        if self.ai_mode == 'minimax':
            mv = minimax_best_move(self.board)
            time.sleep(0.2)
            if mv is None:
                mv = random.choice(available_moves(self.board))
            self.board[mv] = AI
            winner, combo = check_winner(self.board)
            if winner:
                self.is_game_over = True
                self.winner = winner
                self.winning_combo = combo
            else:
                self.current = HUMAN
            return
        elif self.ai_mode == 'nn':
            # prepare input and forward
            x = board_to_input(self.board)
            logits, activations = self.net.predict(x)
            # mask illegal moves
            mask = np.array([ -1e6 if self.board[i] != ' ' else 0 for i in range(9)])
            logits = logits + mask
            # store for animated viz
            self.prev_activations = self.last_activations
            self.last_logits = logits
            self.last_activations = [a.reshape(-1) for a in activations]
            self.viz_anim_start = time.time()
            # visual step-by-step: brief pause to let animation show
            time.sleep(0.25)
            mv = int(np.argmax(logits))
            if self.board[mv] != ' ':
                mv = random.choice(available_moves(self.board))
            self.board[mv] = AI
            winner, combo = check_winner(self.board)
            if winner:
                self.is_game_over = True
                self.winner = winner
                self.winning_combo = combo
            else:
                self.current = HUMAN
            return

    def train_nn_quick(self, samples=500, epochs=80):
        print('Génération dataset...')
        X, Y = generate_dataset(samples)
        print('Entraînement rapide du réseau...')
        self.net.train_supervised(X, Y, epochs=epochs, lr=0.01, batch=32)
        print('Entraînement terminé.')

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_m:
                        self.ai_mode = 'minimax'
                    elif event.key == pygame.K_n:
                        self.ai_mode = 'nn'
                    elif event.key == pygame.K_p:
                        # toggle backend between numpy and torch
                        target = 'torch' if self.net_backend == 'numpy' else 'numpy'
                        self.switch_backend(target)
                    elif event.key == pygame.K_t:
                        # train quickly
                        self.train_nn_quick(samples=300, epochs=400)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    if MARGIN <= mx <= MARGIN + BOARD_SIZE and MARGIN <= my <= MARGIN + BOARD_SIZE:
                        col = (mx - MARGIN) // CELL_SIZE
                        row = (my - MARGIN) // CELL_SIZE
                        pos = int(row*3 + col)
                        self.human_move_at(pos)

            # AI turn
            if not self.is_game_over and self.current == AI:
                self.ai_move()

            # Draw
            self.screen.fill(BG)
            self.draw_board()
            self.draw_panel()
            pygame.display.flip()

        pygame.quit()

# ----------------------------
# Main
# ----------------------------

def main():
    ui = GameUI()
    ui.run()

if __name__ == '__main__':
    main()

