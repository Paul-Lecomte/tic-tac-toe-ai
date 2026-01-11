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

# UI theme centralisé pour réglages rapides
UI_THEME = {
    'bg': BG,
    'panel': PANEL,
    'panel_inner': (8,12,18),
    'panel_block': (18,26,40),
    'accent': ACCENT,
    'success': SUCCESS,
    'warning': (234,179,8),
    'error': (239,68,68),
    'muted': (168,178,194),  # un peu plus contrasté
    'text': WHITE,
    'chip_bg': (45,60,88),
    'chip_active_bg': (36,99,235),
    'chip_text': WHITE,
    'disabled': (90,90,90),
    'hover': (52,77,110),
    'grid_line': MUTED,
    'node_stroke': (12,18,28),
    'connection': (26,38,56),
    'bar_bg': (32,48,72),
    'bar_fill': ACCENT,
    'bar_illegal': (85,85,95),
    'radius': 12,
    'chip_h': 26,
    'chip_pad': 10,
}

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
        # polices harmonisées et contrastées
        self.font = pygame.font.SysFont('Segoe UI', 16)
        self.small_font = pygame.font.SysFont('Segoe UI', 12)
        self.large_font = pygame.font.SysFont('Segoe UI', max(36, int(CELL_SIZE * 0.45)), bold=True)

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
        # Toasts simples
        self.toasts = []  # list of (message, color, expire_time)

    def _push_toast(self, message, color=None, duration=1.8):
        color = color or UI_THEME['accent']
        self.toasts.append((message, color, time.time() + duration))

    def switch_backend(self, backend):
        """Switch net backend to 'numpy' or 'torch' (if available)."""
        backend = backend.lower()
        if backend == self.net_backend:
            print(f'Backend already set to {backend}')
            return
        if backend == 'torch':
            if not USE_TORCH:
                print('PyTorch is not installed. Install torch to use this backend.')
                self._push_toast('PyTorch non dispo', UI_THEME['warning'])
                return
            try:
                self.net = PyTorchNet()
                self.net_backend = 'torch'
                print('Switched backend to PyTorch')
                self._push_toast('Backend: PyTorch', UI_THEME['success'])
            except Exception as e:
                print('Failed to initialize PyTorch backend:', e)
                self._push_toast('Erreur backend torch', UI_THEME['error'])
        elif backend == 'numpy':
            self.net = NumpyNet()
            self.net_backend = 'numpy'
            print('Switched backend to Numpy')
            self._push_toast('Backend: Numpy', UI_THEME['success'])
        else:
            print('Unknown backend:', backend)
            self._push_toast('Backend inconnu', UI_THEME['warning'])

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
        self._push_toast('Réinitialisé', UI_THEME['accent'])

    def _draw_panel_header(self, panel_rect):
        # Status avec badge
        dots = '.' * int((time.time()*2) % 3 + 1)
        status = 'Tour: Vous (X)'
        if self.is_game_over:
            if self.winner == 'draw':
                status = 'Match nul'
                badge_color = UI_THEME['muted']
            elif self.winner == HUMAN:
                status = 'Vous avez gagné'
                badge_color = UI_THEME['success']
            else:
                status = 'IA a gagné'
                badge_color = UI_THEME['error']
        else:
            if self.current == AI:
                status = f'IA pense{dots}'
                badge_color = UI_THEME['accent']
            else:
                badge_color = UI_THEME['success']
        title = self.font.render('Tic-Tac-Toe AI', True, UI_THEME['text'])
        self.screen.blit(title, (panel_rect.x + 16, panel_rect.y + 12))
        # badge
        surf = self.font.render(status, True, (255,255,255))
        badge = pygame.Rect(panel_rect.x + 16, panel_rect.y + 36, surf.get_width()+18, UI_THEME['chip_h'])
        pygame.draw.rect(self.screen, badge_color, badge, border_radius=UI_THEME['radius'])
        self.screen.blit(surf, (badge.x+9, badge.y + UI_THEME['chip_h']/2 - surf.get_height()/2))
        # Chips Mode/Backend
        chip_h = UI_THEME['chip_h']
        chip_pad = UI_THEME['chip_pad']
        def draw_chip(text_s, x, y, active=False):
            color = UI_THEME['chip_active_bg'] if active else UI_THEME['chip_bg']
            surf = self.font.render(text_s, True, UI_THEME['chip_text'])
            rect = pygame.Rect(x, y, surf.get_width()+18, chip_h)
            pygame.draw.rect(self.screen, color, rect, border_radius=UI_THEME['radius'])
            self.screen.blit(surf, (rect.x+9, rect.y+chip_h/2 - surf.get_height()/2))
            return rect
        right_chip = draw_chip(f"Mode: {self.ai_mode} (M/N)", panel_rect.x + 16, panel_rect.y + 64, active=True)
        backend_chip = draw_chip(f"Backend: {self.net_backend} (P)", right_chip.right + chip_pad, panel_rect.y + 64, active=False)
        # tooltips simples au survol
        mx, my = pygame.mouse.get_pos()
        for chip_rect, tip in [
            (right_chip, 'Basculer minimax/NN'),
            (backend_chip, 'Changer backend numpy/torch')
        ]:
            if chip_rect.collidepoint(mx, my):
                tip_surf = self.small_font.render(tip, True, UI_THEME['text'])
                tip_bg = pygame.Rect(mx+12, my+12, tip_surf.get_width()+10, tip_surf.get_height()+6)
                pygame.draw.rect(self.screen, UI_THEME['panel_block'], tip_bg, border_radius=8)
                self.screen.blit(tip_surf, (tip_bg.x+5, tip_bg.y+3))

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
        # Draw as grouped rows with bars
        base_x = panel_rect.x + 10
        base_y = panel_rect.y + 34
        bar_w = panel_rect.width - 32
        bar_h = 12
        gap = 6
        for i in range(9):
            r = i // 3
            c = i % 3
            label = self.small_font.render(f'R{r+1}C{c+1}', True, UI_THEME['muted'])
            y = base_y + i*(bar_h+gap)
            self.screen.blit(label, (base_x, y))
            bar_rect = pygame.Rect(base_x + 70, y, bar_w - 100, bar_h)
            pygame.draw.rect(self.screen, UI_THEME['bar_bg'], bar_rect, border_radius=6)
            p = float(probs[i])
            fill_w = int((bar_rect.width) * p)
            is_legal = (self.board[i] == ' ')
            fill_color = (UI_THEME['bar_fill'] if is_legal else UI_THEME['bar_illegal'])
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_h)
            pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=6)
            # hachures si illégal
            if not is_legal:
                for hx in range(bar_rect.x, bar_rect.right, 6):
                    pygame.draw.line(self.screen, (60,60,70), (hx, bar_rect.y), (hx-4, bar_rect.bottom), 1)
            val = self.small_font.render(f'{p:.2f}', True, UI_THEME['text'])
            self.screen.blit(val, (bar_rect.right - val.get_width() - 6, y))
            # seuil visuel > 0.5
            if p >= 0.5 and is_legal:
                pygame.draw.rect(self.screen, UI_THEME['success'], (bar_rect.right-4, y-2, 4, bar_h+4), border_radius=2)

    def draw_panel(self):
        panel_rect = pygame.Rect(BOARD_SIZE + 2*MARGIN, MARGIN, WIDTH - BOARD_SIZE - 3*MARGIN, HEIGHT - 2*MARGIN)
        pygame.draw.rect(self.screen, UI_THEME['panel_inner'], panel_rect, border_radius=UI_THEME['radius'])
        inner = panel_rect.inflate(-8, -8)
        pygame.draw.rect(self.screen, UI_THEME['panel'], inner, border_radius=UI_THEME['radius'])

        self._draw_panel_header(inner)

        # layout
        probs_h = max(100, int(inner.height * 0.24))
        block_h = max(170, inner.height - probs_h - 44)
        block = pygame.Rect(inner.x + 12, inner.y + 60, inner.width - 24, block_h)
        pygame.draw.rect(self.screen, UI_THEME['panel_block'], block, border_radius=10)
        title = self.small_font.render('Neural Network Activations', True, UI_THEME['text'])
        self.screen.blit(title, (block.x + 12, block.y + 10))

        # Draw layers
        if self.last_activations is not None:
            elapsed = time.time() - self.viz_anim_start
            alpha = self._ease(min(1.0, elapsed / self.viz_anim_duration))
            layers = self._interp_layers(self.prev_activations, self.last_activations, alpha)
            col_count = len(layers)
            col_spacing = max(52, int((block.width - 40) / max(1, col_count-1)))
            start_x = block.x + 14
            start_y = block.y + 24
            node_radius = max(3, int(CELL_SIZE * 0.03))
            max_nodes = max(len(np.array(a).reshape(-1)) for a in layers)
            vertical_spacing = max(10, int((block.height - 50) / max(1, max_nodes)))
            positions = []
            for li, layer in enumerate(layers):
                arr = np.array(layer).reshape(-1)
                n = len(arr)
                lx = start_x + li*col_spacing
                ly0 = start_y
                total_h = (n-1)*vertical_spacing
                ly_start = ly0 + (max_nodes*vertical_spacing - total_h)/2
                pos_layer = []
                for ni in range(n):
                    ly = int(ly_start + ni*vertical_spacing)
                    pos_layer.append((int(lx), ly))
                positions.append(pos_layer)
            # connections atténuées
            for li in range(col_count-1):
                left = positions[li]
                right = positions[li+1]
                for p in left:
                    for q in right:
                        pygame.draw.line(self.screen, UI_THEME['connection'], p, q, 1)
            # nodes colorés
            for li, layer in enumerate(layers):
                arr = np.array(layer).reshape(-1)
                for ni, val in enumerate(arr):
                    v = float(val)
                    v = max(min(v, 5.0), -5.0)
                    intensity = (v + 5.0) / 10.0
                    r = int(40 + 20*(1-intensity))
                    g = int(160 + 80*intensity)
                    b = int(220 - 80*intensity)
                    color = (r, g, b)
                    pygame.draw.circle(self.screen, color, positions[li][ni], node_radius)
                    pygame.draw.circle(self.screen, UI_THEME['node_stroke'], positions[li][ni], node_radius, 2)

        # Probabilities block
        probs_block = pygame.Rect(inner.x + 12, block.bottom + 12, inner.width - 24, probs_h - 6)
        pygame.draw.rect(self.screen, UI_THEME['panel_block'], probs_block, border_radius=10)
        ptitle = self.small_font.render('Predicted Move Probabilities', True, UI_THEME['text'])
        self.screen.blit(ptitle, (probs_block.x + 12, probs_block.y + 8))
        self._draw_probabilities(probs_block)

        # render toasts (coin supérieur droit du panel)
        now = time.time()
        self.toasts = [t for t in self.toasts if t[2] > now]
        ty = inner.y + 6
        for msg, color, exp in self.toasts:
            tsurf = self.small_font.render(msg, True, (255,255,255))
            trect = pygame.Rect(inner.right - tsurf.get_width() - 24, ty, tsurf.get_width()+16, tsurf.get_height()+8)
            pygame.draw.rect(self.screen, color, trect, border_radius=8)
            self.screen.blit(tsurf, (trect.x+8, trect.y+4))
            ty += trect.height + 6

    def draw_board(self):
        base_x = MARGIN
        base_y = MARGIN
        board_rect = pygame.Rect(base_x, base_y, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, UI_THEME['bg'], board_rect, border_radius=8)
        # grid lines
        for i in range(1,3):
            x = base_x + i*CELL_SIZE
            pygame.draw.line(self.screen, UI_THEME['grid_line'], (x, base_y+8), (x, base_y+BOARD_SIZE-8), 2)
            y = base_y + i*CELL_SIZE
            pygame.draw.line(self.screen, UI_THEME['grid_line'], (base_x+8, y), (base_x+BOARD_SIZE-8, y), 2)
        # hover cell
        mx, my = pygame.mouse.get_pos()
        hover_idx = None
        if MARGIN <= mx <= MARGIN + BOARD_SIZE and MARGIN <= my <= MARGIN + BOARD_SIZE:
            col = (mx - MARGIN) // CELL_SIZE
            row = (my - MARGIN) // CELL_SIZE
            hover_idx = int(row*3 + col)
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
                pygame.draw.rect(self.screen, UI_THEME['panel'], rect, border_radius=8)
            # hover accent si libre
            if hover_idx == i and self.board[i] == ' ' and not self.is_game_over and self.current == HUMAN:
                pygame.draw.rect(self.screen, UI_THEME['hover'], rect, 2, border_radius=8)
            # draw X/O
            if self.board[i] == HUMAN:
                txt = self.large_font.render('X', True, UI_THEME['accent'])
                self.screen.blit(txt, (cx + CELL_SIZE/2 - txt.get_width()/2, cy + CELL_SIZE/2 - txt.get_height()/2))
            elif self.board[i] == AI:
                txt = self.large_font.render('O', True, UI_THEME['success'])
                self.screen.blit(txt, (cx + CELL_SIZE/2 - txt.get_width()/2, cy + CELL_SIZE/2 - txt.get_height()/2))

        # Hints en 2 lignes
        hint1 = 'R: reset   T: train NN   Q: quit'
        hint2 = 'M: Minimax   N: NN   P: toggle backend'
        h1 = self.small_font.render(hint1, True, UI_THEME['muted'])
        h2 = self.small_font.render(hint2, True, UI_THEME['muted'])
        hint_x = MARGIN
        preferred_y = MARGIN + BOARD_SIZE + 8
        y1 = min(preferred_y, HEIGHT - MARGIN - h1.get_height() - h2.get_height() - 10)
        y2 = y1 + h1.get_height() + 4
        self.screen.blit(h1, (hint_x, y1))
        self.screen.blit(h2, (hint_x, y2))

    def human_move_at(self, pos):
        if self.is_game_over: return
        if self.board[pos] != ' ': return
        self.board[pos] = HUMAN
        winner, combo = check_winner(self.board)
        if winner:
            self.is_game_over = True
            self.winner = winner
            self.winning_combo = combo
            self._push_toast('Fin de partie', UI_THEME['accent'])
            return
        self.current = AI
        # Si le mode NN est actif, on met à jour la viz juste après le coup humain
        if self.ai_mode == 'nn':
            self.update_nn_viz(force=True)

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
                self._push_toast('Fin de partie', UI_THEME['accent'])
            else:
                self.current = HUMAN
            return
        elif self.ai_mode == 'nn':
            x = board_to_input(self.board)
            logits, activations = self.net.predict(x)
            mask = np.array([ -1e6 if self.board[i] != ' ' else 0 for i in range(9)])
            logits = logits + mask
            self.prev_activations = self.last_activations
            self.last_logits = logits
            self.last_activations = [a.reshape(-1) for a in activations]
            self.viz_anim_start = time.time()
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
                self._push_toast('Fin de partie', UI_THEME['accent'])
            else:
                self.current = HUMAN
            return

    def train_nn_quick(self, samples=500, epochs=80):
        print('Génération dataset...')
        X, Y = generate_dataset(samples)
        print('Entraînement rapide du réseau...')
        self._push_toast('Entraînement...', UI_THEME['accent'])
        self.net.train_supervised(X, Y, epochs=epochs, lr=0.01, batch=32)
        self._push_toast('Entraînement terminé', UI_THEME['success'])
        print('Entraînement terminé.')

    def update_nn_viz(self, force=False):
        """Met à jour last_activations/last_logits pour la visualisation NN.
        Appelée en continu quand le mode NN est actif, et forcée après un coup humain.
        """
        if self.ai_mode != 'nn':
            return
        # si la partie est finie, ne pas mettre à jour
        if self.is_game_over:
            return
        x = board_to_input(self.board)
        try:
            logits, activations = self.net.predict(x)
        except Exception:
            return
        # masque des coups illégaux pour les logits
        mask = np.array([ -1e6 if self.board[i] != ' ' else 0 for i in range(9)])
        new_logits = logits + mask
        # détection de changement significatif ou forcé
        changed = force or (self.last_logits is None) or (np.any(np.abs(np.array(self.last_logits) - np.array(new_logits)) > 1e-6))
        if changed:
            self.prev_activations = self.last_activations
            self.last_logits = new_logits
            self.last_activations = [a.reshape(-1) for a in activations]
            self.viz_anim_start = time.time()

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
                        self._push_toast('Mode: minimax', UI_THEME['success'])
                    elif event.key == pygame.K_n:
                        self.ai_mode = 'nn'
                        self._push_toast('Mode: NN', UI_THEME['success'])
                    elif event.key == pygame.K_p:
                        target = 'torch' if self.net_backend == 'numpy' else 'numpy'
                        self.switch_backend(target)
                    elif event.key == pygame.K_t:
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

            # Mise à jour continue de la viz NN si actif et tour du joueur
            if not self.is_game_over and self.ai_mode == 'nn' and self.current == HUMAN:
                self.update_nn_viz()

            # Draw
            self.screen.fill(UI_THEME['bg'])
            self.draw_board()
            self.draw_panel()
            # FPS discret (coin bas droit)
            fps_text = self.small_font.render(f"{int(self.clock.get_fps())} FPS", True, UI_THEME['muted'])
            self.screen.blit(fps_text, (WIDTH - fps_text.get_width() - MARGIN, HEIGHT - fps_text.get_height() - MARGIN))
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

