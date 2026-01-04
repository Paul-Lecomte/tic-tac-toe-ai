#!/usr/bin/env python3
"""
Tic-Tac-Toe (Console) — Human (X) vs AI (O) using Minimax
Single-file Python implementation.

Usage:
  - Run normally: python tic_tac_toe.py
  - Run automated tests: python tic_tac_toe.py --test

Features:
  - Human is X and plays first.
  - AI is O and uses Minimax (optimal play).
  - Prevents illegal moves.
  - Highlights winning combo in the console using ANSI colors.
  - After game ends, option to play again.

"""

import random
import sys
import argparse

# ANSI colors (may work in most terminals)
CSI = "\033["
RESET = CSI + "0m"
GREEN = CSI + "32m"
YELLOW = CSI + "33m"
CYAN = CSI + "36m"
RED = CSI + "31m"
BOLD = CSI + "1m"

HUMAN = 'X'
AI = 'O'

WIN_COMBO = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]


def available_moves(board):
    return [i for i, v in enumerate(board) if v == ' ']


def check_winner(board):
    """Return (winner, combo) where winner is 'X' or 'O' or 'draw' or None"""
    for a,b,c in WIN_COMBO:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a], (a,b,c)
    if all(cell != ' ' for cell in board):
        return 'draw', None
    return None, None


def display_board(board, highlight=None):
    """Display a 3x3 board in console. highlight is a tuple of indices to color."""
    def cell_text(i):
        v = board[i]
        if v == ' ':
            # show position numbers 1-9 for empty cells
            txt = str(i+1)
            return CYAN + txt + RESET
        if highlight and i in highlight:
            color = GREEN if v == AI else YELLOW
            return color + BOLD + v + RESET
        return BOLD + v + RESET if v != ' ' else ' '

    rows = []
    for r in range(3):
        row = ' | '.join(cell_text(r*3 + c) for c in range(3))
        rows.append(' ' + row + ' ')
    sep = '\n' + ('-' * 11) + '\n'
    print('\n' + sep.join(rows) + '\n')


# Minimax implementation

def minimax(board, depth, is_maximizing):
    winner, _ = check_winner(board)
    if winner == AI:
        return 10 - depth
    if winner == HUMAN:
        return depth - 10
    if winner == 'draw':
        return 0

    if is_maximizing:
        best = -10_000
        for i in available_moves(board):
            board[i] = AI
            score = minimax(board, depth+1, False)
            board[i] = ' '
            if score > best:
                best = score
        return best
    else:
        best = 10_000
        for i in available_moves(board):
            board[i] = HUMAN
            score = minimax(board, depth+1, True)
            board[i] = ' '
            if score < best:
                best = score
        return best


def best_move(board):
    # If first move for AI and center free, take center
    if all(c == ' ' for c in board):
        return 4
    best_score = -10_000
    move = None
    for i in available_moves(board):
        board[i] = AI
        score = minimax(board, 0, False)
        board[i] = ' '
        if score > best_score:
            best_score = score
            move = i
    return move


def human_turn(board):
    while True:
        try:
            raw = input('Votre coup (1-9) > ').strip()
            if raw.lower() in ('q','quit','exit'):
                print('Abandon...')
                sys.exit(0)
            pos = int(raw) - 1
            if pos < 0 or pos > 8:
                print('Veuillez entrer un nombre entre 1 et 9.')
                continue
            if board[pos] != ' ':
                print('Case déjà occupée, choisissez une autre case.')
                continue
            return pos
        except ValueError:
            print('Entrée invalide. Utilisez un chiffre entre 1 et 9.')


def ai_turn(board):
    mv = best_move(board)
    return mv


def play_interactive():
    board = [' '] * 9
    current = HUMAN
    print('\nTic-Tac-Toe — Vous êtes X (jouez premier). Tapez q pour quitter.')
    display_board(board)
    while True:
        if current == HUMAN:
            print('\nTour: Vous (X)')
            pos = human_turn(board)
            board[pos] = HUMAN
        else:
            print('\nTour: IA (O) — réflexion...')
            pos = ai_turn(board)
            # Safety: in unlikely case of None
            if pos is None:
                pos = random.choice(available_moves(board))
            board[pos] = AI
            print(f'IA joue en {pos+1}')

        winner, combo = check_winner(board)
        display_board(board, highlight=combo)
        if winner:
            if winner == 'draw':
                print(BOLD + 'Match nul.' + RESET)
            elif winner == HUMAN:
                print(BOLD + GREEN + 'Vous avez gagné !' + RESET)
            else:
                print(BOLD + RED + 'L\'IA a gagné.' + RESET)
            break
        current = AI if current == HUMAN else HUMAN

    # Play again?
    ans = input('\nRecommencer ? (y/n) > ').strip().lower()
    if ans.startswith('y'):
        play_interactive()
    else:
        print('Merci d\'avoir joué !')


# Automated quick tests to ensure AI does not lose

def automated_tests(simulations=100):
    print(f'Exécution de {simulations} simulations aléatoires pour vérifier que l\'IA ne perd pas...')
    losses = 0
    for s in range(simulations):
        board = [' '] * 9
        current = HUMAN
        while True:
            if current == HUMAN:
                # random human move
                mv = random.choice(available_moves(board))
                board[mv] = HUMAN
            else:
                mv = ai_turn(board)
                if mv is None:
                    mv = random.choice(available_moves(board))
                board[mv] = AI
            winner, _ = check_winner(board)
            if winner:
                if winner == HUMAN:
                    losses += 1
                break
            current = AI if current == HUMAN else HUMAN
    print(f'Simulations terminées. Pertes de l\'IA (victoires humaines) : {losses} / {simulations}')
    if losses == 0:
        print(GREEN + 'OK — L\'IA n\'a perdu aucune partie dans ces simulations.' + RESET)
        return 0
    else:
        print(RED + f'ÉCHEC — L\'IA a perdu {losses} fois.' + RESET)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run automated tests')
    parser.add_argument('--sims', type=int, default=100, help='Number of simulations for tests')
    args = parser.parse_args()

    if args.test:
        code = automated_tests(args.sims)
        sys.exit(code)
    else:
        try:
            play_interactive()
        except KeyboardInterrupt:
            print('\nInterrompu. À bientôt!')


if __name__ == '__main__':
    main()
