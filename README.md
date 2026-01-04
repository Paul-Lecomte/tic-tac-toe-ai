# Tic-Tac-Toe AI (Minimax + NN Visualization)

This project contains several implementations of Tic-Tac-Toe:

- `tic_tac_toe.py`: Console game (Human X vs AI O using Minimax). Includes automated tests (`--test`).
- `tic_tac_toe_gui.py`: Pygame-based UI with two AI modes: Minimax (perfect) and a simple Neural Network (numpy) with real-time visualization of activations and move scores. The NN can be trained quickly against Minimax to mimic its choices.

Requirements
```
pip install -r requirements.txt
```

Run the GUI
```
python tic_tac_toe_gui.py
```

Controls in the GUI
- Click a cell to play (you are X, play first)
- R: reset game
- M: switch AI to Minimax
- N: switch AI to NeuralNet
- T: train the NeuralNet quickly (generates dataset vs Minimax and trains for a short time)
- Q: quit

Notes
- The NN used is a lightweight numpy implementation and is intended for visualization and learning, not production performance.
- If you prefer, you can implement a PyTorch version (optional). The code will fall back to numpy if PyTorch is not installed.

