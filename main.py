from game.board import Board
from gui.pygame_gui import CatanGame

def main():
    board = Board()
    gui = CatanGame(board)
    gui.run()
    
if __name__ == "__main__":
    main()
