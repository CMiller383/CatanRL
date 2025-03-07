from game.board import Board
from visualization.board_graph import visualize_board

def main():
    board = Board()
    print("Generated Unique Catan Game:")
    print("\nHexes:")
    for hex_obj in board.hexes.values():
        print(hex_obj)
    print("\nSpots:")
    for spot in board.spots.values():
        print(spot)
    print("\nRoads:")
    for road in board.roads.values():
        print(road)
    
    visualize_board(board)

if __name__ == "__main__":
    main()
