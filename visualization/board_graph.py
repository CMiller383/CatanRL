import networkx as nx
import matplotlib.pyplot as plt
from game.board import Board

def rotate_point(point):
    x, y = point
    return (y, -x)

def visualize_board(board: Board):
    G = nx.Graph()
    pos = {}
    # Add spots as nodes, applying the rotation.
    for s_id, spot in board.spots.items():
        rotated = rotate_point(spot.position)
        G.add_node(s_id)
        pos[s_id] = rotated
    # Add roads as edges.
    for road in board.roads.values():
        G.add_edge(road.spot1_id, road.spot2_id)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue')
    
    # Annotate each hex with its resource and dice number.
    for hex_obj in board.hexes.values():
        rx, ry = rotate_point(hex_obj.center)
        plt.text(rx, ry, f"{hex_obj.resource.name}\n{hex_obj.number}", 
                 horizontalalignment='center', verticalalignment='center', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title("Unique Catan Board")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    board = Board()
    print("Generated Unique Catan Game:")
    for hex_obj in board.hexes.values():
        print(hex_obj)
    visualize_board(board)
