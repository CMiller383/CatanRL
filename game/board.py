import math
import random
from .hex import Hex
from .spot import Spot
from .road import Road
from .player import Player
from .resource import Resource

class Board:
    def __init__(self):
        self.hexes = {}   # hex_id -> Hex
        self.spots = {}   # spot_id -> Spot
        self.roads = {}   # road_id -> Road
        self.players = {} # player_id -> Player
        self._init_hexes()
        self._init_spots()
        self._init_roads()
    
    def _init_hexes(self):
        # Create a full Catan board (radius 2: 19 hexes) using axial coordinates for flat-topped hexes.
        hex_size = 1.0
        hexes_list = []
        hex_id = 1
        for q in range(-2, 3):
            r1 = max(-2, -q - 2)
            r2 = min(2, -q + 2)
            for r in range(r1, r2 + 1):
                x = hex_size * 3/2 * q
                y = hex_size * math.sqrt(3) * (r + q/2)
                center = (round(x, 2), round(y, 2))
                hexes_list.append((hex_id, center))
                hex_id += 1

        # Standard resource counts: 4 wood, 3 brick, 4 wheat, 4 sheep, 3 ore, 1 desert (total 19)
        resources = ([Resource.WOOD] * 4 + [Resource.BRICK] * 3 +
                     [Resource.WHEAT] * 4 + [Resource.SHEEP] * 4 +
                     [Resource.ORE] * 3 + [Resource.DESERT])
        random.shuffle(resources)
        # Dice numbers for non-desert hexes (18 numbers)
        dice_numbers = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 3, 6, 11, 4, 12]
        random.shuffle(dice_numbers)
        
        for hex_info in hexes_list:
            h_id, center = hex_info
            resource = resources.pop(0)
            number = 0 if resource == Resource.DESERT else dice_numbers.pop(0)
            self.hexes[h_id] = Hex(h_id, number, resource, center)
    
    def _init_spots(self):
        # Derive spots by calculating hex vertices and merging those very close together.
        hex_size = 1.0
        vertex_to_hexes = {}
        tol = 0.05  # tolerance for merging vertices
        
        def find_close_vertex(v):
            for key in vertex_to_hexes.keys():
                if math.hypot(v[0] - key[0], v[1] - key[1]) < tol:
                    return key
            return None

        for hex_obj in self.hexes.values():
            cx, cy = hex_obj.center
            for angle_deg in [0, 60, 120, 180, 240, 300]:
                angle_rad = math.radians(angle_deg)
                vx = cx + hex_size * math.cos(angle_rad)
                vy = cy + hex_size * math.sin(angle_rad)
                v = (vx, vy)
                key = find_close_vertex(v)
                if key is None:
                    key = (round(v[0], 2), round(v[1], 2))
                    vertex_to_hexes[key] = [hex_obj.hex_id]
                else:
                    vertex_to_hexes[key].append(hex_obj.hex_id)
        
        spot_id = 1
        for vertex, hex_ids in vertex_to_hexes.items():
            self.spots[spot_id] = Spot(spot_id, hex_ids, position=vertex)
            spot_id += 1
    
    def _init_roads(self):
        # Derive roads by connecting spots that form edges on each hex.
        road_set = set()
        hex_size = 1.0
        tol = 0.05  # same tolerance used in _init_spots

        def find_spot_by_vertex(v, tol=tol):
            for s_id, spot in self.spots.items():
                if ( (v[0] - spot.position[0])**2 + (v[1] - spot.position[1])**2 )**0.5 < tol:
                    return s_id
            return None

        for hex_obj in self.hexes.values():
            cx, cy = hex_obj.center
            vertices = []
            for angle_deg in [0, 60, 120, 180, 240, 300]:
                angle_rad = math.radians(angle_deg)
                vx = cx + hex_size * math.cos(angle_rad)
                vy = cy + hex_size * math.sin(angle_rad)
                # Instead of rounding, we directly find the matching spot.
                spot_id = find_spot_by_vertex((vx, vy))
                if spot_id is not None:
                    vertices.append(spot_id)
            # Connect consecutive vertices (wrap-around)
            for i in range(len(vertices)):
                s1 = vertices[i]
                s2 = vertices[(i + 1) % len(vertices)]
                if s1 is not None and s2 is not None:
                    road_set.add(tuple(sorted((s1, s2))))
        road_id = 1
        for road_pair in road_set:
            s1, s2 = road_pair
            hexes1 = set(self.spots[s1].adjacent_hex_ids)
            hexes2 = set(self.spots[s2].adjacent_hex_ids)
            adjacent_hexes = list(hexes1.intersection(hexes2))
            self.roads[road_id] = Road(s1, s2, adjacent_hexes)
            road_id += 1

    
    def add_player(self, player_id: int, name: str = ""):
        player = Player(player_id, name)
        self.players[player_id] = player
    
    def get_hex(self, hex_id: int):
        return self.hexes.get(hex_id)
    
    def get_spot(self, spot_id: int):
        return self.spots.get(spot_id)
    
    def get_road(self, road_id: int):
        return self.roads.get(road_id)
    
    def __repr__(self):
        return (f"Board(hexes={list(self.hexes.keys())}, spots={list(self.spots.keys())}, "
                f"roads={list(self.roads.keys())}, players={list(self.players.keys())})")
