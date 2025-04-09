def calculate_longest_path(roads):
    """Optimized algorithm for longest path calculation"""
    if not roads:
        return 0
    
    # Build adjacency list
    adjacency = {}
    for spot1, spot2 in roads:
        if spot1 not in adjacency:
            adjacency[spot1] = []
        if spot2 not in adjacency:
            adjacency[spot2] = []
        
        adjacency[spot1].append(spot2)
        adjacency[spot2].append(spot1)
    
    # Get the longest path using two BFS passes
    def find_furthest(start):
        """Find the furthest node and its distance from start using BFS"""
        distances = {start: 0}
        queue = [start]
        furthest = (start, 0)  # (node, distance)
        
        while queue:
            node = queue.pop(0)
            dist = distances[node]
            
            if dist > furthest[1]:
                furthest = (node, dist)
            
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
        
        return furthest
    
    # Find connected components (separate road networks)
    visited = set()
    max_path_length = 0
    
    # Process each connected component
    for start_node in adjacency:
        if start_node in visited:
            continue
            
        # Find furthest node from arbitrary start node
        furthest1, _ = find_furthest(start_node)
        visited.add(start_node)
        
        # Find furthest node from the furthest node found
        furthest2, distance = find_furthest(furthest1)
        
        # The distance between these two nodes is the diameter (longest path)
        max_path_length = max(max_path_length, distance)
        
        # Mark all nodes in this component as visited
        component_queue = [start_node]
        component_visited = {start_node}
        while component_queue:
            node = component_queue.pop(0)
            visited.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in component_visited:
                    component_visited.add(neighbor)
                    component_queue.append(neighbor)
    
    return max_path_length


def update_longest_road(state):
    """
    Calculate the longest road for each player and update the longest_road_player.
    Returns True if the longest road owner changed.
    """
    min_road_length = 5  # Minimum length required for longest road
    previous_owner = state.longest_road_player
    
    # Get longest road for each player
    longest_roads = {}
    for player_idx, player in enumerate(state.players):
        if len(player.roads) < min_road_length:
            continue
        
        # Collect roads as (spot1_id, spot2_id) tuples
        player_roads = []
        for road_id in player.roads:
            road = state.board.get_road(road_id)
            if road is not None:
                player_roads.append((road.spot1_id, road.spot2_id))
        
        # Calculate longest path for this player
        if player_roads:
            longest_path = calculate_longest_path(player_roads)
            if longest_path >= min_road_length:
                longest_roads[player_idx] = longest_path
    
    # Determine who gets the longest road
    new_owner = None
    new_length = 0
    
    for player_idx, length in longest_roads.items():
        if length > new_length:
            new_length = length
            new_owner = player_idx
        elif length == new_length and new_owner is not None:
            # If there's a tie, the current owner keeps it
            if state.longest_road_player == player_idx:
                new_owner = player_idx
    
    # Update the game state
    if new_owner != state.longest_road_player:
        # Remove VPs from previous owner if any
        if state.longest_road_player is not None:
            state.players[state.longest_road_player].victory_points -= 2
        
        # Add VPs to new owner if any
        if new_owner is not None:
            state.players[new_owner].victory_points += 2
        
        state.longest_road_player = new_owner
        state.longest_road_length = new_length if new_owner is not None else 4
        return True
    
    return False
