from gui.pygame_gui import CatanGame, display_player_setup_menu

def main():
    # Show player setup menu
    num_human_players, agent_types = display_player_setup_menu()
    
    # Exit if menu was closed without starting
    if num_human_players is None:
        return
    
    # Start the game with selected setup
    game = CatanGame(
        num_human_players=num_human_players,
        agent_types=agent_types
    )
    game.run()

if __name__ == "__main__":
    main()