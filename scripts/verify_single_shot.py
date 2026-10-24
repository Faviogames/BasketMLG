
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_api import LiveProcessor

async def test_single_shot_execution():
    print("Testing Single Shot Execution...")
    
    # Mock trained data
    trained_data = {
        'league_name': 'NBA',
        'team_names': ['Team A', 'Team B'],
        'features_used': [],
        'std_dev': 10.0
    }
    
    # Initialize processor
    processor = LiveProcessor(trained_data)
    
    # Mock client methods
    processor.client.get_live_game_data = AsyncMock(return_value={
        'q_scores': {'q1_home': 25, 'q1_away': 25, 'q2_home': 25, 'q2_away': 25, 'q3_home': 0, 'q3_away': 0},
        'time_info': {'current_quarter_display': 'Halftime'},
        'game_finished': False
    })
    processor.client.get_game_statistics = AsyncMock(return_value={
        'statistics': [] # Empty stats for simplicity
    })
    processor.client.clear_cache = MagicMock()
    
    # Mock prediction function to avoid importing real one
    processor._get_prediction_function = MagicMock(return_value=lambda *args, **kwargs: (200.0, [], {}))
    
    # Mock internal methods to avoid complex logic
    processor._should_update_prediction = MagicMock(return_value=(True, "Test Update"))
    processor.compute_live_signals = MagicMock(return_value={})
    processor._detect_significant_changes = MagicMock(return_value={})
    processor._generate_live_alerts = MagicMock(return_value=[])
    processor._calculate_live_context = MagicMock(return_value={})
    
    # Mock global metamodel function if it's imported inside monitor_game
    # (It is imported inside, so we might need to mock it in sys.modules or just let it fail/mock the call)
    # Actually, monitor_game imports apply_metamodel_ai_adjustment. 
    # We can mock it by patching analysis.live_mode.apply_metamodel_ai_adjustment if needed.
    # But let's try running it. The import might fail if not in path, but we added root to path.
    
    # Run monitor_game with single_shot=True
    game_info = {
        'home_team': 'Team A', 
        'away_team': 'Team B',
        'home_score': 50,
        'away_score': 50,
        'status': 'Halftime'
    }
    
    print("Calling monitor_game(single_shot=True)...")
    try:
        await processor.monitor_game(
            game_id="123", 
            game_info=game_info, 
            update_interval=1, 
            max_updates=5, # Should stop after 1
            single_shot=True
        )
        print("[SUCCESS] monitor_game returned successfully.")
    except Exception as e:
        print(f"[FAILURE] monitor_game failed: {e}")
        import traceback
        traceback.print_exc()

    # Verify clear_cache was called
    if processor.client.clear_cache.called:
        print("[SUCCESS] clear_cache() was called.")
    else:
        print("[FAILURE] clear_cache() was NOT called.")

async def test_halftime_enforcement():
    print("\nTesting Halftime Enforcement...")
    
    # Mock trained data
    trained_data = {
        'league_name': 'NBA',
        'team_names': ['Team A', 'Team B'],
        'features_used': [],
        'std_dev': 10.0
    }
    
    processor = LiveProcessor(trained_data)
    
    # Mock client to return incomplete scores (Q2 missing)
    processor.client.get_live_game_data = AsyncMock(return_value={
        'q_scores': {'q1_home': 25, 'q1_away': 25, 'q2_home': 0, 'q2_away': 0}, # Q2 missing
        'time_info': {'current_quarter_display': 'Q2'},
        'game_finished': False
    })
    processor.client.clear_cache = MagicMock()
    
    game_info = {
        'home_team': 'Team A', 
        'away_team': 'Team B',
        'home_score': 25,
        'away_score': 25,
        'status': 'Q2'
    }
    
    print("Calling monitor_game(single_shot=True) with incomplete data...")
    try:
        await processor.monitor_game(
            game_id="123", 
            game_info=game_info, 
            update_interval=1, 
            max_updates=5,
            single_shot=True
        )
        print("[SUCCESS] monitor_game returned (likely due to enforcement return).")
    except Exception as e:
        print(f"[FAILURE] monitor_game failed unexpectedly: {e}")

if __name__ == "__main__":
    asyncio.run(test_single_shot_execution())
    asyncio.run(test_halftime_enforcement())
