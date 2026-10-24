
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_api import LiveProcessor

async def run_full_halftime_test():
    print("==================================================")
    print("FULL HALFTIME SYSTEM TEST")
    print("==================================================")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Setup Mock Data (Realistic NBA Halftime Scenario)
    print("\n[1] Setting up Mock Data (NBA Halftime)...")
    
    # Mock trained data
    trained_data = {
        'league_name': 'NBA',
        'team_names': ['Lakers', 'Celtics'],
        'features_used': ['possessions_avg', 'ortg_home', 'ortg_away', 'ts_home', 'ts_away'], # Minimal set
        'std_dev': 10.0,
        'historical_df': [] # Empty for mock
    }
    
    # Game Info
    game_info = {
        'id': 'test_game_123',
        'home_team': 'Lakers', 
        'away_team': 'Celtics',
        'home_score': 58,
        'away_score': 55,
        'status': 'Halftime',
        'quarter_info': {
            'q1_home': 30, 'q1_away': 28,
            'q2_home': 28, 'q2_away': 27,
            'q3_home': 0, 'q3_away': 0,
            'q4_home': 0, 'q4_away': 0
        }
    }
    
    # Mock Live Data Response
    mock_live_data = {
        'q_scores': game_info['quarter_info'],
        'time_info': {
            'current_quarter_display': 'Halftime',
            'minutes_played': 24.0,
            'total_remaining_minutes': 24.0,
            'completion_percentage': 50.0,
            'current_pace_per_minute': (58+55)/24.0 # ~4.7 pts/min
        },
        'game_finished': False,
        'context': {'estimated_quarter': 'Halftime'}
    }
    
    # Mock Stats Response (Split stats for correct TS% calc)
    # Lakers: 58 pts. FG: 22/45, 3PT: 6/18, FT: 8/10. ORB: 5, DRB: 20. AST: 15. TO: 6.
    # Celtics: 55 pts. FG: 20/42, 3PT: 5/15, FT: 10/12. ORB: 4, DRB: 18. AST: 12. TO: 8.
    mock_stats = {
        'statistics': [{
            'period': 'ALL',
            'groups': [{
                'statisticsItems': [
                    {'name': 'Field goals', 'home': '22/45', 'away': '20/42'},
                    {'name': '3 pointers', 'home': '6/18', 'away': '5/15'},
                    {'name': 'Free throws', 'home': '8/10', 'away': '10/12'},
                    {'name': 'Rebounds', 'home': '25', 'away': '22'},
                    {'name': 'Assists', 'home': '15', 'away': '12'},
                    {'name': 'Turnovers', 'home': '6', 'away': '8'},
                    {'name': 'Fouls', 'home': '9', 'away': '10'},
                    {'name': 'Steals', 'home': '4', 'away': '3'},
                    {'name': 'Blocks', 'home': '2', 'away': '1'}
                ]
            }]
        }]
    }

    # 2. Initialize Processor
    print("[2] Initializing LiveProcessor...")
    processor = LiveProcessor(trained_data)
    
    # 3. Mock Client Methods
    processor.client.get_live_game_data = AsyncMock(return_value=mock_live_data)
    processor.client.get_game_statistics = AsyncMock(return_value=mock_stats)
    processor.client.clear_cache = MagicMock()
    
    # IMPORTANT: We want to use the REAL logic for `monitor_game` and `_should_update_prediction`
    # But we need to mock the actual prediction function to avoid loading heavy ML models
    # However, to test "reaction", we should mock the return of `get_predictions_with_alerts` 
    # to return a value that triggers the safeguard logic.
    
    # Let's say the "Model" predicts something wild, like 250 pts (Current pace projects ~226)
    # Safe Projection: 113 + (24 * 4.7) = ~226
    # If Model says 250, Safeguard should pull it down.
    
    def mock_get_predictions(*args, **kwargs):
        print("    [MOCK] Running Mock Prediction Model...")
        # Return base_prediction, predictions_list, alerts_data
        return 250.0, [], {'live_alerts': []}
        
    processor._get_prediction_function = MagicMock(return_value=mock_get_predictions)
    
    # We also need to mock `aplicar_metamodelo_ai` since it's imported inside monitor_game
    # We can't easily mock a local import without patching sys.modules or the function itself if it was a method.
    # Since `monitor_game` imports it, we'll have to rely on the real one or patch it globally.
    # For this test, let's assume the real one works or fails gracefully (which the code handles).
    # To be safe, let's try to patch it in `analysis.live_mode` if possible, but `monitor_game` imports it directly.
    # Actually, `monitor_game` does: `from analysis.live_mode import apply_context_adjustment` (in the static helper)
    # and `from analysis.metamodel_integration import aplicar_metamodelo_ai` (implied or direct).
    # Let's just run it. If it fails to load the model, it catches exception and uses base prediction.
    
    # 4. Run Test
    print("\n[3] Running Single-Shot Prediction...")
    print(f"    Current Score: {game_info['home_score']} - {game_info['away_score']} (Total: {game_info['home_score'] + game_info['away_score']})")
    print(f"    Status: {game_info['status']}")
    
    try:
        await processor.monitor_game(
            game_id=game_info['id'], 
            game_info=game_info, 
            update_interval=1, 
            max_updates=1,
            single_shot=True
        )
        print("\n[SUCCESS] Test Completed Successfully")
        
    except Exception as e:
        print(f"\n[FAILURE] Test Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Verify Health Checks
    print("\n[4] System Health Verification")
    if processor.client.clear_cache.called:
        print("    [OK] Cache Cleared (Fresh Data Guaranteed)")
    else:
        print("    [FAIL] Cache NOT Cleared")
        
    # Check if we can inspect the last prediction to see if safeguard worked
    if processor.last_prediction:
        print(f"    [OK] Final Prediction Generated: {processor.last_prediction:.1f}")
        if processor.last_prediction < 250.0:
             print(f"    [OK] Safeguard Active (Reduced 250.0 -> {processor.last_prediction:.1f})")
        else:
             print(f"    [WARN] Safeguard Inactive or Model Trusted (Prediction: {processor.last_prediction:.1f})")
    else:
        print("    [FAIL] No Prediction Generated")

if __name__ == "__main__":
    asyncio.run(run_full_halftime_test())
