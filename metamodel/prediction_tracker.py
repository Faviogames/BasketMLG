# ===========================================
# storage/prediction_tracker.py
# Módulo para tracking y análisis de predicciones
# ===========================================

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

def detect_prediction_timing(q_scores: dict, time_info: dict = None) -> str:
    """
    Detecta automáticamente el momento de la predicción basado en los scores de los cuartos
    """
    # Extraer scores de cuartos
    q1_home = q_scores.get('q1_home', 0)
    q1_away = q_scores.get('q1_away', 0)
    q2_home = q_scores.get('q2_home', 0)
    q2_away = q_scores.get('q2_away', 0)
    q3_home = q_scores.get('q3_home', 0)
    q3_away = q_scores.get('q3_away', 0)
    q4_home = q_scores.get('q4_home', 0)
    q4_away = q_scores.get('q4_away', 0)
    
    # Determinar qué cuartos tienen datos
    q1_has_data = q1_home > 0 or q1_away > 0
    q2_has_data = q2_home > 0 or q2_away > 0
    q3_has_data = q3_home > 0 or q3_away > 0
    q4_has_data = q4_home > 0 or q4_away > 0
    
    # Lógica de detección automática
    if not q1_has_data and not q2_has_data and not q3_has_data and not q4_has_data:
        return "pre_game"
    elif q1_has_data and not q2_has_data and not q3_has_data and not q4_has_data:
        return "q1"
    elif q1_has_data and q2_has_data and not q3_has_data and not q4_has_data:
        return "q2"
    elif q1_has_data and q2_has_data and not q3_has_data and not q4_has_data:
        return "halftime"
    elif q1_has_data and q2_has_data and q3_has_data and not q4_has_data:
        return "q3_progress"
    elif q1_has_data and q2_has_data and q3_has_data and q4_has_data:
        # Juego terminado - verificar si hay overtime
        total_score = q1_home + q1_away + q2_home + q2_away + q3_home + q3_away + q4_home + q4_away
        if total_score < 150:
            return "final"
        else:
            return "q4_progress"
    else:
        return "pre_game"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def json_dumps_safe(data: Dict, **kwargs) -> str:
    """Safe JSON dumps that converts numpy types to native Python types"""
    return json.dumps(data, cls=NumpyEncoder, ensure_ascii=False, indent=2, **kwargs)

class PredictionTracker:
    """
    🎯 Sistema de seguimiento de predicciones de Over/Under
    Registra, analiza y rastrea el rendimiento de las predicciones del modelo
    """
    
    def __init__(self, storage_dir: str = "prediction_records"):
        self.storage_dir = storage_dir
        self.ensure_storage_structure()
    
    def ensure_storage_structure(self):
        """Crea la estructura de directorios necesaria"""
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(f"{self.storage_dir}/daily", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/seasonal", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/monthly_indices", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/analytics", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/analytics/monthly_reports", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/analytics/season_analysis", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/exports", exist_ok=True)
    
    def track_prediction(self,
                        home_team: str,
                        away_team: str,
                        odds_line: Optional[float],
                        model_prediction: float,
                        context_adjusted_prediction: float,
                        final_prediction: float,
                        predictions_list: List[Dict],
                        confidence: float,
                        recommended_side: str,
                        edge_magnitude: float,
                        context: Optional[Dict] = None,
                        league: str = "NBA",
                        prediction_timing: str = None) -> str:
        """Registra una nueva predicción para seguimiento posterior"""
        
        # Auto-detect prediction timing if not provided
        if prediction_timing is None:
            if context and 'quarter_scores' in context:
                prediction_timing = detect_prediction_timing(context['quarter_scores'], context.get('time_info'))
            else:
                prediction_timing = "pre_game"
        
        # Generar ID y Timestamp
        prediction_id = self._generate_prediction_id()
        timestamp = datetime.now().isoformat()
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Construir objeto de predicción
        prediction = {
            "prediction_id": prediction_id,
            "metadata": {
                "timestamp": timestamp,
                "date": date_str,
                "version": "2.0",
                "prediction_timing": prediction_timing
            },
            "matchup": {
                "home_team": home_team,
                "away_team": away_team,
                "league": league
            },
            "odds_information": {
                "totals_line": odds_line,
                "bookmaker": "consensus"
            },
            "predictions": {
                "model_prediction": model_prediction,
                "context_adjusted_prediction": context_adjusted_prediction,
                "final_prediction": final_prediction,
                "predictions_list": predictions_list
            },
            "analysis": {
                "confidence": confidence,
                "recommended_bet": recommended_side,
                "edge_percentage": (edge_magnitude / odds_line * 100) if odds_line else 0.0,
                "edge_magnitude": edge_magnitude
            },
            "context": self._clean_game_context(context) if context else {},
            "result_tracking": {
                "status": "pending",
                "actual_total": None,
                "actual_result": None,
                "prediction_accuracy": None,
                "result_timestamp": None,
                "notes": []
            }
        }
        
        # 1. Guardar en archivo diario
        daily_predictions = self._load_daily_predictions(date_str)
        daily_predictions.append(prediction)
        self._save_daily_predictions(date_str, daily_predictions)
        
        # 2. Guardar archivo individual
        individual_file = self._get_individual_file_path(prediction_id, home_team, away_team, date_str)
        with open(individual_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(prediction))
            
        # 3. Actualizar archivo de temporada
        try:
            season_year = self._get_season_year(date_str)
            season_file = self.get_season_file_path(league, season_year)
            
            if os.path.exists(season_file):
                with open(season_file, 'r', encoding='utf-8') as f:
                    season_data = json.load(f)
            else:
                season_data = {
                    "season_metadata": {
                        "league": league,
                        "season_year": season_year,
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "total_predictions": 0
                    },
                    "predictions": []
                }
            
            season_data["predictions"].append(prediction)
            season_data["season_metadata"]["total_predictions"] += 1
            season_data["season_metadata"]["last_updated"] = datetime.now().isoformat()
            
            with open(season_file, 'w', encoding='utf-8') as f:
                f.write(json_dumps_safe(season_data))
        except Exception as e:
            print(f"Error actualizando archivo de temporada: {e}")
        
        # 4. Actualizar índice mensual
        try:
            self.update_monthly_index(prediction)
        except Exception as e:
            print(f"Error actualizando índice mensual: {e}")
        
        print(f"Predicción registrada: {prediction_id} ({home_team} vs {away_team})")
        return prediction_id

    def get_predictions_by_date(self, date_str: str) -> List[Dict]:
        return self._load_daily_predictions(date_str)
    
    def get_pending_predictions(self) -> List[Dict]:
        all_predictions = self.get_all_predictions()
        return [p for p in all_predictions if p['result_tracking']['status'] == 'pending']
    
    def get_all_predictions(self) -> List[Dict]:
        all_predictions = []
        daily_dir = f"{self.storage_dir}/daily"
        if os.path.exists(daily_dir):
            for filename in os.listdir(daily_dir):
                if filename.endswith('_predictions.json'):
                    file_path = os.path.join(daily_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            predictions = json.load(f)
                            all_predictions.extend(predictions)
                    except (json.JSONDecodeError, IOError):
                        continue
        return all_predictions
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        all_predictions = self.get_all_predictions()
        completed_predictions = [p for p in all_predictions if p['result_tracking']['status'] == 'completed']
        
        if not completed_predictions:
            return {
                "total_predictions": len(all_predictions),
                "completed_predictions": 0,
                "pending_predictions": len(all_predictions),
                "overall_accuracy": 0.0
            }
        
        total = len(completed_predictions)
        correct = len([p for p in completed_predictions if p['result_tracking']['prediction_accuracy']])
        
        return {
            "total_predictions": len(all_predictions),
            "completed_predictions": total,
            "pending_predictions": len(all_predictions) - total,
            "overall_accuracy": (correct / total * 100) if total > 0 else 0.0
        }
    
    def export_summary_report(self, output_file: str = None) -> str:
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.storage_dir}/analytics/prediction_summary_{timestamp}.json"
        
        metrics = self.calculate_performance_metrics()
        report = {
            "report_metadata": {"generated_at": datetime.now().isoformat()},
            "performance_metrics": metrics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(report))
        return output_file
    
    # Métodos privados de utilidad
    def _generate_prediction_id(self) -> str:
        return str(uuid.uuid4())[:8].upper()
    
    def _load_daily_predictions(self, date_str: str) -> List[Dict]:
        daily_file = f"{self.storage_dir}/daily/{date_str}_predictions.json"
        if os.path.exists(daily_file):
            try:
                with open(daily_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []
    
    def _save_daily_predictions(self, date_str: str, predictions: List[Dict]):
        daily_file = f"{self.storage_dir}/daily/{date_str}_predictions.json"
        with open(daily_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(predictions))
    
    def _get_individual_file_path(self, prediction_id: str, home_team: str, away_team: str, date_str: str) -> str:
        home_clean = "".join(c for c in home_team if c.isalnum() or c in (' ', '-', '_')).rstrip()
        away_clean = "".join(c for c in away_team if c.isalnum() or c in (' ', '-', '_')).rstrip()
        return f"{self.storage_dir}/daily/{prediction_id}_{home_clean}_vs_{away_clean}_{date_str}.json"
    
    def _clean_game_context(self, context: Dict) -> Dict:
        fields_to_remove = ['live_alerts', 'update_count', 'prediction_stability']
        clean_context = {}
        for key, value in context.items():
            if key not in fields_to_remove:
                clean_context[key] = value
        return clean_context
    
    # 🆕 Métodos para Sistema de Almacenamiento Híbrido
    def get_season_file_path(self, league: str, season_year: str) -> str:
        return f"{self.storage_dir}/seasonal/{league}_{season_year}_season.json"

    def _get_season_year(self, date_str: str) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            year = date_obj.year
            month = date_obj.month
            if month >= 9:
                return f"{year}-{year+1}"
            else:
                return f"{year-1}-{year}"
        except ValueError:
            return "unknown"

    def get_monthly_index_path(self, league: str, year: str, month: str) -> str:
        return f"{self.storage_dir}/monthly_indices/{league}_{year}-{month}_index.json"

    def update_monthly_index(self, prediction_data: Dict):
        league = prediction_data['matchup']['league']
        date_str = prediction_data['metadata']['date']
        date_parts = date_str.split('-')
        year = date_parts[0]
        month = date_parts[1]
        
        index_file = self.get_monthly_index_path(league, year, month)
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                monthly_index = json.load(f)
        else:
            monthly_index = {
                "index_metadata": {
                    "league": league,
                    "year": year,
                    "month": month,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_predictions": 0
                },
                "prediction_references": []
            }
        
        prediction_ref = {
            "prediction_id": prediction_data["prediction_id"],
            "date": date_str,
            "home_team": prediction_data["matchup"]["home_team"],
            "away_team": prediction_data["matchup"]["away_team"],
            "final_prediction": prediction_data["predictions"]["final_prediction"],
            "actual_total": prediction_data["result_tracking"]["actual_total"],
            "status": prediction_data["result_tracking"]["status"],
            "accuracy": prediction_data["result_tracking"]["prediction_accuracy"]
        }
        
        monthly_index["prediction_references"].append(prediction_ref)
        monthly_index["index_metadata"]["total_predictions"] += 1
        monthly_index["index_metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(monthly_index))

    def get_season_predictions(self, league: str, season_year: str) -> List[Dict]:
        season_file = self.get_season_file_path(league, season_year)
        if os.path.exists(season_file):
            with open(season_file, 'r', encoding='utf-8') as f:
                season_data = json.load(f)
                return season_data.get("predictions", [])
        return []

    def get_monthly_predictions(self, league: str, year: str, month: str) -> List[Dict]:
        index_file = self.get_monthly_index_path(league, year, month)
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                monthly_data = json.load(f)
                return monthly_data.get("prediction_references", [])
        return []

    def generate_monthly_report(self, league: str, year: str, month: str) -> str:
        monthly_refs = self.get_monthly_predictions(league, year, month)
        if not monthly_refs:
            return f"No hay predicciones para {league} {year}-{month}"
        
        total = len(monthly_refs)
        completed = len([r for r in monthly_refs if r['status'] == 'completed'])
        correct = len([r for r in monthly_refs if r['accuracy'] is True])
        accuracy = (correct / completed * 100) if completed > 0 else 0
        
        report = {
            "report_metadata": {
                "league": league,
                "year": year,
                "month": month,
                "generated_at": datetime.now().isoformat(),
                "report_type": "monthly_performance"
            },
            "summary": {
                "total_predictions": total,
                "completed_predictions": completed,
                "correct_predictions": correct,
                "accuracy_percentage": accuracy
            },
            "predictions": monthly_refs
        }
        
        report_file = f"{self.storage_dir}/analytics/monthly_reports/{league}_{year}-{month}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(report))
        return report_file

    def generate_season_analysis(self, league: str, season_year: str) -> str:
        season_predictions = self.get_season_predictions(league, season_year)
        if not season_predictions:
            return None
        
        completed = [p for p in season_predictions if p['result_tracking']['status'] == 'completed']
        correct = [p for p in completed if p['result_tracking']['prediction_accuracy']]
        
        analysis = {
            "analysis_metadata": {
                "league": league,
                "season_year": season_year,
                "generated_at": datetime.now().isoformat()
            },
            "summary": {
                "total_predictions": len(season_predictions),
                "completed_predictions": len(completed),
                "correct_predictions": len(correct),
                "overall_accuracy": (len(correct) / len(completed) * 100) if completed else 0
            },
            "detailed_predictions": season_predictions
        }
        
        analysis_file = f"{self.storage_dir}/analytics/season_analysis/{league}_{season_year}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(json_dumps_safe(analysis))
        return analysis_file

# Función de conveniencia para uso directo
def track_basketball_prediction(home_team: str, away_team: str, **kwargs) -> str:
    tracker = PredictionTracker()
    return tracker.track_prediction(home_team, away_team, **kwargs)

if __name__ == "__main__":
    print("=== TEST DEL SISTEMA DE PREDICTION TRACKING ===")
    tracker = PredictionTracker()
    # Test simple
    try:
        pred_id = tracker.track_prediction(
            home_team="Lakers", away_team="Warriors", odds_line=225.5,
            model_prediction=228.0, context_adjusted_prediction=228.0, final_prediction=228.0,
            predictions_list=[], confidence=0.8, recommended_side="OVER", edge_magnitude=2.5,
            context={"quarter_scores": {"q1_home": 30, "q1_away": 30}}, league="NBA"
        )
        print(f"Test exitoso. ID: {pred_id}")
    except Exception as e:
        print(f"Test fallido: {e}")