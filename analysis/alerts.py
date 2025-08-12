# ===========================================
# Archivo: alerts.py (v1.2) - VERSIÓN ROBUSTA CONTRA DATOS CORRUPTOS
# Sistema de alertas inteligentes con detección de balance de juego
# ===========================================
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 🆕 Importar los nuevos tipos de alerta desde config
from config import ALERT_TYPES, ALERT_THRESHOLDS

class BasketballAlerts:
    """Motor de alertas inteligentes para baloncesto"""
    
    def __init__(self, historical_df: pd.DataFrame):
        self.historical_df = historical_df
        self.team_patterns = {}
        self._calculate_team_patterns()
    
    def _calculate_team_patterns(self):
        """Calcula patrones históricos por equipo"""
        print("🔍 Calculando patrones históricos por equipo...")
        
        for team in self._get_all_teams():
            self.team_patterns[team] = self._analyze_team_patterns(team)
    
    def _get_all_teams(self) -> List[str]:
        """Obtiene lista de todos los equipos únicos"""
        if self.historical_df.empty:
            return []
        
        home_teams = self.historical_df['home_team'].unique().tolist()
        away_teams = self.historical_df['away_team'].unique().tolist()
        return list(set(home_teams + away_teams))
    
    def _analyze_team_patterns(self, team_name: str) -> Dict[str, Any]:
        """Analiza patrones históricos específicos de un equipo"""
        team_matches = self._get_team_matches(team_name)
        
        if team_matches.empty:
            return self._get_default_patterns()
        
        patterns = {}
        
        # 📊 PROMEDIOS POR CUARTO (AHORA ROBUSTO)
        patterns['quarter_averages'] = self._calculate_quarter_averages(team_matches)
        patterns['quarter_std'] = self._calculate_quarter_std(team_matches)
        
        # 🎯 PATRONES DE MITADES
        patterns['first_half_avg'] = self._safe_mean(team_matches, 'first_half_points', 50)
        patterns['second_half_avg'] = self._safe_mean(team_matches, 'second_half_points', 50)
        patterns['second_half_surge'] = patterns['second_half_avg'] - patterns['first_half_avg']
        
        # 🚀 TENDENCIAS DE RECUPERACIÓN
        patterns['slow_start_recovery_rate'] = self._calculate_recovery_rate(team_matches)
        patterns['comeback_strength'] = self._calculate_comeback_strength(team_matches)
        
        # ⚡ PATRONES DE PACE
        patterns['avg_pace'] = self._safe_mean(team_matches, 'live_pace_estimate', 100)
        patterns['pace_std'] = self._safe_std(team_matches, 'live_pace_estimate', 10)
        
        # 🎪 CONSISTENCIA POR CUARTOS
        patterns['consistency_score'] = self._calculate_consistency(team_matches)
        patterns['closing_strength'] = patterns['quarter_averages'].get('q4', 25)  # Fuerza en Q4
        
        return patterns
    
    def _safe_mean(self, df: pd.DataFrame, column: str, default_value: float) -> float:
        """🛡️ CALCULA MEAN DE FORMA SEGURA, manejando datos corruptos"""
        if column not in df.columns or df.empty:
            return default_value
        
        try:
            # Convertir a numérico, forzando errores a NaN
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            if numeric_values.isna().all():
                return default_value
            return numeric_values.mean()
        except (ValueError, TypeError, AttributeError):
            return default_value
    
    def _safe_std(self, df: pd.DataFrame, column: str, default_value: float) -> float:
        """🛡️ CALCULA STD DE FORMA SEGURA, manejando datos corruptos"""
        if column not in df.columns or df.empty:
            return default_value
        
        try:
            # Convertir a numérico, forzando errores a NaN
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            if numeric_values.isna().all() or len(numeric_values.dropna()) < 2:
                return default_value
            return numeric_values.std()
        except (ValueError, TypeError, AttributeError):
            return default_value
    
    def _get_team_matches(self, team_name: str) -> pd.DataFrame:
        """Obtiene todos los partidos de un equipo con datos por cuartos"""
        if self.historical_df.empty:
            return pd.DataFrame()
        
        team_matches = self.historical_df[
            (self.historical_df['home_team'] == team_name) | 
            (self.historical_df['away_team'] == team_name)
        ].copy()
        
        if team_matches.empty:
            return pd.DataFrame()
        
        # Añadir datos específicos del equipo (MEJORADO)
        for idx, match in team_matches.iterrows():
            is_home = match['home_team'] == team_name
            
            # 🛡️ MANEJO ROBUSTO DE RAW_MATCH
            try:
                if 'raw_match' in match and match['raw_match'] and isinstance(match['raw_match'], dict):
                    quarter_scores = match['raw_match'].get('quarter_scores', {})
                    
                    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if quarter in quarter_scores and isinstance(quarter_scores[quarter], dict):
                            team_key = 'home_score' if is_home else 'away_score'
                            quarter_points = quarter_scores[quarter].get(team_key, 0)
                            
                            # 🛡️ CONVERSIÓN SEGURA A NÚMERO
                            try:
                                team_matches.at[idx, f'{quarter.lower()}_points'] = pd.to_numeric(quarter_points, errors='coerce')
                            except:
                                team_matches.at[idx, f'{quarter.lower()}_points'] = 0
                        else:
                            team_matches.at[idx, f'{quarter.lower()}_points'] = 0
                    
                    # Calcular mitades de forma segura
                    q1_pts = team_matches.at[idx, 'q1_points'] if not pd.isna(team_matches.at[idx, 'q1_points']) else 0
                    q2_pts = team_matches.at[idx, 'q2_points'] if not pd.isna(team_matches.at[idx, 'q2_points']) else 0
                    q3_pts = team_matches.at[idx, 'q3_points'] if not pd.isna(team_matches.at[idx, 'q3_points']) else 0
                    q4_pts = team_matches.at[idx, 'q4_points'] if not pd.isna(team_matches.at[idx, 'q4_points']) else 0
                    
                    team_matches.at[idx, 'first_half_points'] = q1_pts + q2_pts
                    team_matches.at[idx, 'second_half_points'] = q3_pts + q4_pts
                else:
                    # Si no hay raw_match, usar valores por defecto
                    for quarter in ['q1', 'q2', 'q3', 'q4']:
                        team_matches.at[idx, f'{quarter}_points'] = 0
                    team_matches.at[idx, 'first_half_points'] = 0
                    team_matches.at[idx, 'second_half_points'] = 0
            
            except Exception as e:
                # En caso de cualquier error, usar valores por defecto
                print(f"⚠️ Error procesando datos de cuartos para {team_name}: {e}")
                for quarter in ['q1', 'q2', 'q3', 'q4']:
                    team_matches.at[idx, f'{quarter}_points'] = 0
                team_matches.at[idx, 'first_half_points'] = 0
                team_matches.at[idx, 'second_half_points'] = 0
        
        return team_matches
    
    def _calculate_quarter_averages(self, team_matches: pd.DataFrame) -> Dict[str, float]:
        """🛡️ CALCULA PROMEDIOS POR CUARTO DE FORMA ROBUSTA"""
        averages = {}
        for quarter in ['q1', 'q2', 'q3', 'q4']:
            col_name = f'{quarter}_points'
            averages[quarter] = self._safe_mean(team_matches, col_name, 25.0)
        return averages
    
    def _calculate_quarter_std(self, team_matches: pd.DataFrame) -> Dict[str, float]:
        """🛡️ CALCULA DESVIACIONES ESTÁNDAR POR CUARTO DE FORMA ROBUSTA"""
        std_devs = {}
        for quarter in ['q1', 'q2', 'q3', 'q4']:
            col_name = f'{quarter}_points'
            std_devs[quarter] = self._safe_std(team_matches, col_name, 5.0)
        return std_devs
    
    def _calculate_recovery_rate(self, team_matches: pd.DataFrame) -> float:
        """Calcula tasa de recuperación tras inicios lentos"""
        q1_avg = self._safe_mean(team_matches, 'q1_points', 25.0)
        second_half_avg = self._safe_mean(team_matches, 'second_half_points', 50.0)
        
        if q1_avg == 25.0 or second_half_avg == 50.0:  # Valores por defecto
            return 0.5
        
        try:
            # Convertir columnas a numéricas de forma segura
            q1_numeric = pd.to_numeric(team_matches['q1_points'], errors='coerce')
            second_half_numeric = pd.to_numeric(team_matches['second_half_points'], errors='coerce')
            
            # Filtrar valores válidos
            valid_mask = ~(q1_numeric.isna() | second_half_numeric.isna())
            if valid_mask.sum() == 0:
                return 0.5
            
            slow_starts = q1_numeric[valid_mask] < q1_avg * 0.8
            if slow_starts.sum() == 0:
                return 0.5
            
            recoveries = second_half_numeric[valid_mask & slow_starts] > second_half_avg
            return recoveries.sum() / slow_starts.sum()
            
        except Exception:
            return 0.5
    
    def _calculate_comeback_strength(self, team_matches: pd.DataFrame) -> float:
        """Calcula fuerza de remontada"""
        first_half_avg = self._safe_mean(team_matches, 'first_half_points', 50.0)
        second_half_avg = self._safe_mean(team_matches, 'second_half_points', 50.0)
        
        if first_half_avg <= 0:
            return 1.0
        
        return second_half_avg / first_half_avg
    
    def _calculate_consistency(self, team_matches: pd.DataFrame) -> float:
        """Calcula índice de consistencia entre cuartos"""
        quarter_cols = [f'q{i}_points' for i in range(1, 5)]
        
        # Verificar que las columnas existen
        existing_cols = [col for col in quarter_cols if col in team_matches.columns]
        if len(existing_cols) < 2:
            return 0.5
        
        try:
            variances = []
            for _, match in team_matches.iterrows():
                quarter_points = []
                for col in existing_cols:
                    try:
                        val = pd.to_numeric(match[col], errors='coerce')
                        if not pd.isna(val):
                            quarter_points.append(val)
                    except:
                        continue
                
                if len(quarter_points) > 1:
                    variances.append(np.var(quarter_points))
            
            if not variances:
                return 0.5
            
            avg_variance = np.mean(variances)
            return 1 / (1 + avg_variance)  # Invertir para que mayor consistencia = mayor score
            
        except Exception:
            return 0.5
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Patrones por defecto cuando no hay datos suficientes"""
        return {
            'quarter_averages': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25},
            'quarter_std': {'q1': 5, 'q2': 5, 'q3': 5, 'q4': 5},
            'first_half_avg': 50,
            'second_half_avg': 50,
            'second_half_surge': 0,
            'slow_start_recovery_rate': 0.5,
            'comeback_strength': 1.0,
            'avg_pace': 100,
            'pace_std': 10,
            'consistency_score': 0.5,
            'closing_strength': 25
        }
    
    def generate_pre_game_alerts(self, home_team: str, away_team: str) -> List[str]:
        """Genera alertas predictivas antes del partido"""
        alerts = []
        
        home_patterns = self.team_patterns.get(home_team, self._get_default_patterns())
        away_patterns = self.team_patterns.get(away_team, self._get_default_patterns())
        
        # 🚀 ALERTAS DE TENDENCIAS DE SEGUNDA MITAD
        if home_patterns['second_half_surge'] > 5:
            alerts.append(ALERT_TYPES['SECOND_HALF_SURGE'].format(
                team=home_team, surge=home_patterns['second_half_surge']
            ))
        
        if away_patterns['second_half_surge'] > 5:
            alerts.append(ALERT_TYPES['SECOND_HALF_SURGE'].format(
                team=away_team, surge=away_patterns['second_half_surge']
            ))
        
        # 🎯 ALERTAS DE FUERZA EN CLOSING
        if home_patterns['closing_strength'] > 28:
            alerts.append(ALERT_TYPES['CLOSING_STRENGTH'].format(
                team=home_team, q4_avg=home_patterns['closing_strength']
            ))
        
        if away_patterns['closing_strength'] > 28:
            alerts.append(ALERT_TYPES['CLOSING_STRENGTH'].format(
                team=away_team, q4_avg=away_patterns['closing_strength']
            ))
        
        # 📈 ALERTAS DE RECUPERACIÓN
        if home_patterns['slow_start_recovery_rate'] > ALERT_THRESHOLDS['RECOVERY_THRESHOLD']:
            alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                team=home_team, recovery_rate=home_patterns['slow_start_recovery_rate']
            ))
        
        if away_patterns['slow_start_recovery_rate'] > ALERT_THRESHOLDS['RECOVERY_THRESHOLD']:
            alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                team=away_team, recovery_rate=away_patterns['slow_start_recovery_rate']
            ))
        
        return alerts
    
    def analyze_live_performance(self, home_team: str, away_team: str, q_scores: Dict[str, int], 
                               quarter_stage: str, balance_features: Dict[str, float] = None) -> List[str]:
        """Analiza rendimiento en vivo y genera alertas, incluyendo las de balance"""
        alerts = []
        
        home_patterns = self.team_patterns.get(home_team, self._get_default_patterns())
        away_patterns = self.team_patterns.get(away_team, self._get_default_patterns())
        
        # Determinar cuartos completados
        quarters_completed = ['q1', 'q2'] if quarter_stage == 'halftime' else ['q1', 'q2', 'q3']
        
        # 🔍 ANÁLISIS POR CUARTO COMPLETADO
        for quarter in quarters_completed:
            home_key = f'{quarter}_home'
            away_key = f'{quarter}_away'
            
            if home_key in q_scores and away_key in q_scores:
                # Analizar equipo local
                home_alerts = self._analyze_quarter_performance(
                    home_team, quarter, q_scores[home_key], home_patterns
                )
                alerts.extend(home_alerts)
                
                # Analizar equipo visitante
                away_alerts = self._analyze_quarter_performance(
                    away_team, quarter, q_scores[away_key], away_patterns
                )
                alerts.extend(away_alerts)
        
        # 📊 ANÁLISIS DE PACE ACTUAL
        current_pace = self._calculate_current_pace(q_scores, quarter_stage)
        avg_pace = (home_patterns['avg_pace'] + away_patterns['avg_pace']) / 2
        
        if abs(current_pace - avg_pace) > ALERT_THRESHOLDS['PACE_DIFF_THRESHOLD']:
            direction = "acelerado" if current_pace > avg_pace else "desacelerado"
            alerts.append(ALERT_TYPES['PACE_SHIFT'].format(
                direction=direction, current_pace=current_pace, avg_pace=avg_pace
            ))
        
        # 🔥 ANÁLISIS DE RACHAS
        home_streak_alerts = self._detect_performance_streaks(home_team, q_scores, home_patterns, quarters_completed, 'home')
        away_streak_alerts = self._detect_performance_streaks(away_team, q_scores, away_patterns, quarters_completed, 'away')
        
        alerts.extend(home_streak_alerts)
        alerts.extend(away_streak_alerts)
        
        # ⚖️ ANÁLISIS DE BALANCE
        if balance_features:
            if balance_features.get('is_game_unbalanced', 0) == 1:
                home_pts = sum(q_scores.get(f'q{i}_home', 0) for i in range(1, 4))
                away_pts = sum(q_scores.get(f'q{i}_away', 0) for i in range(1, 4))
                lead = abs(home_pts - away_pts)
                alerts.append(ALERT_TYPES['GAME_UNBALANCED'].format(lead=lead))
            
            if balance_features.get('intensity_drop_factor', 1.0) < 0.8:
                drop_pct = 1 - balance_features['intensity_drop_factor']
                alerts.append(ALERT_TYPES['INTENSITY_DROP'].format(drop=drop_pct))

            if balance_features.get('blowout_momentum', 0.0) > 0.5:
                 alerts.append(ALERT_TYPES['BLOWOUT_MOMENTUM'])
        
        return alerts
    
    def _analyze_quarter_performance(self, team: str, quarter: str, current_points: int, patterns: Dict[str, Any]) -> List[str]:
        """Analiza el rendimiento de un equipo en un cuarto específico"""
        alerts = []
        
        expected_avg = patterns['quarter_averages'][quarter]
        expected_std = patterns['quarter_std'][quarter]
        
        diff = current_points - expected_avg
        z_score = diff / max(expected_std, 1)  # Evitar división por cero
        
        # 🚨 DETECCIÓN DE ANOMALÍAS
        if abs(z_score) > ALERT_THRESHOLDS['ANOMALY_THRESHOLD']:
            if diff > 0:
                alerts.append(ALERT_TYPES['OVER_PERFORMANCE'].format(
                    team=team, diff=diff, quarter=quarter.upper(),
                    current=current_points, avg=expected_avg
                ))
            else:
                alerts.append(ALERT_TYPES['UNDER_PERFORMANCE'].format(
                    team=team, diff=abs(diff), quarter=quarter.upper(),
                    current=current_points, avg=expected_avg
                ))
                
                # Si está muy por debajo y tiene buena tasa de recuperación, avisar
                if patterns['slow_start_recovery_rate'] > ALERT_THRESHOLDS['RECOVERY_THRESHOLD']:
                    alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                        team=team, recovery_rate=patterns['slow_start_recovery_rate']
                    ))
        
        return alerts
    
    def _detect_performance_streaks(self, team: str, q_scores: Dict[str, int], patterns: Dict[str, Any], 
                                  quarters_completed: List[str], team_suffix: str) -> List[str]:
        """Detecta rachas de rendimiento (hot/cold streaks)"""
        alerts = []
        
        if len(quarters_completed) < ALERT_THRESHOLDS['STREAK_MIN']:
            return alerts
        
        over_performances = 0
        under_performances = 0
        
        for quarter in quarters_completed:
            key = f'{quarter}_{team_suffix}'
            if key in q_scores:
                current_points = q_scores[key]
                expected_avg = patterns['quarter_averages'][quarter]
                expected_std = patterns['quarter_std'][quarter]
                
                diff = current_points - expected_avg
                z_score = diff / max(expected_std, 1)
                
                if z_score > ALERT_THRESHOLDS['ANOMALY_THRESHOLD']:
                    over_performances += 1
                elif z_score < -ALERT_THRESHOLDS['ANOMALY_THRESHOLD']:
                    under_performances += 1
        
        # 🔥 HOT STREAK
        if over_performances >= ALERT_THRESHOLDS['STREAK_MIN']:
            alerts.append(ALERT_TYPES['HOT_STREAK'].format(
                team=team, consecutive=over_performances
            ))
        
        # 🧊 COLD STREAK
        if under_performances >= ALERT_THRESHOLDS['STREAK_MIN']:
            alerts.append(ALERT_TYPES['COLD_STREAK'].format(
                team=team, consecutive=under_performances
            ))
        
        return alerts
    
    def _calculate_current_pace(self, q_scores: Dict[str, int], quarter_stage: str) -> float:
        """Calcula el pace actual basado en los puntos anotados"""
        total_points = sum(q_scores.values())
        quarters_played = 2 if quarter_stage == 'halftime' else 3
        minutes_played = quarters_played * 12
        
        # Estimación de posesiones (aproximación)
        estimated_possessions = total_points / 1.08
        pace = (estimated_possessions / minutes_played) * 48
        
        return pace
    
    def get_team_summary(self, team_name: str) -> Dict[str, Any]:
        """Obtiene resumen de patrones de un equipo"""
        if team_name not in self.team_patterns:
            return self._get_default_patterns()
        
        patterns = self.team_patterns[team_name]
        
        return {
            'team': team_name,
            'avg_by_quarter': patterns['quarter_averages'],
            'second_half_tendency': patterns['second_half_surge'],
            'recovery_ability': patterns['slow_start_recovery_rate'],
            'closing_strength': patterns['closing_strength'],
            'consistency': patterns['consistency_score'],
            'typical_pace': patterns['avg_pace']
        }

# Función de conveniencia para crear el sistema de alertas
def create_alerts_system(historical_df: pd.DataFrame) -> BasketballAlerts:
    """Crea y configura el sistema de alertas"""
    return BasketballAlerts(historical_df)