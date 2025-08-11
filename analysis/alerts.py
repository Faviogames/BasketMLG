# ===========================================
# Archivo: alerts.py (v1.0)
# Sistema de alertas inteligentes basadas en patrones históricos
# Detecta anomalías y patrones de anotación por cuartos
# ===========================================
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Tipos de alertas disponibles
ALERT_TYPES = {
    'UNDER_PERFORMANCE': "⚠️ {team} anotando {diff:.1f} puntos menos que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'OVER_PERFORMANCE': "🔥 {team} anotando {diff:.1f} puntos más que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'SURGE_EXPECTED': "🚀 {team} históricamente tiene repunte en {period} (promedio: +{surge:.1f} pts)",
    'COLD_STREAK': "🧊 {team} en racha fría: {consecutive} cuartos consecutivos por debajo del promedio",
    'HOT_STREAK': "🔥 {team} en racha caliente: {consecutive} cuartos consecutivos por encima del promedio",
    'SLOW_START_RECOVERY': "📈 {team} suele recuperarse tras inicios lentos (probabilidad {recovery_rate:.0%})",
    'DEFENSIVE_COLLAPSE': "🛡️ {team} permitiendo {diff:.1f} pts más de lo usual - posible colapso defensivo",
    'PACE_SHIFT': "⚡ Ritmo de juego {direction} ({current_pace:.1f} vs promedio {avg_pace:.1f} posesiones/48min)",
    'SECOND_HALF_SURGE': "💪 {team} promedia {surge:.1f} pts más en segunda mitad - considerar ajustes",
    'CLOSING_STRENGTH': "🎯 {team} muy fuerte en cuartos finales (promedio Q4: {q4_avg:.1f} pts)"
}

# Umbrales para detectar anomalías
THRESHOLDS = {
    'SIGNIFICANT_DIFF': 4.0,  # Diferencia significativa en puntos
    'ANOMALY_THRESHOLD': 1.5,  # Desviaciones estándar para anomalías
    'STREAK_MIN': 2,  # Mínimo de cuartos para considerar racha
    'PACE_DIFF_THRESHOLD': 5.0,  # Diferencia significativa en pace
    'RECOVERY_THRESHOLD': 0.6  # Umbral para probabilidad de recuperación
}

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
        
        # 📊 PROMEDIOS POR CUARTO
        patterns['quarter_averages'] = self._calculate_quarter_averages(team_matches)
        patterns['quarter_std'] = self._calculate_quarter_std(team_matches)
        
        # 🎯 PATRONES DE MITADES
        patterns['first_half_avg'] = team_matches['first_half_points'].mean() if 'first_half_points' in team_matches.columns else 0
        patterns['second_half_avg'] = team_matches['second_half_points'].mean() if 'second_half_points' in team_matches.columns else 0
        patterns['second_half_surge'] = patterns['second_half_avg'] - patterns['first_half_avg']
        
        # 🚀 TENDENCIAS DE RECUPERACIÓN
        patterns['slow_start_recovery_rate'] = self._calculate_recovery_rate(team_matches)
        patterns['comeback_strength'] = self._calculate_comeback_strength(team_matches)
        
        # ⚡ PATRONES DE PACE
        patterns['avg_pace'] = team_matches['live_pace_estimate'].mean() if 'live_pace_estimate' in team_matches.columns else 100
        patterns['pace_std'] = team_matches['live_pace_estimate'].std() if 'live_pace_estimate' in team_matches.columns else 10
        
        # 🎪 CONSISTENCIA POR CUARTOS
        patterns['consistency_score'] = self._calculate_consistency(team_matches)
        patterns['closing_strength'] = patterns['quarter_averages'].get('q4', 25)  # Fuerza en Q4
        
        return patterns
    
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
        
        # Añadir datos específicos del equipo
        for idx, match in team_matches.iterrows():
            is_home = match['home_team'] == team_name
            
            # Extraer puntos por cuarto del raw_match
            if 'raw_match' in match and match['raw_match']:
                quarter_scores = match['raw_match'].get('quarter_scores', {})
                
                for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if quarter in quarter_scores:
                        team_key = 'home_score' if is_home else 'away_score'
                        team_matches.at[idx, f'{quarter.lower()}_points'] = quarter_scores[quarter].get(team_key, 0)
                
                # Calcular mitades
                q1_pts = team_matches.at[idx, 'q1_points'] if 'q1_points' in team_matches.columns else 0
                q2_pts = team_matches.at[idx, 'q2_points'] if 'q2_points' in team_matches.columns else 0
                q3_pts = team_matches.at[idx, 'q3_points'] if 'q3_points' in team_matches.columns else 0
                q4_pts = team_matches.at[idx, 'q4_points'] if 'q4_points' in team_matches.columns else 0
                
                team_matches.at[idx, 'first_half_points'] = q1_pts + q2_pts
                team_matches.at[idx, 'second_half_points'] = q3_pts + q4_pts
        
        return team_matches
    
    def _calculate_quarter_averages(self, team_matches: pd.DataFrame) -> Dict[str, float]:
        """Calcula promedios por cuarto"""
        averages = {}
        for quarter in ['q1', 'q2', 'q3', 'q4']:
            col_name = f'{quarter}_points'
            if col_name in team_matches.columns:
                averages[quarter] = team_matches[col_name].mean()
            else:
                averages[quarter] = 25.0  # Valor por defecto
        return averages
    
    def _calculate_quarter_std(self, team_matches: pd.DataFrame) -> Dict[str, float]:
        """Calcula desviaciones estándar por cuarto"""
        std_devs = {}
        for quarter in ['q1', 'q2', 'q3', 'q4']:
            col_name = f'{quarter}_points'
            if col_name in team_matches.columns:
                std_devs[quarter] = team_matches[col_name].std()
            else:
                std_devs[quarter] = 5.0  # Valor por defecto
        return std_devs
    
    def _calculate_recovery_rate(self, team_matches: pd.DataFrame) -> float:
        """Calcula tasa de recuperación tras inicios lentos"""
        if 'q1_points' not in team_matches.columns or 'second_half_points' not in team_matches.columns:
            return 0.5
        
        q1_avg = team_matches['q1_points'].mean()
        slow_starts = team_matches[team_matches['q1_points'] < q1_avg * 0.8]  # 20% por debajo del promedio
        
        if len(slow_starts) == 0:
            return 0.5
        
        recoveries = slow_starts[slow_starts['second_half_points'] > slow_starts['second_half_points'].mean()]
        return len(recoveries) / len(slow_starts)
    
    def _calculate_comeback_strength(self, team_matches: pd.DataFrame) -> float:
        """Calcula fuerza de remontada"""
        if 'first_half_points' not in team_matches.columns or 'second_half_points' not in team_matches.columns:
            return 1.0
        
        return team_matches['second_half_points'].mean() / max(team_matches['first_half_points'].mean(), 1)
    
    def _calculate_consistency(self, team_matches: pd.DataFrame) -> float:
        """Calcula índice de consistencia entre cuartos"""
        quarter_cols = [f'q{i}_points' for i in range(1, 5) if f'q{i}_points' in team_matches.columns]
        
        if len(quarter_cols) < 2:
            return 0.5
        
        variances = []
        for _, match in team_matches.iterrows():
            quarter_points = [match[col] for col in quarter_cols if not pd.isna(match[col])]
            if len(quarter_points) > 1:
                variances.append(np.var(quarter_points))
        
        if not variances:
            return 0.5
        
        avg_variance = np.mean(variances)
        return 1 / (1 + avg_variance)  # Invertir para que mayor consistencia = mayor score
    
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
        if home_patterns['slow_start_recovery_rate'] > THRESHOLDS['RECOVERY_THRESHOLD']:
            alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                team=home_team, recovery_rate=home_patterns['slow_start_recovery_rate']
            ))
        
        if away_patterns['slow_start_recovery_rate'] > THRESHOLDS['RECOVERY_THRESHOLD']:
            alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                team=away_team, recovery_rate=away_patterns['slow_start_recovery_rate']
            ))
        
        return alerts
    
    def analyze_live_performance(self, home_team: str, away_team: str, q_scores: Dict[str, int], quarter_stage: str) -> List[str]:
        """Analiza rendimiento en vivo y genera alertas"""
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
        
        if abs(current_pace - avg_pace) > THRESHOLDS['PACE_DIFF_THRESHOLD']:
            direction = "acelerado" if current_pace > avg_pace else "desacelerado"
            alerts.append(ALERT_TYPES['PACE_SHIFT'].format(
                direction=direction, current_pace=current_pace, avg_pace=avg_pace
            ))
        
        # 🔥 ANÁLISIS DE RACHAS
        home_streak_alerts = self._detect_performance_streaks(home_team, q_scores, home_patterns, quarters_completed, 'home')
        away_streak_alerts = self._detect_performance_streaks(away_team, q_scores, away_patterns, quarters_completed, 'away')
        
        alerts.extend(home_streak_alerts)
        alerts.extend(away_streak_alerts)
        
        return alerts
    
    def _analyze_quarter_performance(self, team: str, quarter: str, current_points: int, patterns: Dict[str, Any]) -> List[str]:
        """Analiza el rendimiento de un equipo en un cuarto específico"""
        alerts = []
        
        expected_avg = patterns['quarter_averages'][quarter]
        expected_std = patterns['quarter_std'][quarter]
        
        diff = current_points - expected_avg
        z_score = diff / max(expected_std, 1)  # Evitar división por cero
        
        # 🚨 DETECCIÓN DE ANOMALÍAS
        if abs(z_score) > THRESHOLDS['ANOMALY_THRESHOLD']:
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
                if patterns['slow_start_recovery_rate'] > THRESHOLDS['RECOVERY_THRESHOLD']:
                    alerts.append(ALERT_TYPES['SLOW_START_RECOVERY'].format(
                        team=team, recovery_rate=patterns['slow_start_recovery_rate']
                    ))
        
        return alerts
    
    def _detect_performance_streaks(self, team: str, q_scores: Dict[str, int], patterns: Dict[str, Any], 
                                  quarters_completed: List[str], team_suffix: str) -> List[str]:
        """Detecta rachas de rendimiento (hot/cold streaks)"""
        alerts = []
        
        if len(quarters_completed) < THRESHOLDS['STREAK_MIN']:
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
                
                if z_score > THRESHOLDS['ANOMALY_THRESHOLD']:
                    over_performances += 1
                elif z_score < -THRESHOLDS['ANOMALY_THRESHOLD']:
                    under_performances += 1
        
        # 🔥 HOT STREAK
        if over_performances >= THRESHOLDS['STREAK_MIN']:
            alerts.append(ALERT_TYPES['HOT_STREAK'].format(
                team=team, consecutive=over_performances
            ))
        
        # 🧊 COLD STREAK
        if under_performances >= THRESHOLDS['STREAK_MIN']:
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