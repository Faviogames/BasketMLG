# ===========================================
# core/odds_api.py
# Cliente para API de cuotas de the-odds-api.com
# Obtiene líneas de Over/Under automáticamente
# ===========================================

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import sys
import os

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ODDS_API_KEY

class OddsApiClient:
    """
    Cliente para the-odds-api.com
    Obtiene cuotas Over/Under para partidos de NBA
    """
    
    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.cache = {}
        self.cache_duration = 300  # 5 minutos
        self.last_request_time = 0
        
        # 🏆 BOOKMAKERS PRIORITARIOS (orden de prioridad)
        self.priority_bookmakers = [
            'betfair',      # Prioridad 1
            'pinnacle',     # Prioridad 2
            'unibet_eu',    # Prioridad 3
            'unibet',       # Variante sin _eu
            'bet365',       # Fallback
            'williamhill',  # Fallback
            'generic'       # Cualquier otro bookmaker
        ]
        
        # Mapeo de nombres de bookmakers para normalización
        self.bookmaker_mappings = {
            'bet365': 'bet365',
            'william hill': 'williamhill',
            'marathonbet': 'marathonbet',
            'betvictor': 'betvictor',
            '888sport': 'sport888',
            'unibet europe': 'unibet_eu',
            'unibet eu': 'unibet_eu'
        }
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si el cache es válido"""
        if cache_key not in self.cache:
            return False
        cached_time = self.cache[cache_key]['timestamp']
        return (datetime.now().timestamp() - cached_time) < self.cache_duration
    
    def get_live_odds(self) -> Optional[List[Dict]]:
        """
        Obtiene cuotas live para NBA con totals (Over/Under)
        """
        cache_key = 'live_nba_odds'
        
        # Verificar cache
        if self._is_cache_valid(cache_key):
            print("Usando datos cacheados de cuotas")
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/basketball_nba/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'eu',
                'markets': 'totals',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            print(f"Obteniendo cuotas de {url}")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: mostrar respuesta
                print(f"Respuesta API recibida: {len(data)} partidos")
                
                # Procesar y normalizar datos
                processed_games = self._process_odds_data(data)
                
                # Cache los resultados
                self.cache[cache_key] = {
                    'data': processed_games,
                    'timestamp': datetime.now().timestamp()
                }
                
                print(f"Obtenidas cuotas para {len(processed_games)} partidos")
                return processed_games
            else:
                print(f"Error obteniendo cuotas: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error en get_live_odds: {e}")
            return None
    
    def _process_odds_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Procesa y normaliza los datos de la API priorizando bookmakers específicos
        """
        processed_games = []
        
        for game in raw_data:
            try:
                # Información básica del partido
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                commence_time = game.get('commence_time', '')
                
                # 🏆 PRIORIZAR BOOKMAKERS ESPECÍFICOS
                bookmakers = self._prioritize_bookmakers(game.get('bookmakers', []))
                
                # Buscar cuotas de totals (Over/Under) priorizando bookmakers específicos
                totals_odds = self._extract_totals_with_priority(bookmakers)
                
                if totals_odds and totals_odds['line'] > 0:
                    processed_game = {
                        'home_team': self._normalize_team_name(home_team),
                        'away_team': self._normalize_team_name(away_team),
                        'commence_time': commence_time,
                        'totals_line': totals_odds['line'],
                        'over_odds': totals_odds.get('over_odds', 0),
                        'under_odds': totals_odds.get('under_odds', 0),
                        'bookmaker': totals_odds['bookmaker'],
                        'bookmaker_priority': totals_odds.get('priority', 99),
                        'raw_home_team': home_team,
                        'raw_away_team': away_team
                    }
                    processed_games.append(processed_game)
                    
            except Exception as e:
                print(f"Warning: Error procesando partido: {e}")
                continue
        
        return processed_games
    
    def _prioritize_bookmakers(self, bookmakers: List[Dict]) -> List[Dict]:
        """
        🏆 PRIORIZA BOOKMAKERS ESPECÍFICOS
        Ordena los bookmakers según la prioridad definida
        """
        prioritized = []
        
        # Primero agregar bookmakers prioritarios en orden
        for priority_bookmaker in self.priority_bookmakers:
            for bookmaker in bookmakers:
                bookmaker_title = bookmaker.get('title', '').lower().strip()
                normalized_name = self._normalize_bookmaker_name(bookmaker_title)
                
                if normalized_name == priority_bookmaker:
                    prioritized.append({
                        **bookmaker,
                        'priority': self.priority_bookmakers.index(priority_bookmaker),
                        'normalized_name': normalized_name
                    })
        
        # Agregar bookmakers no prioritarios al final
        used_bookmakers = {b.get('normalized_name') for b in prioritized if 'normalized_name' in b}
        
        for bookmaker in bookmakers:
            bookmaker_title = bookmaker.get('title', '').lower().strip()
            normalized_name = self._normalize_bookmaker_name(bookmaker_title)
            
            if normalized_name not in used_bookmakers and normalized_name != 'generic':
                prioritized.append({
                    **bookmaker,
                    'priority': len(self.priority_bookmakers),  # Prioridad baja
                    'normalized_name': normalized_name
                })
        
        return prioritized
    
    def _normalize_bookmaker_name(self, bookmaker_title: str) -> str:
        """
        Normaliza nombres de bookmakers para matching consistente
        """
        if not bookmaker_title:
            return 'generic'
        
        # Limpiar y normalizar
        normalized = bookmaker_title.lower().strip()
        
        # Aplicar mapeos conocidos
        for old_name, new_name in self.bookmaker_mappings.items():
            if old_name.lower() in normalized:
                return new_name
        
        # Mapeos específicos para los prioritarios
        if 'betfair' in normalized:
            return 'betfair'
        elif 'pinnacle' in normalized:
            return 'pinnacle'
        elif 'unibet' in normalized:
            return 'unibet_eu' if 'eu' in normalized else 'unibet'
        
        return 'generic'
    
    def _extract_totals_with_priority(self, bookmakers: List[Dict]) -> Optional[Dict]:
        """
        Extrae cuotas de totals priorizando bookmakers específicos
        """
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'totals':
                    # Extraer la línea (total points) y las cuotas
                    line = market.get('details', '')
                    total_line = 0
                    
                    if line:
                        # La línea suele estar en el campo 'details' como "Over/Under 215.5"
                        line_match = re.search(r'(\d+\.?\d*)', line)
                        if line_match:
                            total_line = float(line_match.group(1))
                    else:
                        # Si no hay 'details', intentar extraer de los outcomes
                        outcomes = market.get('outcomes', [])
                        if outcomes and len(outcomes) >= 2:
                            # Buscar el 'point' field en los outcomes
                            for outcome in outcomes:
                                if 'point' in outcome:
                                    total_line = float(outcome.get('point', 0))
                                    break
                    
                    # Recopilar cuotas
                    over_odds = None
                    under_odds = None
                    
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == 'Over':
                            over_odds = float(outcome.get('price', 0))
                        elif outcome.get('name') == 'Under':
                            under_odds = float(outcome.get('price', 0))
                    
                    # Solo devolver si tenemos ambas cuotas
                    if over_odds is not None and under_odds is not None:
                        return {
                            'line': total_line,
                            'bookmaker': bookmaker.get('title', 'Unknown'),
                            'over_odds': over_odds,
                            'under_odds': under_odds,
                            'priority': bookmaker.get('priority', 99)
                        }
        
        return None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normaliza nombres de equipos para mejor matching
        """
        if not team_name:
            return ""
        
        # Mapeo de nombres comunes
        name_mappings = {
            'Los Angeles Clippers': 'LA Clippers',
            'Los Angeles Lakers': 'LA Lakers',
            'New York Knicks': 'NY Knicks',
            'Golden State Warriors': 'GS Warriors',
            'San Antonio Spurs': 'SA Spurs',
            'New Orleans Pelicans': 'NO Pelicans',
            'Oklahoma City Thunder': 'OKC Thunder',
            'Portland Trail Blazers': 'Portland Trailblazers',
        }
        
        return name_mappings.get(team_name, team_name)
    
    def find_matching_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Busca cuotas que coincidan con los equipos especificados
        """
        print(f"Buscando cuotas para: {home_team} vs {away_team}")
        
        # Obtener cuotas live
        live_odds = self.get_live_odds()
        if not live_odds:
            return None
        
        # Normalizar nombres para comparación
        home_normalized = self._normalize_team_name(home_team).lower()
        away_normalized = self._normalize_team_name(away_team).lower()
        
        # Buscar coincidencia exacta primero
        for game in live_odds:
            game_home = game['home_team'].lower()
            game_away = game['away_team'].lower()
            
            # Verificar coincidencia exacta (en cualquier orden)
            if ((home_normalized == game_home and away_normalized == game_away) or
                (home_normalized == game_away and away_normalized == game_home)):
                print(f"Coincidencia exacta encontrada: {game['home_team']} vs {game['away_team']}")
                return game
        
        # Si no hay coincidencia exacta, buscar coincidencia parcial
        for game in live_odds:
            game_home = game['home_team'].lower()
            game_away = game['away_team'].lower()
            
            # Verificar si los nombres contienen las palabras clave
            home_match = any(word in game_home for word in home_normalized.split() if len(word) > 3)
            away_match = any(word in game_away for word in away_normalized.split() if len(word) > 3)
            
            # Verificar también en el orden inverso
            home_match_reverse = any(word in game_away for word in home_normalized.split() if len(word) > 3)
            away_match_reverse = any(word in game_home for word in away_normalized.split() if len(word) > 3)
            
            if (home_match and away_match) or (home_match_reverse and away_match_reverse):
                print(f"Coincidencia parcial encontrada: {game['home_team']} vs {game['away_team']}")
                print(f"   Original: {home_team} vs {away_team}")
                return game
        
        print(f"No se encontraron cuotas para {home_team} vs {away_team}")
        return None
    
    def get_odds_for_teams(self, home_team: str, away_team: str) -> Optional[float]:
        """
        Función principal: obtiene la línea de totales para los equipos especificados
        Retorna la línea de Over/Under o None si no se encuentra
        """
        matching_game = self.find_matching_odds(home_team, away_team)
        
        if matching_game:
            line = matching_game['totals_line']
            print(f"Linea encontrada: {line}")
            print(f"   Over {line}: {matching_game['over_odds']:.2f}")
            if matching_game.get('under_odds'):
                print(f"   Under {line}: {matching_game['under_odds']:.2f}")
            print(f"   Casa: {matching_game['bookmaker']}")
            return line
        else:
            return None
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con la API
        """
        try:
            print("Probando conexión con API de cuotas...")
            
            # Intentar obtener datos básicos
            url = f"{self.base_url}/basketball_nba/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'eu',
                'markets': 'totals',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Conexión exitosa. Partidos disponibles: {len(data)}")
                return True
            else:
                print(f"Error en conexión: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error probando conexión: {e}")
            return False
    
    def clear_cache(self):
        """Limpia el cache manualmente"""
        self.cache.clear()
        print("Cache de cuotas limpiado")


# Función de utilidad para uso directo
def get_totals_line(home_team: str, away_team: str, api_key: str = None) -> Optional[float]:
    """
    Función de utilidad para obtener línea de totales
    """
    client = OddsApiClient(api_key) if api_key else OddsApiClient()
    return client.get_odds_for_teams(home_team, away_team)


if __name__ == "__main__":
    # Test básico
    client = OddsApiClient()
    
    if client.test_connection():
        print("\nTest: Buscando cuotas para Lakers vs Warriors")
        line = client.get_odds_for_teams("Los Angeles Lakers", "Golden State Warriors")
        if line:
            print(f"Linea obtenida: {line}")
        else:
            print("No se pudo obtener la línea")
    else:
        print("Falló el test de conexión")