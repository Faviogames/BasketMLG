# ===========================================
# core/live_api.py (v4.1 RESTAURADO + FASE 1)
# ✅ RESTAURADO: Implementación completa de scraping de v3.0
# ✅ INTEGRADO: Con arquitectura v4.0 sin betting protection
# ✅ FUNCIONAL: Detección de partidos + mapeo + alertas
# 🆕 FASE 1: Señales live avanzadas implementadas
# ===========================================

"""
🆕 FASE 1: SEÑALES LIVE AVANZADAS - DOCUMENTACIÓN

🎯 OBJETIVO: Mejorar precisión de predicciones live con señales contextuales dinámicas

📊 SEÑALES IMPLEMENTADAS:

1. 🏀 FOUL TROUBLE INDEX (FTI)
   - Nativo: Usa datos de 'personal_fouls' de SofaScore
   - Proxy: FT_rate por minuto cuando faltan datos nativos
   - Fórmula: FTI = (pf_rate - 0.5) / 1.5 (normalizado 0-1)
   - Umbrales: >0.7 = Foul trouble alto, >0.8 = Extremo

2. 🎯 TRUE SHOOTING % (TS%) + eFG%
   - TS% = Puntos / (2 * (FGA + 0.44 * FTA))
   - eFG% = (FGM + 0.5 * 3PM) / FGA
   - Shrinkage: alpha = min(1.0, FGA/40) para estabilidad
   - Umbrales: Diferencia >0.15 = Shooting efficiency significativa

3. 🏃 RUN DETECTOR
   - Detecta rachas de puntuación en ventana deslizante
   - Umbral: 6-8 pts en 3-5 min (ajustable por tiempo)
   - Fuerza: 0.0-1.0 (normalizada por tiempo transcurrido)
   - Ventana: Último cuarto jugado vs anterior

🔧 INTEGRACIÓN EN SISTEMA:

- compute_live_signals(): Calcula todas las señales en LiveProcessor
- apply_context_adjustment(): Usa señales para escalado dinámico
- _generate_live_alerts(): Genera alertas basadas en señales
- Telemetría: Muestra señales aplicadas en logs

⚙️ AJUSTES DINÁMICOS:

- Garbage Time: Escala reducción por foul trouble (+10% max)
- Partidos Cerrados: Escala por shooting efficiency (+20% max)
- Intensidad: Escala por run detector (+15% max)
- Fouling Strategy: Escala por FTI directo (+25% max)
- Run Activo: Ajuste directo por fuerza de run (+20% max)
- Foul Trouble Extremo: Ajuste por FTI >0.7 (+30% max)

🚨 ALERTAS NUEVAS:

- RUN DETECTED: Fuerza >0.3 (moderada) o >0.5 (fuerte)
- FOUL TROUBLE: FTI >0.7 (alto) o >0.8 (extremo)
- FT HEAVY GAME: FTA >20 (moderado) o >30 (alto)
- SHOOTING EFFICIENCY: Diferencia TS% >0.2

📈 MEJORAS ESPERADAS:

- +15-25% precisión en juegos con foul trouble
- +10-20% precisión en detección de runs
- +5-15% precisión en juegos con shooting efficiency extrema
- Alertas proactivas para cambios de momentum
"""

import asyncio
import json
import time
import random
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None  # La extension no necesita Playwright instalado.
from analysis.metamodel_integration import aplicar_metamodelo_ai
from core.live_features import calculate_detailed_quarter_metrics
from core.sofascore_session import browser_context_options, load_browser_session, save_browser_session
from core.browser_bridge import BrowserBridge

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with fallback for zero denominator"""
    try:
        if denominator == 0 or denominator is None:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

class SofaScoreClient:
    """
    🎯 Cliente LIVE SofaScore - Versión restaurada funcional
    """
    
    def __init__(self, headless: bool = True):
        self.browser = None
        self.context = None
        self.page = None
        self.headless = headless
        self.playwright_instance = None
        self.debug_mode = True # Modo debug activado por defecto
        self.last_discovery_error = None
        
        # URLs
        self.base_url = "https://www.sofascore.com"
        self.api_base = "https://www.sofascore.com/api/v1"
        
        # Cache y rate limiting
        self.cache = {}
        self.cache_duration = 30  # segundos - más tiempo por estabilidad
        self.last_request_time = 0
        self.min_request_interval = 2.0  # más conservador para evitar detección
        
        # 🎯 Stats que nuestro modelo necesita (ACTUALIZADO según API real)
        self.required_stats = {
            # Mapeo basado en la estructura real de la API de SofaScore
            'Field goals': 'field_goals_made',  # API: "Field goals" -> homeValue/awayValue = made
            '2 pointers': '2_point_field_goals_made',  # API: "2 pointers" -> homeValue/awayValue = made
            '3 pointers': '3_point_field_goals_made',  # API: "3 pointers" -> homeValue/awayValue = made
            'Free throws': 'free_throws_made',  # API: "Free throws" -> homeValue/awayValue = made
            'Rebounds': 'total_rebounds',  # API: "Rebounds" -> homeValue/awayValue = total
            'Offensive rebounds': 'offensive_rebounds',  # API: "Offensive rebounds"
            'Defensive rebounds': 'defensive_rebounds',  # API: "Defensive rebounds"
            'Assists': 'assists',  # API: "Assists"
            'Turnovers': 'turnovers',  # API: "Turnovers"
            'Steals': 'steals',  # API: "Steals"
            'Blocks': 'blocks',  # API: "Blocks"
            'Fouls': 'personal_fouls',  # API: "Fouls" -> totalFouls
            # Alias para compatibilidad
            'Field goals made': 'field_goals_made',
            '3-point field goals made': '3_point_field_goals_made',
            'Free throws made': 'free_throws_made',
            'Total rebounds': 'total_rebounds',
            'Personal fouls': 'personal_fouls',
            # 🆕 CARACTERÍSTICAS COMBINADAS QUE EL MODELO ESPERA
            'field_goals_attempted': 'field_goals_attempted',
            'free_throws_attempted': 'free_throws_attempted',
            'total_fouls': 'personal_fouls'  # Para compatibilidad con el modelo
        }
    
    async def initialize(self):
        """Inicializa el navegador oculto y reutiliza estado validado si existe."""
        try:
            print('[LIVE] Iniciando cliente SofaScore en segundo plano...')
            
            self.playwright_instance = await async_playwright().start()
            
            try:
                self.browser = await self.playwright_instance.chromium.launch(
                    channel='chrome',
                    headless=self.headless,
                )
            except Exception as e:
                print(f'[WARN] Chrome no disponible ({type(e).__name__}). Probando Chromium...')
                self.browser = await self.playwright_instance.chromium.launch(
                    headless=self.headless,
                )

            options = browser_context_options(self.browser.version)
            state = load_browser_session()
            if state is not None:
                options['storage_state'] = state
                print('[SESSION] Reutilizando cookies y almacenamiento guardados.')
            else:
                print('[SESSION] Sin sesion validada guardada.')
            self.context = await self.browser.new_context(**options)
            self.page = await self.context.new_page()
            
            print('[LIVE] Cliente SofaScore preparado.')
            return True
            
        except Exception as e:
            detail = str(e).encode('ascii', errors='replace').decode('ascii')
            print(f'[ERROR] No se pudo iniciar el cliente: {detail}')
            return False
    
    async def cleanup(self):
        """Limpia recursos del navegador"""
        try:
            if self.page:
                try:
                    await self.page.close()
                except Exception:
                    pass
                self.page = None
            if self.context:
                try:
                    await self.context.close()
                except Exception:
                    pass
                self.context = None
            if self.browser:
                try:
                    await self.browser.close()
                except Exception:
                    pass
                self.browser = None
            if self.playwright_instance:
                try:
                    await self.playwright_instance.stop()
                except Exception:
                    pass
                self.playwright_instance = None
            print("Playwright cleaned up successfully")
        except Exception as e:
            print(f"Warning: Error in cleanup: {e}")
            import traceback
            traceback.print_exc()
    
    async def _rate_limit(self):
        """Rate limiting no-bloqueante."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)  # ✅ NO BLOQUEA EVENT LOOP

        self.last_request_time = time.time()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si cache es válido"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        return (time.time() - cached_time) < self.cache_duration
    
    async def _establish_basketball_session(self) -> bool:
        """Espera la verificacion visible antes de evaluar el estado HTTP inicial."""
        if not self.page and not await self.initialize():
            self.last_discovery_error = 'No se pudo iniciar el navegador de SofaScore.'
            return False

        # Una pestana preparada manualmente ya puede tener acceso al JSON.
        # No abandonarla para repetir innecesariamente la pagina de entrada.
        if self.page.url.startswith(self.api_base + '/'):
            try:
                ready = await self.page.evaluate("""() => {
                    try {
                        const data = JSON.parse(document.body.innerText);
                        return !data.error && (Array.isArray(data.events) ||
                            Array.isArray(data.statistics) || !!data.event);
                    } catch { return false; }
                }""")
                if ready:
                    print('[SESSION] Reutilizando la pestana de SofaScore con datos validos.')
                    return True
            except Exception:
                pass

        response = await self.page.goto(
            f"{self.base_url}/basketball",
            wait_until="domcontentloaded", timeout=30000
        )
        if self.debug_mode:
            final_url = self.page.url.split('?', 1)[0]
            status = response.status if response else 'sin respuesta'
            print(f'[FLOW] /basketball -> HTTP {status}; destino: {final_url}')
        challenged = 'captcha' in self.page.url.lower()
        if challenged:
            if self.headless:
                print('[SESSION] Esperando la comprobacion automatica de SofaScore en segundo plano...')
            else:
                print('[SESSION] SofaScore solicita verificacion en la ventana de Chrome.')
                print('[SESSION] Esperando hasta 60 segundos a que completes la verificacion.')
            wait_seconds = 20 if self.headless else 60
            for _ in range(wait_seconds):
                await asyncio.sleep(1)
                if 'captcha' not in self.page.url.lower():
                    break
            else:
                self.last_discovery_error = (
                    'SofaScore mantiene la verificacion pendiente; la sesion no tiene acceso a los datos.'
                )
                print(f'[ERROR] {self.last_discovery_error}')
                return False
        elif response and response.status != 200:
            self.last_discovery_error = f'La pagina de baloncesto devuelve HTTP {response.status}.'
            print(f'[ERROR] {self.last_discovery_error}')
            return False

        # DOMContentLoaded no implica que la aplicacion haya cargado los partidos.
        # Esperar una senal real de la pagina en vez de depender de una pausa fija.
        try:
            await self.page.wait_for_function(
                "() => !!document.querySelector('a[href*=\"/basketball/match/\"]')",
                timeout=20000
            )
        except Exception:
            self.last_discovery_error = (
                'La pagina de baloncesto no termino de mostrar los enlaces de partidos. '
                'Revisa la ventana del navegador antes de consultar la API.'
            )
            print(f'[ERROR] {self.last_discovery_error}')
            return False
        return True

    async def health_check(self) -> bool:
        """
        ✅ Health check que prueba el endpoint JSON real (como v3.0)
        """
        try:
            if not self.page:
                await self.initialize()
            
            print("SEARCH: Health check...")
            
            await self._rate_limit()
            
            if not await self._establish_basketball_session():
                return False
            
            # 3. Probar endpoint JSON con fetch interno (respeta cookies de sesión)
            json_content = await self.page.evaluate("""
                async () => {
                    try {
                        const r = await fetch('/api/v1/sport/basketball/events/live');
                        if (!r.ok) return { success: false, status: r.status };
                        const data = await r.json();
                        if (!Array.isArray(data.events)) return { success: false, error: 'Respuesta sin lista events' };
                        return {
                            success: true,
                            events_count: data.events?.length || 0
                        };
                    } catch (e) {
                        return { success: false, error: e.message };
                    }
                }
            """)
            
            print(f"   [DEBUG] Health check in-page fetch result: {json_content}")
            if json_content and json_content.get('success'):
                print(f"SUCCESS: Health check passed - {json_content.get('events_count', 0)} events")
                return True
            
            # Si fetch no tuvo éxito, fallback a navegación directa
            response = await self.page.goto(
                f"{self.api_base}/sport/basketball/events/live",
                wait_until="domcontentloaded"
            )
            if response.status == 200:
                json_content = await self.page.evaluate("""
                    () => {
                        try {
                            const bodyText = document.body.textContent || document.body.innerText;
                            const data = JSON.parse(bodyText);
                            return {
                                success: true,
                                events_count: data.events?.length || 0
                            };
                        } catch (e) {
                            return { success: false, error: e.message };
                        }
                    }
                """)
                
                if json_content.get('success'):
                    print(f"SUCCESS: Health check passed - {json_content.get('events_count', 0)} events")
                    return True
                else:
                    print(f"WARNING: Invalid JSON: {json_content.get('error')}")
                    return False
            else:
                print(f"ERROR: Health check failed: Status {response.status}")
                return False

        except Exception as e:
            print(f"ERROR: Health check error: {e}")
            return False
    
    async def get_live_basketball_games(self) -> Optional[List[Dict]]:
        """
        🎯 MÉTODO PRINCIPAL RESTAURADO: Playwright con Sesión (EXITOSO en test)
        """
        cache_key = 'live_games'
        self.last_discovery_error = None
        
        # Check cache primero
        if self._is_cache_valid(cache_key):
            print("Using cached data for live games")
            return self.cache[cache_key]['data']
        
        await self._rate_limit()
        
        try:
            if not self.page:
                await self.initialize()
            
            print("Getting live games - session method...")
            
            # 🎯 MÉTODO PRINCIPAL: Con Sesión (EXITOSO en test)
            debug_mode = getattr(self, 'debug_mode', True)
            games_data = await self._get_games_with_session(debug_mode)
            
            if games_data is not None:
                games = self._filter_supported_games(games_data)
                
                if games:
                    # Cache los resultados
                    self.cache[cache_key] = {
                        'data': games,
                        'timestamp': time.time()
                    }
                    
                    print(f"SUCCESS: {len(games)} live games obtained (main method)")
                    return games
                return []

            if self.last_discovery_error:
                # Repetir la misma URL no resuelve una sesion bloqueada.
                return None
            
            # FALLBACK: Método básico
            print("🔄 Intentando método fallback...")
            fallback_data = await self._get_games_basic_method()
            
            if fallback_data is not None:
                games = self._filter_supported_games(fallback_data)
                
                if games:
                    # Cache los resultados
                    self.cache[cache_key] = {
                        'data': games,
                        'timestamp': time.time()
                    }
                    
                    print(f"SUCCESS: {len(games)} live games obtained (fallback method)")
                    return games
                return []
            
            print("ERROR: Could not obtain live games")
            self.last_discovery_error = 'No se pudo consultar el listado live de SofaScore.'
            return None
            
        except Exception as e:
            self.last_discovery_error = f'Error consultando el listado live: {e}'
            print(f"ERROR: Error obtaining live games: {e}")
            return None
    
    async def _get_games_with_session(self, debug_mode: bool = True) -> Optional[Dict]:
        """🎯 MÉTODO PRINCIPAL: Con sesión establecida (EXITOSO en test)"""
        try:
            if debug_mode:
                print("   Main method: With session...")
            
            if not await self._establish_basketball_session():
                return None
            
            # 3. Intentar fetch interno primero
            json_content = await self.page.evaluate("""
                async () => {
                    try {
                        const r = await fetch('/api/v1/sport/basketball/events/live');
                        if (!r.ok) return { success: false, status: r.status };
                        const data = await r.json();
                        if (!Array.isArray(data.events)) return { success: false, error: 'Respuesta sin lista events' };
                        return {
                            success: true,
                            data: data,
                            events_count: data.events?.length || 0
                        };
                    } catch (e) {
                        return { success: false, error: e.message };
                    }
                }
            """)
            
            if not json_content.get('success'):
                # Fallback a navegación directa
                api_response = await self.page.goto(
                    f"{self.api_base}/sport/basketball/events/live",
                    wait_until="domcontentloaded"
                )
                if api_response and api_response.status == 200:
                    json_content = await self.page.evaluate("""
                        () => {
                            try {
                                const bodyText = document.body.textContent || document.body.innerText;
                                const data = JSON.parse(bodyText);
                                if (!Array.isArray(data.events)) return { success: false, error: 'Respuesta sin lista events' };
                                return {
                                    success: true,
                                    data: data,
                                    events_count: data.events?.length || 0
                                };
                            } catch (e) {
                                return { success: false, error: e.message };
                            }
                        }
                    """)
                else:
                    json_content = {'success': False, 'status': api_response.status if api_response else 'sin respuesta'}
            
            if json_content.get('success'):
                if debug_mode:
                    print(f"   SUCCESS: {json_content['events_count']} eventos (con sesión)")
                if self.context is not None:
                    try:
                        await save_browser_session(self.context)
                    except Exception as error:
                        print(f'[WARN] No se pudo guardar la sesion ({type(error).__name__}).')
                return json_content['data']
            else:
                self.last_discovery_error = (
                    'El endpoint /sport/basketball/events/live no es accesible: '
                    f"{json_content.get('error') or json_content.get('status')}. "
                    'Esto no significa que no haya partidos en vivo.'
                )
                print(f"   ERROR: Status or JSON error: {json_content.get('error') or json_content.get('status')}")
                return None
                
        except Exception as e:
            print(f"   ERROR: Main method error: {e}")
            return None
    
    async def _get_games_basic_method(self) -> Optional[Dict]:
        """🔄 MÉTODO FALLBACK: Básico directo (EXITOSO en test)"""
        try:
            print("   Fallback method: Basic...")
            
            response = await self.page.goto(
                f"{self.api_base}/sport/basketball/events/live",
                wait_until="domcontentloaded"
            )
            
            if response.status == 200:
                json_content = await self.page.evaluate("""
                    () => {
                        try {
                            const bodyText = document.body.textContent || document.body.innerText;
                            const data = JSON.parse(bodyText);
                            return {
                                success: true,
                                data: data,
                                events_count: data.events?.length || 0
                            };
                        } catch (e) {
                            return { success: false, error: e.message };
                        }
                    }
                """)
                
                if json_content['success']:
                    print(f"   SUCCESS: {json_content['events_count']} eventos (básico)")
                    return json_content['data']
                else:
                    print(f"   ERROR: JSON error: {json_content['error']}")
                    return None
            else:
                print(f"   ERROR: Status {response.status}")
                return None
                
        except Exception as e:
            print(f"   ERROR: Basic method error: {e}")
            return None

    
    async def get_live_game_data(self, game_id: str, game_info: Dict) -> Optional[Dict]:
        """
        📊 Obtiene datos live de un juego específico
        🆕 AHORA INCLUYE: Información precisa de tiempo del SofaScore API
        """
        cache_key = f'game_data_{game_id}'

        if self._is_cache_valid(cache_key):
            print(f"Using cached data for game {game_id}")
            return self.cache[cache_key]['data']

        await self._rate_limit()

        try:
            if not self.page and not getattr(self, 'bridge', None):
                await self.initialize()

            # 🆕 PASO 1: Obtener datos del evento con información de tiempo precisa
            event_data = await self.get_game_event_data(game_id)

            if event_data:
                # Extraer información de tiempo precisa
                time_info = self.extract_time_info(event_data)
                if not time_info.get('clock_valid', False):
                    print(f"[CLOCK] {time_info.get('clock_error', 'Reloj no valido')}. Se omite la prediccion.")
                    return None
                print(f"[TIME] Time info extracted: {time_info['current_quarter_display']}, "
                      f"{time_info['minutes_played']:.1f}min played, "
                      f"{time_info['total_remaining_minutes']:.1f}min remaining")
            else:
                print("[WARN] Could not get event data, falling back to basic estimation")
                time_info = None

            # PASO 2: Obtener scores actuales del evento
            q_scores = {}
            status_desc = "Live"
            game_finished = False

            # Intentar extraer directamente de event_data (más rápido y directo)
            if event_data and ('homeScore' in event_data or 'awayScore' in event_data):
                home_s = event_data.get('homeScore', {})
                away_s = event_data.get('awayScore', {})
                q_scores = {
                    'q1_home': home_s.get('period1', 0),
                    'q1_away': away_s.get('period1', 0),
                    'q2_home': home_s.get('period2', 0),
                    'q2_away': away_s.get('period2', 0),
                    'q3_home': home_s.get('period3', 0),
                    'q3_away': away_s.get('period3', 0),
                    'q4_home': home_s.get('period4', 0),
                    'q4_away': away_s.get('period4', 0)
                }
                status_desc = event_data.get('status', {}).get('description', '')
                game_finished = event_data.get('status', {}).get('type') in ['finished', 'final']
            else:
                # Fallback: consultar listado global si event_data no tenía scores
                current_games = await self.get_live_basketball_games()
                current_game = None
                if current_games:
                    for game in current_games:
                        if game.get('id') == game_id:
                            current_game = game
                            break

                if current_game:
                    q_scores = self._extract_quarter_scores(current_game)
                    status_desc = current_game.get('status', '')
                    game_finished = current_game.get('status', '').lower() in ['finished', 'final']
                else:
                    print(f"ERROR: Game {game_id} not found")
                    return None

            # Convertir a formato requerido
            live_data = {
                'q_scores': q_scores,
                'game_finished': game_finished,
                'context': {
                    'estimated_quarter': self._estimate_quarter_from_status(status_desc),
                    'estimated_pace': self._estimate_pace({'quarter_info': q_scores}),
                    'offensive_efficiency': 1.0
                }
            }

            # 🆕 PASO 3: Agregar información de tiempo precisa si está disponible
            if time_info:
                live_data['time_info'] = time_info
                # Actualizar contexto con datos precisos
                live_data['context'].update({
                    'current_quarter': time_info['current_quarter_display'],
                    'minutes_played': time_info['minutes_played'],
                    'remaining_minutes': time_info['total_remaining_minutes'],
                    'current_pace': time_info['current_pace_per_minute'],
                    'projected_final': time_info['projected_final_score'],
                    'completion_percentage': time_info['completion_percentage']
                })

            # 🆕 PASO 4: Obtener estadísticas detalladas
            current_stats = await self.get_game_statistics(game_id)
            if current_stats:
                live_data['current_stats'] = current_stats
                print(f"[STATS] Statistics obtained: {len(current_stats)} metrics")
            else:
                live_data['current_stats'] = {}
                print("[WARN] Could not get detailed statistics")

            # Cache el resultado
            self.cache[cache_key] = {
                'data': live_data,
                'timestamp': time.time()
            }

            return live_data

        except Exception as e:
            print(f"ERROR: Error obtaining game data: {e}")
            return None
    
    def _extract_quarter_scores(self, game: Dict) -> Dict[str, int]:
        """Extrae scores por cuarto del juego"""
        quarter_info = game.get('quarter_info', {})
        
        return {
            'q1_home': quarter_info.get('q1_home', 0),
            'q1_away': quarter_info.get('q1_away', 0),
            'q2_home': quarter_info.get('q2_home', 0),
            'q2_away': quarter_info.get('q2_away', 0),
            'q3_home': quarter_info.get('q3_home', 0),
            'q3_away': quarter_info.get('q3_away', 0),
            'q4_home': quarter_info.get('q4_home', 0),
            'q4_away': quarter_info.get('q4_away', 0)
        }
    
    def _estimate_quarter_from_status(self, status: str) -> str:
        """Estima el cuarto actual basado en status"""
        status_lower = status.lower()
        if 'halftime' in status_lower:
            return "Halftime"
        elif '1st' in status_lower or '1' in status_lower:
            return "Q1"
        elif '2nd' in status_lower or '2' in status_lower:
            return "Q2"
        elif '3rd' in status_lower or '3' in status_lower:
            return "Q3"
        elif '4th' in status_lower or '4' in status_lower:
            return "Q4"
        else:
            return "Live"
    
    def _estimate_pace(self, game: Dict) -> int:
        """Estima el pace del juego"""
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        total_score = home_score + away_score
        
        # Estimación básica de pace
        if total_score > 0:
            return min(120, max(80, int((total_score / 48) * 100)))
        
        return 100  # Default pace
    
    def _estimate_efficiency(self, game: Dict) -> int:
        """Estima la eficiencia ofensiva"""
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        total_score = home_score + away_score
        
        # Estimación básica de eficiencia
        if total_score > 0:
            return min(130, max(90, int(total_score * 2.3)))
        
        return 110  # Default efficiency
    
    async def get_game_event_data(self, game_id: str) -> Optional[Dict]:
        """
        🕐 Obtiene datos del evento específico incluyendo información de tiempo
        ✅ Extrae time.played, time.periodLength y otros datos temporales
        """
        cache_key = f'event_data_{game_id}'

        if self._is_cache_valid(cache_key):
            print(f"Using cached event data for game {game_id}")
            return self.cache[cache_key]['data']

        await self._rate_limit()

        try:
            if not self.page:
                await self.initialize()

            print(f"Getting event data for game {game_id}...")

            # 🏀 PASO 1: Navegar al partido para establecer sesión
            print(f"   Navigating to game {game_id}...")
            game_url = f"{self.base_url}/event/{game_id}"
            # Usa 'domcontentloaded' que es más rápido y menos propenso a timeouts
            try:
                game_response = await self.page.goto(game_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                print(f"   [WARN] Navigation timeout (domcontentloaded) but proceeding: {e}")
                # Si falla el load pero la página está ahí, intentamos seguir
                game_response = None

            if game_response and game_response.status != 200:
                print(f"   ERROR: Error navigating to game: Status {game_response.status}")
                return None

            # Esperar a que se cargue la página del partido
            await asyncio.sleep(2)
            print(f'   Pagina del partido {game_id} cargada; comprobando acceso a la API...')

            # 🏀 PASO 2: Acceder a la API del evento
            print(f"   Accessing event data...")
            event_url = f"{self.api_base}/event/{game_id}"

            # 1. Intentar mediante fetch interno en la sesión establecida
            json_content = await self.page.evaluate(f"""
                async () => {{
                    try {{
                        const r = await fetch('{event_url}');
                        if (!r.ok) return {{ success: false, status: r.status }};
                        const data = await r.json();
                        return {{ success: true, data: data }};
                    }} catch (e) {{
                        return {{ success: false, error: e.message }};
                    }}
                }}
            """)

            # 2. Si fetch no tuvo éxito, fallback a navegación directa
            if not json_content.get('success'):
                response = await self.page.goto(event_url, wait_until="domcontentloaded")
                if response and response.status == 200:
                    json_content = await self.page.evaluate("""
                        () => {
                            try {
                                const bodyText = document.body.textContent || document.body.innerText;
                                const data = JSON.parse(bodyText);
                                return { success: true, data: data };
                            } catch (e) {
                                return { success: false, error: e.message };
                            }
                        }
                    """)

            if json_content and json_content.get('success'):
                event_data = json_content['data']

                # Handle API response structure - sometimes wrapped in 'event' key
                if isinstance(event_data, dict) and 'event' in event_data:
                    # Response is {"event": {...}}
                    actual_event_data = event_data['event']
                else:
                    # Response is the event data directly
                    actual_event_data = event_data

                # Cache the processed event data
                self.cache[cache_key] = {
                    'data': actual_event_data,
                    'timestamp': time.time()
                }

                print(f"SUCCESS: Event data obtained for game {game_id}")
                return actual_event_data
            else:
                err_msg = json_content.get('error', f"Status {json_content.get('status')}") if json_content else "No response"
                print(f"ERROR: Error obtaining event data: {err_msg}")
                return None

        except Exception as e:
            print(f"ERROR: Error in get_game_event_data: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def get_game_statistics(self, game_id: str) -> Optional[Dict]:
        """
        📊 Obtiene estadísticas detalladas del juego (rebotes, faltas, tiros, etc.)
        """
        cache_key = f'game_stats_{game_id}'
        if self._is_cache_valid(cache_key):
            print(f"Using cached statistics for game {game_id}")
            return self.cache[cache_key]['data']

        await self._rate_limit()

        try:
            if not self.page:
                await self.initialize()

            print(f"   Getting statistics for game {game_id}...")
            stats_url = f"{self.api_base}/event/{game_id}/statistics"
            
            # Navegar directamente a la API endpoint
            response = await self.page.goto(stats_url, wait_until="domcontentloaded")

            if response.status == 200:
                json_content = await self.page.evaluate("""
                    () => {
                        try {
                            const bodyText = document.body.textContent || document.body.innerText;
                            const data = JSON.parse(bodyText);
                            return { success: true, data: data };
                        } catch (e) {
                            return { success: false, error: e.message };
                        }
                    }
                """)

                if json_content['success']:
                    stats_data = json_content['data']
                    parsed_stats = self._parse_statistics(stats_data)
                    
                    self.cache[cache_key] = {
                        'data': parsed_stats,
                        'timestamp': time.time()
                    }
                    return parsed_stats
                else:
                    print(f"   ERROR: JSON error in statistics: {json_content['error']}")
                    return None
            else:
                print(f"   ERROR: Status {response.status} fetching statistics")
                return None

        except Exception as e:
            print(f"ERROR: Error getting game statistics: {e}")
            return None

    def _parse_statistics(self, stats_data: Dict) -> Dict[str, float]:
        """Parsea la respuesta de estadísticas de SofaScore"""
        parsed = {}
        
        if not stats_data or 'statistics' not in stats_data:
            print("⚠️ No statistics found in response")
            return parsed

        # Debug: Print available periods
        periods = [item.get('period') for item in stats_data['statistics']]
        print(f"   [DEBUG] Available stats periods: {periods}")

        # Buscar estadísticas del periodo "ALL"
        all_period_stats = None
        for item in stats_data['statistics']:
            if item.get('period') == 'ALL':
                all_period_stats = item
                break
        
        if not all_period_stats:
            print("⚠️ 'ALL' period statistics not found")
            return parsed

        # Mapeo de keys de SofaScore a nuestro formato interno
        found_keys = []
        for group in all_period_stats.get('groups', []):
            for item in group.get('statisticsItems', []):
                key = item.get('key')
                found_keys.append(key)
                
                # Flexible matching for keys
                if key in ['freeThrowsScored', 'freeThrows']:
                    parsed['home_free_throws_made'] = item.get('homeValue', 0)
                    parsed['away_free_throws_made'] = item.get('awayValue', 0)
                    parsed['home_free_throws_attempted'] = item.get('homeTotal', 0)
                    parsed['away_free_throws_attempted'] = item.get('awayTotal', 0)
                
                elif key in ['twoPointersScored', 'twoPointers']:
                    parsed['home_2_point_field_goals_made'] = item.get('homeValue', 0)
                    parsed['away_2_point_field_goals_made'] = item.get('awayValue', 0)
                    parsed['home_2_point_field_goals_attempted'] = item.get('homeTotal', 0)
                    parsed['away_2_point_field_goals_attempted'] = item.get('awayTotal', 0)
                
                elif key in ['threePointersScored', 'threePointers']:
                    parsed['home_3_point_field_goals_made'] = item.get('homeValue', 0)
                    parsed['away_3_point_field_goals_made'] = item.get('awayValue', 0)
                    parsed['home_3_point_field_goals_attempted'] = item.get('homeTotal', 0)
                    parsed['away_3_point_field_goals_attempted'] = item.get('awayTotal', 0)
                
                elif key in ['fieldGoalsScored', 'fieldGoals']:
                    parsed['home_field_goals_made'] = item.get('homeValue', 0)
                    parsed['away_field_goals_made'] = item.get('awayValue', 0)
                    parsed['home_field_goals_attempted'] = item.get('homeTotal', 0)
                    parsed['away_field_goals_attempted'] = item.get('awayTotal', 0)
                
                elif key == 'rebounds':
                    parsed['home_total_rebounds'] = item.get('homeValue', 0)
                    parsed['away_total_rebounds'] = item.get('awayValue', 0)
                
                elif key == 'offensiveRebounds':
                    parsed['home_offensive_rebounds'] = item.get('homeValue', 0)
                    parsed['away_offensive_rebounds'] = item.get('awayValue', 0)
                
                elif key == 'defensiveRebounds':
                    parsed['home_defensive_rebounds'] = item.get('homeValue', 0)
                    parsed['away_defensive_rebounds'] = item.get('awayValue', 0)
                
                elif key == 'assists':
                    parsed['home_assists'] = item.get('homeValue', 0)
                    parsed['away_assists'] = item.get('awayValue', 0)
                
                elif key == 'turnovers':
                    parsed['home_turnovers'] = item.get('homeValue', 0)
                    parsed['away_turnovers'] = item.get('awayValue', 0)
                
                elif key == 'steals':
                    parsed['home_steals'] = item.get('homeValue', 0)
                    parsed['away_steals'] = item.get('awayValue', 0)
                
                elif key == 'blocks':
                    parsed['home_blocks'] = item.get('homeValue', 0)
                    parsed['away_blocks'] = item.get('awayValue', 0)
                
                elif key in ['totalFouls', 'fouls']: 
                    parsed['home_personal_fouls'] = item.get('homeValue', 0)
                    parsed['away_personal_fouls'] = item.get('awayValue', 0)
                
                elif key == 'maxPointsInARow':
                    parsed['home_max_run'] = item.get('homeValue', 0)
                    parsed['away_max_run'] = item.get('awayValue', 0)
                
                elif key == 'biggestLead':
                    parsed['home_biggest_lead'] = item.get('homeValue', 0)
                    parsed['away_biggest_lead'] = item.get('awayValue', 0)
        
        print(f"   [DEBUG] Parsed {len(parsed)} metrics from keys: {found_keys[:5]}...")
        return parsed

    def extract_time_info(self, event_data: Dict) -> Dict[str, float]:
        """Normalize the event clock using its period and selected league."""
        from core.game_clock import normalize_event_clock
        return normalize_event_clock(event_data, getattr(self, 'league_name', None))

    async def get_game_statistics(self, game_id: str) -> Optional[Dict]:
        """
        📊 Obtiene estadísticas de juego específico
        ✅ Navega primero al partido para establecer sesión, luego accede a la API
        🆕 MEJORA: Sistema de fallback inteligente cuando la calidad de datos es pobre
        """


        cache_key = f'stats_{game_id}'

        if self._is_cache_valid(cache_key):
            print(f"Using cached stats for game {game_id}")
            return self.cache[cache_key]['data']

        await self._rate_limit()

        try:
            if not self.page:
                await self.initialize()

            print(f"Getting statistics for game {game_id}...")

            # 🏀 PASO 1: Navegar al partido para establecer sesión
            print(f"   Navigating to game {game_id}...")
            game_url = f"{self.base_url}/event/{game_id}"  # Navigate to game page first
            
            try:
                game_response = await self.page.goto(game_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                print(f"   [WARN] Navigation timeout (domcontentloaded) but proceeding: {e}")
                game_response = None

            if game_response and game_response.status != 200:
                print(f"   ERROR: Error navigating to game: Status {game_response.status}")
                return None

            # Esperar a que se cargue la página del partido
            await asyncio.sleep(2)
            print(f'   Pagina del partido {game_id} cargada; comprobando acceso a la API...')

            # 🏀 PASO 2: Ahora acceder a la API de estadísticas
            print(f"   Accessing statistics...")
            stats_url = f"{self.api_base}/event/{game_id}/statistics"

            # 1. Intentar mediante fetch interno en la sesión establecida
            json_content = await self.page.evaluate(f"""
                async () => {{
                    try {{
                        const r = await fetch('{stats_url}');
                        if (!r.ok) return {{ success: false, status: r.status }};
                        const data = await r.json();
                        return {{ success: true, data: data }};
                    }} catch (e) {{
                        return {{ success: false, error: e.message }};
                    }}
                }}
            """)

            # 2. Si fetch no tuvo éxito, fallback a navegación directa
            if not json_content.get('success'):
                response = await self.page.goto(stats_url, wait_until="domcontentloaded")
                if response and response.status == 200:
                    json_content = await self.page.evaluate("""
                        () => {
                            try {
                                const bodyText = document.body.textContent || document.body.innerText;
                                const data = JSON.parse(bodyText);
                                return { success: true, data: data };
                            } catch (e) {
                                return { success: false, error: e.message };
                            }
                        }
                    """)

            if json_content and json_content.get('success'):
                stats_data = json_content['data']

                # 🆕 MEJORA: Validar y aplicar fallback si es necesario
                parsed_stats = self._parse_massive_stats_json(stats_data)
               
                # 🆕 MEJORA: Verificar si hay suficientes datos para features críticas
                critical_features_present = self._validate_critical_features(parsed_stats)
               
                if not critical_features_present:
                    parsed_stats = await self._apply_intelligent_fallback(parsed_stats, game_id)

                # Cache el resultado
                self.cache[cache_key] = {
                    'data': parsed_stats,
                    'timestamp': time.time(),
                    'data_quality_score': getattr(self, '_last_data_quality_score', 0.5),
                    'fallback_applied': not critical_features_present
                }

                return parsed_stats
            else:
                err = json_content.get('error', f"Status {json_content.get('status')}") if json_content else "No stats"
                print(f"ERROR: Error obtaining stats: {err}")
                return None

        except Exception as e:
            print(f"ERROR: Error in get_game_statistics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _validate_critical_features(self, stats_data: Dict) -> bool:
        """
        🆕 MEJORA: Valida si hay suficientes datos para features críticas
        Returns True si hay suficientes datos, False si se necesita fallback
        """
        try:
            # Features críticas mínimas necesarias para cálculo de características live
            required_critical_stats = [
                'home_field_goals_made', 'away_field_goals_made',
                'home_2_point_field_goals_made', 'away_2_point_field_goals_made',
                'home_3_point_field_goals_made', 'away_3_point_field_goals_made',
                'home_field_goals_attempted', 'away_field_goals_attempted',
                'home_free_throws_made', 'away_free_throws_made',
                'home_free_throws_attempted', 'away_free_throws_attempted',
                'home_total_rebounds', 'away_total_rebounds',
                'home_offensive_rebounds', 'away_offensive_rebounds',
                'home_defensive_rebounds', 'away_defensive_rebounds',
                'home_assists', 'away_assists',
                'home_turnovers', 'away_turnovers',
                'home_steals', 'away_steals',
                'home_blocks', 'away_blocks',
                'home_personal_fouls', 'away_personal_fouls'
            ]
            
            # Stats que PUEDEN ser cero legítimamente durante un partido
            allow_zero_stats = [
                'home_3_point_field_goals_made', 'away_3_point_field_goals_made',
                'home_free_throws_made', 'away_free_throws_made',
                'home_free_throws_attempted', 'away_free_throws_attempted',
                'home_offensive_rebounds', 'away_offensive_rebounds',
                'home_defensive_rebounds', 'away_defensive_rebounds',
                'home_assists', 'away_assists',
                'home_turnovers', 'away_turnovers',
                'home_steals', 'away_steals',
                'home_blocks', 'away_blocks',
                'home_personal_fouls', 'away_personal_fouls',
                'home_2_point_field_goals_made', 'away_2_point_field_goals_made'
            ]

            # Verificar si todas las estadísticas críticas están presentes
            for stat in required_critical_stats:
                # 1. Verificar existencia (CRÍTICO: Debe existir siempre)
                if stat not in stats_data:
                    # print(f"[DEBUG] Falta stat crítica: {stat}")
                    return False
                
                # 2. Verificar valor cero (Solo para stats mayores)
                if stats_data[stat] == 0:
                    if stat not in allow_zero_stats:
                        # Es una stat mayor (FG Attempts, Total Rebounds) que NO debería ser 0 en Live
                        # print(f"[DEBUG] Stat mayor es cero: {stat}")
                        return False
            
            # print(f"[CRITICAL VALIDATION] All critical features present and non-zero")
            return True
            
        except Exception as e:
            print(f"[CRITICAL VALIDATION] Error validating critical features: {e}")
            return False

            return self._basic_fallback(partial_stats)

    def _basic_fallback(self, partial_stats: Dict) -> Dict:
        """
        🆕 MEJORA: Fallback básico con valores conservadores
        """
        # print(f"[FALLBACK] Aplicando fallback básico con valores conservadores")
        
        fallback_stats = partial_stats.copy()
        
        # Valores mínimos para evitar errores críticos
        minimal_stats = {
            'home_field_goals_made': 10, 'away_field_goals_made': 10,
            'home_field_goals_attempted': 20, 'away_field_goals_attempted': 20,
            'home_free_throws_made': 5, 'away_free_throws_made': 5,
            'home_free_throws_attempted': 7, 'away_free_throws_attempted': 7,
            'home_total_rebounds': 15, 'away_total_rebounds': 15,
            'home_personal_fouls': 8, 'away_personal_fouls': 8,
            'home_assists': 5, 'away_assists': 5,
            'home_turnovers': 8, 'away_turnovers': 8,
            'home_steals': 2, 'away_steals': 2,
            'home_blocks': 1, 'away_blocks': 1
        }
        
        # Solo agregar valores que faltan completamente
        for key, value in minimal_stats.items():
            if key not in fallback_stats or fallback_stats[key] == 0:
                fallback_stats[key] = value
                # print(f"[FALLBACK] Agregando valor mínimo para {key}: {value}")
        
        return fallback_stats
    
    def _filter_supported_games(self, data: Dict) -> List[Dict]:
        """Filtra y procesa los juegos soportados - IMPLEMENTACIÓN COMPLETA RESTAURADA"""
        
        events = data.get('events', [])
        
        if not events:
            print("Warning: No events in response")
            return []
        
        supported_games = []
        
        for event in events:
            try:
                # Información básica
                game_info = {
                    'id': str(event.get('id', '')),
                    'home_team': event.get('homeTeam', {}).get('name', 'Unknown'),
                    'away_team': event.get('awayTeam', {}).get('name', 'Unknown'),
                    'home_score': self._safe_int(event.get('homeScore', {}).get('current', 0)),
                    'away_score': self._safe_int(event.get('awayScore', {}).get('current', 0)),
                    'status': event.get('status', {}).get('description', 'Unknown'),
                    'tournament_name': event.get('tournament', {}).get('name', 'Unknown'),
                    'detected_league': self._detect_league(event),
                    'quarter_info': self._extract_quarter_info(event),
                    'time_info': self._extract_time_info(event),
                    'priority': self._calculate_game_priority(event)
                }
                
                # Solo agregar si tenemos información mínima
                if game_info['id'] and game_info['home_team'] != 'Unknown':
                    supported_games.append(game_info)
                    
            except Exception as e:
                print(f"Warning: Error processing event {event.get('id', 'unknown')}: {e}")
                continue
        
        # Ordenar por prioridad
        supported_games.sort(key=lambda x: x['priority'], reverse=True)
        
        print(f"Filtered {len(supported_games)} supported games from {len(events)} events")
        
        return supported_games
    
    def _safe_int(self, value) -> int:
        """Convierte valor a int de forma segura"""
        try:
            if value is None:
                return 0
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _detect_league(self, event: Dict) -> str:
        """Detecta la liga basado en el torneo"""
        tournament = event.get('tournament', {})
        tournament_name = tournament.get('name', '').upper()

        # Mapeo de ligas conocidas
        if 'NBA' in tournament_name:
            return 'NBA'
        elif 'WNBA' in tournament_name:
            return 'WNBA'
        elif 'NBL' in tournament_name or 'NATIONAL BASKETBALL LEAGUE' in tournament_name:
            return 'NBL'
        elif 'EUROLEAGUE' in tournament_name or 'EURO LEAGUE' in tournament_name:
            return 'EuroLeague'
        elif 'NCAA' in tournament_name:
            return 'NCAA'
        elif 'G LEAGUE' in tournament_name or 'G-LEAGUE' in tournament_name:
            return 'G League'
        else:
            return 'Other Basketball'
    
    def _calculate_game_priority(self, event: Dict) -> int:
        """Calcula prioridad del juego para ordenamiento"""
        priority = 0
        
        # Prioridad por liga
        league = self._detect_league(event)
        if league == 'NBA':
            priority += 100
        elif league == 'WNBA':
            priority += 90
        elif league == 'EuroLeague':
            priority += 80
        elif league == 'NCAA':
            priority += 70
        else:
            priority += 50
        
        # Prioridad por estado del juego
        status = event.get('status', {}).get('type', '')
        if status == 'inprogress':
            priority += 50
        elif status == 'notstarted':
            priority += 30
        
        return priority
    
    def _extract_quarter_info(self, event: Dict) -> Dict:
        """Extrae información de cuartos del evento"""
        try:
            status = event.get('status', {})
            quarter_info = {
                'current_period': status.get('period', ''),
                'status_type': status.get('type', ''),
                'description': status.get('description', ''),
                'is_live': status.get('type') == 'inprogress'
            }
            
            # Extraer scores por período si disponibles
            home_score = event.get('homeScore', {})
            away_score = event.get('awayScore', {})
            
            for period in ['period1', 'period2', 'period3', 'period4']:
                if period in home_score and period in away_score:
                    quarter_num = period.replace('period', 'q')
                    quarter_info[f'{quarter_num}_home'] = self._safe_int(home_score[period])
                    quarter_info[f'{quarter_num}_away'] = self._safe_int(away_score[period])
            
            return quarter_info
            
        except Exception as e:
            print(f"Warning: Error extracting quarter info: {e}")
            return {'current_period': 'unknown', 'is_live': False}
    
    def _extract_time_info(self, event: Dict) -> Dict:
        """Extrae información de tiempo del juego"""
        try:
            return {
                'start_timestamp': event.get('startTimestamp'),
                'last_updated': datetime.now().isoformat(),
                'time_status': event.get('status', {}).get('description', '')
            }
        except Exception as e:
            print(f"⚠️ Error extracting time info: {e}")
            return {'start_timestamp': None, 'last_updated': datetime.now().isoformat()}
    
    def _parse_massive_stats_json(self, massive_json) -> Dict:
        """Parser del JSON de estadísticas de SofaScore"""
        extracted_stats = {}

        try:
            # Handle both dict and list inputs
            if isinstance(massive_json, dict):
                statistics = massive_json.get('statistics', [])
            elif isinstance(massive_json, list):
                statistics = massive_json
            else:
                print(f"ERROR: Unexpected statistics format: {type(massive_json)}")
                return {}

            # Initialize lists
            available_stats = []
            missing_stats = []
            all_stat_names = set()

            # Find ALL period stats
            all_period_stats = None
            for item in statistics:
                if item.get('period') == 'ALL':
                    all_period_stats = item
                    break
            
            if not all_period_stats:
                # Fallback to first period if ALL not found (unlikely)
                if statistics:
                    all_period_stats = statistics[0]
            
            if not all_period_stats:
                print("   [DEBUG] No stats found in any period")
                return {}

            # Collect all stat names for debug
            # for group in all_period_stats.get('groups', []):
            #     for item in group.get('statisticsItems', []):
            #         stat_name = item.get('name', '')
            #         all_stat_names.add(stat_name)
            
            # print(f"   [DEBUG] Available stat names (ALL period): {list(all_stat_names)}")

            # Parse stats from ALL period groups
            for group in all_period_stats.get('groups', []):
                for item in group.get('statisticsItems', []):
                    stat_name = item.get('name', '')

                    if stat_name in self.required_stats:
                        model_name = self.required_stats[stat_name]
                        
                        home_val = item.get('home', 0)
                        away_val = item.get('away', 0)
                        
                        # Initialize variables to avoid UnboundLocalError
                        home_made = 0
                        away_made = 0

                        # Check if values are strings with '/' (e.g. "20/45" or "20/45 (44%)")
                        is_fraction = isinstance(home_val, str) and '/' in home_val
                        
                        if is_fraction:
                            try:
                                # Clean values (remove percentage if present)
                                clean_home = home_val.split('(')[0].strip()
                                clean_away = away_val.split('(')[0].strip()
                                
                                home_made, home_attempted = map(int, clean_home.split('/'))
                                away_made, away_attempted = map(int, clean_away.split('/'))
                                
                                extracted_stats[f'home_{model_name}'] = home_made
                                extracted_stats[f'away_{model_name}'] = away_made
                                
                                # Handle attempted if applicable
                                if '_made' in model_name:
                                    attempted_name = model_name.replace('_made', '_attempted')
                                    extracted_stats[f'home_{attempted_name}'] = home_attempted
                                    extracted_stats[f'away_{attempted_name}'] = away_attempted
                                    
                                    # print(f"Stat {stat_name}: Home {home_made}/{home_attempted}, Away {away_made}/{away_attempted}")
                            except ValueError:
                                # print(f"   [WARN] Failed to parse fraction for {stat_name}: {home_val}, {away_val}")
                                is_fraction = False # Treat as non-fraction if parsing fails
                        else:
                            # Para otras estadísticas: usar home/away directamente
                            other_home_value = self._safe_int(home_val)
                            other_away_value = self._safe_int(away_val)
                            
                            extracted_stats[f'home_{model_name}'] = other_home_value
                            extracted_stats[f'away_{model_name}'] = other_away_value
                            
                            # print(f"Stat {stat_name}: Home {other_home_value}, Away {other_away_value}")
                        
                        available_stats.append(stat_name)

                        # 🆕 MEJORA: Validar que los valores no sean ceros cuando no deberían
                        if stat_name in ['Field goals', '2 pointers', '3 pointers', 'Free throws']:
                            if is_fraction and home_made == 0 and away_made == 0:
                                print(f"[API WARNING] Stat {stat_name} tiene valores ceros para ambos equipos")
                        else:
                            if not is_fraction and other_home_value == 0 and other_away_value == 0:
                                print(f"[API WARNING] Stat {stat_name} tiene valores ceros para ambos equipos")

            # 🆕 MEJORA: Calcular estadísticas combinadas que el modelo espera
            # Estas son estadísticas que no vienen directamente de la API pero el modelo necesita
            self._calculate_combined_statistics(extracted_stats)

            # 🆕 MEJORA: Reporte detallado de calidad de datos
            # print(f"[API DATA QUALITY] Total stats disponibles: {len(all_stat_names)}")
            # print(f"[API DATA QUALITY] Stats requeridas encontradas: {len(available_stats)}/{len(self.required_stats)}")
            
            # Identificar estadísticas críticas faltantes
            # Mapeo correcto entre nombres de API y nombres de características del modelo
            critical_api_to_model = {
                'Field goals': 'field_goals_made',
                '2 pointers': '2_point_field_goals_made',
                '3 pointers': '3_point_field_goals_made',
                'Free throws': 'free_throws_made',
                'Fouls': 'personal_fouls',
                'Rebounds': 'total_rebounds',
                'Assists': 'assists',
                'Turnovers': 'turnovers',
                'Steals': 'steals',
                'Blocks': 'blocks'
            }
            
            # Verificar estadísticas críticas usando el mapeo correcto
            # Verificar estadísticas críticas usando el mapeo correcto
            for api_stat, model_stat in critical_api_to_model.items():
                # Verificar si existen las versiones home/away
                home_stat = f'home_{model_stat}'
                away_stat = f'away_{model_stat}'
                
                if home_stat not in extracted_stats or away_stat not in extracted_stats:
                    missing_stats.append(model_stat)
            
            # Verificar estadísticas combinadas críticas
            combined_stats = ['field_goals_attempted', 'free_throws_attempted', 'total_rebounds']
            for stat in combined_stats:
                home_stat = f'home_{stat}'
                away_stat = f'away_{stat}'
                if home_stat not in extracted_stats or away_stat not in extracted_stats:
                    missing_stats.append(stat)
            
            if missing_stats:
                print(f"[API CRITICAL] Estadísticas críticas faltantes: {missing_stats}")
                print(f"[API CRITICAL] Esto puede causar alta imputación de características")
            
            # Calcular calidad de datos basada en estadísticas críticas
            total_critical_features = len(critical_api_to_model) + len(combined_stats)
            found_critical_features = total_critical_features - len(missing_stats)
            data_quality_score = found_critical_features / total_critical_features
            # print(f"[API QUALITY SCORE] {data_quality_score:.2f} ({data_quality_score*100:.1f}% de datos críticos)")
            # print(f"[API QUALITY] Estadísticas críticas encontradas: {found_critical_features}/{total_critical_features}")
            
            if data_quality_score < 0.7:
                print(f"[API ALERT] Calidad de datos pobre ({data_quality_score*100:.1f}%) - esperada alta imputación")

            return extracted_stats

        except Exception as e:
            print(f"ERROR: Error parsing statistics: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_combined_statistics(self, extracted_stats: Dict) -> None:
        """
        🆕 CALCULA ESTADÍSTICAS COMBINADAS QUE EL MODELO ESPERA
        Estas estadísticas no vienen directamente de la API pero son necesarias para el modelo
        """
        try:
            # 1. Calcular field_goals_attempted si no existe
            if 'home_field_goals_attempted' not in extracted_stats or extracted_stats['home_field_goals_attempted'] == 0:
                # Primero intentar usar homeTotal/awayTotal si existen
                home_fga = extracted_stats.get('home_field_goals_attempted', 0)
                away_fga = extracted_stats.get('away_field_goals_attempted', 0)
                
                if home_fga == 0:
                    # Sumar 2 pointers + 3 pointers
                    home_2pa = extracted_stats.get('home_2_point_field_goals_attempted', 0)
                    home_3pa = extracted_stats.get('home_3_point_field_goals_attempted', 0)
                    home_fga = home_2pa + home_3pa
                    extracted_stats['home_field_goals_attempted'] = home_fga
                
                if away_fga == 0:
                    away_2pa = extracted_stats.get('away_2_point_field_goals_attempted', 0)
                    away_3pa = extracted_stats.get('away_3_point_field_goals_attempted', 0)
                    away_fga = away_2pa + away_3pa
                    extracted_stats['away_field_goals_attempted'] = away_fga
                
                # print(f"[COMBINED STATS] FG attempts calculated: Home {home_fga}, Away {away_fga}")
            
            # 2. Calcular field_goals_made si no existe
            if 'home_field_goals_made' not in extracted_stats or extracted_stats['home_field_goals_made'] == 0:
                # Sumar 2 pointers + 3 pointers made
                home_2pm = extracted_stats.get('home_2_point_field_goals_made', 0)
                home_3pm = extracted_stats.get('home_3_point_field_goals_made', 0)
                extracted_stats['home_field_goals_made'] = home_2pm + home_3pm
                
                away_2pm = extracted_stats.get('away_2_point_field_goals_made', 0)
                away_3pm = extracted_stats.get('away_3_point_field_goals_made', 0)
                extracted_stats['away_field_goals_made'] = away_2pm + away_3pm
                
                # print(f"[COMBINED STATS] FG made calculated: Home {extracted_stats['home_field_goals_made']}, Away {extracted_stats['away_field_goals_made']}")
            
            # 3. Verificar que tenemos las estadísticas de tiros libres
            if 'home_free_throws_made' not in extracted_stats or extracted_stats['home_free_throws_made'] == 0:
                # Si no tenemos free throws made, usar el valor de 'Free throws' si existe
                if 'home_free_throws' in extracted_stats and extracted_stats['home_free_throws'] > 0:
                    extracted_stats['home_free_throws_made'] = extracted_stats['home_free_throws']
                    print(f"[COMBINED STATS] FT made mapped from Free throws: Home {extracted_stats['home_free_throws_made']}")
                
                if 'away_free_throws' in extracted_stats and extracted_stats['away_free_throws'] > 0:
                    extracted_stats['away_free_throws_made'] = extracted_stats['away_free_throws']
                    print(f"[COMBINED STATS] FT made mapped from Free throws: Away {extracted_stats['away_free_throws_made']}")
            
            # 3b. Calcular free_throws_attempted si no existe o es cero
            if 'home_free_throws_attempted' not in extracted_stats or extracted_stats['home_free_throws_attempted'] == 0:
                # Usar homeTotal si existe, o estimar basado en made
                home_fta = extracted_stats.get('home_free_throws_attempted', 0)
                if home_fta == 0:
                    home_ftm = extracted_stats.get('home_free_throws_made', 0)
                    # Estimar basado en porcentaje típico de FT (~75%)
                    home_fta = max(home_ftm, int(home_ftm * 1.33)) if home_ftm > 0 else 0
                    extracted_stats['home_free_throws_attempted'] = home_fta
                
                away_fta = extracted_stats.get('away_free_throws_attempted', 0)
                if away_fta == 0:
                    away_ftm = extracted_stats.get('away_free_throws_made', 0)
                    away_fta = max(away_ftm, int(away_ftm * 1.33)) if away_ftm > 0 else 0
                    extracted_stats['away_free_throws_attempted'] = away_fta
                
                # print(f"[COMBINED STATS] FT attempts calculated: Home {home_fta}, Away {away_fta}")
            
            # 4. Asegurar que tenemos personal_fouls
            if 'home_personal_fouls' not in extracted_stats or extracted_stats['home_personal_fouls'] == 0:
                # Usar 'Fouls' si existe
                if 'home_fouls' in extracted_stats and extracted_stats['home_fouls'] > 0:
                    extracted_stats['home_personal_fouls'] = extracted_stats['home_fouls']
                    print(f"[COMBINED STATS] Personal fouls mapped from Fouls: Home {extracted_stats['home_personal_fouls']}")
                
                if 'away_fouls' in extracted_stats and extracted_stats['away_fouls'] > 0:
                    extracted_stats['away_personal_fouls'] = extracted_stats['away_fouls']
                    print(f"[COMBINED STATS] Personal fouls mapped from Fouls: Away {extracted_stats['away_personal_fouls']}")
            
            # 4b. Asegurar mapeo bidireccional
            if 'home_fouls' not in extracted_stats and 'home_personal_fouls' in extracted_stats:
                extracted_stats['home_fouls'] = extracted_stats['home_personal_fouls']
                
            if 'away_fouls' not in extracted_stats and 'away_personal_fouls' in extracted_stats:
                extracted_stats['away_fouls'] = extracted_stats['away_personal_fouls']
            
            # 5. Verificar total_rebounds
            if 'home_total_rebounds' not in extracted_stats or extracted_stats['home_total_rebounds'] == 0:
                # Sumar offensive + defensive rebounds
                home_orb = extracted_stats.get('home_offensive_rebounds', 0)
                home_drb = extracted_stats.get('home_defensive_rebounds', 0)
                total_home_reb = home_orb + home_drb
                if total_home_reb > 0:  # Solo actualizar si hay datos reales
                    extracted_stats['home_total_rebounds'] = total_home_reb
                
                away_orb = extracted_stats.get('away_offensive_rebounds', 0)
                away_drb = extracted_stats.get('away_defensive_rebounds', 0)
                total_away_reb = away_orb + away_drb
                if total_away_reb > 0:  # Solo actualizar si hay datos reales
                    extracted_stats['away_total_rebounds'] = total_away_reb
                
                # if total_home_reb > 0 or total_away_reb > 0:
                #     print(f"[COMBINED STATS] Total rebounds calculated: Home {total_home_reb}, Away {total_away_reb}")
            
            # print(f"[COMBINED STATS] Estadísticas combinadas calculadas exitosamente")
            
        except Exception as e:
            print(f"[COMBINED STATS] Error calculando estadísticas combinadas: {e}")
    
    def check_model_compatibility(self, models_folder: str = "models") -> bool:
        """Verifica si hay modelos entrenados disponibles"""
        
        exists = os.path.exists(models_folder) and bool(os.listdir(models_folder))
        
        if not exists:
            print(f"Warning: Models folder not found or empty")
            try:
                available_files = os.listdir(models_folder)
                print(f"📋 Archivos disponibles en models/:")
                for file in available_files:
                    if file.endswith('.joblib'):
                        print(f"   - {file}")
            except FileNotFoundError:
                print(f"ERROR: Models folder does not exist")
        
        return exists
    
    def clear_cache(self):
        """Limpia cache manualmente"""
        self.cache.clear()
        print("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Info del estado actual del cache"""
        return {
            'cache_entries': len(self.cache),
            'cache_duration': self.cache_duration,
            'last_request': self.last_request_time,
            'methods_available': ['session_method', 'basic_method']
        }


# ===========================================
# LIVE PROCESSOR COMPLETO - VERSIÓN LIMPIA
# ===========================================

    def _apply_context_adjustment_static(self, prediction, balance_features, live_pace_metrics, quarter_stage, live_signals=None):
        """
        Static version of apply_context_adjustment for testing purposes.
        This is a copy of the function from live_mode.py to avoid circular imports.
        """
        from analysis.live_mode import apply_context_adjustment
        return apply_context_adjustment(prediction, balance_features, live_pace_metrics, quarter_stage, live_signals)
class BrowserExtensionClient(SofaScoreClient):
    """Keep existing model parsers; replace browser automation with a local bridge."""

    def __init__(self, headless=True):
        super().__init__(headless=headless)
        self.bridge = BrowserBridge()
        self.last_error = None

    async def initialize(self):
        if self.bridge.server:
            return True
        try:
            self.bridge.start()
        except OSError:
            self.last_discovery_error = 'No se puede abrir el puerto local 8766. Cierra otra ejecucion de BasketMLG y reintenta.'
            print(f'[ERROR] {self.last_discovery_error}')
            return False
        print('[LIVE] Transporte: extension del navegador del usuario.')
        print('[LIVE] Abre el panel de la extension, selecciona SofaScore y conecta este codigo:')
        print(f'\n{self.bridge.token}\n')
        print('[LIVE] Manten el panel abierto. Espera maxima por consulta: 90 segundos.')
        return True

    async def cleanup(self):
        await asyncio.to_thread(self.bridge.close)
        self.clear_cache()
        print('[LIVE] Conexion local cerrada.')

    async def _request_browser(self, path):
        self.last_error = None
        if not self.bridge.server and not await self.initialize():
            self.last_error = self.last_discovery_error
            return None
        try:
            return await self.bridge.request(path)
        except (RuntimeError, TimeoutError, ValueError) as error:
            self.last_error = str(error)
            print(f'[ERROR] {self.last_error}')
            return None

    async def health_check(self):
        return await self.get_live_basketball_games() is not None

    async def get_live_basketball_games(self):
        self.last_discovery_error = None
        data = await self._request_browser('/api/v1/sport/basketball/events/live')
        if data is None:
            self.last_discovery_error = self.last_error
            return None
        return self._filter_supported_games(data)

    async def get_game_event_data(self, game_id):
        game_id = str(game_id)
        if not game_id.isascii() or not game_id.isdecimal():
            self.last_error = 'El ID del partido debe ser numerico.'
            print(f'[ERROR] {self.last_error}')
            return None
        key = f'event_data_{game_id}'
        if self._is_cache_valid(key):
            return self.cache[key]['data']
        data = await self._request_browser(f'/api/v1/event/{game_id}')
        if data is None:
            return None
        self.cache[key] = {'data': data['event'], 'timestamp': time.time()}
        return data['event']

    async def get_game_statistics(self, game_id):
        game_id = str(game_id)
        if not game_id.isascii() or not game_id.isdecimal():
            self.last_error = 'El ID del partido debe ser numerico.'
            return None
        key = f'stats_{game_id}'
        if self._is_cache_valid(key):
            return self.cache[key]['data']
        data = await self._request_browser(f'/api/v1/event/{game_id}/statistics')
        if data is None:
            return None
        # Never reinterpret a single quarter as whole-game statistics.
        totals = next((p for p in data['statistics']
                       if isinstance(p, dict) and p.get('period') == 'ALL'), None)
        if not totals or not totals.get('groups'):
            self.last_error = 'SofaScore aun no ofrece estadisticas acumuladas (ALL) del partido.'
            print(f'[ERROR] {self.last_error}')
            return None
        provided = {self.required_stats.get(item.get('name'))
                    for group in totals['groups'] if isinstance(group, dict)
                    for item in group.get('statisticsItems', []) if isinstance(item, dict)}
        needed = {'field_goals_made', '2_point_field_goals_made',
                  '3_point_field_goals_made', 'free_throws_made', 'total_rebounds',
                  'offensive_rebounds', 'defensive_rebounds', 'assists', 'turnovers',
                  'steals', 'blocks', 'personal_fouls'}
        if needed - provided:
            self.last_error = 'Estadisticas incompletas para el modelo: ' + ', '.join(sorted(needed - provided))
            print(f'[ERROR] {self.last_error}')
            return None
        parsed = self._parse_massive_stats_json(data)
        if not parsed:
            self.last_error = 'No se pudieron interpretar las estadisticas del partido.'
            return None
        self.cache[key] = {'data': parsed, 'timestamp': time.time()}
        return parsed

    async def get_live_game_data(self, game_id, game_info):
        # Do not calculate a live prediction using missing event/statistics data.
        if not await self.get_game_event_data(game_id):
            return None
        if not await self.get_game_statistics(game_id):
            return None
        return await super().get_live_game_data(game_id, game_info)


class LiveProcessor:
    """
    🎯 Procesador unificado que integra SofaScoreClient + modelo + alertas
    Usa BrowserExtensionClient para recibir los datos del navegador del usuario.
    ✅ Mapea API data → formato modelo 
    ✅ Genera features live
    ✅ Integra con sistema alertas actual
    🚫 SIN betting protection ni Kelly Criterion
    """
    
    def __init__(self, trained_data: Dict, headless: bool = True):
        self.client = BrowserExtensionClient(headless=headless)
        self.client.league_name = trained_data.get('league_name', 'NBA')
        self.trained_data = trained_data

        # Sistema de alertas eliminado - integrado en el meta-modelo
        self.alerts_system = None
        self.alerts_available = False
        print("[INFO] Sistema de alertas integrado en el meta-modelo")

        # Cache para tracking cambios
        self.last_prediction = None
        self.last_stats = None
        self.last_q_scores = None
        self.last_game_info = None
        self.update_count = 0
        self.session_start_time = time.time()

        # Lazy import para evitar importación circular
        self._get_predictions_func = None

        # Configuración de alertas y estabilidad
        self.alert_thresholds = {
            'prediction_change': 2.0,  # Cambio mínimo para alerta
            'shooting_hot_threshold': 0.55,  # 55% FG
            'shooting_cold_threshold': 0.35,  # 35% FG
            'turnover_spike': 2,  # +2 turnovers
            'scoring_run': 6  # +6 puntos seguidos
        }

        # 🆕 PREDICTION STABILITY CONFIGURATION
        self.stability_config = {
            'min_score_change_for_update': 3,  # Solo actualizar si cambio ≥3 pts
            'prediction_damping_factor': 0.7,  # Amortiguar cambios grandes
            'max_prediction_swing': 8.0,  # Máximo cambio por update
            'quarter_in_progress_penalty': 0.5,  # Penalizar estabilidad en cuarto en progreso
            'update_frequency_quarter_active': 60,  # 60s cuando cuarto activo
            'update_frequency_quarter_break': 30,  # 30s en descansos
        }

        print("LiveProcessor initialized")
        print(f"Model: {trained_data['league_name']}")
        print(f"Available teams: {len(trained_data['team_names'])}")
        print(f"Alerts: {'Available' if self.alerts_available else 'Not available'}")
        print(f"Stability: Enabled (min {self.stability_config['min_score_change_for_update']} pts)")

    def _get_prediction_function(self):
        """🔧 Lazy import para evitar importación circular"""
        if self._get_predictions_func is None:
            try:
                from analysis.live_mode import get_predictions_with_alerts
                self._get_predictions_func = get_predictions_with_alerts
                print("Prediction function loaded successfully")
            except ImportError as e:
                print(f"❌ Error cargando función de predicción: {e}")
                # Fallback a función dummy
                def dummy_predictions(home_team, away_team, q_scores, trained_data):
                    return 200.0, [], {'pre_game_alerts': [], 'live_alerts': []}
                self._get_predictions_func = dummy_predictions
                print("Warning: Using dummy prediction function")
        
        return self._get_predictions_func
    
    async def initialize(self):
        """Inicializa el cliente SofaScore"""
        return await self.client.initialize()
    
    async def cleanup(self):
        """Limpia recursos"""
        await self.client.cleanup()
    
    async def get_available_games(self) -> List[Dict]:
        """
        Obtiene juegos live usando SofaScoreClient + validación completa de modelo
        """
        print("Obteniendo juegos live de SofaScore...")
        
        # Consultar mediante la extension y conservar el contrato de datos.
        games = await self.client.get_live_basketball_games()
        
        if games is None:
            print(f"[ERROR] {self.client.last_discovery_error or 'Fallo al consultar SofaScore.'}")
            return []
        if not games:
            print('[INFO] SofaScore devolvio un listado live vacio.')
            return []
        
        print(f"Recibidos {len(games)} juegos de la API")
        
        # Validar que tenemos modelo para estas ligas
        validated_games = []
        
        for game in games:
            detected_league = game['detected_league']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # 1. Verificar si tenemos modelo para esta liga
            league_compatible = self._check_league_compatibility(detected_league)
            
            # 2. Verificar si conocemos estos equipos
            teams_compatible = self._check_teams_compatibility(home_team, away_team)
            
            # 3. Estimar cuartos restantes para priorizar
            remaining_quarters = self._estimate_remaining_quarters(game)
            
            # 4. Calcular score de compatibilidad
            compatibility_score = self._calculate_compatibility_score(
                league_compatible, teams_compatible, remaining_quarters, game
            )
            
            if compatibility_score > 0.5:  # Umbral mínimo
                game['compatibility_score'] = compatibility_score
                game['league_compatible'] = league_compatible
                game['teams_compatible'] = teams_compatible
                game['remaining_quarters'] = remaining_quarters
                validated_games.append(game)
        
        # Ordenar por compatibilidad
        validated_games.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        print(f"{len(validated_games)} juegos validos para prediccion")
        
        return validated_games
    
    def _show_simple_analysis(self, prediction: float, bookie_line: float = None):
        """🎯 ANÁLISIS SIMPLE - Solo predicción sin betting protection"""
        std_dev = self.trained_data.get('std_dev', 10.0)
        
        print(f"\n[PREDICTION] PREDICCIÓN DE LA IA: {prediction:.1f} pts")
        
        if bookie_line:
            diff = prediction - bookie_line
            print(f"[LINE] Línea de casa: {bookie_line:.1f}")
            print(f"[DIFF] Diferencia: {diff:+.1f} pts")
            
            # Análisis de probabilidades en 3 líneas: L-2, L, L+2
            from scipy import stats
            lines = [bookie_line - 2, bookie_line, bookie_line + 2]
            print(f"[PROBS] Probabilidades:")
            for L in lines:
                z = (L - prediction) / std_dev
                under_prob = stats.norm.cdf(z)
                over_prob = 1 - under_prob
                print(f"   {L:.1f}: Over {over_prob:.1%} | Under {under_prob:.1%}")
            
            # Lado favorecido en la línea central
            z_center = (bookie_line - prediction) / std_dev
            under_center = stats.norm.cdf(z_center)
            over_center = 1 - under_center
            if over_center > under_center:
                edge = over_center - 0.5
                print(f"[EDGE] Lado favorecido: OVER (+{edge:.1%} edge)")
            else:
                edge = under_center - 0.5
                print(f"[EDGE] Lado favorecido: UNDER (+{edge:.1%} edge)")
        
        print("-" * 50)

    async def monitor_game(self, game_id: str, game_info: Dict,
                           update_interval: int = 30, max_updates: int = 120,
                           bookie_line: float = None, single_shot: bool = False):
        """
        📄 MONITOREO LIVE - Versión limpia sin betting protection
        🚀 OPTIMIZADO: Caché pre-partido para ~70% menos CPU
        🆕 SINGLE SHOT: Si single_shot=True, ejecuta una vez y sale
        """
        # Cada selección inicia un seguimiento independiente sobre el mismo cliente.
        self.last_prediction = None
        self.last_stats = None
        self.last_q_scores = None
        self.last_game_info = None
        self.update_count = 0
        self.session_start_time = time.time()

        mode_str = "SINGLE SHOT PREDICTION" if single_shot else "MONITOREO LIVE"
        print(f"[START] INICIANDO {mode_str}")
        print(f"Match: {game_info['home_team']} vs {game_info['away_team']}")
        print(f"Interval: {update_interval}s | Max updates: {max_updates}")
        if bookie_line:
            print(f"Line: {bookie_line}")
        print("-" * 60)

        # 🚀 CONSTRUIR CACHÉ PRE-PARTIDO UNA SOLA VEZ
        try:
            from analysis.live_mode import build_pre_game_cache
            self.pre_game_cache = build_pre_game_cache(
                game_info['home_team'],
                game_info['away_team'],
                self.trained_data
            )
            print("✅ Caché pre-partido construido para optimización live")
        except Exception as e:
            print(f"[WARN] Error construyendo caché: {e} - continuando sin caché")
            self.pre_game_cache = None

        get_predictions = self._get_prediction_function()

        for update_num in range(1, max_updates + 1):
            try:
                print(f"\n[UPDATE #{update_num}] ({datetime.now().strftime('%H:%M:%S')})")
                
                # 🆕 FORCE FRESH DATA: Limpiar caché del cliente para asegurar datos frescos
                self.client.clear_cache()
                print("[CACHE] Caché limpiado para asegurar datos frescos")
                
                # 1. Obtener datos live actuales
                live_data = await self.client.get_live_game_data(game_id, game_info)
                
                if not live_data or not live_data.get('q_scores'):
                    print("❌ No se pudieron obtener datos live")
                    await asyncio.sleep(update_interval)
                    continue
                
                if live_data.get('game_finished', False):
                    print('[END] JUEGO TERMINADO - sin nueva prediccion')
                    break

                # 2. Convertir a formato del modelo
                q_scores = live_data['q_scores']
                home_team = game_info['home_team']
                away_team = game_info['away_team']
                
                # 🆕 HALFTIME ENFORCEMENT (RELAXED)
                # Verificar que se tengan datos de Q1 y Q2 completos
                # Si faltan parciales pero tenemos un total coherente (>50pts), permitimos continuar con warnings
                current_total_score = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0) + \
                                    q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0) + \
                                    q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0) + \
                                    q_scores.get('q4_home', 0) + q_scores.get('q4_away', 0)
                
                sum_q1_q2 = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0) + \
                           q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
                
                # Check estricto solo si el total es muy bajo (indicando que realmente faltan datos)
                if (q_scores['q1_home'] == 0 or q_scores['q2_home'] == 0) and current_total_score < 50:
                    print("\n[ERROR] REQUISITO DE HALFTIME NO CUMPLIDO")
                    print(f"   Datos insuficientes: Total {current_total_score} pts. Se requiere Q1 y Q2.")
                    print("   Por favor, espere al descanso para usar esta herramienta.")
                    if single_shot:
                        return
                    else:
                        await asyncio.sleep(update_interval)
                        continue
                elif (q_scores['q1_home'] == 0 or q_scores['q2_home'] == 0):
                    print(f"[WARN] Datos parciales de Q1/Q2 incompletos, pero Total ({current_total_score}) es válido. Procediendo con estimaciones...")

                
                # 3. Verificar si debe actualizarse la predicción
                should_update, update_reason = self._should_update_prediction(q_scores, game_info)

                if should_update:
                    # 1. Obtener estadísticas live del juego
                    current_stats = await self.client.get_game_statistics(game_id)

                    # 2. Calcular señales live avanzadas
                    time_info = live_data.get('time_info', {})
                    live_signals = self.compute_live_signals(current_stats or {}, q_scores, game_info, time_info=time_info)
                    
                    # 🆕 PASAR ESTADÍSTICAS ACTUALES EN live_signals PARA INTEGRACIÓN
                    if current_stats:
                        live_signals['current_stats'] = current_stats
                        print(f"[STATS] {len(current_stats)} estadísticas pasadas a live_signals")

                    # 3. Generar predicción BASE (con caché para optimización)
                    base_prediction, predictions_list, alerts_data = get_predictions(
                        home_team, away_team, q_scores, self.trained_data,
                        pre_game_cache=self.pre_game_cache, live_signals=live_signals, silent_mode=True
                    )
                    
                    # 4. 🤖 METAMODELO YA APLICADO EN get_predictions
                    # La función get_predictions (live_mode.py) ya aplica:
                    # 1. Chronos Metamodel
                    # 2. Safeguards (Physics Anchor)
                    # 3. Alerts
                    
                    final_prediction = base_prediction
                    meta_adjustment = 0.0 # Ya incluido en base_prediction
                    meta_metadata = {} # Se podría extraer de alerts_data si fuera necesario


                    # ✅ EXPORTAR PREDICCIÓN AUTOMÁTICAMENTE DESPUÉS DE GENERARLA
                    try:
                        from metamodel.prediction_tracker import PredictionTracker
                        
                        # Calcular confidence y edge para el tracking
                        confidence = 0.75  # Valor por defecto para live
                        if bookie_line:
                            edge = abs(final_prediction - bookie_line)
                            recommended_side = "OVER" if final_prediction > bookie_line else "UNDER"
                        else:
                            edge = 0.0
                            recommended_side = "NEUTRAL"
                        
                        # Registrar la predicción con todos los parámetros requeridos
                        tracker = PredictionTracker()
                        prediction_id = tracker.track_prediction(
                            home_team=home_team,
                            away_team=away_team,
                            odds_line=bookie_line,
                            model_prediction=base_prediction,  # Predicción base del modelo XGBoost
                            context_adjusted_prediction=final_prediction,  # Predicción final (Metamodelo)
                            final_prediction=final_prediction,  # Predicción final
                            predictions_list=predictions_list,  # Lista de probabilidades de las 3 líneas
                            confidence=confidence,
                            recommended_side=recommended_side,
                            edge_magnitude=edge,
                            prediction_timing=None,  # Auto-detectar timing basado en quarter_scores
                            context={
                                'quarter_scores': q_scores,
                                'live_alerts': alerts_data.get('live_alerts', []),
                                'update_count': update_num,
                                'prediction_stability': update_reason,
                                'meta_adjustment': meta_adjustment,
                                'meta_metadata': meta_metadata
                            },
                            league=self.trained_data.get('league_name', 'NBA')
                        )
                        print("[OK] Predicción exportada automáticamente")
                    except Exception as e:
                        print(f"[WARN] Error exportando predicción: {e}")
                        # No fallar el monitoreo por errores de exportación

                    # Aplicar estabilidad a la predicción FINAL
                    stabilized_prediction = self._apply_prediction_stability(final_prediction, q_scores, game_info)

                    # Calcular cambios significativos
                    change_info = self._calculate_prediction_changes(stabilized_prediction)

                    # Extraer contexto live
                    live_context = live_data.get('context', {})
                    live_alerts = alerts_data.get('live_alerts', [])

                    # Mostrar update con información de tiempo y DUAL OUTPUT
                    time_context = live_data.get('time_info', {})
                    self._show_live_update(
                        stabilized_prediction, live_data, game_info, change_info,
                        live_context, live_alerts, bookie_line, {
                            'update_count': update_num,
                            'total_updates': max_updates,
                            'stability_reason': update_reason,
                            'time_info': time_context,
                            'base_prediction': base_prediction,
                            'meta_adjustment': meta_adjustment
                        }
                    )

                    # Actualizar tracking
                    self.last_prediction = stabilized_prediction
                    self.last_q_scores = q_scores.copy()
                    self.update_count = update_num

                else:
                    # No actualizar predicción - mostrar mensaje de estabilidad
                    print(f"🛡️ PREDICCIÓN ESTABLE: {update_reason}")
                    print(f"   📊 Última predicción: {self.last_prediction:.1f} pts")

                    # Mostrar estado aún cuando no recalculamos predicción
                    try:
                        ctx = live_data.get('context', {}) if isinstance(live_data, dict) else {}
                        quarter = ctx.get('estimated_quarter', 'Live')
                        quarter_status = self._detect_quarter_status(game_info)
                        stability_indicator = "[OK]" if not quarter_status['in_progress'] else "[LIVE]"
                        quarter_display = f"{quarter} {stability_indicator}"
                        print(f"[TIME] Estado: {quarter_display}")
                    except Exception:
                        pass

                    print("-" * 60)

                    # Calcular intervalo adaptativo
                    adaptive_interval = self._calculate_adaptive_update_interval(game_info)
                    if adaptive_interval != update_interval:
                        print(f"[TIME] Intervalo adaptado: {adaptive_interval}s (cuarto en progreso)")
                        update_interval = adaptive_interval
                
                # 8. Verificar si el juego terminó
                if live_data.get('game_finished', False):
                    print("[END] JUEGO TERMINADO")
                    break
                
                # 9. Calcular y usar intervalo adaptativo
                adaptive_interval = self._calculate_adaptive_update_interval(game_info)
                actual_interval = adaptive_interval if adaptive_interval != update_interval else update_interval

                if adaptive_interval != update_interval:
                    print(f"[TIME] Intervalo adaptado: {actual_interval}s")
                    update_interval = adaptive_interval

                # 🆕 SINGLE SHOT EXIT
                if single_shot:
                    print("\n[SINGLE SHOT] Predicción completada - Regresando al menú")
                    break

                await asyncio.sleep(actual_interval)
                
            except KeyboardInterrupt:
                print("\n⌨️ Monitoreo interrumpido por usuario")
                break
            except Exception as e:
                print(f"[ERROR] Error en update #{update_num}: {e}")
                await asyncio.sleep(update_interval)
                continue
        
        print(f"\n[END] MONITOREO COMPLETADO")
        print(f"Total updates: {self.update_count}")
        print(f"Duration: {(time.time() - self.session_start_time)/60:.1f} minutes")
    
    def _show_live_update(self, prediction: float, live_data: Dict, game_info: Dict,
                           change_info: Dict, live_context: Dict, live_alerts: List[str],
                           bookie_line: float, result: Dict):
        """📺 Muestra update live - versión con estabilidad mejorada"""

        # Scores actuales - usar datos frescos del game_info en lugar de calcular de quarters
        home_score = game_info.get('home_score', live_data['q_scores']['q1_home'] + live_data['q_scores']['q2_home'] + live_data['q_scores']['q3_home'])
        away_score = game_info.get('away_score', live_data['q_scores']['q1_away'] + live_data['q_scores']['q2_away'] + live_data['q_scores']['q3_away'])
        current_total = home_score + away_score

        home_team = game_info['home_team']
        away_team = game_info['away_team']

        print(f"Match: {home_team} {home_score} - {away_team} {away_score}")
        print(f"Total: {current_total} pts")

        # 🛡️ MOSTRAR ESTADO DE ESTABILIDAD
        stability_reason = result.get('stability_reason', '')
        if stability_reason:
            if "Cambio mínimo" in stability_reason or "cuarto en progreso" in stability_reason.lower():
                print(f"[STABLE] {stability_reason}")
            else:
                print(f"[UPDATE] {stability_reason}")

        # ANÁLISIS SIMPLIFICADO PARA PREDICTORES
        base_prediction = result.get('base_prediction', prediction)
        meta_adjustment = result.get('meta_adjustment', 0.0)
        
        print(f"\n[STATS] Base Model (XGBoost): {base_prediction:.1f} pts")
        if abs(meta_adjustment) > 0.1:
            sign = "+" if meta_adjustment >= 0 else ""
            print(f"[AI] Metamodel Adj: {sign}{meta_adjustment:.1f} pts")
        
        if bookie_line:
            self._show_simple_analysis(prediction, bookie_line)
        else:
            # Predicción simple si no hay línea
            print(f"[FINAL] Final Prediction: {prediction:.1f} pts")
            if change_info['significant_prediction']:
                change = change_info['prediction_change']
                direction = "UP" if change > 0 else "DOWN"
                print(f"[{direction}] Cambio: {change:+.1f} pts")

        # 🆕 Contexto live con información de tiempo precisa
        time_info = result.get('time_info', {})
        if time_info:
            current_quarter = time_info.get('current_quarter_display', 'Live')
            minutes_played = time_info.get('minutes_played', 0)
            remaining_minutes = time_info.get('total_remaining_minutes', 0)
            completion_pct = time_info.get('completion_percentage', 0)
            current_pace = time_info.get('current_pace_per_minute', 0)

            quarter_display = f"{current_quarter} ({minutes_played:.1f}min jugados, {remaining_minutes:.1f}min restantes)"
            print(f"[TIME] Estado preciso: {quarter_display}")
            print(f"[PROG] Progreso: {completion_pct:.1f}% completado | Pace actual: {current_pace:.1f} pts/min")
        else:
            # Fallback al método anterior
            quarter = live_context.get('estimated_quarter', 'Live')
            quarter_status = self._detect_quarter_status(game_info)
            stability_indicator = "[OK]" if not quarter_status['in_progress'] else "[LIVE]"
            quarter_display = f"{quarter} {stability_indicator}"
            print(f"[TIME] Estado estimado: {quarter_display}")
 
        # 🚨 Alertas centralizadas: deltas por equipo + top 3 alertas live
        alerts_lines = []

        # Calcular deltas por equipo desde el último update usando scores totales
        try:
            last_home = self.last_game_info.get('home_score') if hasattr(self, 'last_game_info') and self.last_game_info else None
            last_away = self.last_game_info.get('away_score') if hasattr(self, 'last_game_info') and self.last_game_info else None
        except Exception:
            last_home, last_away = None, None

        delta_home = home_score - last_home if last_home is not None else 0
        delta_away = away_score - last_away if last_away is not None else 0

        # Guardar game_info actual para el próximo update
        self.last_game_info = game_info.copy()

        # Cambio de predicción desde el último update
        pred_change = 0.0
        if self.last_prediction is not None:
            try:
                pred_change = prediction - self.last_prediction
            except Exception:
                pred_change = 0.0

        # Formatear deltas por equipo
        def fmt_delta(val: int) -> str:
            sign = "+" if val >= 0 else ""
            return f"{sign}{val}"

        def fmt_pred_change(pc: float) -> str:
            sign = "+" if pc >= 0 else ""
            return f"{sign}{pc:.1f}"

        # Solo mostrar deltas si hay cambios significativos
        if last_home is not None and abs(delta_home) >= 1:
            alerts_lines.append(f"• {home_team}: {fmt_delta(delta_home)} puntos — ajuste del modelo: {fmt_pred_change(pred_change)} pts")
        if last_away is not None and abs(delta_away) >= 1:
            alerts_lines.append(f"• {away_team}: {fmt_delta(delta_away)} puntos — ajuste del modelo: {fmt_pred_change(pred_change)} pts")

        # Añadir top 3 alertas live
        for alert in (live_alerts or [])[:3]:
            alerts_lines.append(f"• {alert}")

        if alerts_lines:
            print("🚨 Alertas:")
            for line in alerts_lines:
                print(f"   {line}")

        print("-" * 60)

    def _calculate_prediction_changes(self, current_prediction: float) -> Dict:
        """📊 Calcula cambios en predicción"""
        if self.last_prediction is None:
            return {
                'prediction_change': 0.0,
                'significant_prediction': False,
                'first_prediction': True
            }

        change = current_prediction - self.last_prediction
        significant = abs(change) >= self.alert_thresholds['prediction_change']

        return {
            'prediction_change': change,
            'significant_prediction': significant,
            'first_prediction': False
        }

    def _should_update_prediction(self, q_scores: Dict, game_info: Dict) -> Tuple[bool, str]:
        """
        🛡️ DETERMINA SI DEBE ACTUALIZARSE LA PREDICCIÓN
        Solo actualizar si hay cambios significativos en el score
        """
        if self.last_q_scores is None:
            return True, "Primera actualización"

        # Calcular cambio total en scores
        current_home = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        current_away = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']

        last_home = self.last_q_scores['q1_home'] + self.last_q_scores['q2_home'] + self.last_q_scores['q3_home']
        last_away = self.last_q_scores['q1_away'] + self.last_q_scores['q2_away'] + self.last_q_scores['q3_away']

        total_change = abs(current_home - last_home) + abs(current_away - last_away)
        min_change = self.stability_config['min_score_change_for_update']

        if total_change < min_change:
            return False, f"Cambio mínimo no alcanzado ({total_change:.0f} < {min_change} pts)"

        # Verificar si estamos en un cuarto en progreso
        quarter_status = self._detect_quarter_status(game_info)
        if quarter_status['in_progress']:
            # Ser más conservador durante cuarto en progreso
            if total_change < min_change * 1.5:  # Requiere cambio mayor
                return False, f"Cuarto en progreso - cambio insuficiente ({total_change:.0f} pts)"

        return True, f"Cambio significativo detectado (+{total_change:.0f} pts)"

    def _apply_prediction_stability(self, raw_prediction: float, q_scores: Dict, game_info: Dict) -> float:
        """
        🛡️ APLICA ESTABILIDAD A LA PREDICCIÓN
        Amortigua cambios grandes y considera contexto del cuarto
        """
        if self.last_prediction is None:
            return raw_prediction

        change = raw_prediction - self.last_prediction
        abs_change = abs(change)

        # Aplicar damping base
        damping = self.stability_config['prediction_damping_factor']
        damped_change = change * damping

        # Aplicar límite máximo de cambio
        max_swing = self.stability_config['max_prediction_swing']
        if abs(damped_change) > max_swing:
            damped_change = max_swing if damped_change > 0 else -max_swing

        # Penalizar estabilidad durante cuarto en progreso
        quarter_status = self._detect_quarter_status(game_info)
        if quarter_status['in_progress']:
            penalty = self.stability_config['quarter_in_progress_penalty']
            damped_change *= penalty

        stabilized_prediction = self.last_prediction + damped_change

        # Logging de estabilización
        if abs(change) > 1.0:  # Solo log cambios significativos
            print(f"🛡️ Estabilización: {change:+.1f} → {damped_change:+.1f} pts "
                  f"(damping: {damping:.1f}, max: ±{max_swing:.1f})")

        return stabilized_prediction

    def _detect_quarter_status(self, game_info: Dict) -> Dict:
        """
        🎯 DETECTA ESTADO DEL CUARTO CON MEJOR PRECISIÓN
        """
        status = game_info.get('status', '').lower()
        quarter_info = game_info.get('quarter_info', {})

        # Estado base
        in_progress = False
        current_quarter = 'unknown'
        confidence = 'low'

        # Detectar por status string
        if 'halftime' in status:
            in_progress = False
            current_quarter = 'halftime'
            confidence = 'high'
        elif 'q1' in status or '1st' in status:
            in_progress = True
            current_quarter = 'Q1'
            confidence = 'high'
        elif 'q2' in status or '2nd' in status:
            in_progress = True
            current_quarter = 'Q2'
            confidence = 'high'
        elif 'q3' in status or '3rd' in status:
            in_progress = True
            current_quarter = 'Q3'
            confidence = 'high'
        elif 'q4' in status or '4th' in status:
            in_progress = True
            current_quarter = 'Q4'
            confidence = 'high'
        elif 'finished' in status or 'final' in status:
            in_progress = False
            current_quarter = 'finished'
            confidence = 'high'
        else:
            # Fallback: estimar por quarter_info
            if quarter_info:
                # Si tenemos quarter_info, usar para detectar
                q3_home = quarter_info.get('q3_home', 0)
                q3_away = quarter_info.get('q3_away', 0)
                q2_home = quarter_info.get('q2_home', 0)
                q2_away = quarter_info.get('q2_away', 0)

                if q3_home > 0 or q3_away > 0:
                    in_progress = True
                    current_quarter = 'Q3'
                    confidence = 'medium'
                elif q2_home > 0 or q2_away > 0:
                    in_progress = True
                    current_quarter = 'Q2'
                    confidence = 'medium'
                else:
                    in_progress = False
                    current_quarter = 'Q1'
                    confidence = 'low'

        return {
            'in_progress': in_progress,
            'current_quarter': current_quarter,
            'confidence': confidence,
            'status_string': status
        }


    def _calculate_adaptive_update_interval(self, game_info: Dict) -> int:
        """
        ⏱️ CALCULA INTERVALO DE ACTUALIZACIÓN ADAPTATIVO
        Más frecuente en descansos, menos frecuente durante cuarto
        """
        quarter_status = self._detect_quarter_status(game_info)

        if quarter_status['in_progress']:
            return self.stability_config['update_frequency_quarter_active']
        else:
            return self.stability_config['update_frequency_quarter_break']

    async def process_game_update(self, game_id: str, game_info: Dict) -> Optional[Dict]:
        """
        🎯 Procesa update de juego específico usando SofaScoreClient + código existente
        🚀 OPTIMIZADO: Soporta caché pre-partido para ~70% menos CPU
        """
        self.update_count += 1

        # 🚀 CONSTRUIR CACHÉ PRE-PARTIDO UNA SOLA VEZ (si no existe)
        if not hasattr(self, 'pre_game_cache') or self.pre_game_cache is None:
            try:
                from analysis.live_mode import build_pre_game_cache
                self.pre_game_cache = build_pre_game_cache(
                    game_info['home_team'],
                    game_info['away_team'],
                    self.trained_data
                )
                print("✅ Caché pre-partido construido para process_game_update")
            except Exception as e:
                print(f"⚠️ Error construyendo caché: {e} - continuando sin caché")
                self.pre_game_cache = None

        # Silenciar los mensajes de debug después de la tercera actualización
        if self.update_count > 3:
            self.client.debug_mode = False

        try:
            # 1. Get live game data including time info
            live_data = await self.client.get_live_game_data(game_id, game_info)

            if not live_data:
                print("⚠️ No se pudieron obtener datos live del juego")
                return None

            # 2. Get current stats using SofaScoreClient
            current_stats = await self.client.get_game_statistics(game_id)

            if not current_stats:
                print("⚠️ No se pudieron obtener estadísticas")
                return None

            # 3. Obtener info actualizada del juego (scores, status)
            updated_games = await self.client.get_live_basketball_games()
            current_game_info = None

            for game in updated_games or []:
                if game.get('id') == game_id:
                    current_game_info = game
                    break

            if current_game_info:
                # Actualizar game_info con datos frescos
                game_info.update({
                    'home_score': current_game_info.get('home_score', game_info.get('home_score', 0)),
                    'away_score': current_game_info.get('away_score', game_info.get('away_score', 0)),
                    'status': current_game_info.get('status', game_info.get('status', '')),
                    'quarter_info': current_game_info.get('quarter_info', game_info.get('quarter_info', {}))
                })

                # Actualizar q_scores con datos frescos si están disponibles
                fresh_quarter_info = current_game_info.get('quarter_info', {})
                if fresh_quarter_info and any(fresh_quarter_info.get(k, 0) > 0 for k in ['q1_home', 'q1_away', 'q2_home', 'q2_away', 'q3_home', 'q3_away']):
                    q_scores.update(fresh_quarter_info)

            # 4. Convertir stats API → formato modelo
            q_scores = self._convert_stats_to_quarters(current_stats, game_info)

            # Parse stats from SofaScore API format to model format
            if current_stats and 'statistics' in current_stats:
                parsed_stats = self.client._parse_massive_stats_json(current_stats['statistics'])
            else:
                parsed_stats = {}

            # Calculate advanced live signals with time awareness
            time_info = live_data.get('time_info', {})
            live_signals = self.compute_live_signals(parsed_stats or {}, q_scores, game_info, time_info=time_info)
            
            # 🆕 PASAR ESTADÍSTICAS ACTUALES EN live_signals PARA INTEGRACIÓN
            if parsed_stats:
                live_signals['current_stats'] = parsed_stats
                print(f"✅ [STATS INTEGRATION] {len(parsed_stats)} estadísticas pasadas a live_signals")

            # 4. Generar predicción usando función existente (con caché y señales live)
            get_predictions = self._get_prediction_function()
            prediction, predictions_list, alerts_data = get_predictions(
                game_info['home_team'], game_info['away_team'], q_scores, self.trained_data,
                pre_game_cache=self.pre_game_cache, live_signals=live_signals, silent_mode=True
            )
            
            # 5. Detectar cambios significativos
            change_info = self._detect_significant_changes(prediction, current_stats, q_scores)
            
            # 6. Generar alertas live específicas con señales avanzadas
            live_alerts = self._generate_live_alerts(change_info, current_stats, game_info, live_signals, alerts_data)
            
            # 7. Agregar contexto live adicional
            live_context = self._calculate_live_context(current_stats, q_scores, game_info)
            
            # 8. Actualizar cache
            self.last_prediction = prediction
            self.last_stats = current_stats
            self.last_q_scores = q_scores
            
            return {
                'prediction': prediction,
                'predictions_list': predictions_list,
                'alerts_data': alerts_data,
                'live_alerts': live_alerts,
                'change_info': change_info,
                'live_context': live_context,
                'current_stats': current_stats,
                'q_scores': q_scores,
                'game_info': game_info,
                'update_count': self.update_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error procesando predicción: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_stats_to_quarters(self, api_stats: Dict, game_info: Dict) -> Dict[str, int]:
        """
        📄 Convierte stats API → formato q_scores para código existente
        """
        # Usar quarter info extraída por SofaScoreClient
        quarter_info = game_info.get('quarter_info', {})
        
        q_scores = {
            'q1_home': quarter_info.get('q1_home', 0),
            'q1_away': quarter_info.get('q1_away', 0),
            'q2_home': quarter_info.get('q2_home', 0),
            'q2_away': quarter_info.get('q2_away', 0),
            'q3_home': quarter_info.get('q3_home', 0),
            'q3_away': quarter_info.get('q3_away', 0),
            'q4_home': quarter_info.get('q4_home', 0),
            'q4_away': quarter_info.get('q4_away', 0)
        }
        
        # Si no tenemos quarter info, usar scores actuales como fallback
        if all(score == 0 for score in q_scores.values()):
            home_score = game_info.get('home_score', 0)
            away_score = game_info.get('away_score', 0)
            
            # Distribución estimada por cuartos
            q_scores['q1_home'] = int(home_score * 0.25)
            q_scores['q1_away'] = int(away_score * 0.25)
            q_scores['q2_home'] = int(home_score * 0.25)
            q_scores['q2_away'] = int(away_score * 0.25)
        
        return q_scores

    # Métodos de compatibilidad (mantenidos sin cambios)
    def _check_league_compatibility(self, detected_league: str) -> float:
        """Verifica compatibilidad de liga"""
        model_league = self.trained_data.get('league_name', '').lower()
        detected_lower = detected_league.lower()

        if model_league in detected_lower or detected_lower in model_league:
            return 1.0
        elif 'nba' in model_league and 'nba' in detected_lower:
            return 1.0
        elif 'wnba' in model_league and 'wnba' in detected_lower:
            return 1.0
        elif 'euroleague' in model_league and 'euro' in detected_lower:
            return 0.8
        else:
            return 0.3
    
    def _check_teams_compatibility(self, home_team: str, away_team: str) -> float:
        """Verifica si conocemos estos equipos"""
        model_teams = [name.lower() for name in self.trained_data.get('team_names', [])]
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        home_found = any(team in home_lower or home_lower in team for team in model_teams)
        away_found = any(team in away_lower or away_lower in team for team in model_teams)
        
        if home_found and away_found:
            return 1.0
        elif home_found or away_found:
            return 0.7
        else:
            return 0.4
    
    def _estimate_remaining_quarters(self, game: Dict) -> int:
        """Estima cuartos restantes - VERSIÓN ROBUSTA para ambos formatos"""
        
        # Intentar usar status_obj (dict) primero
        status_obj = game.get('status_obj', {})
        period = None
        
        if isinstance(status_obj, dict) and status_obj:
            period = status_obj.get('period', None)
            status_type = status_obj.get('type', '')
            
            # Si tenemos período numérico, usarlo
            if period is not None:
                try:
                    p = int(period)
                    if p <= 1:
                        return 3  # Q1 -> 3 cuartos restantes
                    elif p == 2:
                        return 2  # Q2 -> 2 cuartos restantes  
                    elif p == 3:
                        return 1  # Q3 -> 1 cuarto restante
                    elif p >= 4:
                        return 0  # Q4+ -> juego casi terminado
                except (ValueError, TypeError):
                    pass
            
            # Si no hay period pero hay type
            if status_type == 'finished':
                return 0
            elif status_type == 'inprogress':
                return 2  # Estimación conservadora
        
        # Fallback: usar status (string) con heurísticas de texto
        status_str = str(game.get('status', '')).lower()
        
        if 'final' in status_str or 'finished' in status_str:
            return 0
        elif 'halftime' in status_str:
            return 2
        elif '4th' in status_str or 'quarter 4' in status_str or 'q4' in status_str:
            return 0
        elif '3rd' in status_str or 'quarter 3' in status_str or 'q3' in status_str:
            return 1  
        elif '2nd' in status_str or 'quarter 2' in status_str or 'q2' in status_str:
            return 2
        elif '1st' in status_str or 'quarter 1' in status_str or 'q1' in status_str:
            return 3
        else:
            # Default: asumir que está en progreso con 2 cuartos restantes
            return 2
    
    def _calculate_compatibility_score(self, league_compat: float, teams_compat: float,
                                     remaining_quarters: int, game: Dict) -> float:
        """Calcula score de compatibilidad total"""
        base_score = (league_compat * 0.4) + (teams_compat * 0.4)
        
        # Bonus por cuartos restantes
        quarter_bonus = min(remaining_quarters * 0.05, 0.2)
        
        return min(base_score + quarter_bonus, 1.0)
    
    def _detect_significant_changes(self, prediction: float, current_stats: Dict, q_scores: Dict) -> Dict:
        """Detecta cambios significativos"""
        changes = {
            'prediction_changed': False,
            'stats_changed': False,
            'scoring_run': False,
            'significant_events': []
        }
        
        if self.last_prediction:
            pred_change = abs(prediction - self.last_prediction)
            changes['prediction_changed'] = pred_change >= 2.0
            
        if self.last_q_scores:
            # Detectar rachas de puntuación
            home_change = (q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']) - \
                         (self.last_q_scores['q1_home'] + self.last_q_scores['q2_home'] + self.last_q_scores['q3_home'])
            away_change = (q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']) - \
                         (self.last_q_scores['q1_away'] + self.last_q_scores['q2_away'] + self.last_q_scores['q3_away'])
            
            if home_change >= 6:
                changes['significant_events'].append(f"🏀 Racha local: +{home_change} pts")
            if away_change >= 6:
                changes['significant_events'].append(f"🏀 Racha visitante: +{away_change} pts")
        
        return changes
    
    def _generate_live_alerts(self, change_info: Dict, current_stats: Dict, game_info: Dict,
                              live_signals: Dict[str, float] = None, alerts_data: Dict = None) -> List[str]:
        """Genera alertas live específicas incluyendo señales avanzadas"""
        alerts = []

        if change_info.get('prediction_changed'):
            alerts.append("📊 Cambio significativo en predicción")

        for event in change_info.get('significant_events', []):
            alerts.append(event)

        # Alertas de estado del juego
        status = game_info.get('status', '')
        if 'timeout' in status.lower():
            alerts.append("⸗ Timeout en el juego")
        elif 'halftime' in status.lower():
            alerts.append("🏃 Medio tiempo")

        # 🆕 FASE 1: Alertas basadas en señales live avanzadas
        if live_signals:
            # Alerta de run detector
            if live_signals.get('run_active', False):
                run_side = live_signals.get('run_side', 'none')
                run_strength = live_signals.get('run_strength', 0.0)
                if run_strength > 0.5:
                    alerts.append(f"🏃 RUN DETECTADO ({run_side}): Racha fuerte (+{run_strength:.1f})")
                elif run_strength > 0.3:
                    alerts.append(f"🏃 Run detectado ({run_side}): Racha moderada (+{run_strength:.1f})")

            # Alerta de foul trouble
            home_fti = live_signals.get('home_fti', 0.5)
            away_fti = live_signals.get('away_fti', 0.5)
            max_fti = max(home_fti, away_fti)

            if max_fti > 0.8:
                alerts.append("🏀 FOUL TROUBLE EXTREMO: Alto riesgo de tiros libres")
            elif max_fti > 0.7:
                alerts.append("🏀 Foul trouble alto: Más FT esperados")

            # Alerta de FT heavy game
            home_fta = current_stats.get('home_free_throws_attempted', 0)
            away_fta = current_stats.get('away_free_throws_attempted', 0)
            total_fta = home_fta + away_fta

            if total_fta > 30:  # Más de 30 FT intentados
                alerts.append("🎯 FT HEAVY GAME: Muchos tiros libres esperados")
            elif total_fta > 20:
                alerts.append("🎯 Juego con muchos FT: Estrategia fouling activa")

            # Alerta de shooting efficiency extrema
            diff_ts = live_signals.get('diff_ts_live', 0.0)
            if abs(diff_ts) > 0.2:
                direction = "superior" if diff_ts > 0 else "inferior"
                alerts.append(f"🎯 Shooting efficiency {direction}: Diferencia TS% significativa ({diff_ts:.2f})")

            # 🆕 FASE 2: Alertas O/U específicas

            # Alerta de A/TO ratio extrema - AFECTA TOTALES
            diff_ato = live_signals.get('diff_ato_ratio', 0.0)
            if abs(diff_ato) > 0.3:
                direction = "mejor control" if diff_ato > 0 else "peor control"
                alerts.append(f"🏀 Control de balón {direction}: Más posesiones eficientes ({diff_ato:.2f})")

            # Alerta de rebound efficiency extrema - AFECTA TOTALES
            home_treb_diff = live_signals.get('home_treb_diff', 0.0)
            if abs(home_treb_diff) > 0.5:
                direction = "dominando rebotes" if home_treb_diff > 0 else "siendo dominado en rebotes"
                alerts.append(f"🏀 Rebotes {direction}: Más posesiones disponibles ({home_treb_diff:.2f})")

            # Alerta de OREB% extrema - AFECTA TOTALES
            diff_oreb = live_signals.get('diff_oreb_pct', 0.0)
            if abs(diff_oreb) > 0.15:
                direction = "superior" if diff_oreb > 0 else "inferior"
                alerts.append(f"🏀 Rebotes ofensivos {direction}: Más segundas oportunidades ({diff_oreb:.1%})")

        # Add pre-game alerts if available
        if alerts_data and alerts_data.get('pre_game_alerts'):
            alerts.extend([f"📋 {alert}" for alert in alerts_data['pre_game_alerts'][:2]])

        return alerts
    
    def _calculate_live_context(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict:
        """Calcula contexto live del juego"""
        context = {
            'estimated_quarter': 'Live',
            'estimated_pace': 0,
            'offensive_efficiency': 0,
            'total_points': 0
        }
        
        # Calcular total actual
        home_total = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        away_total = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
        context['total_points'] = home_total + away_total
        
        # Estimar cuarto actual basado en scores
        if q_scores['q3_home'] > 0 or q_scores['q3_away'] > 0:
            context['estimated_quarter'] = 'Q3+'
        elif q_scores['q2_home'] > 0 or q_scores['q2_away'] > 0:
            context['estimated_quarter'] = 'Q2+'
        else:
            context['estimated_quarter'] = 'Q1'
        
        # Estimar pace (puntos por minuto aproximado)
        if context['estimated_quarter'] == 'Q3+':
            minutes_played = 30  # Aproximadamente
            context['estimated_pace'] = (context['total_points'] / minutes_played) * 48
        elif context['estimated_quarter'] == 'Q2+':
            minutes_played = 20
            context['estimated_pace'] = (context['total_points'] / minutes_played) * 48
        
        return context

    def compute_live_signals(self, current_stats: Dict, q_scores: Dict, game_info: Dict,
                            historical_baseline: Dict = None, time_info: Dict = None) -> Dict[str, float]:
        """
        🎯 FASE 2: Señales live enfocadas en OVER/UNDER totals
        - Foul Trouble Index (FTI) - Más FTs = Más puntos
        - In-game Shooting Efficiency (TS%, eFG%) - Mejor tiro = Más puntos
        - Run Detector - Rachas de anotación = Más puntos
        🆕 FASE 2: Señales O/U específicas usando stats completos de SofaScore
        - Assist-to-Turnover Ratio (A/TO) - Mejor control = Más posesiones eficientes
        - Rebound Efficiency metrics - Más rebotes = Más posesiones = Más puntos
        """
        signals = {}

        try:
            # Foul Trouble Index (FTI) - Critical for O/U
            fti_signals = self._calculate_foul_trouble_index(current_stats, q_scores, game_info, historical_baseline, time_info)
            signals.update(fti_signals)

            # In-game Shooting Efficiency - Critical for O/U
            shooting_signals = self._calculate_live_shooting_efficiency(current_stats, q_scores, game_info)
            signals.update(shooting_signals)

            # Run Detector - Critical for O/U
            run_signals = self._calculate_run_detector(q_scores, game_info)
            signals.update(run_signals)

            # Phase 2: O/U Specific Signals
            ou_signals = self._calculate_live_ou_signals(current_stats)
            signals.update(ou_signals)

            # [NEW] Add time_info to signals for downstream use
            if time_info:
                signals['time_info'] = time_info

        except Exception as e:
            print(f"Error calculating live signals: {e}")

        return signals

    def _calculate_live_ou_signals(self, current_stats: Dict) -> Dict:
        """Calcula señales específicas O/U: A/TO, Rebounds, etc."""
        
        # Assist to Turnover Ratio
        home_ast = current_stats.get('home_assists', 0)
        home_to = current_stats.get('home_turnovers', 0)
        home_ato = home_ast / max(1, home_to)
        
        away_ast = current_stats.get('away_assists', 0)
        away_to = current_stats.get('away_turnovers', 0)
        away_ato = away_ast / max(1, away_to)
        
        # Rebound Efficiency
        home_treb = current_stats.get('home_total_rebounds', 0)
        away_treb = current_stats.get('away_total_rebounds', 0)
        
        # Normalizar diferencia de rebotes (aprox 50 rebotes por juego)
        # Diferencia simple de rebotes totales
        home_treb_diff = (home_treb - away_treb)
        
        # OREB % (estimado)
        home_orb = current_stats.get('home_offensive_rebounds', 0)
        away_drb = current_stats.get('away_defensive_rebounds', 0)
        home_oreb_pct = home_orb / max(1, home_orb + away_drb)
        
        away_orb = current_stats.get('away_offensive_rebounds', 0)
        home_drb = current_stats.get('home_defensive_rebounds', 0)
        away_oreb_pct = away_orb / max(1, away_orb + home_drb)
        
        return {
            'home_ato_ratio': home_ato,
            'away_ato_ratio': away_ato,
            'diff_ato_ratio': home_ato - away_ato,
            'home_treb_diff': home_treb_diff,
            'home_oreb_pct': home_oreb_pct,
            'away_oreb_pct': away_oreb_pct,
            'diff_oreb_pct': home_oreb_pct - away_oreb_pct
        }


    def _calculate_foul_trouble_index(self, current_stats, q_scores, game_info, historical_baseline, time_info):
        """Calcula el índice de problemas de faltas (FTI)"""
        # Intentar datos nativos de faltas
        home_pf = current_stats.get('home_personal_fouls', 0)
        away_pf = current_stats.get('away_personal_fouls', 0)

        # 🆕 Usar tiempo preciso si está disponible, sino estimar
        if time_info and time_info.get('minutes_played', 0) > 0:
            minutes_played = time_info['minutes_played']
        else:
            minutes_played = self._estimate_minutes_played(q_scores, game_info)

        if minutes_played > 0:
            # Si tenemos datos nativos de faltas
            if home_pf > 0 or away_pf > 0:
                home_pf_rate = home_pf / minutes_played
                away_pf_rate = away_pf / minutes_played

                # Normalizar vs baseline liga (usar percentiles típicos NBA/WNBA)
                # NBA/WNBA: promedio ~4-5 faltas por cuarto (12 min) = ~0.17-0.21 pf/min
                # Alto foul trouble: >0.25 pf/min, Extremo: >0.33 pf/min
                home_fti = min(1.0, max(0.0, (home_pf_rate - 0.17) / 0.25))  # p50=0.17, p90=0.42
                away_fti = min(1.0, max(0.0, (away_pf_rate - 0.17) / 0.25))
            else:
                # Fallback proxy: usar FT_rate como indicador de foul trouble
                home_fti, away_fti = self._calculate_fti_proxy(current_stats, minutes_played)
        else:
            home_fti = away_fti = 0.5

        # Aplicar modificadores contextuales
        home_fti = self._apply_fti_modifiers(home_fti, q_scores, game_info, 'home')
        away_fti = self._apply_fti_modifiers(away_fti, q_scores, game_info, 'away')

        return {
            'home_fti': home_fti,
            'away_fti': away_fti,
            'diff_fti': home_fti - away_fti
        }

    def _calculate_fti_proxy(self, current_stats: Dict, minutes_played: float) -> Tuple[float, float]:
        """Proxy FTI usando FT_rate cuando no hay datos de faltas"""

        home_fta = current_stats.get('home_free_throws_attempted', 0)
        away_fta = current_stats.get('away_free_throws_attempted', 0)

        if minutes_played > 0:
            home_ft_rate = home_fta / minutes_played
            away_ft_rate = away_fta / minutes_played

            # FT_rate alto indica foul trouble (más FT por faltas)
            # NBA/WNBA: promedio ~1-1.5 FT/min, Alto: >2.0 FT/min
            home_fti_proxy = min(1.0, max(0.0, (home_ft_rate - 1.0) / 2.0))  # p50=1.0, p90=3.0
            away_fti_proxy = min(1.0, max(0.0, (away_ft_rate - 1.0) / 2.0))
        else:
            home_fti_proxy = away_fti_proxy = 0.5

        return home_fti_proxy, away_fti_proxy

    def _apply_fti_modifiers(self, base_fti: float, q_scores: Dict, game_info: Dict, team_side: str) -> float:
        """Aplica modificadores contextuales al FTI base"""

        modified_fti = base_fti

        # Modificador por cuarto (más importante en Q4)
        quarter_stage = game_info.get('status', '').lower()
        if 'q4' in quarter_stage or '4th' in quarter_stage:
            modified_fti *= 1.2  # +20% importancia en Q4
        elif 'q3' in quarter_stage or '3rd' in quarter_stage:
            modified_fti *= 1.1  # +10% importancia en Q3

        # Modificador por diferencia de score (más fouls cuando juego cerrado)
        home_score = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        away_score = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
        score_diff = abs(home_score - away_score)

        if score_diff <= 10:  # Juego cerrado
            modified_fti *= 1.15  # +15% importancia

        return min(1.0, max(0.0, modified_fti))

    def _calculate_live_shooting_efficiency(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict[str, float]:
        """Calcula TS% y eFG% live con shrinkage para estabilidad"""

        # Extraer stats necesarias
        home_fgm = current_stats.get('home_field_goals_made', 0)
        home_fga = current_stats.get('home_field_goals_attempted', 0)
        home_3pm = current_stats.get('home_3_point_field_goals_made', 0)
        home_ftm = current_stats.get('home_free_throws_made', 0)
        home_fta = current_stats.get('home_free_throws_attempted', 0)

        away_fgm = current_stats.get('away_field_goals_made', 0)
        away_fga = current_stats.get('away_field_goals_attempted', 0)
        away_3pm = current_stats.get('away_3_point_field_goals_made', 0)
        away_ftm = current_stats.get('away_free_throws_made', 0)
        away_fta = current_stats.get('away_free_throws_attempted', 0)

        # Debug stats
        # print(f"   [DEBUG] Shooting Stats: Home FGM={home_fgm}, FGA={home_fga}, 3PM={home_3pm}, FTM={home_ftm}, FTA={home_fta}")
        # print(f"   [DEBUG] Shooting Stats: Away FGM={away_fgm}, FGA={away_fga}, 3PM={away_3pm}, FTM={away_ftm}, FTA={away_fta}")

        # Calcular eFG%
        home_efg = (home_fgm + 0.5 * home_3pm) / max(home_fga, 1) if home_fga > 0 else 0.5
        away_efg = (away_fgm + 0.5 * away_3pm) / max(away_fga, 1) if away_fga > 0 else 0.5

        # Calcular TS%
        home_ts_attempts = 2 * (home_fga + 0.44 * home_fta)
        away_ts_attempts = 2 * (away_fga + 0.44 * away_fta)

        # Calcular puntos totales actuales sumando todos los cuartos disponibles
        home_points = sum(q_scores.get(f'q{i}_home', 0) for i in range(1, 5))
        away_points = sum(q_scores.get(f'q{i}_away', 0) for i in range(1, 5))
        
        home_ts = home_points / max(home_ts_attempts, 1) if home_ts_attempts > 0 else 0.5
        away_ts = away_points / max(away_ts_attempts, 1) if away_ts_attempts > 0 else 0.5
        
        # Aplicar shrinkage basado en volumen de tiros
        shrinkage_alpha = min(1.0, max(home_fga, away_fga) / 40.0)  # Confianza aumenta con FGA

        home_efg_adj = 0.5 + shrinkage_alpha * (home_efg - 0.5)
        away_efg_adj = 0.5 + shrinkage_alpha * (away_efg - 0.5)
        home_ts_adj = 0.5 + shrinkage_alpha * (home_ts - 0.5)
        away_ts_adj = 0.5 + shrinkage_alpha * (away_ts - 0.5)

        return {
            'home_ts_live': home_ts_adj,
            'away_ts_live': away_ts_adj,
            'diff_ts_live': home_ts_adj - away_ts_adj,
            'home_efg_live': home_efg_adj,
            'away_efg_live': away_efg_adj,
            'diff_efg_live': home_efg_adj - away_efg_adj
        }

    def _calculate_run_detector(self, q_scores: Dict, game_info: Dict) -> Dict:
        """Detecta rachas de puntuación con ventana deslizante"""

        # Calcular deltas desde último update
        if not hasattr(self, 'last_q_scores') or not self.last_q_scores:
            return {'run_active': False, 'run_side': 'none', 'run_strength': 0.0}

        # Calcular puntos en la última ventana (último cuarto jugado)
        current_home = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        current_away = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']

        last_home = self.last_q_scores['q1_home'] + self.last_q_scores['q2_home'] + self.last_q_scores['q3_home']
        last_away = self.last_q_scores['q1_away'] + self.last_q_scores['q2_away'] + self.last_q_scores['q3_away']

        delta_home = current_home - last_home
        delta_away = current_away - last_away

        # Estimar tiempo transcurrido (aproximado)
        time_delta = self._estimate_time_delta(game_info)

        # Detectar run basado en puntos por tiempo
        run_active = False
        run_side = 'none'
        run_strength = 0.0

        if time_delta > 0:
            home_rate = delta_home / time_delta
            away_rate = delta_away / time_delta

            # Umbrales para detectar run (ajustables)
            run_threshold = 6.0 / time_delta if time_delta <= 180 else 8.0 / time_delta  # 6-8 pts en 3-5 min

            if home_rate >= run_threshold:
                run_active = True
                run_side = 'home'
                run_strength = min(1.0, home_rate / (12.0 / time_delta))  # Max 12 pts en ventana
            elif away_rate >= run_threshold:
                run_active = True
                run_side = 'away'
                run_strength = min(1.0, away_rate / (12.0 / time_delta))

        return {
            'run_active': run_active,
            'run_side': run_side,
            'run_strength': run_strength
        }

    def _estimate_minutes_played(self, q_scores: Dict, game_info: Dict) -> float:
        """Estima minutos jugados basado en scores y status"""

        # Contar cuartos con datos
        quarters_played = 0
        if q_scores['q1_home'] > 0 or q_scores['q1_away'] > 0:
            quarters_played += 1
        if q_scores['q2_home'] > 0 or q_scores['q2_away'] > 0:
            quarters_played += 1
        if q_scores['q3_home'] > 0 or q_scores['q3_away'] > 0:
            quarters_played += 1

        # Estimar minutos (12 min por cuarto NBA/WNBA)
        minutes_played = quarters_played * 12

        # Ajuste por status
        status = game_info.get('status', '').lower()
        if 'halftime' in status:
            minutes_played = 24  # Exactamente halftime
        elif 'q4' in status or '4th' in status:
            minutes_played = min(minutes_played, 36)  # No más de 36 min

        return max(minutes_played, 1)  # Mínimo 1 min

    def _estimate_time_delta(self, game_info: Dict) -> float:
        """Estima tiempo transcurrido desde último update (en segundos)"""

        # Estimación simple basada en status del juego
        status = game_info.get('status', '').lower()

        if 'q1' in status or '1st' in status:
            return 180  # ~3 min ventana típica
        elif 'q2' in status or '2nd' in status:
            return 180
        elif 'halftime' in status:
            return 300  # ~5 min halftime
        elif 'q3' in status or '3rd' in status:
            return 180
        elif 'q4' in status or '4th' in status:
            return 120  # Q4 más rápido
        else:
            return 180  # Default

    # 🆕 FASE 2: NUEVAS SEÑALES LIVE USANDO STATS COMPLETOS DE SOFASCORE

    def _calculate_assist_turnover_ratio(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict[str, float]:
        """Calcula Assist-to-Turnover Ratio (A/TO) - indicador de control de balón"""

        home_assists = current_stats.get('home_assists', 0)
        away_assists = current_stats.get('away_assists', 0)
        home_turnovers = current_stats.get('home_turnovers', 0)
        away_turnovers = current_stats.get('away_turnovers', 0)

        # Calcular ratios con manejo de división por cero
        home_ato = safe_divide(home_assists, max(home_turnovers, 1), 1.0)
        away_ato = safe_divide(away_assists, max(away_turnovers, 1), 1.0)

        # Normalizar vs baseline NBA/WNBA (A/TO promedio ~1.2)
        # Shrinkage para estabilidad con pocos turnovers
        total_turnovers = home_turnovers + away_turnovers
        shrinkage = min(1.0, total_turnovers / 20.0)  # Confianza aumenta con TO totales

        home_ato_norm = 0.5 + shrinkage * (min(home_ato, 3.0) - 1.2) / 1.8  # Normalizar 0-3 range
        away_ato_norm = 0.5 + shrinkage * (min(away_ato, 3.0) - 1.2) / 1.8

        # Clamp a rango 0-1
        home_ato_norm = max(0.0, min(1.0, home_ato_norm))
        away_ato_norm = max(0.0, min(1.0, away_ato_norm))

        return {
            'home_ato_ratio': home_ato_norm,
            'away_ato_ratio': away_ato_norm,
            'diff_ato_ratio': home_ato_norm - away_ato_norm
        }

    def _calculate_rebound_efficiency(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict[str, float]:
        """Calcula métricas de eficiencia en rebotes (OREB%, DREB%, TREB differential)"""

        # Datos de rebotes
        home_oreb = current_stats.get('home_offensive_rebounds', 0)
        away_oreb = current_stats.get('away_offensive_rebounds', 0)
        home_dreb = current_stats.get('home_defensive_rebounds', 0)
        away_dreb = current_stats.get('away_defensive_rebounds', 0)
        home_treb = current_stats.get('home_total_rebounds', home_oreb + home_dreb)
        away_treb = current_stats.get('away_total_rebounds', away_oreb + away_dreb)

        # Calcular porcentajes de rebotes ofensivos
        total_oreb_opportunities = home_oreb + away_oreb
        if total_oreb_opportunities > 0:
            home_oreb_pct = home_oreb / total_oreb_opportunities
            away_oreb_pct = away_oreb / total_oreb_opportunities
        else:
            home_oreb_pct = away_oreb_pct = 0.25  # Default NBA/WNBA ~25%

        # Calcular porcentajes de rebotes defensivos
        total_dreb_opportunities = home_dreb + away_dreb
        if total_dreb_opportunities > 0:
            home_dreb_pct = home_dreb / total_dreb_opportunities
            away_dreb_pct = away_dreb / total_dreb_opportunities
        else:
            home_dreb_pct = away_dreb_pct = 0.75  # Default NBA/WNBA ~75%

        # Calcular diferencial de rebotes totales
        total_reb_diff = home_treb - away_treb

        # Normalizar diferencial (típico rango -20 a +20)
        home_treb_diff_norm = max(-1.0, min(1.0, total_reb_diff / 20.0))
        away_treb_diff_norm = -home_treb_diff_norm

        return {
            'home_oreb_pct': home_oreb_pct,
            'away_oreb_pct': away_oreb_pct,
            'diff_oreb_pct': home_oreb_pct - away_oreb_pct,
            'home_dreb_pct': home_dreb_pct,
            'away_dreb_pct': away_dreb_pct,
            'diff_dreb_pct': home_dreb_pct - away_dreb_pct,
            'home_treb_diff': home_treb_diff_norm,
            'away_treb_diff': away_treb_diff_norm
        }

    def _calculate_steal_block_impact(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict[str, float]:
        """Calcula impacto de steals y blocks en el juego"""

        home_steals = current_stats.get('home_steals', 0)
        away_steals = current_stats.get('away_steals', 0)
        home_blocks = current_stats.get('home_blocks', 0)
        away_blocks = current_stats.get('away_blocks', 0)

        # Combinar steals + blocks como indicador defensivo
        home_sb_total = home_steals + home_blocks
        away_sb_total = away_steals + away_blocks

        # Estimar minutos jugados para normalización
        minutes_played = self._estimate_minutes_played(q_scores, game_info)

        if minutes_played > 0:
            # SB por minuto (normalizado vs baseline NBA/WNBA ~0.5 SB/minuto total)
            total_sb_rate = (home_sb_total + away_sb_total) / minutes_played
            home_sb_rate = home_sb_total / minutes_played
            away_sb_rate = away_sb_total / minutes_played

            # Normalizar impacto (0.2-0.8 SB/minuto = rango típico)
            home_sb_impact = max(0.0, min(1.0, (home_sb_rate - 0.2) / 0.6))
            away_sb_impact = max(0.0, min(1.0, (away_sb_rate - 0.2) / 0.6))
        else:
            home_sb_impact = away_sb_impact = 0.5

        return {
            'home_sb_impact': home_sb_impact,
            'away_sb_impact': away_sb_impact,
            'diff_sb_impact': home_sb_impact - away_sb_impact
        }

    def _calculate_lead_momentum_analysis(self, current_stats: Dict, q_scores: Dict, game_info: Dict) -> Dict[str, float]:
        """Analiza tiempo en ventaja y cambios de liderazgo para momentum"""

        # Nota: Esta función requiere parsing especial del JSON de SofaScore
        # Los datos de lead time vienen en formato específico en la API

        # Por ahora, implementación básica usando datos disponibles
        # En producción, esto se expandiría para parsear el "Lead" group del JSON

        # Estimación simple basada en score diferencial actual
        home_score = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        away_score = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
        score_diff = home_score - away_score

        # Estimar tiempo en ventaja basado en diferencial de puntos
        # (esto es una aproximación; en producción usaríamos los datos reales de lead time)
        if abs(score_diff) > 10:
            lead_time_home = 1.0 if score_diff > 0 else 0.0
            lead_time_away = 1.0 if score_diff < 0 else 0.0
        elif abs(score_diff) > 5:
            lead_time_home = 0.7 if score_diff > 0 else 0.3
            lead_time_away = 0.7 if score_diff < 0 else 0.3
        else:
            lead_time_home = lead_time_away = 0.5

        # Estimar cambios de lead (simplificado)
        lead_changes = 0  # En producción: parsear del JSON

        # Calcular momentum shift basado en cambios recientes
        momentum_shift = 0.0
        if hasattr(self, 'last_q_scores') and self.last_q_scores:
            last_home = self.last_q_scores['q1_home'] + self.last_q_scores['q2_home'] + self.last_q_scores['q3_home']
            last_away = self.last_q_scores['q1_away'] + self.last_q_scores['q2_away'] + self.last_q_scores['q3_away']
            last_diff = last_home - last_away

            # Momentum shift = cambio en la ventaja
            momentum_shift = score_diff - last_diff
            momentum_shift = max(-10.0, min(10.0, momentum_shift)) / 10.0  # Normalizar

        # Lead stability (inversa de volatilidad del score)
        lead_stability = 1.0 - min(1.0, abs(momentum_shift))

        return {
            'lead_time_home': lead_time_home,
            'lead_time_away': lead_time_away,
            'lead_changes': lead_changes,
            'momentum_shift': momentum_shift,
            'lead_stability': lead_stability
        }

