"""Orquestación live por extensión y consulta de modelos guardados."""
import asyncio
import os
import sys
import joblib
import random
import traceback
from contextlib import closing, suppress
from typing import Dict, List, Optional

# Mantener la ejecución directa del archivo desde la raíz del proyecto.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_FOLDER


class BasketballLiveSystem:
    """Carga modelos y ejecuta la búsqueda y el seguimiento por extensión."""
    
    def __init__(self):
        self.models = {}
        self.current_processor = None

    def load_models(self):
        """Carga todos los modelos disponibles"""
        if not os.path.exists(MODELS_FOLDER) or not os.listdir(MODELS_FOLDER):
            print("❌ ERROR: No se encontró la carpeta de modelos o está vacía.")
            print("💡 Ejecuta con --train primero para entrenar los modelos.")
            return False

        print("Cargando modelos disponibles...")
        
        for filename in os.listdir(MODELS_FOLDER):
            # Solo archivos de modelos principales (excluir meta-artifacts)
            if not filename.endswith(".joblib"):
                continue
            if filename.endswith("_meta.joblib"):
                # Evitar cargar el meta-modelo como modelo principal
                # El meta se usa solo en tiempo de inferencia desde analysis/live_mode.py
                # y podría no contener claves como 'team_names' o 'historical_df'
                print(f"Saltando meta-artifact: {filename}")
                continue

            league_name = filename.replace(".joblib", "").replace("_", " ")
            file_path = os.path.join(MODELS_FOLDER, filename)
            try:
                model_data = joblib.load(file_path)

                # Validación mínima para evitar archivos incompatibles
                if not isinstance(model_data, dict) or 'team_names' not in model_data or 'historical_df' not in model_data:
                    print(f"Archivo no es un modelo principal compatible, saltando: {filename}")
                    continue

                # 🔒 Fuente única de verdad para la liga: forzar en el payload del modelo
                # Esto asegura que parámetros por liga (NBA/WNBA/NBL) usen el set correcto.
                try:
                    model_data['league_name'] = league_name
                except Exception:
                    pass

                self.models[league_name] = model_data
                print(f"Modelo cargado: {league_name}")
                print(f"   {len(model_data['team_names'])} equipos, {len(model_data['historical_df'])} partidos")
            except Exception as e:
                print(f"❌ Error al cargar {filename}: {e}")
        
        if not self.models:
            print("❌ ERROR: No se encontraron modelos válidos.")
            return False
        
        print(f"\nSistema listo con {len(self.models)} modelo(s)")
        return True

    async def run_live(self):
        """Selecciona liga y partido, sin menú de modos ni entrada por ID."""
        model_data = self._select_league()
        if not model_data:
            return

        # El listado de modelos no necesita importar ni iniciar el transporte.
        from core.live_api import LiveProcessor
        processor = LiveProcessor(model_data, headless=True)
        self.current_processor = processor
        try:
            if not await processor.initialize():
                return

            while True:
                if not await self._select_and_monitor(processor):
                    return
        except (EOFError, KeyboardInterrupt):
            print("\nSeguimiento cancelado.")
        finally:
            try:
                await processor.cleanup()
            finally:
                self.current_processor = None

    async def _select_and_monitor(self, processor):
        """Un partido por vuelta; el puente pertenece a toda la sesión."""
        print("\nBuscando partidos en vivo...")
        games = await processor.get_available_games()
        if not games:
            if processor.client.last_discovery_error:
                print(f"[ERROR] Busqueda no disponible: {processor.client.last_discovery_error}")
            else:
                print("No hay partidos live compatibles con el modelo seleccionado.")
            return

        selected_game = self._select_game(games)
        if not selected_game:
            return
        interval = self._configure_monitoring()

        bookie_line = None
        try:
            from analysis.odds_api import OddsApiClient
            print("\nBuscando línea de apuestas en OddsAPI...")
            odds_client = OddsApiClient()
            bookie_line = odds_client.get_odds_for_teams(
                selected_game["home_team"], selected_game["away_team"]
            )
            if bookie_line:
                print(f"Línea encontrada automáticamente: {bookie_line}")
            else:
                print("No se encontró línea automática.")
        except Exception as e:
            print(f"Error consultando OddsAPI: {e}")

        if bookie_line is None:
            try:
                line_input = input("\nIngrese línea de la casa (opcional, Enter para omitir): ").strip()
                if line_input:
                    bookie_line = float(line_input)
            except ValueError:
                print("Valor inválido, ignorando línea.")

        print(f"\nIniciando monitoreo para {selected_game['home_team']} vs {selected_game['away_team']}...")
        return await self._monitor_with_navigation(processor,
            game_id=selected_game["id"],
            game_info=selected_game,
            update_interval=interval,
            bookie_line=bookie_line,
        )

    async def _monitor_with_navigation(self, processor, **kwargs):
        """Cancela sólo el monitor y espera su salida antes de otra selección."""
        if not sys.stdin.isatty():
            await processor.monitor_game(**kwargs)
            return False

        from prompt_toolkit.input import create_input
        print("[TECLAS] V: volver al listado | Q: salir (sin Enter).")
        # Input ofrece close(), pero Win32Input no implementa __exit__.
        with closing(create_input()) as keyboard, keyboard.raw_mode():
            monitor = asyncio.create_task(processor.monitor_game(**kwargs))
            try:
                while not monitor.done():
                    for key in keyboard.read_keys():
                        command = key.data.lower()
                        if command in ('v', 'q', '\x03'):
                            monitor.cancel()
                            with suppress(asyncio.CancelledError):
                                await monitor
                            if command == 'v':
                                print("\nVolviendo al listado; conexión de la extensión conservada.")
                                return True
                            return False
                    await asyncio.sleep(0.1)
                await monitor  # Propagar errores reales del seguimiento.
                return False
            finally:
                if not monitor.done():
                    monitor.cancel()
                    with suppress(asyncio.CancelledError):
                        await monitor


    def _select_league(self) -> Optional[Dict]:
        """Selecciona liga/modelo para usar"""

        if not self.models:
            print("No hay modelos cargados")
            return None

        print(f"\nLigas Disponibles:")
        leagues = list(self.models.keys())

        for i, league in enumerate(leagues, 1):
            model_info = self.models[league]
            teams_count = len(model_info['team_names'])
            matches_count = len(model_info['historical_df'])
            print(f"{i}. {league}: {teams_count} equipos, {matches_count} partidos")

        try:
            choice = int(input(f"\nSelecciona liga (1-{len(leagues)}): "))

            if 1 <= choice <= len(leagues):
                selected_league = leagues[choice - 1]
                selected_model = self.models[selected_league]

                print(f"Liga seleccionada: {selected_league}")
                return selected_model
            else:
                print("Seleccion invalida")
                return None

        except (ValueError, EOFError, KeyboardInterrupt):
            print("Seleccion cancelada")
            return None
    
    def _select_game(self, available_games: List[Dict]) -> Optional[Dict]:
        """Selecciona juego específico para monitorear"""

        print(f"\nJuegos Live Disponibles:")
        print("="*50)

        for i, game in enumerate(available_games, 1):
            home_team = game['home_team']
            away_team = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']
            status = game['status']
            confidence = game.get('model_confidence', 0.8)
            remaining_quarters = game.get('estimated_remaining_quarters', 2)

            print(f"{i}. {home_team} {home_score} - {away_team} {away_score}")
            print(f"   Estado: {status}")
            print(f"   Confianza: {confidence:.1%}")
            print(f"   Cuartos restantes: ~{remaining_quarters}")
            print()

        if not available_games:
            print("No hay juegos disponibles")
            return None

        try:
            choice = int(input(f"Selecciona juego (1-{len(available_games)}): "))

            if 1 <= choice <= len(available_games):
                selected_game = available_games[choice - 1]

                print(f"Juego seleccionado: {selected_game['home_team']} vs {selected_game['away_team']}")
                return selected_game
            else:
                print("Seleccion invalida")
                return None

        except (ValueError, EOFError, KeyboardInterrupt):
            print("Seleccion cancelada")
            return None
    
    def _configure_monitoring(self) -> int:
        """Configura intervalo de monitoreo automáticamente"""
        # Intervalo automático entre 45-60 segundos para optimizar rendimiento
        interval = random.randint(45, 60)
        print(f"Monitoreo automatico configurado: updates cada {interval}s")
        return interval
    
    def show_available_models(self):
        """Muestra información detallada de modelos"""

        print(f"\nModelos Disponibles")
        print("="*40)

        for league_name, model_data in self.models.items():
            print(f"\n{league_name}")
            print(f"   Equipos: {len(model_data['team_names'])}")
            print(f"   Partidos: {len(model_data['historical_df'])}")
            print(f"   Features: {len(model_data['features_used'])}")
            print(f"   MAE: {model_data.get('mae', 'N/A')}")
            print(f"   Modelo: {model_data.get('model_type', 'No especificado')}")

            # Mostrar algunos equipos como ejemplo
            teams = model_data['team_names'][:5]
            if len(model_data['team_names']) > 5:
                teams_display = f"{', '.join(teams)}, ... (+{len(model_data['team_names']) - 5} mas)"
            else:
                teams_display = ', '.join(teams)
            print(f"   Equipos: {teams_display}")

    async def cleanup(self):
        """Limpia recursos del sistema"""
        if self.current_processor:
            await self.current_processor.cleanup()
            self.current_processor = None
        self.models.clear()


# ===========================================
# FUNCIÓN PRINCIPAL DEL SISTEMA
# ===========================================

async def run_basketball_live_system():
    """Funcion principal del sistema live"""

    system = BasketballLiveSystem()

    try:
        # Cargar modelos
        if not system.load_models():
            return

        # Ir directamente a selección de liga y búsqueda
        await system.run_live()

    except Exception as e:
        print(f"Error critico del sistema: {e}")
        traceback.print_exc()
    finally:
        # Cleanup siempre
        await system.cleanup()


# ===========================================
# PUNTO DE ENTRADA DIRECTO
# ===========================================

if __name__ == "__main__":
    print("Basketball Live System v3.0")
    asyncio.run(run_basketball_live_system())
