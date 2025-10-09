# ===========================================
# analysis/live_system.py (v3.0 PRODUCTION)
# SISTEMA LIVE COMPLETO: Integra modo manual + live real
# ✅ PARCHES CRÍTICOS #2 y #3 APLICADOS
# ✅ BASADO EN: test exitoso Playwright + prototipos funcionales
# ✅ MANTIENE: modo manual existente
# ✅ AÑADE: modo live real con monitoreo automático
# ===========================================

import asyncio
import os
import joblib
import random
from typing import Dict, List, Optional
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from datetime import datetime
import traceback

# Imports del sistema existente
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_FOLDER, UI_MESSAGES
from core.live_api import SofaScoreClient, LiveProcessor

# Import del modo manual existente (para mantener compatibilidad)
try:
    from analysis.live_mode import live_mode_with_alerts
    MANUAL_MODE_AVAILABLE = True
except ImportError:
    print("⚠️ Modo manual no disponible")
    MANUAL_MODE_AVAILABLE = False

class BasketballLiveSystem:
    """
    🎯 Sistema Live Completo de Basketball

    CARACTERÍSTICAS:
    ✅ Modo Manual: predicción única (existente)
    ✅ Modo Live Real: monitoreo automático cada 45-60s (automático)
    ✅ Playwright: navegador real para evitar bloqueos
    ✅ Mapeo inteligente: SofaScore → modelo
    ✅ Alertas avanzadas: cambios significativos + contexto
    ✅ Multi-liga: NBA, WNBA, EuroLeague, etc.
    """
    
    def __init__(self):
        self.models = {}
        self.current_processor = None
        
        print("🚀 Basketball Live System v3.0 inicializado")
        print("🎭 Playwright: Control total del navegador")
        print("🔄 Modos: Manual + Live Real")
        
    def load_models(self):
        """Carga todos los modelos disponibles"""
        if not os.path.exists(MODELS_FOLDER) or not os.listdir(MODELS_FOLDER):
            print("❌ ERROR: No se encontró la carpeta de modelos o está vacía.")
            print("💡 Ejecuta con --train primero para entrenar los modelos.")
            return False

        print("📂 Cargando modelos disponibles...")
        
        for filename in os.listdir(MODELS_FOLDER):
            # Solo archivos de modelos principales (excluir meta-artifacts)
            if not filename.endswith(".joblib"):
                continue
            if filename.endswith("_meta.joblib"):
                # Evitar cargar el meta-modelo como modelo principal
                # El meta se usa solo en tiempo de inferencia desde analysis/live_mode.py
                # y podría no contener claves como 'team_names' o 'historical_df'
                print(f"↪️ Saltando meta-artifact: {filename}")
                continue

            league_name = filename.replace(".joblib", "").replace("_", " ")
            file_path = os.path.join(MODELS_FOLDER, filename)
            try:
                model_data = joblib.load(file_path)

                # Validación mínima para evitar archivos incompatibles
                if not isinstance(model_data, dict) or 'team_names' not in model_data or 'historical_df' not in model_data:
                    print(f"ℹ️ Archivo no es un modelo principal compatible, saltando: {filename}")
                    continue

                # 🔒 Fuente única de verdad para la liga: forzar en el payload del modelo
                # Esto asegura que parámetros por liga (NBA/WNBA/NBL) usen el set correcto.
                try:
                    model_data['league_name'] = league_name
                except Exception:
                    pass

                self.models[league_name] = model_data
                print(f"✅ Modelo cargado: {league_name}")
                print(f"   📊 {len(model_data['team_names'])} equipos, {len(model_data['historical_df'])} partidos")
            except Exception as e:
                print(f"❌ Error al cargar {filename}: {e}")
        
        if not self.models:
            print("❌ ERROR: No se encontraron modelos válidos.")
            return False
        
        print(f"\n✅ Sistema listo con {len(self.models)} modelo(s)")
        return True
    
    async def run_interactive_menu(self):
        """Ejecuta el menú interactivo principal"""
        
        print(f"\n🏀 BASKETBALL LIVE SYSTEM v3.0")
        print("="*60)
        
        while True:
            try:
                print(f"\n📋 OPCIONES DISPONIBLES:")
                print("1. 🎯 Modo Manual - Predicción única")
                print("2. 📡 Modo Live Real - Monitoreo automático") 
                print("3. 🧪 Modo Test - Verificar sistema")
                print("4. 📊 Mostrar modelos disponibles")
                print("5. 🚪 Salir")
                
                choice = input("\nSelecciona opción (1-5): ").strip()
                
                if choice == "1":
                    await self._run_manual_mode()
                elif choice == "2":
                    await self._run_live_real_mode()
                elif choice == "3":
                    await self._run_test_mode()
                elif choice == "4":
                    self._show_available_models()
                elif choice == "5":
                    print("👋 ¡Hasta luego!")
                    break
                else:
                    print("❌ Opción inválida. Selecciona 1-5.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Sistema cerrado por usuario")
                break
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                traceback.print_exc()
    
    async def _run_manual_mode(self):
        """🎯 Modo Manual: predicción única (ejecutado en hilo separado para evitar conflictos de asyncio)"""
        
        print(f"\n🎯 MODO MANUAL - Predicción Única")
        print("="*50)
        
        if not MANUAL_MODE_AVAILABLE:
            print("❌ Modo manual no disponible")
            return
        
        # Seleccionar liga
        selected_model = self._select_league()
        if not selected_model:
            return
        
        try:
            # Ejecutar la función sincrónica en un hilo para evitar:
            # RuntimeError: asyncio.run() cannot be called from a running event loop
            # prompt_toolkit usa asyncio.run internamente; en un hilo no hay loop corriendo.
            print("🧵 Ejecutando Modo Manual en hilo de trabajo (evita conflictos con asyncio)...")
            await asyncio.to_thread(live_mode_with_alerts, selected_model)
            print("✅ Modo Manual finalizado")
        except Exception as e:
            print(f"❌ Error en modo manual: {e}")
            traceback.print_exc()
    
    # ✅ CRÍTICO #2: FUNCIÓN _run_live_real_mode CORREGIDA
    async def _run_live_real_mode(self):
        """📡 Modo Live Real: monitoreo automático (NUEVO)"""
        
        print(f"\n📡 MODO LIVE REAL - Monitoreo Automático")
        print("="*50)
        print("🎭 Usando Playwright para evitar bloqueos")
        print("🔄 Updates automáticos cada 45-60 segundos")
        print("🚨 Alertas en tiempo real")
        
        # Seleccionar liga
        selected_model = self._select_league()
        if not selected_model:
            return
        
        # Inicializar procesador live
        processor = LiveProcessor(selected_model)
        self.current_processor = processor
        
        try:
            # 1. Inicializar sistema
            print("\n🔧 Inicializando sistema live...")
            init_success = await processor.initialize()
            
            if not init_success:
                print("❌ Error inicializando sistema live")
                return
            
            # 2. Health check
            print("🔍 Verificando conectividad...")
            is_healthy = await processor.client.health_check()
            
            if not is_healthy:
                print("❌ Sistema no funcional")
                return
            
            # 3. Obtener juegos disponibles
            print("📡 Obteniendo juegos live...")
            available_games = await processor.get_available_games()
            
            if not available_games:
                print("⚠️ No hay juegos live disponibles para esta liga")
                print("💡 Intenta más tarde o selecciona otra liga")
                return
            
            # 4. Mostrar juegos y seleccionar
            selected_game = self._select_game(available_games)
            if not selected_game:
                return
            
            # 5. Configurar monitoreo
            update_interval = self._configure_monitoring()
            
            # ✅ CRÍTICO #2: OBTENER LÍNEA DE APUESTAS ANTES DEL LOOP ASYNC
            bookie_line = None
            try:
                line_input = input("📈 Línea de apuestas (Enter para omitir): ")
                if line_input.strip():
                    bookie_line = float(line_input)
            except:
                pass
            
            # 6. INICIAR MONITOREO LIVE
            print(f"\n🚀 INICIANDO MONITOREO LIVE")
            print("⚠️ Presiona Ctrl+C en cualquier momento para detener")
            
            # ✅ CRÍTICO #2: LLAMADA CORREGIDA CON bookie_line COMO PARÁMETRO
            await processor.monitor_game(
                selected_game['id'], 
                selected_game, 
                update_interval=update_interval,
                max_updates=120,  # Máximo 1 hora
                bookie_line=bookie_line  # ✅ PASADO COMO PARÁMETRO
            )
            
        except KeyboardInterrupt:
            print(f"\n⌨️ Monitoreo detenido por usuario")
        except Exception as e:
            print(f"❌ Error en modo live: {e}")
            traceback.print_exc()
        finally:
            # Cleanup siempre
            if processor:
                await processor.cleanup()
                self.current_processor = None
    
    async def _run_test_mode(self):
        """🧪 Modo Test: verifica funcionalidad del sistema"""
        
        print(f"\n🧪 MODO TEST - Verificación del Sistema")
        print("="*50)
        
        # Seleccionar liga para test
        selected_model = self._select_league()
        if not selected_model:
            return
        
        # Importar función de test
        try:
            from tests.test_live_api import test_live_system
            
            print("🔬 Ejecutando test completo del sistema...")
            
            success = await test_live_system(selected_model)
            
            if success:
                print("✅ Sistema live funcional")
                print("💡 Puedes usar el Modo Live Real con confianza")
            else:
                print("❌ Sistema live tiene problemas")
                print("💡 Revisa la configuración o intenta más tarde")
                
        except Exception as e:
            print(f"❌ Error en test: {e}")
            traceback.print_exc()
    
    def _select_league(self) -> Optional[Dict]:
        """Selecciona liga/modelo para usar"""
        
        if not self.models:
            print("❌ No hay modelos cargados")
            return None
        
        print(f"\n🏆 Ligas Disponibles:")
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
                
                print(f"✅ Liga seleccionada: {selected_league}")
                return selected_model
            else:
                print("❌ Selección inválida")
                return None
                
        except (ValueError, EOFError, KeyboardInterrupt):
            print("❌ Selección cancelada")
            return None
    
    def _select_game(self, available_games: List[Dict]) -> Optional[Dict]:
        """Selecciona juego específico para monitorear"""
        
        print(f"\n🏀 Juegos Live Disponibles:")
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
            print(f"   📊 Estado: {status}")
            print(f"   🎯 Confianza: {confidence:.1%}")
            print(f"   ⏰ Cuartos restantes: ~{remaining_quarters}")
            print()
        
        if not available_games:
            print("⚠️ No hay juegos disponibles")
            return None
        
        try:
            choice = int(input(f"Selecciona juego (1-{len(available_games)}): "))
            
            if 1 <= choice <= len(available_games):
                selected_game = available_games[choice - 1]
                
                print(f"✅ Juego seleccionado: {selected_game['home_team']} vs {selected_game['away_team']}")
                return selected_game
            else:
                print("❌ Selección inválida")
                return None
                
        except (ValueError, EOFError, KeyboardInterrupt):
            print("❌ Selección cancelada")
            return None
    
    def _configure_monitoring(self) -> int:
        """Configura intervalo de monitoreo automáticamente"""
        # Intervalo automático entre 45-60 segundos para optimizar rendimiento
        interval = random.randint(45, 60)
        print(f"✅ Monitoreo automático configurado: updates cada {interval}s")
        return interval
    
    def _show_available_models(self):
        """Muestra información detallada de modelos"""
        
        print(f"\n📊 MODELOS DISPONIBLES")
        print("="*60)
        
        for league_name, model_data in self.models.items():
            print(f"\n🏆 {league_name}")
            print(f"   👥 Equipos: {len(model_data['team_names'])}")
            print(f"   📊 Partidos: {len(model_data['historical_df'])}")
            print(f"   🎯 Features: {len(model_data['features_used'])}")
            print(f"   📈 MAE: {model_data.get('mae', 'N/A')}")
            print(f"   🤖 Modelo: {model_data.get('model_type', 'Random Forest')}")
            
            # Mostrar algunos equipos como ejemplo
            teams = model_data['team_names'][:5]
            if len(model_data['team_names']) > 5:
                teams_display = f"{', '.join(teams)}, ... (+{len(model_data['team_names']) - 5} más)"
            else:
                teams_display = ', '.join(teams)
            print(f"   🏀 Equipos: {teams_display}")
        
        print("="*60)
    
    async def cleanup(self):
        """Limpia recursos del sistema"""
        if self.current_processor:
            await self.current_processor.cleanup()
            self.current_processor = None


# ===========================================
# FUNCIÓN PRINCIPAL DEL SISTEMA
# ===========================================

async def run_basketball_live_system():
    """🚀 Función principal del sistema live"""
    
    system = BasketballLiveSystem()
    
    try:
        # Cargar modelos
        if not system.load_models():
            return
        
        # Ejecutar menú interactivo
        await system.run_interactive_menu()
        
    except Exception as e:
        print(f"❌ Error crítico del sistema: {e}")
        traceback.print_exc()
    finally:
        # Cleanup siempre
        await system.cleanup()


# ===========================================
# INTEGRACIÓN CON main.py EXISTENTE
# ===========================================

def add_live_mode_to_main():
    """
    🔗 Función para integrar en main.py existente
    
    INSTRUCCIONES DE INTEGRACIÓN:
    
    1. En main.py, agregar import:
       from analysis.live_system import run_basketball_live_system
    
    2. En el argparse, agregar nueva opción:
       parser.add_argument("--live", action="store_true", 
                          help="Ejecuta el sistema live completo")
    
    3. En main(), agregar condición:
       if args.live:
           asyncio.run(run_basketball_live_system())
           return
    
    4. Mantener lógica existente para --train y modo normal
    """
    pass


# ===========================================
# PUNTO DE ENTRADA DIRECTO
# ===========================================

if __name__ == "__main__":
    print("Basketball Live System v3.0")
    print("Sistema completo: Manual + Live Real")
    print("Powered by Playwright + SofaScore")

    # Ejecutar sistema
    asyncio.run(run_basketball_live_system())