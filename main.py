# ===========================================
# Archivo: main.py (v2.1)
# Punto de entrada principal con sistema de alertas inteligentes
# Estructura modular mejorada
# ===========================================
import os
import joblib
import argparse
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import traceback
from datetime import datetime

# Imports de los módulos reorganizados
from config import MODELS_FOLDER, UI_MESSAGES
from core.training import train_models_by_league
from analysis.live_mode import live_mode_with_alerts

def log_crash(e):
    """Genera un archivo de log cuando ocurre un error inesperado."""
    with open("crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Crash Report ---\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error: {str(e)}\n")
        f.write("Traceback:\n")
        traceback.print_exc(file=f)
        f.write("\n\n")
    print("\n¡Ups! Ocurrió un error inesperado. Se ha generado un archivo 'crash_log.txt' con los detalles.")

def main():
    """Función principal para manejar la ejecución del programa."""
    parser = argparse.ArgumentParser(description="Herramienta de predicción de baloncesto v2.1 - Con Alertas Inteligentes.")
    parser.add_argument("--train", action="store_true", help="Entrena los modelos con los datos disponibles.")
    args = parser.parse_args()

    print(UI_MESSAGES['welcome'])
    print(UI_MESSAGES['features'])
    
    if args.train:
        print(f"\n{UI_MESSAGES['training_start']}")
        try:
            train_models_by_league()
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            log_crash(e)
    else:
        print(f"\n{UI_MESSAGES['prediction_mode']}")
        
        if not os.path.exists(MODELS_FOLDER) or not os.listdir(MODELS_FOLDER):
            print("❌ ERROR: No se encontró la carpeta de modelos o está vacía.")
            print("💡 Ejecuta con --train primero para entrenar los modelos.")
            return

        # Cargar todos los modelos disponibles
        all_trained_data = {}
        for filename in os.listdir(MODELS_FOLDER):
            if filename.endswith(".joblib"):
                league_name = filename.replace(".joblib", "").replace("_", " ")
                try:
                    all_trained_data[league_name] = joblib.load(os.path.join(MODELS_FOLDER, filename))
                    print(f"✅ Modelo cargado: {league_name}")
                except Exception as e:
                    print(f"❌ Error al cargar el modelo {filename}: {e}")
                    log_crash(e)
        
        if not all_trained_data:
            print("❌ ERROR: No se encontraron modelos válidos.")
            return

        # Mostrar ligas disponibles
        leagues = list(all_trained_data.keys())
        print(f"\n🏆 Ligas disponibles ({len(leagues)} modelos entrenados):")
        for league in leagues:
            model_info = all_trained_data[league]
            print(f"  📊 {league}: {len(model_info['team_names'])} equipos, {len(model_info['historical_df'])} partidos")
        
        # Seleccionar liga
        league_completer = WordCompleter(leagues, ignore_case=True)
        selected_league = prompt("\n🔽 Selecciona la liga: ", completer=league_completer)
        
        if selected_league in all_trained_data:
            try:
                # 🚀 USAR EL NUEVO MODO EN VIVO CON ALERTAS
                live_mode_with_alerts(all_trained_data[selected_league])
            except Exception as e:
                print(f"❌ Error en el modo en vivo: {e}")
                log_crash(e)
        else:
            print("❌ Selección inválida.")

if __name__ == "__main__":
    main()