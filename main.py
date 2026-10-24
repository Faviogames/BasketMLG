"""Punto de entrada: live por extensión, modelos disponibles y entrenamiento."""
import argparse
import asyncio
import traceback
from datetime import datetime


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
    parser = argparse.ArgumentParser(description="BasketMLG: predicción live de baloncesto.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true",
                       help="Selecciona liga y busca partidos mediante la extensión de Chrome.")
    modes.add_argument("--models", action="store_true",
                       help="Muestra los modelos disponibles y termina.")
    modes.add_argument("--train", action="store_true",
                       help="Entrena los modelos con los datos disponibles.")
    args = parser.parse_args()

    if not any((args.live, args.models, args.train)):
        parser.print_help()
        return 0

    try:
        if args.live:
            from analysis.live_system import run_basketball_live_system
            asyncio.run(run_basketball_live_system())
        elif args.models:
            from analysis.live_system import BasketballLiveSystem
            system = BasketballLiveSystem()
            if not system.load_models():
                return 1
            system.show_available_models()
        elif args.train:
            from config import UI_MESSAGES
            from core.training import train_models_by_league
            print(UI_MESSAGES["training_start"])
            train_models_by_league()
    except (EOFError, KeyboardInterrupt):
        print("\nEjecución cancelada.")
        return 130
    except Exception as e:
        print(f"[ERROR] {e}")
        log_crash(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
