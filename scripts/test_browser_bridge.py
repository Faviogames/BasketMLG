"""Prueba manual de la extension sin cargar modelos ni Playwright."""
import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.browser_bridge import BrowserBridge


async def main(event_id=None):
    bridge = BrowserBridge(timeout=120)
    bridge.start()
    print('Abre el panel de la extension y conecta este codigo (valido solo para esta ejecucion):')
    print(f'\n{bridge.token}\n')
    try:
        live = await bridge.request('/api/v1/sport/basketball/events/live')
        print(f"Listado recibido: {len(live['events'])} partidos.")
        for event in live['events']:
            print(f"{event.get('id')}: {event.get('homeTeam', {}).get('name')} vs {event.get('awayTeam', {}).get('name')}")
        if event_id:
            event = await bridge.request(f'/api/v1/event/{event_id}')
            print(f"Evento recibido: {event['event'].get('id')}")
            stats = await bridge.request(f'/api/v1/event/{event_id}/statistics')
            print('Periodos recibidos:', [p.get('period') for p in stats['statistics']])
        print('Prueba terminada. El listado vacio es valido si no hay partidos en vivo.')
    except (RuntimeError, ValueError, TimeoutError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(1)
    finally:
        await asyncio.to_thread(bridge.close)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', type=int, help='ID de partido para comprobar evento y estadisticas')
    args = parser.parse_args()
    asyncio.run(main(args.event))
