"""Diagnostico de acceso sin cargar modelos ML ni solicitar cuotas.

Uso: python scripts/diagnose_sofascore.py --event 16685169
Ejecuta el diagnostico sin abrir una ventana de navegador.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def diagnose(event_id):
    profile = Path(__file__).resolve().parents[1] / '.browser_profile'
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            str(profile), channel='chrome', headless=True,
            args=['--no-first-run', '--no-default-browser-check'],
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            response = await page.goto('https://www.sofascore.com/basketball',
                                       wait_until='domcontentloaded', timeout=30000)
            print('SESSION', response.status if response else None, page.url)
            if 'captcha' in page.url.lower():
                print('[ERROR] SofaScore solicita verificacion y bloquea la sesion oculta.')
                return
            await asyncio.sleep(2)
            # Misma sesion, mismo origen y mismas peticiones que utiliza el cliente.
            for path in ['/api/v1/sport/basketball/events/live',
                         f'/api/v1/event/{event_id}/statistics']:
                result = await page.evaluate('''async path => {
                    try {
                        const response = await fetch(path);
                        const text = await response.text();
                        let data;
                        try { data = JSON.parse(text); } catch {}
                        return {
                            status: response.status,
                            content_type: response.headers.get('content-type'),
                            keys: data ? Object.keys(data) : [],
                            events_count: Array.isArray(data?.events) ? data.events.length : null,
                            statistics_periods: Array.isArray(data?.statistics) ? data.statistics.length : null,
                            error: data?.error,
                            sample_events: Array.isArray(data?.events) ? data.events.slice(0, 5).map(e => ({
                                id: e.id, home: e.homeTeam?.name, away: e.awayTeam?.name,
                                status: e.status, tournament: e.tournament?.name
                            })) : []
                        };
                    } catch (error) { return {error: error.message}; }
                }''', path)
                print(path, json.dumps(result, ensure_ascii=True))
        finally:
            await context.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', default='16685169')
    args = parser.parse_args()
    if not args.event.isdecimal():
        parser.error('--event debe ser un ID numerico')
    asyncio.run(diagnose(args.event))
