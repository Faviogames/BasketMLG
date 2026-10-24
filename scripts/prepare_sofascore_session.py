"""Preparar sesion: python scripts/prepare_sofascore_session.py
Comprobar reutilizacion oculta: python scripts/prepare_sofascore_session.py --check
La preparacion abre una ventana solo al ejecutar este comando expresamente.
"""
import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.sofascore_session import browser_context_options, load_browser_session, save_browser_session
from playwright.async_api import async_playwright, Error as PlaywrightError


async def prepare(check=False):
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(channel='chrome', headless=check)
        except Exception:
            browser = await playwright.chromium.launch(headless=check)
        try:
            options = browser_context_options(browser.version)
            state = load_browser_session()
            if state is not None:
                options['storage_state'] = state
            context = await browser.new_context(**options)
            page = await context.new_page()
            await page.goto('https://www.sofascore.com/basketball',
                            wait_until='domcontentloaded', timeout=30000)
            if not check:
                print('En la ventana de SofaScore, completa la verificacion si aparece.')
                print('Espera a ver los partidos. No necesitas iniciar sesion en una cuenta.')
                await asyncio.to_thread(input, 'Cuando esten visibles, pulsa Enter aqui: ')
            else:
                for _ in range(20):
                    if 'captcha' not in page.url.lower():
                        break
                    await asyncio.sleep(1)
            result = await page.evaluate('''async () => {
                try {
                    const response = await fetch('/api/v1/sport/basketball/events/live',
                        {signal: AbortSignal.timeout(15000)});
                    const data = await response.json();
                    return {status: response.status, valid: response.ok && Array.isArray(data.events),
                        events: Array.isArray(data.events) ? data.events.length : null};
                } catch(error) { return {valid: false, error: error.message}; }
            }''')
            if not result.get('valid'):
                print('[ERROR] No se ha validado el acceso al listado. Estado HTTP:', result.get('status'))
                print('[INFO] La sesion anterior se conserva; no se guarda una verificacion pendiente.')
                return False
            await save_browser_session(context)
            print('[OK] Listado accesible:', result['events'], 'partidos. Sesion guardada localmente.')
            if not check:
                print('Ahora comprueba: python scripts/prepare_sofascore_session.py --check')
            return True
        except PlaywrightError:
            print('[ERROR] La ventana se cerro o SofaScore interrumpio la preparacion.')
            print('[INFO] No se ha completado la validacion del listado.')
            return False
        finally:
            await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validar la sesion sin abrir ventana')
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(prepare(args.check)) else 1)
