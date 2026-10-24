"""Estado local de SofaScore. No registra ni exporta valores de cookies."""
import json
import os
from pathlib import Path
import time


STATE_PATH = Path(__file__).resolve().parents[1] / '.browser_profile' / 'sofascore_session.json'


def browser_context_options(version):
    # Misma identidad, correspondiente al navegador instalado, en ambos modos.
    return {
        'user_agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       f'Chrome/{version} Safari/537.36'),
        'viewport': {'width': 1280, 'height': 720},
        'locale': 'en-US',
    }


def load_browser_session(path=None):
    target = Path(path) if path is not None else STATE_PATH
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('cookies'), list) or not isinstance(data.get('origins'), list):
            raise ValueError('Invalid storage state')
        now = time.time()
        cookies = []
        for cookie in data['cookies']:
            if not isinstance(cookie, dict):
                raise ValueError('Invalid cookie')
            expires = cookie.get('expires', -1)
            if not isinstance(expires, (int, float)):
                raise ValueError('Invalid cookie expiry')
            if expires <= 0 or expires > now:
                cookies.append(cookie)
        return {'cookies': cookies, 'origins': data['origins']}
    except (OSError, ValueError, TypeError):
        print('[SESSION] El estado guardado no es legible. No se ha borrado ni sobrescrito.')
        return None


async def save_browser_session(context, path=None):
    target = Path(path) if path is not None else STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    state = await context.storage_state()
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(state), encoding='utf-8')
    os.replace(temporary, target)
