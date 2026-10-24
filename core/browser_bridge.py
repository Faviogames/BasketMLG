"""Authenticated, loopback-only transport for the user's browser extension."""
import asyncio
import hmac
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROUTE = re.compile(r"/api/v1/(?:sport/basketball/events/live|event/[0-9]+(?:/statistics)?)\Z")
ORIGIN = re.compile(r"chrome-extension://[a-p]{32}\Z")


def validate_payload(path, data):
    if not ROUTE.fullmatch(path) or not isinstance(data, dict):
        raise ValueError('Respuesta o ruta no valida.')
    key, kind = ('events', list) if path.endswith('/live') else (
        ('statistics', list) if path.endswith('/statistics') else ('event', dict))
    if not isinstance(data.get(key), kind):
        raise ValueError(f'La respuesta no contiene {key} valido; puede ser una verificacion o un cambio de API.')
    if key == 'event' and str(data[key].get('id')) != path.split('/')[-1]:
        raise ValueError('El evento recibido no corresponde al solicitado.')
    return data


class BrowserBridge:
    def __init__(self, port=8766, timeout=90):
        self.port = port
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.server = None
        self.thread = None
        self.lock = threading.Lock()
        self.serial = asyncio.Lock()
        self.pending = None

    def start(self):
        if self.server:
            return
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Never log pairing credentials or payloads.

            def send_json(self, code, body):
                raw = json.dumps(body).encode('utf-8')
                self.send_response(code)
                origin = self.headers.get('Origin', '')
                if ORIGIN.fullmatch(origin):
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def allowed(self):
                origin = self.headers.get('Origin', '')
                expected = 'Bearer ' + bridge.token
                return (self.headers.get('Host') == f'127.0.0.1:{bridge.port}'
                        and (not origin or ORIGIN.fullmatch(origin))
                        and hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), expected.encode('utf-8')))

            def do_OPTIONS(self):
                if (self.headers.get('Host') != f'127.0.0.1:{bridge.port}'
                        or not ORIGIN.fullmatch(self.headers.get('Origin', ''))):
                    return self.send_json(403, {'error': 'Origen no permitido'})
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
                self.end_headers()

            def do_GET(self):
                if not self.allowed():
                    return self.send_json(403, {'error': 'Emparejamiento no valido'})
                if self.path == '/status':
                    return self.send_json(200, {'ready': True})
                if self.path != '/jobs':
                    return self.send_json(404, {})
                with bridge.lock:
                    item = bridge.pending
                    job = ({'id': item['id'], 'path': item['path']}
                           if item and not item['done'].is_set() else None)
                self.send_json(200, {'job': job})

            def do_POST(self):
                self.connection.settimeout(10)
                if not self.allowed():
                    return self.send_json(403, {'error': 'Emparejamiento no valido'})
                if self.path != '/result':
                    return self.send_json(404, {})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 8 * 1024 * 1024:
                        return self.send_json(413, {'error': 'Tamano no valido'})
                    result = json.loads(self.rfile.read(length))
                    if not isinstance(result, dict):
                        raise ValueError()
                except (ValueError, OSError):
                    return self.send_json(400, {'error': 'JSON no valido'})
                with bridge.lock:
                    item = bridge.pending
                    if not item or result.get('id') != item['id'] or item['done'].is_set():
                        return self.send_json(409, {'error': 'Solicitud caducada'})
                    item['result'] = result
                    item['done'].set()
                self.send_json(200, {'ok': True})

        self.server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    async def request(self, path):
        if not ROUTE.fullmatch(path):
            raise ValueError('Ruta no permitida')
        if not self.server:
            raise RuntimeError('El puente no esta iniciado')
        async with self.serial:
            item = {'id': secrets.token_hex(16), 'path': path, 'done': threading.Event()}
            with self.lock:
                self.pending = item
            try:
                deadline = time.monotonic() + self.timeout
                while not item['done'].is_set():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('La extension no respondio. Abre su panel, conecta el codigo y mantenlo abierto con una pestana de SofaScore seleccionada.')
                    await asyncio.sleep(0.1)
                result = item.get('result', {})
                if result.get('error'):
                    raise RuntimeError(str(result['error'])[:350])
                return validate_payload(path, result.get('data'))
            finally:
                with self.lock:
                    if self.pending is item:
                        self.pending = None

    def close(self):
        with self.lock:
            if self.pending:
                self.pending['result'] = {'error': 'Puente cerrado'}
                self.pending['done'].set()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None
