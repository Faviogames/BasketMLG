'use strict';
const base = 'http://127.0.0.1:8766';
const route = /^\/api\/v1\/(?:sport\/basketball\/events\/live|event\/[0-9]+(?:\/statistics)?)$/;
const el = id => document.getElementById(id);
let connected = false;
let token = '';
let selectedTab = null;
let transport = 'navigation';
const usedDocuments = new Set();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function local(path, body) {
  const response = await fetch(base + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {Authorization: 'Bearer ' + token, 'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
    cache: 'no-store'
  });
  // Al cambiar de partido Python descarta la consulta anterior. Su respuesta
  // tardía no debe desconectar el panel ni exigir emparejar otra vez.
  if (path === '/result' && response.status === 409) return {expired: true};
  if (!response.ok) throw new Error(response.status === 403
    ? 'Código rechazado. Copia el código de la ejecución actual de Python.'
    : `El puente local devuelve HTTP ${response.status}.`);
  return response.json();
}

async function refreshTabs() {
  const previous = el('tab').value;
  const tabs = await chrome.tabs.query({url: 'https://www.sofascore.com/*'});
  el('tab').replaceChildren();
  for (const tab of tabs) {
    const option = document.createElement('option');
    option.value = String(tab.id);
    option.textContent = tab.title || tab.url;
    el('tab').append(option);
  }
  if (tabs.some(tab => String(tab.id) === previous)) el('tab').value = previous;
  if (!tabs.length) el('status').textContent = 'Abre SofaScore y pulsa Actualizar pestañas.';
}

async function resolveSofaTab() {
  try {
    const tab = await chrome.tabs.get(selectedTab);
    if (tab.url && new URL(tab.url).origin === 'https://www.sofascore.com') return tab;
  } catch {
    // Chrome may replace/close a tab after the dropdown was populated.
  }
  const candidates = (await chrome.tabs.query({url: 'https://www.sofascore.com/*'}))
    .filter(tab => tab.url && new URL(tab.url).origin === 'https://www.sofascore.com');
  if (candidates.length !== 1) {
    throw new Error(candidates.length
      ? 'La pestaña seleccionada ya no está disponible y hay varias de SofaScore. Desconecta, actualiza pestañas y selecciona la que quieres usar.'
      : 'La pestaña seleccionada ya no existe. Abre SofaScore en este navegador, actualiza pestañas y vuelve a conectar.');
  }
  selectedTab = candidates[0].id;
  await refreshTabs();
  el('tab').value = String(selectedTab);
  return candidates[0];
}

async function executeJob(job) {
  if (!job || typeof job.id !== 'string' || !route.test(job.path)) {
    throw new Error('Solicitud local no válida.');
  }
  try {
    await resolveSofaTab();
    if (transport === 'navigation') return {id: job.id, ...await navigateAndRead(job.path)};
    const results = await chrome.scripting.executeScript({
      target: {tabId: selectedTab},
      world: 'ISOLATED',
      // Same-origin request in the user's tab. Cookies stay inside Chrome.
      func: async path => {
        try {
          if (location.origin !== 'https://www.sofascore.com') throw new Error('Origen incorrecto.');
          if (location.pathname.includes('captcha')) throw new Error('Completa la verificación en la pestaña de SofaScore.');
          const response = await fetch(path, {
            credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(20000)
          });
          if (!response.ok) throw new Error(`SofaScore devuelve HTTP ${response.status}. Revisa su pestaña y cualquier verificación pendiente.`);
          if (response.url.includes('captcha')) throw new Error('SofaScore solicita una verificación en su pestaña.');
          try { return {data: await response.json()}; }
          catch { throw new Error('SofaScore no ha devuelto JSON. Comprueba su pestaña.'); }
        } catch (error) { return {error: error.message}; }
      },
      args: [job.path]
    });
    return {id: job.id, ...(results[0]?.result || {error: 'La pestaña no respondió.'})};
  } catch (error) {
    return {id: job.id, error: error.message};
  }
}

async function navigateTab(url) {
  const tabId = selectedTab;
  // Listen before updating, so a fast response cannot race the load listener.
  await new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer);
      chrome.tabs.onUpdated.removeListener(updated);
      chrome.tabs.onRemoved.removeListener(removed);
      if (error) reject(error); else resolve();
    };
    const updated = (id, change) => {
      if (id === tabId && change.status === 'complete') finish();
    };
    const removed = id => {
      if (id === tabId) finish(new Error('Se ha cerrado la pestaña de SofaScore durante la consulta.'));
    };
    const timer = setTimeout(() => finish(new Error('SofaScore no terminó de cargar el enlace en 25 segundos. Revisa su pestaña.')), 25000);
    chrome.tabs.onUpdated.addListener(updated);
    chrome.tabs.onRemoved.addListener(removed);
    chrome.tabs.update(tabId, {url}).catch(finish);
  });
}

async function navigateAndRead(path) {
  const tab = await chrome.tabs.get(selectedTab);
  const target = 'https://www.sofascore.com' + path;
  const key = `${selectedTab}:${path}`;
  // Reuse a JSON document the user has already opened once. Later requests
  // navigate again to avoid treating an old scoreboard as a live update.
  if (tab.url !== target || usedDocuments.has(key) || tab.status === 'loading') {
    el('status').textContent = `Abriendo en SofaScore: ${path}…`;
    await navigateTab(target);
  }
  const results = await chrome.scripting.executeScript({
    target: {tabId: selectedTab}, world: 'ISOLATED',
    func: expectedPath => {
      if (location.origin !== 'https://www.sofascore.com') {
        return {error: 'La navegación salió de SofaScore. Revisa la pestaña seleccionada.'};
      }
      if (location.pathname.includes('captcha')) {
        return {error: 'La navegación también ha llegado a la verificación de SofaScore. Complétala manualmente en su pestaña y vuelve a intentarlo.'};
      }
      if (location.pathname !== expectedPath) {
        return {error: 'SofaScore redirigió a otra página; no se ha recibido el enlace de datos solicitado.'};
      }
      const text = document.querySelector('pre')?.textContent || document.body?.innerText || '';
      if (text.length > 8 * 1024 * 1024) return {error: 'La respuesta de SofaScore supera el tamaño permitido.'};
      let data;
      try { data = JSON.parse(text); }
      catch { return {error: 'El enlace abierto no muestra JSON válido. Revisa si la pestaña indica Access denied o una verificación.'}; }
      const key = expectedPath.endsWith('/live') ? 'events'
        : expectedPath.endsWith('/statistics') ? 'statistics' : 'event';
      if (!data || (key === 'event'
          ? !data.event || typeof data.event !== 'object' || Array.isArray(data.event)
          : !Array.isArray(data[key]))) {
        return {error: `El enlace muestra JSON pero no contiene ${key} válido. Puede ser una respuesta de acceso denegado.`};
      }
      return {data};
    },
    args: [path]
  });
  const result = results[0]?.result || {error: 'No se pudo leer la página de datos.'};
  if (result.data) usedDocuments.add(key);
  return result;
}

function controls(active) {
  for (const id of ['token', 'tab', 'mode', 'connect', 'refresh', 'open']) el(id).disabled = active;
  el('disconnect').disabled = !active;
}

async function connect() {
  if (connected) return;
  token = el('token').value.trim();
  selectedTab = Number(el('tab').value);
  transport = el('mode').value || 'navigation';
  if (!token || !selectedTab) {
    el('status').textContent = 'Introduce el código y selecciona una pestaña de SofaScore.';
    return;
  }
  connected = true;
  controls(true);
  let lastOutcome = '';
  let stage = 'conectar con Python';
  try {
    await local('/status');
    el('status').textContent = 'Conectado. Esperando consultas de Python…';
    while (connected) {
      stage = 'esperar una consulta de Python';
      const {job} = await local('/jobs');
      if (job) {
        stage = 'consultar SofaScore';
        el('status').textContent = `Consultando ${job.path}…`;
        const result = await executeJob(job);
        lastOutcome = result.error
          ? `Última consulta: ${result.error}`
          : `Datos obtenidos de SofaScore · ${job.path}`;
        stage = 'entregar la respuesta a Python';
        const delivery = await local('/result', result);
        if (delivery.expired) {
          el('status').textContent = 'Consulta anterior cancelada. Esperando el partido seleccionado…';
          continue;
        }
        el('status').textContent = result.error
          ? result.error
          : `Datos enviados a Python · ${new Date().toLocaleTimeString()} · ${job.path}`;
        // Python ends discovery on an access error. Keep the original cause
        // visible instead of overwriting it with the following socket failure.
        if (result.error) break;
      }
      await pause(1000);
    }
  } catch (error) {
    el('status').textContent = `Conexión detenida al ${stage}: ${error.message}. ${lastOutcome} Comprueba que Python sigue abierto y esperando consultas.`;
  } finally {
    connected = false;
    token = '';
    controls(false);
  }
}

el('refresh').addEventListener('click', () => refreshTabs().catch(e => {el('status').textContent = e.message;}));
el('open').addEventListener('click', () => chrome.tabs.create({url: 'https://www.sofascore.com/basketball'}));
el('connect').addEventListener('click', connect);
el('disconnect').addEventListener('click', () => {
  connected = false;
  el('status').textContent = 'Desconectando…';
});
refreshTabs().catch(e => {el('status').textContent = e.message;});
