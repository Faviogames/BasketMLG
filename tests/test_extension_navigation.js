// Sin navegador ni red: respuesta tardía, nueva consulta y autenticación.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id === 'token' ? 'test-pairing' : id === 'mode' ? 'fetch' : '1',
    addEventListener() {}, replaceChildren() {}, appendChild() {},
  });
  return elements.get(id);
}
let jobs = 0;
let deliveries = 0;
const context = vm.createContext({
  document: {getElementById: element},
  chrome: {tabs: {query: async () => []}},
  AbortSignal, Date, Set, Number, String, Error,
  setTimeout: callback => { callback(); },
  fetch: async (url, options) => {
    assert.equal(options.headers.Authorization, 'Bearer test-pairing');
    let status = 200;
    let data = {};
    if (url.endsWith('/jobs')) {
      jobs++;
      if (jobs <= 2) data = {job: {id: String(jobs), path: '/api/v1/event/' + jobs}};
      else {
        vm.runInContext('connected = false', context);
        data = {job: null};
      }
    }
    if (url.endsWith('/result')) status = ++deliveries === 1 ? 409 : 200;
    return {status, ok: status === 200, json: async () => data};
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../browser_extension/panel.js'), 'utf8'), context);
vm.runInContext('executeJob = async job => ({id: job.id, data: {event: {id: job.id}}})', context);
(async () => {
  await vm.runInContext('connect()', context);
  assert.equal(deliveries, 2, 'El HTTP 409 no debe desconectar el panel');
  assert.equal(jobs, 3);
  assert.match(element('status').textContent, /Datos enviados/);
  // Otros rechazos conservan el tratamiento de error.
  context.fetch = async () => ({status: 403, ok: false});
  await assert.rejects(vm.runInContext("local('/result', {})", context), /Código rechazado/);
  console.log('OK: respuesta caducada, siguiente partido y rechazo de credencial.');
})().catch(error => { console.error(error); process.exitCode = 1; });
