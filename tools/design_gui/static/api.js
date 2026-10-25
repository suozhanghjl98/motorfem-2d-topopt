"use strict";
/* tools/design_gui/static/api.js —— fetch 封装 + token。 */

const DESIGN_TOKEN =
  (document.querySelector('meta[name="design-gui-token"]') || {}).content || "";

async function apiFetch(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (DESIGN_TOKEN) headers["X-Design-GUI-Token"] = DESIGN_TOKEN;
  const r = await fetch(path, Object.assign({}, opts, {headers}));
  const ctype = r.headers.get("Content-Type") || "";
  let data;
  if (ctype.includes("application/json")) {
    data = await r.json();
  } else {
    const text = await r.text();
    data = {ok: r.ok, text};
  }
  if (!r.ok) {
    const msg = data && data.error ? data.error : `HTTP ${r.status}`;
    const err = new Error(msg);
    err.status = r.status;
    err.data = data;
    throw err;
  }
  return data;
}

function apiGet(path) {
  return apiFetch(path, {method: "GET"});
}

function apiPost(path, body) {
  return apiFetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

function fmt(x, n = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(n);
}
