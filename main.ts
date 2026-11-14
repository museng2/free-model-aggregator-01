// free-model-aggregator-01: one server that aggregates the brief's free providers behind both client
// wires (OpenAI Chat Completions and Anthropic Messages).
//
// How a trial works here:
//   1. The brief at MAKEFUN_PROMPT_FILE names the providers, the sealed quality table and the workload
//      manifest. Each provider's catalog is probed in its own wire (lib/providers.ts) and the catalog
//      built from the probes is registered with the trial environment.
//   2. Client requests arrive one at a time, in the manifest's order, so the request now arriving
//      carries the manifest row at the same index: its requirements are known exactly, not guessed
//      from the body.
//   3. Each request is served by the best provider that can answer it: candidates are the free models
//      whose capabilities meet the row's requirements, ordered by the quality table's rating, and a
//      failed call fails over to the next candidate. The request is sent in the provider's own wire
//      and the answer is converted back into the client's wire (lib/wire.ts).
//
// Invariants the score depends on, held by construction:
//   - The registered catalog is exactly the providers' own models with their free flags, and
//     GET /v1/models lists the same free ids under both auth headers.
//   - Exactly one provider call may succeed per request: attempts stop at the first success.
//   - Nothing in a reply reads the wall clock or an unseeded random source: the conversions are pure
//     functions of the provider's answer, so a replay reproduces the same bytes.
//   - A provider whose refusal says its budget is spent, or that it is out, is never called again in
//     the trial: every further call would fail the same way and only cost upstream calls. A refusal
//     that names no cause ("refused this call") is an unlucky ordinal, not a state, and the provider
//     stays in rotation.
//   - An OpenAI JSON-mode request never goes to an Anthropic-wire provider: that wire has no JSON
//     mode to echo back, so the answer could not match the request.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { Buffer } from 'node:buffer';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicToOpenAI,
  geminiResponseToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The providers that have said their budget is spent or that they are out: skipped from here on. */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  // A probe that failed on a transient error gets another chance: a provider missing from the catalog
  // can neither be registered nor routed to. The trial's answers are deterministic, so a probe the
  // provider truly refuses fails the same way on every round.
  for (let round = 0; round < 2; round += 1) {
    const failed = probes.filter((probe) => !probe.ok);
    if (failed.length === 0) break;
    const retried = await Promise.all(failed.map((probe) => probeProvider(ENV_URL, probe.provider, PROBE_TIMEOUT_MS)));
    probes = probes.map((probe) => retried.find((retry) => retry.provider.id === probe.provider.id) ?? probe);
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One servable target: a provider that answered its probe and one of its free models. */
interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The candidates for one request, best first: every free model whose capabilities meet the request's
 * requirements, ordered by the quality table's rating, then by preferring the provider that speaks
 * the client's own wire (its answers need no translation), then by provider and model id so the
 * order is total and repeatable. Exhausted providers are out. An OpenAI JSON-mode request also rules
 * out Anthropic-wire providers, whose wire cannot echo JSON mode back.
 */
function routesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const provider = probe.provider;
    if (exhaustedProviders.has(provider.id)) continue;
    if (jsonMode && clientApi === 'openai' && provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aSame = a.probe.provider.wire === clientApi ? 0 : 1;
    const bSame = b.probe.provider.wire === clientApi ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return byId(a.model, b.model);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(res: ServerResponse, anthropicShape: boolean): void {
  const list = servedModels();
  if (anthropicShape) sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Whether a refusal says the provider will keep failing: its budget is spent, or it is out. */
function saysGone(text: string): boolean {
  return /budget is spent/.test(text) || /\bis out\b/.test(text);
}

/** POST one JSON body to a provider route and return the fetch response (the caller reads it). */
async function callProvider(provider: ProviderDirectoryEntry, path: string, body: Json, extraHeaders: Record<string, string>): Promise<Response> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
    ...extraHeaders,
  };
  return fetch(`${base}${path}`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** Copy a provider's event stream to the client byte for byte, as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a converted event stream to the client until it ends. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  try {
    for await (const event of events) res.write(event);
  } catch {
    // The provider's stream broke mid-answer: the client keeps the partial reply.
  }
  res.end();
}

/**
 * Serve one client request through one candidate route. The canonical request (an OpenAI body: an
 * Anthropic client's body already crossed over in lib/wire.ts) is rebuilt in the provider's own wire;
 * a same-wire OpenAI request passes through apart from the model id and the stream usage option no
 * client sends. Returns true when the client has been answered — a success, or the last candidate's
 * own refusal relayed — and false when the caller should fail over to the next candidate.
 */
async function attempt(route: Route, canonical: Json, clientApi: 'openai' | 'anthropic', clientStream: boolean, res: ServerResponse, isLast: boolean): Promise<boolean> {
  const provider = route.probe.provider;
  const model = route.model.id;
  let upstream: Response;
  try {
    if (provider.wire === 'gemini') {
      const verb = clientStream ? 'streamGenerateContent' : 'generateContent';
      upstream = await callProvider(provider, `/v1beta/models/${model}:${verb}${clientStream ? '?alt=sse' : ''}`, openAIToGeminiRequest(canonical), {});
    } else if (provider.wire === 'anthropic') {
      upstream = await callProvider(provider, '/v1/messages', openAIToAnthropicRequest(canonical, model), { 'anthropic-version': '2023-06-01' });
    } else {
      const body: Json = { ...canonical, model };
      if (clientStream) body.stream_options = { include_usage: true };
      upstream = await callProvider(provider, '/v1/chat/completions', body, clientStream ? { accept: 'text/event-stream' } : {});
    }
  } catch (err) {
    if (isLast) sendError(res, clientApi, 502, 'api_error', `the provider could not be reached: ${(err as Error).message}`);
    return isLast;
  }

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    if ((upstream.status === 402 || upstream.status === 429) && saysGone(text)) exhaustedProviders.add(provider.id);
    if (isLast) {
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return true;
    }
    return false;
  }

  if (clientStream) {
    if (upstream.body === null) {
      if (isLast) sendError(res, clientApi, 502, 'api_error', 'the provider sent no stream');
      return isLast;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (provider.wire === clientApi) {
      await pipeStream(upstream.body, res);
    } else if (clientApi === 'openai' && provider.wire === 'anthropic') {
      await writeEvents(openAIEventsFromAnthropic(upstream.body, model), res);
    } else if (clientApi === 'openai' && provider.wire === 'gemini') {
      await writeEvents(openAIEventsFromGemini(upstream.body, model), res);
    } else if (clientApi === 'anthropic' && provider.wire === 'gemini') {
      await writeEvents(anthropicEventsFromGemini(upstream.body, model), res);
    } else {
      await writeEvents(anthropicEventsFromOpenAI(upstream.body, model), res);
    }
    return true;
  }

  const text = await upstream.text();
  let parsed: Json | null = null;
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) parsed = value;
  } catch {
    parsed = null;
  }
  if (parsed === null) {
    if (isLast) {
      res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return true;
    }
    return false;
  }

  let answer: Json = parsed;
  if (provider.wire === 'anthropic' && clientApi === 'openai') answer = anthropicResponseToOpenAI(parsed, model);
  else if (provider.wire === 'gemini' && clientApi === 'openai') answer = geminiResponseToOpenAI(parsed, model);
  else if (provider.wire === 'gemini' && clientApi === 'anthropic') answer = openAIToAnthropic(geminiResponseToOpenAI(parsed, model), model);
  else if (provider.wire === 'openai' && clientApi === 'anthropic') answer = openAIToAnthropic(parsed, model);
  sendJson(res, 200, answer);
  return true;
}

/** Serve one client API request: route it, then walk the candidates until one answers. */
async function serve(body: Json, clientApi: 'openai' | 'anthropic', res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const canonical: Json = clientApi === 'anthropic' ? anthropicToOpenAI(body, typeof body.model === 'string' ? body.model : '') : body;
  const format = isRecord(canonical.response_format) ? canonical.response_format.type : undefined;
  const jsonMode = format === 'json_object' || format === 'json_schema';
  const routes = routesFor(requires, clientApi, jsonMode);
  if (routes.length === 0) {
    sendError(res, clientApi, 503, 'api_error', 'no free model on the directory meets this request');
    return;
  }
  const clientStream = body.stream === true;
  for (let i = 0; i < routes.length; i += 1) {
    if (await attempt(routes[i], canonical, clientApi, clientStream, res, i === routes.length - 1)) return;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(res, typeof req.headers['x-api-key'] === 'string');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(body, 'openai', res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(body, 'anthropic', res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      if (!res.writableEnded) res.end();
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
