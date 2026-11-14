// The wire translations of the aggregator.
//
// Three upstream wires (OpenAI Chat Completions, Anthropic Messages, Gemini generateContent) and two
// client wires (OpenAI and Anthropic) meet here. Every conversion pivots through one canonical shape —
// an OpenAI Chat Completions request or response — so each direction is written once:
//
//   client body --(anthropicToOpenAI, when the client is Anthropic)--> canonical OpenAI request
//   canonical   --(openAIToAnthropicRequest / openAIToGeminiRequest)--> the provider's own wire
//   provider answer --(anthropicResponseToOpenAI / geminiResponseToOpenAI)--> canonical OpenAI answer
//   canonical   --(openAIToAnthropic, when the client is Anthropic)--> the client's wire
//
// Same-wire traffic never takes this path: main.ts passes it through untouched.
//
// The conversions are written against what a provider can echo back: the system prompt, the last user
// message's text, the tool names in order, max_tokens, the stop sequences, any image (by its base64
// bytes) and JSON mode. Each wire's parser reads those fields from its own spelling, so the spelling
// chosen here is the one that parses back to the same values:
//   - JSON mode exists on the OpenAI wire (response_format) and on Gemini's (generationConfig
//     .responseMimeType); Anthropic's parser never reports it, which is why main.ts never routes an
//     OpenAI JSON-mode request to an Anthropic-wire provider.
//   - A token cap the client did not set stays unset on every wire: an invented default would echo
//     back as a different max_tokens than the client sent.
//   - Stop reasons travel by family: stop/end_turn/STOP, length/max_tokens/MAX_TOKENS,
//     tool_calls/tool_use, content_filter/refusal/SAFETY map onto each other.
//
// Everything is a pure function of its arguments — no clock, no randomness — so a replay that serves
// the same provider answers produces the same bytes. The model listings use fixed timestamps for the
// same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The text blocks of a content value, joined as the wire parsers join them ('\n' between blocks). */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A response text: the string itself, or its text blocks joined with no separator (an answer is one text). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

// ---------------------------------------------------------------------------
// Client Anthropic -> canonical OpenAI (requests), canonical -> client (answers)
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blocksText(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Chat Completions answer as an Anthropic Messages object naming `model` back to the client. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = answerText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

// ---------------------------------------------------------------------------
// Canonical OpenAI -> Anthropic wire (requests), Anthropic wire -> canonical (answers)
// ---------------------------------------------------------------------------

/** The system text of a canonical request, joined exactly as an OpenAI parser joins system messages. */
function systemTextOf(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = blocksText(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** One canonical content part as an Anthropic block: text or image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[3] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * A canonical OpenAI request as an Anthropic Messages request naming `model`. Fields the client did
 * not set stay unset (an invented max_tokens or stop list would echo back wrong), and the tool
 * definitions keep their names and order, which is what an Anthropic parser reports.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const out: Json = { model, messages: [] };
  const system = systemTextOf(body);
  if (system !== '') out.system = system;
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((part) => {
        if (!isRecord(part)) return [];
        const block = anthropicBlockOf(part);
        return block === null ? [] : [block];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  out.messages = messages;
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (cap !== null) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as a canonical Chat Completions answer. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const text = answerText(blocks);
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

// ---------------------------------------------------------------------------
// Canonical OpenAI -> Gemini wire (requests), Gemini wire -> canonical (answers)
// ---------------------------------------------------------------------------

/** One canonical content part as a Gemini part: text, inline image data, a file URI, or nothing. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[3] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/**
 * A canonical OpenAI request as a Gemini generateContent request (the model rides in the URL, which
 * main.ts builds): contents, an optional systemInstruction, the tools as functionDeclarations in
 * their original order, and a generationConfig holding only what the client set — plus
 * responseMimeType when the client asked for JSON mode, the one field Gemini's parser reports back
 * as json_mode.
 */
export function openAIToGeminiRequest(body: Json): Json {
  const out: Json = {};
  const system = systemTextOf(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const callNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      const name = callNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? '';
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: toolResultText(raw.content) } } }] });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const converted = geminiPartOf(part);
        if (converted !== null) parts.push(converted);
      }
    }
    if (raw.role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string') callNames.set(call.id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (cap !== null) config.maxOutputTokens = cap;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishOf(reason: unknown, hasToolCall: boolean): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'PROHIBITED_CONTENT') return 'content_filter';
  if (hasToolCall) return 'tool_calls';
  return 'stop';
}

/** A Gemini generateContent answer as a canonical Chat Completions answer. */
export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: geminiFinishOf(candidate.finishReason, toolCalls.length > 0),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The raw frames of an upstream SSE stream, as {event, data} pairs in arrival order. */
export async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): { event: string | null; data: string }[] => {
    const out: { event: string | null; data: string }[] = [];
    for (const frame of text.split('\n\n')) {
      if (frame.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) out.push({ event, data: data.join('\n') });
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const parsed of framesOf(frame)) yield parsed;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const parsed of framesOf(buffer)) yield parsed;
  } finally {
    reader.releaseLock();
  }
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One OpenAI chunk frame as a string. */
function openAIChunk(id: string, model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, { id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk first, text as content deltas, tool_use
 * blocks as tool_calls deltas, then the finish chunk (the stop reason mapped), a usage chunk carrying
 * the stream's real token counts, and [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let id = 'chatcmpl-anthropic';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolIndexByBlock = new Map<number, number>();
  let nextToolIndex = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(id, model, { role: 'assistant', content: '' }, null)];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(id, model, {}, finishReasonOf(stopReason ?? 'end_turn')),
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message)) {
      if (typeof data.message.id === 'string') id = `chatcmpl-${data.message.id}`;
      if (isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      continue;
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const blockIndex = typeof data.index === 'number' ? data.index : toolIndexByBlock.size;
      const toolIndex = nextToolIndex;
      nextToolIndex += 1;
      toolIndexByBlock.set(blockIndex, toolIndex);
      yield* start();
      yield openAIChunk(
        id,
        model,
        {
          tool_calls: [
            {
              index: toolIndex,
              id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${toolIndex}`,
              type: 'function',
              function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
            },
          ],
        },
        null,
      );
      continue;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield* start();
        yield openAIChunk(id, model, { content: data.delta.text }, null);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const toolIndex = toolIndexByBlock.get(typeof data.index === 'number' ? data.index : -1);
        if (toolIndex !== undefined) {
          yield* start();
          yield openAIChunk(id, model, { tool_calls: [{ index: toolIndex, function: { arguments: data.delta.partial_json } }] }, null);
        }
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
    if (frame.event === 'message_stop') {
      yield* start();
      yield* finish();
    }
  }
  yield* start();
  yield* finish();
}

/**
 * A Gemini SSE stream as OpenAI chunks: text parts as content deltas, functionCall parts as tool_calls
 * deltas, then the finish chunk, the usage chunk from the final frame's usageMetadata, and [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const id = 'chatcmpl-gemini';
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason: string | null = null;
  let toolCount = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(id, model, { role: 'assistant', content: '' }, null)];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(id, model, {}, finishReason ?? 'stop'),
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string' && part.text !== '') {
        yield* start();
        yield openAIChunk(id, model, { content: part.text }, null);
      } else if (isRecord(part.functionCall)) {
        yield* start();
        const index = toolCount;
        toolCount += 1;
        yield openAIChunk(
          id,
          model,
          {
            tool_calls: [
              {
                index,
                id: `call_${index}`,
                type: 'function',
                function: {
                  name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
                  arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
                },
              },
            ],
          },
          null,
        );
      }
    }
    if (typeof candidate.finishReason === 'string') finishReason = geminiFinishOf(candidate.finishReason, toolCount > 0);
  }
  yield* start();
  yield* finish();
}

/**
 * A Gemini SSE stream as Anthropic events: message_start (the input count is only known at the end of
 * a Gemini stream, so it opens at 0, as a converted stream must), one text block carrying the pieces,
 * tool blocks for any functionCall parts, then message_delta with the mapped stop reason and the real
 * output count, and message_stop.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let textOpen = false;
  let finished = false;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let toolCount = 0;
  const openTools: number[] = [];

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_gemini',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    ];
  };
  const openText = (): string[] => {
    if (textOpen) return [];
    textOpen = true;
    return [sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textOpen) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
    for (const index of [...openTools].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string' && part.text !== '') {
        yield* start();
        yield* openText();
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } });
      } else if (isRecord(part.functionCall)) {
        yield* start();
        const index = 1 + toolCount;
        toolCount += 1;
        openTools.push(index);
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: `toolu_${index}`,
            name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
            input: {},
          },
        });
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
        });
      }
    }
    if (typeof candidate.finishReason === 'string') stopReason = stopReasonOf(geminiFinishOf(candidate.finishReason, toolCount > 0));
  }
  yield* start();
  yield* finish();
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data.trim() === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  yield* startEvents();
  yield* finishEvents(pendingStop ?? 'end_turn');
}
