import { error, ingest, routeApi } from "./api";
import type { Env } from "./model";
import { validatePayload, ValidationError } from "./validation";

const MAX_INGEST_BYTES = 8 * 1024 * 1024;

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/api/v1/ingest") {
        if (request.method !== "POST") return error(405, "method_not_allowed", "Use POST for ingestion");
        if (!env.BENCHMARK_INGEST_TOKEN) return error(503, "ingestion_unconfigured", "Ingestion is not configured");
        if (!(await authorized(request, env.BENCHMARK_INGEST_TOKEN))) return error(401, "unauthorized", "A valid ingestion bearer token is required");
        const declaredLength = Number.parseInt(request.headers.get("Content-Length") ?? "0", 10);
        if (declaredLength > MAX_INGEST_BYTES) return error(413, "payload_too_large", "Payload exceeds 8 MiB");
        const body = await request.text();
        if (new TextEncoder().encode(body).byteLength > MAX_INGEST_BYTES) return error(413, "payload_too_large", "Payload exceeds 8 MiB");
        let raw: unknown;
        try { raw = JSON.parse(body); } catch { return error(400, "invalid_json", "Request body is not valid JSON"); }
        const payload = validatePayload(raw);
        return ingest(env, payload, await sha256(body));
      }
      if (url.pathname.startsWith("/api/")) return routeApi(request, env, url);
      const response = await env.ASSETS.fetch(request);
      return withSecurityHeaders(response);
    } catch (cause) {
      if (cause instanceof ValidationError) return error(400, "invalid_payload", cause.message);
      console.error("request failed", cause);
      return error(500, "internal_error", "The request could not be completed");
    }
  },
} satisfies ExportedHandler<Env>;

async function authorized(request: Request, expectedToken: string): Promise<boolean> {
  const authorization = request.headers.get("Authorization");
  if (!authorization?.startsWith("Bearer ")) return false;
  const actual = authorization.slice(7);
  if (!actual || actual.length !== expectedToken.length) return false;
  const [actualHash, expectedHash] = await Promise.all([sha256Bytes(actual), sha256Bytes(expectedToken)]);
  let difference = 0;
  for (let index = 0; index < actualHash.length; index += 1) difference |= actualHash[index] ^ expectedHash[index];
  return difference === 0;
}

async function sha256(value: string): Promise<string> {
  return [...await sha256Bytes(value)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Bytes(value: string): Promise<Uint8Array> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return new Uint8Array(digest);
}

function withSecurityHeaders(response: Response): Response {
  const secured = new Response(response.body, response);
  secured.headers.set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
  secured.headers.set("Referrer-Policy", "no-referrer");
  secured.headers.set("X-Content-Type-Options", "nosniff");
  secured.headers.set("X-Frame-Options", "DENY");
  return secured;
}
