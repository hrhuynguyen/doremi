// Doremi – static server + Claude endpoints (session design, session reflection).
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, normalize, join } from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

const PUBLIC_DIR = fileURLToPath(new URL("./public/", import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.CLAUDE_MODEL || "claude-opus-5";
const client = new Anthropic(); // ANTHROPIC_API_KEY or a stored profile

const MODES = ["major pentatonic", "minor pentatonic", "major", "natural minor", "dorian", "lydian", "mixolydian"];
const ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const SceneSchema = z.object({
  name: z.string().describe("Two or three word name for the session"),
  root: z.enum(ROOTS).describe("Root note"),
  mode: z.enum(MODES).describe("Scale / mode"),
  bpm: z.number().describe("Breathing pacer tempo between 48 and 80"),
  reverb: z.number().describe("Reverb amount between 0.3 and 0.9"),
  brightness: z.number().describe("Filter brightness between 0.2 and 0.9"),
  padWave: z.enum(["sine", "triangle", "sawtooth", "square"]).describe("Pad oscillator shape"),
  palette: z.array(z.string()).describe("Exactly three hex colors such as #7dd3fc that suit the mood"),
  guidance: z.string().describe("One or two warm, concrete sentences telling the user how to move, addressed as 'you'"),
  why: z.string().describe("One plain sentence on the music-therapy rationale for these choices"),
});

const DESIGN_SYSTEM = `You are the composer inside Doremi, a music-therapy instrument that people play with their body through a webcam.
How it is played: the left hand plucks a harp (horizontal position picks the note across three octaves of your chosen scale; a closed fist mutes it). The right hand holds a sustained chord (height picks one of four chords; how open the hand is sets brightness). Nodding the head plays a soft heartbeat drum with a low root note, tilting the head pans the sound, turning the head adds space, and smiling adds a shimmer layer. A breathing pacer pulses at the bpm you set, so 48-80 bpm gives 6-10 breaths per minute.
Given how the person says they feel, design one session. Use the iso principle: meet the current mood first, then lean the harmony toward where they want to go. Lower, slower, minor or dorian choices meet heaviness; pentatonic modes guarantee no wrong notes for anxious or self-critical people; lydian or major with brighter filters lift. Keep guidance plain and physical (which hand, how fast, where to start). Never promise clinical outcomes.`;

const REFLECT_SYSTEM = `You are the reflective voice at the end of a Doremi music-therapy session. You receive how the person felt beforehand, statistics from how they moved, and up to two previous sessions from the same device when they exist: how often each hand was visible, average speed, range of motion as a percentage of the screen, a calm score over time (higher is slower, steadier movement), head tilt range, number of head nods (each nod plays a heartbeat drum), time spent smiling, notes plucked and chord changes, and duetPct (how much of the session a second person was in frame playing alongside them).
Write three to five sentences in the second person, warm and specific, grounded in the numbers without listing them all. Notice one thing that stood out (for example a hand that did most of the work, movement that slowed over time, or a wide head-tilt range) and connect it gently to how they said they felt. When previousSessions is not empty, one sentence must explicitly compare today with the most recent previous session using the numbers (for example: last time your right hand was in frame 40% of the time, today 75%). Pick the largest change you can find across hand presence, range, calm, or nods. When duetPct is 20 or more, acknowledge that they played with someone and say something about playing together. End with one concrete suggestion for next time, phrased as an invitation. No headers, no bullet points, no medical claims, no emojis.`;

function clamp(v, a, b) { return Math.min(b, Math.max(a, Number.isFinite(v) ? v : a)); }
function sanitize(scene) {
  const hex = /^#[0-9a-f]{6}$/i;
  const palette = (scene.palette || []).filter((c) => hex.test(c)).slice(0, 3);
  const fallback = ["#7dd3fc", "#a5b4fc", "#f0abfc"];
  while (palette.length < 3) palette.push(fallback[palette.length]);
  return {
    ...scene,
    bpm: Math.round(clamp(scene.bpm, 44, 90)),
    reverb: clamp(scene.reverb, 0.2, 0.95),
    brightness: clamp(scene.brightness, 0.1, 1),
    palette,
  };
}

async function design({ mood }) {
  const response = await client.messages.parse({
    model: MODEL,
    max_tokens: 2000,
    output_config: { effort: "low", format: zodOutputFormat(SceneSchema) },
    system: DESIGN_SYSTEM,
    messages: [{ role: "user", content: `Here is how the person says they feel right now:\n\n"${mood}"\n\nDesign their session.` }],
  });
  if (response.stop_reason === "refusal") throw new Error("Claude declined to design this session.");
  if (!response.parsed_output) throw new Error("Claude returned an unreadable session.");
  return { scene: sanitize(response.parsed_output), model: response.model };
}

async function reflect({ mood, scene, summary }) {
  const response = await client.beta.messages.create({
    model: MODEL,
    max_tokens: 1200,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { effort: "low" },
    system: REFLECT_SYSTEM,
    messages: [{
      role: "user",
      content: JSON.stringify({ howTheyFeltBefore: mood || null, session: scene?.name, sessionStats: summary }, null, 2),
    }],
  });
  if (response.stop_reason === "refusal") throw new Error("Claude declined to reflect on this session.");
  const text = response.content.filter((b) => b.type === "text").map((b) => b.text).join("").trim();
  return { reflection: text, model: response.model };
}

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon" };

function send(res, status, body, type = "application/json") {
  res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  res.end(type === "application/json" ? JSON.stringify(body) : body);
}

async function readJson(req) {
  let size = 0; const chunks = [];
  for await (const chunk of req) { size += chunk.length; if (size > 1e6) throw new Error("Body too large"); chunks.push(chunk); }
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {};
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (req.method === "GET" && url.pathname === "/api/health") return send(res, 200, { ok: true, model: MODEL });
    if (req.method === "POST" && url.pathname === "/api/design") return send(res, 200, await design(await readJson(req)));
    if (req.method === "POST" && url.pathname === "/api/reflect") return send(res, 200, await reflect(await readJson(req)));
    if (req.method !== "GET") return send(res, 405, { error: "Method not allowed" });

    const rel = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    const file = join(PUBLIC_DIR, rel === "/" || rel === "\\" ? "index.html" : rel);
    if (!file.startsWith(PUBLIC_DIR)) return send(res, 403, { error: "Forbidden" });
    const data = await readFile(file);
    return send(res, 200, data, TYPES[extname(file)] || "application/octet-stream");
  } catch (err) {
    if (err?.code === "ENOENT") return send(res, 404, { error: "Not found" });
    const status = err?.status && err.status >= 400 && err.status < 600 ? err.status : 500;
    console.error(`[${req.method} ${url.pathname}]`, err?.message || err);
    return send(res, status, { error: err?.message || "Server error" });
  }
});

server.listen(PORT, () => {
  console.log(`Doremi running at http://localhost:${PORT}  (model: ${MODEL})`);
});
