// Doremi – body-controlled music therapy instrument.
import * as mpVision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const { HandLandmarker, FaceLandmarker, FilesetResolver } = mpVision.default ?? mpVision;

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const SCALES = {
  "major pentatonic": [0, 2, 4, 7, 9],
  "minor pentatonic": [0, 3, 5, 7, 10],
  major: [0, 2, 4, 5, 7, 9, 11],
  "natural minor": [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
};
const ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const DEFAULT_SCENE = {
  name: "Calm Waters", root: "D", mode: "major pentatonic", bpm: 60, reverb: 0.6, brightness: 0.45, padWave: "triangle",
  palette: ["#7dd3fc", "#a5b4fc", "#f0abfc"],
  guidance: "Raise your hands slowly. Your left hand plays the harp, your right hand holds the chords. Tilt your head to move the sound around you.",
  why: "A pentatonic scale has no wrong notes, so every movement you make sounds right.",
};
const PRESETS = {
  calm: DEFAULT_SCENE,
  uplift: { name: "Sunny Side", root: "G", mode: "lydian", bpm: 66, reverb: 0.5, brightness: 0.7, padWave: "triangle",
    palette: ["#fcd34d", "#fdba74", "#86efac"],
    guidance: "Float your right hand up high and open it wide. Skip your left hand across the strings like you're splashing through puddles.",
    why: "Lydian's raised fourth sounds like a question that keeps opening, which lifts a flat mood without forcing it." },
  focus: { name: "Quiet Garden", root: "A", mode: "dorian", bpm: 60, reverb: 0.45, brightness: 0.5, padWave: "sine",
    palette: ["#5eead4", "#a3e635", "#fde68a"],
    guidance: "Rest one slow chord under your right hand. Move your left hand in small, deliberate steps, one string at a time.",
    why: "Dorian is steady and a little serious, and small movements over a steady pulse gather a scattered mind." },
  wake: { name: "Morning Light", root: "C", mode: "major", bpm: 72, reverb: 0.4, brightness: 0.75, padWave: "triangle",
    palette: ["#fde047", "#fb923c", "#f9a8d4"],
    guidance: "Reach both arms up and out like a stretch. Sweep your left hand high across the strings, and nod on every out-breath to set a gentle pulse.",
    why: "A bright major key at a walking tempo wakes the body without the jolt of caffeine or a loud alarm." },
  wind: { name: "Night Drift", root: "F", mode: "major pentatonic", bpm: 48, reverb: 0.85, brightness: 0.3, padWave: "sine",
    palette: ["#818cf8", "#c4b5fd", "#67e8f9"],
    guidance: "Keep your hands low and slow, close to your body. Let each note ring out fully before you move to the next one.",
    why: "Six breaths a minute with soft, low sounds nudges the nervous system toward rest before sleep." },
  play: { name: "Bubble Pop", root: "C", mode: "major pentatonic", bpm: 84, reverb: 0.35, brightness: 0.9, padWave: "square",
    palette: ["#f472b6", "#a3e635", "#38bdf8"],
    guidance: "Wave, wiggle, and jump! Big fast moves make lots of notes. Nod your head to play the drum and smile to make hearts.",
    why: "There are no wrong notes, so kids and sensory-sensitive players get instant success and plenty of movement." },
  release: { name: "Deep Tide", root: "E", mode: "minor pentatonic", bpm: 52, reverb: 0.8, brightness: 0.35, padWave: "sawtooth",
    palette: ["#fb7185", "#c084fc", "#818cf8"],
    guidance: "Start low and heavy. Drag your left hand slowly across the low strings, then let both hands rise as the breath ring grows.",
    why: "Meeting a heavy mood in a minor key first, then rising slowly, is the iso principle: match it, then move it." },
};
const HAND_CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = [4, 8, 12, 16, 20];

let nodThreshold = 0.55; // face-heights per second; [ and ] tune it live
try { const v = parseFloat(localStorage.getItem("doremi.nodThreshold")); if (v > 0) nodThreshold = v; } catch {}
let swapHands = false; // press "h" to put the harp on the right side instead


const state = {
  phase: "intro", scene: DEFAULT_SCENE, scale: [], chords: [], chordNames: [],
  visionReady: false, cameraReady: false,
  body: { left: null, right: null, head: { present: false, roll: 0, yaw: 0, smile: 0, nose: null, noseY: null, noseT: 0, nodVel: 0, lastNod: 0, nod: false, pulse: 0 } },
  energy: 0, calm: 1, breath: 0,
  harp: { lastIdx: -1, lastTime: 0 },
  pad: { chordIdx: -1, lastSeen: 0, playing: false, lastArp: 0, arpStep: 0 },
  strings: [], particles: [], mood: "", lastFrame: -1, model: "", duet: false, recording: null,
};
let stats = newStats();
function newStats() {
  const hand = () => ({ frames: 0, speedSum: 0, minX: 1, maxX: 0, minY: 1, maxY: 0 });
  return { seconds: 0, frames: 0, notes: 0, chordChanges: 0, calmSum: 0, calmTrail: [], lastSample: 0,
    faceFrames: 0, smileFrames: 0, nods: 0, duetFrames: 0, rollMin: 0, rollMax: 0, left: hand(), right: hand() };
}

// ---------- canvas / camera geometry ----------
const video = $("cam"), canvas = $("scene"), ctx = canvas.getContext("2d");
let W = 1, H = 1;
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function coverMap() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const s = Math.max(W / vw, H / vh);
  return { s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2, vw, vh };
}
const toScreen = (p, m) => ({ x: m.ox + (1 - p.x) * m.vw * m.s, y: m.oy + p.y * m.vh * m.s });
const setStatus = (t) => ($("status").textContent = t);

// ---------- vision ----------
let hands, face;
async function initVision() {
  setStatus("Loading hand and face models…");
  const vision = await FilesetResolver.forVisionTasks(MP_WASM);
  [hands, face] = await Promise.all([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" }, runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    }),
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" }, runningMode: "VIDEO", numFaces: 2, outputFaceBlendshapes: true,
    }),
  ]);
  state.visionReady = true;
}
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  await video.play();
  state.cameraReady = true;
}

function handFeatures(lm, prev, dt, m) {
  const pts = lm.map((p) => toScreen(p, m));
  const wrist = pts[0];
  const palmPx = dist(wrist, pts[9]) || 1;
  const cx = (pts[0].x + pts[5].x + pts[9].x + pts[13].x + pts[17].x) / 5 / W;
  const cy = (pts[0].y + pts[5].y + pts[9].y + pts[13].y + pts[17].y) / 5 / H;
  const spread = [8, 12, 16, 20].reduce((s, i) => s + dist(pts[i], wrist), 0) / 4 / palmPx;
  const open = clamp((spread - 1.3) / 0.8, 0, 1);
  let vx = 0, vy = 0, speed = 0;
  if (prev && dt > 0) { vx = (cx - prev.x) / dt; vy = (cy - prev.y) / dt; speed = lerp(prev.speed, Math.hypot(vx, vy), 0.4); }
  return { x: clamp(cx, 0, 1), y: clamp(cy, 0, 1), open, vx, vy, speed, pts, palmPx };
}
function processHands(res, m, dt) {
  // Roles come from screen side, like a mirror: left half = harp hand, right half = chord hand.
  const feats = (res?.landmarks || []).map((lm) => handFeatures(lm, null, dt, m)).sort((a, b) => a.x - b.x);
  const next = { left: null, right: null };
  if (feats.length >= 2) { next.left = feats[0]; next.right = feats[feats.length - 1]; }
  else if (feats.length === 1) next[feats[0].x < 0.5 ? "left" : "right"] = feats[0];
  if (swapHands) [next.left, next.right] = [next.right, next.left];
  for (const key of ["left", "right"]) {
    const f = next[key], prev = state.body[key];
    if (f && prev && dt > 0) { f.vx = (f.x - prev.x) / dt; f.vy = (f.y - prev.y) / dt; f.speed = lerp(prev.speed, Math.hypot(f.vx, f.vy), 0.4); }
  }
  state.body.left = next.left; state.body.right = next.right;
}
function processFace(res, m) {
  const h = state.body.head;
  const lm = res?.faceLandmarks?.[0];
  state.duet = (res?.faceLandmarks?.length || 0) >= 2;
  if (!lm) { h.present = false; h.smile = lerp(h.smile, 0, 0.2); return; }
  h.present = true;
  const P = (i) => toScreen(lm[i], m);
  const a = P(33), b = P(263), nose = P(1), c1 = P(234), c2 = P(454);
  const roll = Math.atan2(a.y - b.y, a.x - b.x);
  const xl = Math.min(c1.x, c2.x), xr = Math.max(c1.x, c2.x);
  const yaw = (nose.x - xl) / Math.max(1, xr - xl) - 0.5;
  h.roll = lerp(h.roll, roll, 0.25); h.yaw = lerp(h.yaw, yaw, 0.2); h.nose = nose;
  // Nod: nose dropping quickly, measured in face-heights per second.
  const faceH = Math.max(1, dist(P(10), P(152)));
  const now = performance.now();
  if (h.noseY !== null && now > h.noseT) {
    const v = ((nose.y - h.noseY) / faceH) / Math.max(0.008, (now - h.noseT) / 1000);
    h.nodVel = lerp(h.nodVel, v, 0.7);
    h.nodPeak = Math.max(h.nodPeak || 0, h.nodVel);
    if (h.nodVel > nodThreshold && now - h.lastNod > 400) { h.nod = true; h.lastNod = now; }
  }
  h.noseY = nose.y; h.noseT = now;
  // Duet partner: their nod plays the drum too, their smile counts too.
  const lm2 = res.faceLandmarks[1];
  if (lm2) {
    const p2 = h.partner || (h.partner = { noseY: null, noseT: 0, nodVel: 0, lastNod: 0 });
    const n2 = toScreen(lm2[1], m), fh2 = Math.max(1, dist(toScreen(lm2[10], m), toScreen(lm2[152], m)));
    if (p2.noseY !== null && now > p2.noseT) {
      p2.nodVel = lerp(p2.nodVel, ((n2.y - p2.noseY) / fh2) / Math.max(0.008, (now - p2.noseT) / 1000), 0.7);
      if (p2.nodVel > nodThreshold && now - p2.lastNod > 400) { h.nod = true; p2.lastNod = now; h.partnerNose = n2; }
    }
    p2.noseY = n2.y; p2.noseT = now;
    const c2 = res.faceBlendshapes?.[1]?.categories;
    if (c2) { const g2 = (n) => c2.find((c) => c.categoryName === n)?.score ?? 0; h.smile = Math.max(h.smile, (g2("mouthSmileLeft") + g2("mouthSmileRight")) / 2); }
  } else h.partner = null;
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (cats) {
    const get = (n) => cats.find((c) => c.categoryName === n)?.score ?? 0;
    h.smile = lerp(h.smile, (get("mouthSmileLeft") + get("mouthSmileRight")) / 2, 0.2);
  }
}

// Mouse fallback when there is no camera: move = right hand, hold click = left hand.
const pointer = { x: 0.5, y: 0.5, down: false, active: false };
window.addEventListener("pointermove", (e) => { pointer.x = e.clientX / W; pointer.y = e.clientY / H; pointer.active = true; });
window.addEventListener("pointerdown", () => (pointer.down = true));
window.addEventListener("pointerup", () => (pointer.down = false));
function applyPointer(dt) {
  if (!pointer.active) return;
  const key = pointer.down ? "left" : "right", other = key === "left" ? "right" : "left";
  const prev = state.body[key];
  const vx = prev ? (pointer.x - prev.x) / dt : 0, vy = prev ? (pointer.y - prev.y) / dt : 0;
  state.body[key] = { x: pointer.x, y: pointer.y, open: 1, vx, vy, speed: prev ? lerp(prev.speed, Math.hypot(vx, vy), 0.4) : 0, pts: null, palmPx: 40 };
  state.body[other] = null;
}

// ---------- audio ----------
const audio = { ready: false };
let audioInit = null;
function initAudio() {
  if (audioInit) return audioInit;
  audioInit = (async () => {
    await Tone.start();
    const master = new Tone.Volume(-4).toDestination();
    const recorder = new Tone.Recorder();
    master.connect(recorder);
    const limiter = new Tone.Limiter(-2).connect(master);
    const reverb = new Tone.Reverb({ decay: 7, preDelay: 0.05, wet: 0.6 }).connect(limiter);
    const panner = new Tone.Panner(0).connect(reverb);
    const filter = new Tone.Filter({ frequency: 900, type: "lowpass", rolloff: -24, Q: 0.8 }).connect(panner);
    const pad = new Tone.PolySynth(Tone.Synth, { oscillator: { type: "triangle" }, envelope: { attack: 1.0, decay: 0.4, sustain: 0.8, release: 3.5 } }).connect(filter);
    pad.volume.value = -10;
    const harp = new Tone.PolySynth(Tone.Synth, { oscillator: { type: "sine" }, envelope: { attack: 0.004, decay: 0.9, sustain: 0, release: 1.4 } }).connect(filter);
    harp.volume.value = -9;
    const harpBright = new Tone.PolySynth(Tone.Synth, { oscillator: { type: "triangle" }, envelope: { attack: 0.004, decay: 0.5, sustain: 0, release: 0.8 } }).connect(filter);
    harpBright.volume.value = -22;
    const bell = new Tone.PolySynth(Tone.Synth, { oscillator: { type: "sine" }, envelope: { attack: 0.01, decay: 1.2, sustain: 0, release: 2 } }).connect(filter);
    bell.volume.value = -20;
    const drone = new Tone.Synth({ oscillator: { type: "sine" }, envelope: { attack: 5, decay: 0, sustain: 1, release: 8 } }).connect(reverb);
    drone.volume.value = -22;
    const kick = new Tone.MembraneSynth({ pitchDecay: 0.05, octaves: 6, envelope: { attack: 0.001, decay: 0.5, sustain: 0, release: 0.6 } }).connect(reverb);
    kick.volume.value = -8;
    const bass = new Tone.Synth({ oscillator: { type: "sine" }, envelope: { attack: 0.01, decay: 0.8, sustain: 0.2, release: 1.2 } }).connect(reverb);
    bass.volume.value = -10;
    const shimmer = new Tone.Synth({ oscillator: { type: "sine" }, envelope: { attack: 2, decay: 0, sustain: 1, release: 4 } }).connect(reverb);
    shimmer.volume.value = -60;
    await reverb.ready;
    Object.assign(audio, { ready: true, master, reverb, panner, filter, pad, harp, harpBright, bell, drone, shimmer, kick, bass, recorder });
  })();
  return audioInit;
}

function sanitizeScene(s) {
  const out = { ...DEFAULT_SCENE, ...s };
  if (!SCALES[out.mode]) out.mode = DEFAULT_SCENE.mode;
  if (!ROOTS.includes(out.root)) out.root = DEFAULT_SCENE.root;
  out.bpm = clamp(Number(out.bpm) || 60, 44, 90);
  out.reverb = clamp(Number(out.reverb) || 0.6, 0.2, 0.95);
  out.brightness = clamp(Number(out.brightness) || 0.45, 0.1, 1);
  if (!["sine", "triangle", "sawtooth", "square"].includes(out.padWave)) out.padWave = "triangle";
  const pal = (out.palette || []).filter((c) => /^#[0-9a-f]{6}$/i.test(c)).slice(0, 3);
  while (pal.length < 3) pal.push(DEFAULT_SCENE.palette[pal.length]);
  out.palette = pal;
  return out;
}
function applyScene(scene) {
  state.scene = scene;
  const intervals = SCALES[scene.mode];
  const root = Tone.Frequency(`${scene.root}3`);
  const notes = [];
  for (let o = 0; o < 3; o++) for (const i of intervals) notes.push(root.transpose(o * 12 + i).toNote());
  state.scale = notes;
  state.strings = notes.map(() => 0);
  const n = intervals.length;
  state.chords = [0, 1, 2, 3].map((k) => {
    const base = Math.round((k / 4) * n);
    const at = (d) => notes[Math.min(base + d, notes.length - 1)];
    return [at(0), at(2), at(4), at(n)];
  });
  state.chordNames = state.chords.map((c) => c[0].replace(/\d+/, ""));
  state.harp.lastIdx = -1; state.pad.chordIdx = -1;
  document.documentElement.style.setProperty("--accent", scene.palette[0]);
  if (audio.ready) {
    audio.reverb.wet.value = scene.reverb;
    audio.pad.set({ oscillator: { type: scene.padWave } });
    audio.drone.triggerAttack(`${scene.root}2`);
    audio.shimmer.triggerAttack(`${scene.root}6`);
  }
}

function updateAudio(now, dt) {
  if (!audio.ready) return;
  const { left, right, head } = state.body;
  const sc = state.scene;

  const cycle = (60 / sc.bpm) * 8;
  const t = (now / 1000) % cycle;
  state.breath = t < cycle * 0.4 ? t / (cycle * 0.4) : 1 - (t - cycle * 0.4) / (cycle * 0.6);
  const breathEase = 0.5 - 0.5 * Math.cos(Math.PI * state.breath);

  const speeds = [left?.speed, right?.speed].filter((v) => v != null);
  const inst = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0;
  state.energy = lerp(state.energy, clamp(inst / 1.5, 0, 1), 0.05);
  state.calm = 1 - state.energy;

  // Left hand: harp. Horizontal position picks the string; a fist mutes.
  const N = state.scale.length;
  if (left && left.open > 0.12) {
    const pos = left.x * N, idx = clamp(Math.floor(pos), 0, N - 1);
    // Held still: a soft pulse on the current string each beat so the hand is never silent.
    if (idx === state.harp.lastIdx && now - state.harp.lastTime >= 60000 / sc.bpm) {
      audio.harp.triggerAttackRelease(state.scale[idx], "8n", undefined, 0.22 + left.open * 0.1);
      state.harp.lastTime = now; state.strings[idx] = Math.max(state.strings[idx], 0.5);
      burst(handPoint(left, 8), "♪", sc.palette[0], 40);
    }
    if (idx !== state.harp.lastIdx && Math.abs(pos - (state.harp.lastIdx + 0.5)) > 0.75 && now - state.harp.lastTime > 60) {
      const note = state.scale[idx];
      const vel = clamp(0.35 + left.speed * 0.5, 0.35, 1);
      audio.harp.triggerAttackRelease(note, "8n", undefined, vel);
      audio.harpBright.triggerAttackRelease(note, "16n", undefined, vel * 0.6);
      state.harp.lastIdx = idx; state.harp.lastTime = now;
      state.strings[idx] = 1; stats.notes++;
      burst(handPoint(left, 8), Math.random() < 0.3 ? "♫" : "♪", sc.palette[0], 64 + left.speed * 30, note.replace(/\d/, ""));
    }
  } else state.harp.lastIdx = -1;

  // Right hand: chords. Height picks the chord, openness sets brightness.
  let targetCutoff = 500;
  if (right) {
    state.pad.lastSeen = now;
    const pos = (1 - right.y) * 4, ci = clamp(Math.floor(pos), 0, 3);
    if (ci !== state.pad.chordIdx && (state.pad.chordIdx < 0 || Math.abs(pos - (state.pad.chordIdx + 0.5)) > 0.7)) {
      if (state.pad.playing) audio.pad.releaseAll();
      audio.pad.triggerAttack(state.chords[ci], undefined, 0.6);
      state.pad.chordIdx = ci; state.pad.playing = true; stats.chordChanges++;
      burst(handPoint(right, 0), "𝄞", sc.palette[1], 110);
    }
    targetCutoff = 250 + right.open * 2600 * (0.5 + sc.brightness);
    const beat = 60000 / sc.bpm;
    if (state.pad.playing && now - state.pad.lastArp >= beat) {
      const chord = state.chords[state.pad.chordIdx];
      const note = Tone.Frequency(chord[state.pad.arpStep % chord.length]).transpose(12).toNote();
      audio.bell.triggerAttackRelease(note, "4n", undefined, 0.12 + right.open * 0.25);
      state.pad.arpStep++; state.pad.lastArp = now;
    }
  } else if (state.pad.playing && now - state.pad.lastSeen > 1200) {
    audio.pad.releaseAll(); state.pad.playing = false; state.pad.chordIdx = -1;
  }
  targetCutoff *= 0.75 + 0.5 * breathEase;
  audio.filter.frequency.value = lerp(audio.filter.frequency.value, targetCutoff, 0.08);
  audio.drone.volume.value = -24 + 6 * breathEase;

  // Head: nod plays a heartbeat, tilt pans, turning adds space, smiling adds shimmer.
  if (head.nod) {
    head.nod = false; head.pulse = 1; stats.nods++;
    audio.kick.triggerAttackRelease("C1", "8n");
    audio.bass.triggerAttackRelease(`${sc.root}1`, "4n", undefined, 0.8);
    if (head.nose) burst({ x: head.nose.x, y: head.nose.y + 40 }, "♥", sc.palette[2], 80);
  }
  const targetPan = head.present ? clamp(head.roll / 0.45, -1, 1) : 0;
  audio.panner.pan.value = lerp(audio.panner.pan.value, targetPan, 0.08);
  const targetWet = clamp(sc.reverb + Math.abs(head.yaw) * 1.2, 0.2, 0.95);
  audio.reverb.wet.value = lerp(audio.reverb.wet.value, targetWet, 0.05);
  audio.shimmer.volume.value = lerp(audio.shimmer.volume.value, -60 + head.smile * 36, 0.05);
}

function updateStats(dt) {
  const s = stats; s.seconds += dt; s.frames++; s.calmSum += state.calm; if (state.duet) s.duetFrames++;
  for (const k of ["left", "right"]) {
    const h = state.body[k]; if (!h) continue;
    const st = s[k]; st.frames++; st.speedSum += h.speed;
    st.minX = Math.min(st.minX, h.x); st.maxX = Math.max(st.maxX, h.x); st.minY = Math.min(st.minY, h.y); st.maxY = Math.max(st.maxY, h.y);
  }
  const head = state.body.head;
  if (head.present) {
    s.faceFrames++; s.rollMin = Math.min(s.rollMin, head.roll); s.rollMax = Math.max(s.rollMax, head.roll);
    if (head.smile > 0.35) s.smileFrames++;
  }
  if (s.seconds - s.lastSample >= 5) { s.calmTrail.push(Math.round(state.calm * 100)); s.lastSample = s.seconds; }
}
function buildSummary() {
  const s = stats;
  const hand = (k) => { const h = s[k]; const ok = h.frames > 0; return {
    visiblePct: s.frames ? Math.round((h.frames / s.frames) * 100) : 0,
    avgSpeed: ok ? +(h.speedSum / h.frames).toFixed(2) : 0,
    rangeXPct: ok ? Math.round((h.maxX - h.minX) * 100) : 0,
    rangeYPct: ok ? Math.round((h.maxY - h.minY) * 100) : 0 }; };
  return {
    durationSec: Math.round(s.seconds), notesPlayed: s.notes, chordChanges: s.chordChanges,
    avgCalmPct: Math.round((100 * s.calmSum) / Math.max(1, s.frames)), calmEvery5s: s.calmTrail,
    leftHand: hand("left"), rightHand: hand("right"),
    headTiltRangeDeg: Math.round(((s.rollMax - s.rollMin) * 180) / Math.PI),
    smilePct: s.faceFrames ? Math.round((100 * s.smileFrames) / s.faceFrames) : 0, headNods: s.nods, duetPct: s.frames ? Math.round((100 * s.duetFrames) / s.frames) : 0,
    scene: `${state.scene.name} (${state.scene.root} ${state.scene.mode}, ${state.scene.bpm} bpm)`,
  };
}

// ---------- visuals ----------
function draw(now, dt) {
  ctx.clearRect(0, 0, W, H);
  const pal = state.scene.palette;
  if (state.phase === "session") { drawStrings(pal); drawChordZones(pal); }
  drawParticles(dt);
  drawHands(pal);
  drawHead(pal);
  if (state.phase === "session") drawBreath(pal);
}
function drawStrings(pal) {
  const N = state.scale.length, per = SCALES[state.scene.mode].length;
  for (let i = 0; i < N; i++) {
    const x = ((i + 0.5) / N) * W, g = state.strings[i];
    state.strings[i] *= 0.93;
    ctx.save();
    ctx.strokeStyle = pal[Math.floor(i / per) % 3]; ctx.globalAlpha = 0.08 + g * 0.8; ctx.lineWidth = 1 + g * 3;
    if (g > 0.05) { ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = g * 30; }
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    if (g > 0.2) { ctx.globalAlpha = g; ctx.fillStyle = "#fff"; ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.fillText(state.scale[i], x, H - 40); }
    ctx.restore();
  }
}
function drawChordZones(pal) {
  const active = state.pad.chordIdx;
  for (let k = 0; k < 4; k++) {
    const y0 = H * (1 - (k + 1) / 4), y1 = H * (1 - k / 4);
    ctx.save();
    if (k === active) {
      const grad = ctx.createLinearGradient(W * 0.55, 0, W, 0);
      grad.addColorStop(0, "rgba(0,0,0,0)"); grad.addColorStop(1, pal[1]);
      ctx.globalAlpha = 0.18; ctx.fillStyle = grad; ctx.fillRect(W * 0.55, y0, W * 0.45, y1 - y0);
    }
    ctx.globalAlpha = k === active ? 0.9 : 0.3; ctx.fillStyle = "#fff"; ctx.font = "13px system-ui"; ctx.textAlign = "right";
    ctx.fillText(state.chordNames[k] || "", W - 20, (y0 + y1) / 2);
    ctx.restore();
  }
}
function pill(x, y, text, color) {
  ctx.save(); ctx.globalCompositeOperation = "source-over"; ctx.shadowBlur = 0;
  ctx.font = `700 12px "Nunito", system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width + 20;
  ctx.fillStyle = color; ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.roundRect(x - w / 2, y - 11, w, 22, 11); ctx.fill();
  ctx.fillStyle = "#1b1740"; ctx.globalAlpha = 1; ctx.fillText(text, x, y + 1); ctx.restore();
}
function drawHands(pal) {
  const m = coverMap();
  for (const [key, color] of [["left", pal[0]], ["right", pal[1]]]) {
    const h = state.body[key]; if (!h) continue;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    if (h.pts) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.35 + h.open * 0.4; ctx.lineWidth = 2; ctx.lineCap = "round";
      ctx.beginPath();
      for (const [a, b] of HAND_CONN) { ctx.moveTo(h.pts[a].x, h.pts[a].y); ctx.lineTo(h.pts[b].x, h.pts[b].y); }
      ctx.stroke();
      ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 16;
      for (const i of TIPS) { ctx.beginPath(); ctx.arc(h.pts[i].x, h.pts[i].y, 5, 0, Math.PI * 2); ctx.fill(); }
      if (h.speed > 0.15 && state.particles.length < 600) for (const i of TIPS) spawn(h.pts[i].x, h.pts[i].y, h, color);
      ctx.globalAlpha = 0.9; ctx.shadowBlur = 0; ctx.fillStyle = "#fff"; ctx.font = "bold 12px system-ui"; ctx.textAlign = "center";
      pill(h.pts[0].x, h.pts[0].y + 26, key === "left" ? "♪ harp hand" : "♫ chord hand", color);
    } else {
      ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 24; ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.arc(h.x * W, h.y * H, 14, 0, Math.PI * 2); ctx.fill();
      if (h.speed > 0.15 && state.particles.length < 600) spawn(h.x * W, h.y * H, h, color);
    }
    ctx.restore();
  }
  void m;
}
function handPoint(h, i) { return h.pts ? h.pts[i] : { x: h.x * W, y: h.y * H }; }
function burst(pt, glyph, color, size, label) {
  if (state.particles.length > 250) state.particles.shift();
  state.particles.push({ x: pt.x, y: pt.y, vx: (Math.random() - 0.5) * 60, vy: -70 - Math.random() * 60, life: 1, decay: 0.32 + Math.random() * 0.2,
    size, glyph, color, rot: (Math.random() - 0.5) * 0.6, spin: (Math.random() - 0.5) * 2, label });
}
function spawn(x, y, h, color) {
  state.particles.push({ x, y, vx: (Math.random() - 0.5) * 40 + h.vx * W * 0.2, vy: -30 - Math.random() * 40, life: 1, decay: 0.8 + Math.random() * 0.6,
    size: 18 + Math.random() * 14, glyph: Math.random() < 0.5 ? "♪" : "·", color, rot: (Math.random() - 0.5), spin: (Math.random() - 0.5) * 3 });
}
function drawParticles(dt) {
  ctx.save(); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const ps = state.particles;
  for (let i = ps.length - 1; i >= 0; i--) {
    const p = ps[i]; p.life -= p.decay * dt;
    if (p.life <= 0) { ps.splice(i, 1); continue; }
    p.x += p.vx * dt; p.y += p.vy * dt; p.vy -= 15 * dt; p.vx *= 0.99; p.rot += p.spin * dt;
    const scale = p.label ? 0.8 + 0.4 * Math.sin((1 - p.life) * Math.PI) : 1;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.globalAlpha = Math.min(1, p.life * 1.4);
    ctx.font = `700 ${Math.round(p.size * scale)}px "Fredoka", system-ui`;
    ctx.lineWidth = Math.max(2, p.size * 0.08); ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineJoin = "round";
    ctx.shadowColor = p.color; ctx.shadowBlur = 26;
    ctx.strokeText(p.glyph, 0, 0);
    ctx.fillStyle = p.color; ctx.fillText(p.glyph, 0, 0);
    if (p.label) {
      ctx.shadowBlur = 0; ctx.font = `800 ${Math.round(p.size * 0.32)}px "Nunito", system-ui`;
      const w = ctx.measureText(p.label).width + 14, lx = p.size * 0.55, ly = -p.size * 0.45;
      ctx.fillStyle = "rgba(255,255,255,.92)"; ctx.beginPath(); ctx.roundRect(lx - w / 2, ly - p.size * 0.2, w, p.size * 0.4, p.size * 0.2); ctx.fill();
      ctx.fillStyle = "#1b1740"; ctx.fillText(p.label, lx, ly + 1);
    }
    ctx.restore();
  }
  ctx.restore();
}
function drawHead(pal) {
  const h = state.body.head; if (!h.present || !h.nose) return;
  if (state.phase === "session") {
    const level = clamp(h.nodVel / nodThreshold, 0, 1.5);
    ctx.save(); ctx.globalAlpha = 0.5; ctx.fillStyle = "rgba(255,255,255,.25)";
    ctx.beginPath(); ctx.roundRect(h.nose.x - 40, h.nose.y - 150, 80, 6, 3); ctx.fill();
    ctx.fillStyle = level >= 1 ? pal[2] : "#fff"; ctx.beginPath(); ctx.roundRect(h.nose.x - 40, h.nose.y - 150, Math.min(80, 80 * level), 6, 3); ctx.fill();
    ctx.globalAlpha = 0.7; ctx.font = `600 11px "Nunito", system-ui`; ctx.textAlign = "center"; ctx.fillText("nod", h.nose.x, h.nose.y - 158); ctx.restore();
  }
  if (h.pulse > 0.01) {
    ctx.save(); ctx.strokeStyle = pal[2]; ctx.lineWidth = 6 * h.pulse; ctx.globalAlpha = h.pulse * 0.9;
    ctx.shadowColor = pal[2]; ctx.shadowBlur = 30 * h.pulse;
    ctx.beginPath(); ctx.arc(h.nose.x, h.nose.y, 90 + (1 - h.pulse) * 160, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    h.pulse *= 0.9;
  }
  const r = 110 * (0.6 + h.smile * 0.9);
  const g = ctx.createRadialGradient(h.nose.x, h.nose.y, 0, h.nose.x, h.nose.y, r);
  g.addColorStop(0, pal[2]); g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.12 + h.smile * 0.3; ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(h.nose.x, h.nose.y, r, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  if (h.smile > 0.5 && Math.random() < 0.08) burst({ x: h.nose.x + (Math.random() - 0.5) * 160, y: h.nose.y - 60 }, "♡", pal[2], 44 + h.smile * 24);
}
function drawBreath(pal) {
  const e = 0.5 - 0.5 * Math.cos(Math.PI * state.breath);
  const x = W / 2, y = H - 110, r = 30 + 30 * e;
  const cycle = (60 / state.scene.bpm) * 8, t = (performance.now() / 1000) % cycle, inhale = t < cycle * 0.4;
  ctx.save();
  ctx.fillStyle = pal[0]; ctx.globalAlpha = 0.18; ctx.beginPath(); ctx.arc(x, y, r + 14, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 0.95; ctx.shadowColor = pal[0]; ctx.shadowBlur = 24; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0; ctx.fillStyle = "#1b1740";
  const eye = inhale ? 3.2 : 2.2;
  ctx.beginPath(); ctx.arc(x - r * 0.32, y - r * 0.12, eye, 0, Math.PI * 2); ctx.arc(x + r * 0.32, y - r * 0.12, eye, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = 2.5; ctx.strokeStyle = "#1b1740"; ctx.lineCap = "round"; ctx.beginPath();
  if (inhale) ctx.arc(x, y + r * 0.18, r * 0.22, 0, Math.PI * 2); else ctx.arc(x, y + r * 0.1, r * 0.3, 0.15 * Math.PI, 0.85 * Math.PI);
  ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.globalAlpha = 0.85; ctx.font = `600 13px "Fredoka", system-ui`; ctx.textAlign = "center";
  ctx.fillText(inhale ? "breathe in…" : "and let it go", x, y + r + 26);
  ctx.restore();
}

// ---------- main loop ----------
let lastVideoTime = -1, frameCount = 0, hudTick = 0;
function loop(now) {
  requestAnimationFrame(loop);
  const dt = state.lastFrame < 0 ? 0.016 : clamp((now - state.lastFrame) / 1000, 0.001, 0.1);
  state.lastFrame = now;
  if (state.visionReady && state.cameraReady && video.videoWidth > 0) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const m = coverMap();
      try {
        processHands(hands.detectForVideo(video, now), m, dt);
        if (frameCount % 2 === 0) processFace(face.detectForVideo(video, now), m);
      } catch (e) { console.warn(e); }
    }
  } else if (!state.cameraReady) applyPointer(dt);
  frameCount++;
  if (state.phase === "session") {
    updateAudio(now, dt); updateStats(dt);
    if (now - hudTick > 100) { hudTick = now; updateHud(); }
  }
  draw(now, dt);
}
function updateHud() {
  const { left, right } = state.body;
  $("mCalm").style.width = `${Math.round(state.calm * 100)}%`;
  $("mLeft").style.width = `${left ? Math.round(30 + left.open * 70) : 0}%`;
  $("mRight").style.width = `${right ? Math.round(30 + right.open * 70) : 0}%`;
  $("duetHud").classList.toggle("hidden", !state.duet);
  const l = stats.left.frames, r = stats.right.frames, tot = l + r;
  $("balance").textContent = tot ? `Left hand ${Math.round((100 * l) / tot)}% · right hand ${Math.round((100 * r) / tot)}% · ${stats.notes} notes` : "Show me your hands…";
}

// ---------- history (local, private) ----------
const HISTORY_KEY = "doremi.history";
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; } }
function saveHistory(list) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(-50))); } catch {} }
function recentForClaude() {
  return loadHistory().slice(-2).map((e) => ({ date: e.date, scene: e.scene, mood: e.mood, avgCalmPct: e.summary.avgCalmPct,
    leftVisiblePct: e.summary.leftHand.visiblePct, rightVisiblePct: e.summary.rightHand.visiblePct,
    leftRangeXPct: e.summary.leftHand.rangeXPct, rightRangeXPct: e.summary.rightHand.rangeXPct, headNods: e.summary.headNods, durationSec: e.summary.durationSec }));
}
function renderJourney() {
  const list = loadHistory(); const card = $("journey");
  if (!list.length) return card.classList.add("hidden");
  card.classList.remove("hidden");
  const week = list.filter((e) => Date.now() - new Date(e.date).getTime() < 7 * 864e5).length;
  const calms = list.map((e) => e.summary.avgCalmPct);
  const first = calms[Math.max(0, calms.length - 6)], last = calms[calms.length - 1];
  const trend = calms.length > 1 ? (last >= first ? `calm ↑ ${first}% → ${last}%` : `calm ↓ ${first}% → ${last}%`) : `calm ${last}%`;
  $("jsummary").textContent = `${list.length} session${list.length > 1 ? "s" : ""} · ${week} this week · ${trend}`;
  const pts = calms.slice(-12), n = pts.length;
  const xy = pts.map((c, i) => [n > 1 ? (i / (n - 1)) * 232 + 4 : 120, 40 - (c / 100) * 34]);
  const path = xy.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7dd3fc";
  $("jspark").innerHTML = `<path d="${path} L${xy[n - 1][0].toFixed(1)},44 L${xy[0][0].toFixed(1)},44 Z" fill="${accent}" opacity=".18"/>
    <path d="${path}" fill="none" stroke="${accent}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    ${xy.map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="#fff"/>`).join("")}`;
  const tot = list.reduce((a, e) => a + e.summary.leftHand.visiblePct + e.summary.rightHand.visiblePct, 0);
  const lshare = tot ? Math.round((100 * list.reduce((a, e) => a + e.summary.leftHand.visiblePct, 0)) / tot) : 50;
  const nods = list.reduce((a, e) => a + (e.summary.headNods || 0), 0), notes = list.reduce((a, e) => a + e.summary.notesPlayed, 0);
  $("jnote").textContent = `Hands: left ${lshare}% · right ${100 - lshare}% · ${notes} notes and ${nods} heartbeats so far. Calm score per session above.`;
}
function exportCsv() {
  const list = loadHistory(); if (!list.length) return;
  const cols = ["date", "scene", "mood", "durationSec", "notesPlayed", "chordChanges", "avgCalmPct", "leftVisiblePct", "rightVisiblePct",
    "leftRangeXPct", "leftRangeYPct", "rightRangeXPct", "rightRangeYPct", "leftAvgSpeed", "rightAvgSpeed", "headTiltRangeDeg", "headNods", "smilePct", "duetPct", "reflection"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = list.map((e) => { const s = e.summary; return [e.date, e.scene, e.mood, s.durationSec, s.notesPlayed, s.chordChanges, s.avgCalmPct,
    s.leftHand.visiblePct, s.rightHand.visiblePct, s.leftHand.rangeXPct, s.leftHand.rangeYPct, s.rightHand.rangeXPct, s.rightHand.rangeYPct,
    s.leftHand.avgSpeed, s.rightHand.avgSpeed, s.headTiltRangeDeg, s.headNods, s.smilePct, s.duetPct, e.reflection].map(q).join(","); });
  const blob = new Blob([cols.join(",") + "\n" + rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "doremi-sessions.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------- session flow ----------
async function startSession(scene) {
  const pending = initAudio();
  setStatus("Starting audio…");
  try { await pending; } catch (e) { setStatus(`Audio failed: ${e.message}`); return; }
  applyScene(sanitizeScene(scene));
  stats = newStats(); state.particles = []; state.duet = false;
  try { audio.recorder.start(); } catch (e) { console.warn("recorder", e); }
  state.phase = "session";
  $("intro").classList.add("hidden"); $("reflection").classList.add("hidden"); $("hud").classList.remove("hidden");
  $("sceneName").textContent = state.scene.name; $("guidance").textContent = state.scene.guidance; $("sceneWhy").textContent = state.scene.why;
  setStatus(`${state.scene.root} ${state.scene.mode} · breath ${Math.round((60 / state.scene.bpm) * 8)}s · esc to end`);
}
async function designSession() {
  state.mood = $("mood").value.trim();
  if (!state.mood) return startSession(DEFAULT_SCENE);
  const btn = $("design"); btn.disabled = true; btn.textContent = "🎼 Claude is composing…";
  initAudio().catch(() => {});
  try {
    const res = await fetch("/api/design", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mood: state.mood }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    $("hudModel").textContent = `Composed by ${data.model}`;
    await startSession(data.scene);
  } catch (e) {
    console.error(e); setStatus(`Claude unavailable (${e.message}). Using Calm Waters.`);
    await startSession(DEFAULT_SCENE);
  } finally { btn.disabled = false; btn.textContent = "✨ Let Claude compose my session"; }
}
function localReflection(s) {
  const l = s.leftHand, r = s.rightHand;
  const lead = l.rangeXPct + l.rangeYPct >= r.rangeXPct + r.rangeYPct ? "left" : "right";
  return `You played for ${s.durationSec} seconds and plucked ${s.notesPlayed} notes. Your ${lead} hand covered more of the space, and your calm level averaged ${s.avgCalmPct}%. Next time, try leading with the other hand for a minute and let each chord breathe a little longer.`;
}
async function endSession() {
  if (state.phase !== "session") return;
  state.phase = "reflection";
  if (audio.ready) { audio.pad.releaseAll(); audio.drone.triggerRelease(); audio.shimmer.triggerRelease(); state.pad.playing = false; }
  const s = buildSummary();
  $("duetBadge").classList.toggle("hidden", s.duetPct < 20);
  $("playback").classList.add("hidden");
  if (audio.ready && audio.recorder.state === "started") {
    setTimeout(async () => {
      try {
        const blob = await audio.recorder.stop();
        if (blob.size > 0) {
          const url = URL.createObjectURL(blob);
          $("audioPlayer").src = url; $("downloadLink").href = url; $("playback").classList.remove("hidden");
        }
      } catch (e) { console.warn("recorder stop", e); }
    }, 2500); // let the reverb tail finish
  }
  const mmss = `${Math.floor(s.durationSec / 60)}:${String(s.durationSec % 60).padStart(2, "0")}`;
  const tiles = [["Duration", mmss], ["Notes", s.notesPlayed], ["Chords", s.chordChanges], ["Calm", `${s.avgCalmPct}%`],
    ["Left reach", `${s.leftHand.rangeXPct}%`], ["Right reach", `${s.rightHand.rangeXPct}%`], ["Nods", s.headNods], ["Smiling", `${s.smilePct}%`]];
  if (s.duetPct >= 20) tiles.push(["Duet", `${s.duetPct}%`]);
  $("stats").innerHTML = tiles.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
  let reflection = "";
  $("reflectText").textContent = "Claude is listening back to your session…";
  $("hud").classList.add("hidden"); $("reflection").classList.remove("hidden");
  setStatus("Session complete");
  try {
    const res = await fetch("/api/reflect", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mood: state.mood, scene: state.scene, summary: s, previousSessions: recentForClaude() }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    $("reflectText").textContent = data.reflection;
    reflection = data.reflection;
  } catch (e) {
    console.error(e); reflection = localReflection(s); $("reflectText").textContent = reflection;
    setStatus(`Claude unavailable (${e.message})`);
  }
  if (s.durationSec >= 10) { const list = loadHistory(); list.push({ date: new Date().toISOString(), scene: state.scene.name, mood: state.mood, summary: s, reflection }); saveHistory(list); renderJourney(); }
}
function resetToIntro() {
  state.phase = "intro";
  $("reflection").classList.add("hidden"); $("hud").classList.add("hidden"); $("intro").classList.remove("hidden");
  setStatus("I can see you. Wave a hand.");
}

$("design").addEventListener("click", designSession);
document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
  state.mood = $("mood").value.trim() || b.querySelector("b").textContent;
  $("hudModel").textContent = `${b.querySelector(".emoji").textContent} ${b.querySelector("b").textContent} preset`;
  startSession(PRESETS[b.dataset.preset]);
}));
$("end").addEventListener("click", endSession);
$("askTile").addEventListener("click", () => $("mood").focus());
$("exportCsv").addEventListener("click", exportCsv);
$("clearHistory").addEventListener("click", () => { if (confirm("Clear your saved sessions on this device?")) { saveHistory([]); renderJourney(); } });
$("again").addEventListener("click", resetToIntro);
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "h") { swapHands = !swapHands; try { localStorage.setItem("doremi.swapHands", swapHands ? "1" : "0"); } catch {} setStatus(swapHands ? "Harp is now on the RIGHT side, chords on the left" : "Harp on the LEFT side, chords on the right"); }
  if (e.key === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  if (e.key === "Escape") endSession();
  if (e.key === "[" || e.key === "]") {
    nodThreshold = clamp(+(nodThreshold + (e.key === "[" ? -0.1 : 0.1)).toFixed(2), 0.2, 3);
    try { localStorage.setItem("doremi.nodThreshold", String(nodThreshold)); } catch {}
    setStatus(`Nod sensitivity: fires above ${nodThreshold} ( [ = easier, ] = harder )`);
  }
});

async function boot() {
  resize(); window.addEventListener("resize", resize);
  if (typeof Tone === "undefined") setStatus("Audio library failed to load. Check the network.");
  applyScene(DEFAULT_SCENE); renderJourney();
  fetch("/api/health").then((r) => r.json()).then((d) => { state.model = d.model; $("hudModel").textContent = `Composed by ${d.model}`; }).catch(() => {});
  requestAnimationFrame(loop);
  try {
    await Promise.all([initCamera(), initVision()]);
    setStatus("I can see you! 👋 Wave a hand.");
  } catch (e) {
    console.error(e);
    setStatus(`Camera or models unavailable (${e.message}). Mouse mode: move = right hand, hold click = left hand.`);
  }
}
boot();
