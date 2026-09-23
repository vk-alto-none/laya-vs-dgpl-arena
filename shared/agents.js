/* Model agents — one driver per side of the arena (Snake & Kombat).
   Classification: PROPRIETARY & CONFIDENTIAL — COMMERCIAL ENTERPRISE (DGPL)
*/
"use strict";

const STEER_VALUES = {
  HARD_LEFT: -1.0,
  LEFT: -0.35,
  STRAIGHT: 0.0,
  RIGHT: 0.35,
  HARD_RIGHT: 1.0
};

const QUESTIONS = {
  steer: {
    type: 'choice',
    instructions: {
      task: 'You are steering a snake in an arena. Choose the steering action for the next moment of travel.',
      goal: 'Reach the nearest apple as quickly as possible. Nothing can kill the snake: it passes harmlessly over its own body, and the edges wrap around, so leaving one side brings it back on the opposite side. Speed to the apple is all that matters.',
      reading_the_state: 'The snake always moves forward along its current heading. Left and right are relative to that heading. The apple bearing is measured along the SHORTEST route, which may run out through one edge and back in the opposite one: a negative bearing is to the left, a positive bearing is to the right.',
      closing_on_the_apple: 'A steering answer is held for a short burst and then the snake runs straight again, so an easing turn barely changes the heading. If the apple is behind the snake or far off to one side, answer with the matching HARD turn and keep answering it until the apple comes near the front; easing in that situation makes the snake circle the apple forever without reaching it.'
    },
    criteria: {
      HARD_LEFT: 'Turn left as sharply as possible. Use when the apple is far to the left or behind on the left.',
      LEFT: 'Ease left. Use when the apple is moderately to the left.',
      STRAIGHT: 'Hold the current heading. Use when the apple is nearly dead ahead.',
      RIGHT: 'Ease right. Use when the apple is moderately to the right.',
      HARD_RIGHT: 'Turn right as sharply as possible. Use when the apple is far to the right or behind on the right.'
    }
  },
  sprint: {
    type: 'noul',
    instructions: 'Sprinting makes the snake travel much faster but turn wider, so it can overshoot an apple that needs a turn. Is sprinting the right call right now?',
    criteria: {
      true: 'The apple is roughly dead ahead and still some distance away, so extra speed gets there sooner.',
      false: 'The apple is off to one side, behind, or very close, so a sprint would carry the snake past it.'
    }
  }
};

export class ModelAgent {
  constructor(arena, opts = {}){
    this.arena = arena;
    this.questions = opts.questions || QUESTIONS;
    this.endpoint = opts.endpoint || '/api/v1/systemone';
    this.model = opts.model || 'dgpl-s1';
    this.label = opts.label || this.model;
    this.lastLatency = 0;
    this.serverLatencyUs = 10.3;
    this.onUpdate = opts.onUpdate || (() => {});
    this.safetyNet = !!opts.safetyNet;

    this.steer = 0;
    this.sprint = false;
    this.stopped = true;
    this.turnBudget = opts.turnBudget || 0.55;
    this.workers = opts.workers || 3;
    this.minInterval = opts.minInterval ?? 40;
    this.angAtDecision = 0;

    this.decisions = 0;
    this.saves = 0;
    this.latencies = [];
    this.lastChoice = '—';
    this.lastProbs = null;
    this.sprintP = 0;
    this.confidence = 0;
    this.error = null;
    this.startedAt = 0;
    this.servedModel = 'DGPL-System1-v2.0';
    this.inputTokens = 0;
    this.outputTokens = 0;
  }

  getApiKey() {
    return localStorage.getItem("dgpl_api_key") || "";
  }

  reset(){
    this.decisions = 0; this.saves = 0; this.latencies = [];
    this.inputTokens = 0; this.outputTokens = 0;
    this.lastChoice = '—'; this.lastProbs = null; this.error = null;
    this.steer = 0; this.sprint = false; this.startedAt = performance.now();
    this.angAtDecision = (this.arena && this.arena.snake) ? this.arena.snake.ang : 0;
    this.onReset();
  }

  onReset(){}
  onAnswers(){}

  get p50(){
    if (!this.latencies.length) return 0;
    const a = this.latencies.slice().sort((x,y) => x-y);
    return a[Math.floor(a.length/2)];
  }
  get p95(){
    if (!this.latencies.length) return 0;
    const a = this.latencies.slice().sort((x,y) => x-y);
    return a[Math.min(a.length-1, Math.floor(a.length*0.95))];
  }
  get rate(){
    const secs = (performance.now() - this.startedAt)/1000;
    return secs > 0.5 ? this.decisions/secs : 0;
  }

  senseState(){ return this.arena ? this.arena.sense() : {}; }

  buildPayload() {
    const state = this.senseState();
    if (this.endpoint.includes('/api/v1/systemone')) {
      // Determine candidates based on questions
      let candidates = ['ADVANCE', 'RETREAT', 'PUNCH', 'KICK', 'BLOCK', 'JUMP'];
      if (this.questions.steer && this.questions.steer.criteria) {
        candidates = Object.keys(this.questions.steer.criteria);
      } else if (this.questions.action && this.questions.action.criteria) {
        candidates = Object.keys(this.questions.action.criteria);
      }
      return {
        task: "choice",
        state: typeof state === 'string' ? state : JSON.stringify(state),
        candidates: candidates
      };
    }
    // Fallback shape
    return { model: this.model, state: state, questions: this.questions };
  }

  async sendRequest() {
    const payload = this.buildPayload();
    const apiKey = this.getApiKey();
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) {
      headers['X-DGPL-API-Key'] = apiKey;
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const t0 = performance.now();
    const res = await fetch(this.endpoint, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(payload)
    });

    const elapsedMs = performance.now() - t0;
    const headerLat = res.headers.get("X-DGPL-Latency-Us");
    if (headerLat) this.serverLatencyUs = parseFloat(headerLat);

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    const body = await res.json();
    return { body, elapsedMs };
  }

  async prime(){
    this.reset();
    this.stopped = false;
    try {
      const { body, elapsedMs } = await this.sendRequest();
      this.apply(body, elapsedMs);
    } catch (err) {
      this.error = String(err.message || err);
      this.onUpdate(this);
    }
    this.startedAt = performance.now();
  }

  start(){
    this.stopped = false;
    if (!this.decisions) this.reset();
    for (let i = 0; i < this.workers; i++){
      setTimeout(() => this.pump(), i * 150);
    }
  }
  stop(){ this.stopped = true; }

  control(){
    if (!this.arena || !this.arena.snake) return {steer: 0, sprint: false};
    let steer = this.steer, sprint = this.sprint;
    const spent = Math.abs(this.arena.snake.ang - this.angAtDecision);
    const perTrip = this.arena.turnRate * Math.max(0.08, (this.p50 || this.lastLatency || 120)/1000);
    const budget = Math.max(perTrip * 1.5, this.turnBudget * (0.7 + 1.3*Math.abs(this.steer)));
    if (spent >= budget) steer = 0;
    if (this.safetyNet){
      const s = this.arena.sense();
      const c = s.clearance_px;
      if (c && c.ahead < 60){
        const left = c['30deg_left'] + c['60deg_left'];
        const right = c['30deg_right'] + c['60deg_right'];
        steer = left > right ? -1 : 1;
        sprint = false;
        this.saves++;
      }
    }
    return {steer, sprint};
  }

  async pump(){
    while (!this.stopped){
      if (this.arena && !this.arena.running){ await sleep(120); continue; }
      try {
        const { body, elapsedMs } = await this.sendRequest();
        this.apply(body, elapsedMs);
        if (elapsedMs < this.minInterval) await sleep(this.minInterval - elapsedMs);
      } catch (err) {
        this.error = String(err.message || err);
        this.onUpdate(this);
        await sleep(500);
      }
    }
  }

  apply(body, ms){
    this.lastLatency = ms;
    this.latencies.push(ms);
    if (this.latencies.length > 60) this.latencies.shift();
    this.decisions++;
    this.error = null;

    // Handle standard DGPL /api/v1/systemone response
    if (body.decision) {
      const d = body.decision;
      this.servedModel = d.model || 'DGPL-System1-v2.0';
      const selected = d.selected || 'ADVANCE';
      const probs = d.distribution || {};
      const conf = d.confidence || 0.98;

      // Construct normalized answers
      const answers = {
        action: { choice: selected, probabilities: probs, confidence: conf },
        steer: { choice: selected, probabilities: probs, confidence: conf },
        commit: { noul: probs['KICK'] || (selected === 'KICK' ? 0.9 : 0.1) },
        sprint: { noul: selected === 'STRAIGHT' ? 0.85 : 0.15 }
      };

      this.answers = answers;
      this.onAnswers(answers);

      if (answers.steer && STEER_VALUES[selected] !== undefined){
        const want = STEER_VALUES[selected];
        this.steer = this.steer * 0.35 + want * 0.65;
        this.angAtDecision = (this.arena && this.arena.snake) ? this.arena.snake.ang : 0;
        this.lastChoice = selected;
        this.lastProbs = probs;
        this.confidence = conf;
      }
      if (answers.sprint){
        this.sprintP = answers.sprint.noul;
        this.sprint = this.sprintP > 0.65;
      }
    } else {
      // Legacy answer structure
      const a = body.answers || {};
      this.servedModel = body.model || this.servedModel;
      this.answers = a;
      this.onAnswers(a);
      if (a.steer){
        const probs = a.steer.probabilities || {};
        const want = STEER_VALUES[a.steer.choice] ?? 0;
        this.steer = this.steer*0.35 + want*0.65;
        this.angAtDecision = (this.arena && this.arena.snake) ? this.arena.snake.ang : 0;
        this.lastChoice = a.steer.choice;
        this.lastProbs = probs;
        this.confidence = a.steer.confidence ?? 0;
      }
      if (a.sprint){
        const p = a.sprint.noul ?? 0;
        this.sprintP = p;
        this.sprint = p > 0.65;
      }
    }

    this.onUpdate(this);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

export class HumanAgent {
  constructor(arena, opts = {}){
    this.arena = arena;
    this.label = opts.label || 'Human';
    this.model = 'human';
    this.keys = opts.keys || new Set();
    this.decisions = 0; this.saves = 0; this.error = null;
    this.lastChoice = '—'; this.lastProbs = null; this.sprintP = 0;
    this.inputTokens = 0; this.outputTokens = 0; this.servedModel = 'keyboard';
    this.steer = 0;
  }

  get p50(){ return 0; }
  get p95(){ return 0; }
  get rate(){ return 0; }
  reset(){ this.decisions = 0; this.steer = 0; }
  async prime(){ this.reset(); }
  start(){}
  stop(){}
  control(){
    let s = 0;
    if (this.keys.has('arrowleft') || this.keys.has('a')) s -= 1;
    if (this.keys.has('arrowright') || this.keys.has('d')) s += 1;
    if (s !== this.steer && s !== 0) this.decisions++;
    this.steer = s;
    return {steer: s, sprint: this.keys.has('space') || this.keys.has('shift')};
  }
}
