/* Model agents — one snake driver per side of the lab.

   HumanAgent reads the keyboard. ModelAgent drives the snake from a TypeSafe
   System One judgment and works for ANY model id, so two ModelAgents with
   different models can race the same seeded arena against each other.
   Runs a pipelined decision loop: as soon as one judgment returns, the next request
   goes out. Between answers the snake keeps steering on the last decision, so the
   round-trip latency is visible as the decision rate rather than as stutter.

   Two independent questions are asked over the same state in one request, so they
   run in parallel: how to steer (choice) and whether it is safe to sprint (noul).
   The snake acts on the chosen option; the full distribution is kept for the HUD
   and the exported run record. */
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
    this.questions = opts.questions || QUESTIONS;   // the game supplies its own
    this.endpoint = opts.endpoint || '/api/jev';
    this.model = opts.model || 'jev-latest';
    this.label = opts.label || this.model;
    this.lastLatency = 0;
    this.onUpdate = opts.onUpdate || (() => {});
    this.safetyNet = !!opts.safetyNet;

    this.steer = 0;
    this.sprint = false;
    this.stopped = true;
    this.turnBudget = opts.turnBudget || 0.55;
    this.workers = opts.workers || 3;
    this.minInterval = opts.minInterval ?? 60;   // ms floor between a worker's calls   // radians of heading change per answer
    this.angAtDecision = 0;

    this.decisions = 0;
    this.saves = 0;
    this.latencies = [];
    this.lastLatency = 0;
    this.lastChoice = '—';
    this.lastProbs = null;
    this.sprintP = 0;
    this.confidence = 0;
    this.error = null;
    this.startedAt = 0;
    this.servedModel = '';
    this.inputTokens = 0;
    this.outputTokens = 0;
  }

  reset(){
    this.decisions = 0; this.saves = 0; this.latencies = [];
    this.inputTokens = 0; this.outputTokens = 0;
    this.lastChoice = '—'; this.lastProbs = null; this.error = null;
    this.steer = 0; this.sprint = false; this.startedAt = performance.now();
    this.angAtDecision = (this.arena && this.arena.snake) ? this.arena.snake.ang : 0;
    this.onReset();
  }

  /** hooks for a subclass: per-round state, and reading a game's own answers */
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

  /** the state sent with every request; a game with a different world overrides it */
  senseState(){ return this.arena ? this.arena.sense() : {}; }

  /** one blocking decision before the round starts, so nobody acts blind */
  async prime(){
    this.reset();
    this.stopped = false;
    const t = performance.now();
    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: this.model, state: this.senseState(), questions: this.questions})
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      this.apply(body, performance.now() - t);
    } catch (err) {
      this.error = String(err.message || err);
      this.onUpdate(this);
    }
    this.startedAt = performance.now();   // don't charge the warm-up to the decision rate
  }

  start(){
    this.stopped = false;
    if (!this.decisions) this.reset();
    // one in-flight request per worker, staggered: a ~1.2s round trip becomes a
    // ~3/s decision rate without any single answer being waited on
    for (let i = 0; i < this.workers; i++){
      setTimeout(() => this.pump(), i * 380);
    }
  }
  stop(){ this.stopped = true; }

  /** called every frame by the arena; returns the steering currently in force
      A decision is held until the next answer arrives, which at ~1s round trips is
      long enough to spin the snake in a full circle. So each answer carries a turn
      budget: once that much heading change has been spent, the snake runs straight
      until Jev speaks again. */
  control(){
    if (!this.arena || !this.arena.snake) return {steer: 0, sprint: false};
    let steer = this.steer, sprint = this.sprint;
    const spent = Math.abs(this.arena.snake.ang - this.angAtDecision);
    // A sharp answer buys a bigger heading change than an easing one. The cap is
    // relative to what the arena's latency-paced turn rate can deliver in one round
    // trip, so it does not silently shut off steering when the world is paced slow.
    const perTrip = this.arena.turnRate * Math.max(0.3, (this.p50 || this.lastLatency || 300)/1000);
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
      const t = performance.now();
      try {
        const res = await fetch(this.endpoint, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({model: this.model, state: this.senseState(), questions: this.questions})
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
        this.apply(body, performance.now() - t);
        const spent = performance.now() - t;
        if (spent < this.minInterval) await sleep(this.minInterval - spent);
      } catch (err) {
        this.error = String(err.message || err);
        this.onUpdate(this);
        await sleep(600);
      }
    }
  }

  apply(body, ms){
    const a = body.answers || {};
    this.servedModel = body.model || this.servedModel;
    if (body.usage){
      this.inputTokens += body.usage.input_tokens || 0;
      this.outputTokens += body.usage.output_tokens || 0;
    }
    this.lastLatency = ms;
    this.latencies.push(ms);
    if (this.latencies.length > 60) this.latencies.shift();
    this.decisions++;
    this.error = null;

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
    this.onUpdate(this);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));


/** keyboard driver, same interface as ModelAgent so either can sit on either side */
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
    this.lastChoice = s < 0 ? 'LEFT' : s > 0 ? 'RIGHT' : 'STRAIGHT';
    return {steer: s, sprint: this.keys.has('arrowup') || this.keys.has('w') || this.keys.has('shift')};
  }
}

// exported so a probe can ask the models the exact questions the game asks
export { QUESTIONS as SNAKE_QUESTIONS };
