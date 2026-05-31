// Retro sound effects — Web Audio API chiptune synthesis (no external files)

const SFX = (function () {
  'use strict';

  let ctx = null;
  let muted = false;
  const MASTER_VOL = 0.18;

  function _ctx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return ctx;
  }

  // Resume context on first user interaction (browser autoplay policy)
  function _ensureResumed() {
    const ac = _ctx();
    if (ac.state === 'suspended') ac.resume();
  }

  function _osc(type, freq, duration, volume, detune) {
    if (muted) return;
    _ensureResumed();
    const ac = _ctx();
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = type;
    o.frequency.value = freq;
    if (detune) o.detune.value = detune;
    g.gain.setValueAtTime(volume * MASTER_VOL, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + duration);
    o.connect(g);
    g.connect(ac.destination);
    o.start(ac.currentTime);
    o.stop(ac.currentTime + duration);
    return { osc: o, gain: g };
  }

  // ─── Sound effects ───────────────────────────────────────────

  function walk() {
    // Soft footstep blip
    _osc('square', 180 + Math.random() * 40, 0.06, 0.3);
  }

  function arrive() {
    // Two-tone arrival chime
    _osc('square', 440, 0.08, 0.5);
    setTimeout(() => _osc('square', 660, 0.12, 0.5), 80);
  }

  function speech() {
    // Typewriter-style speech blip (Undertale vibes)
    const base = 260 + Math.random() * 80;
    _osc('square', base, 0.04, 0.35);
  }

  function sceneTransition() {
    // Swoosh — descending noise burst
    if (muted) return;
    _ensureResumed();
    const ac = _ctx();
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = 'sawtooth';
    o.frequency.setValueAtTime(800, ac.currentTime);
    o.frequency.exponentialRampToValueAtTime(200, ac.currentTime + 0.15);
    g.gain.setValueAtTime(0.12 * MASTER_VOL, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + 0.15);
    o.connect(g);
    g.connect(ac.destination);
    o.start();
    o.stop(ac.currentTime + 0.15);
  }

  function hypothesis() {
    // Curious rising arpeggio
    const notes = [330, 440, 550, 660];
    notes.forEach((f, i) => {
      setTimeout(() => _osc('square', f, 0.1, 0.4), i * 60);
    });
  }

  function training() {
    // Rapid beep-boop computing sounds
    for (let i = 0; i < 4; i++) {
      setTimeout(() => {
        _osc('square', 200 + Math.random() * 600, 0.05, 0.25);
      }, i * 70);
    }
  }

  function evaluation() {
    // Drumroll-ish anticipation
    for (let i = 0; i < 6; i++) {
      setTimeout(() => {
        _osc('square', 220, 0.03, 0.2 + i * 0.05);
      }, i * 50);
    }
    // Final reveal note
    setTimeout(() => _osc('triangle', 523, 0.3, 0.5), 350);
  }

  function newBest() {
    // Victory fanfare! Rising major arpeggio
    const notes = [523, 659, 784, 1047];
    notes.forEach((f, i) => {
      setTimeout(() => {
        _osc('square', f, 0.2, 0.5);
        _osc('square', f * 0.5, 0.2, 0.2); // octave below for thickness
      }, i * 100);
    });
    // Sparkle on top
    setTimeout(() => _osc('square', 1319, 0.4, 0.3), 450);
  }

  function depart() {
    // Teleport out — descending sweep
    if (muted) return;
    _ensureResumed();
    const ac = _ctx();
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = 'square';
    o.frequency.setValueAtTime(880, ac.currentTime);
    o.frequency.exponentialRampToValueAtTime(110, ac.currentTime + 0.4);
    g.gain.setValueAtTime(0.3 * MASTER_VOL, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + 0.4);
    o.connect(g);
    g.connect(ac.destination);
    o.start();
    o.stop(ac.currentTime + 0.4);
  }

  function spawn() {
    // Teleport in — ascending sweep + chime
    if (muted) return;
    _ensureResumed();
    const ac = _ctx();
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = 'square';
    o.frequency.setValueAtTime(110, ac.currentTime);
    o.frequency.exponentialRampToValueAtTime(880, ac.currentTime + 0.3);
    g.gain.setValueAtTime(0.3 * MASTER_VOL, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + 0.35);
    o.connect(g);
    g.connect(ac.destination);
    o.start();
    o.stop(ac.currentTime + 0.35);
    // Arrival chime
    setTimeout(() => {
      _osc('triangle', 880, 0.15, 0.4);
      _osc('triangle', 1100, 0.2, 0.3);
    }, 300);
  }

  function finale() {
    // Epic ending — full major chord + sweep
    const chord = [523, 659, 784];
    chord.forEach(f => {
      _osc('square', f, 0.6, 0.35);
      _osc('triangle', f, 0.8, 0.2);
    });
    // Rising sparkle
    setTimeout(() => {
      [1047, 1175, 1319, 1568].forEach((f, i) => {
        setTimeout(() => _osc('square', f, 0.15, 0.25), i * 80);
      });
    }, 400);
  }

  function intro() {
    // Boot-up sequence
    _osc('square', 220, 0.08, 0.3);
    setTimeout(() => _osc('square', 330, 0.08, 0.3), 100);
    setTimeout(() => _osc('square', 440, 0.15, 0.4), 200);
    setTimeout(() => _osc('triangle', 660, 0.25, 0.35), 350);
  }

  function toggleMute() {
    muted = !muted;
    return muted;
  }

  return {
    walk,
    arrive,
    speech,
    sceneTransition,
    hypothesis,
    training,
    evaluation,
    newBest,
    depart,
    spawn,
    finale,
    intro,
    toggleMute,
    get muted() { return muted; },
  };
})();
