// UI — text overlays, leaderboard panel, playback controls, progress bar, summary

const UI_FONT = '12px monospace';
const UI_FONT_TITLE = 'bold 15px monospace';
const UI_FONT_SMALL = '10px monospace';
const CANVAS_W = 600;
const CANVAS_H = 400;

// ─── Shared panel drawing with glass effect ─────────────────────

function _drawPanel(ctx, x, y, w, h, radius) {
  // Background with subtle vertical gradient
  const grad = ctx.createLinearGradient(x, y, x, y + h);
  grad.addColorStop(0, 'rgba(20, 16, 40, 0.94)');
  grad.addColorStop(1, 'rgba(12, 10, 25, 0.96)');
  ctx.fillStyle = grad;
  roundRect(ctx, x, y, w, h, radius);
  ctx.fill();
  // Border
  ctx.strokeStyle = 'rgba(80, 60, 140, 0.5)';
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, radius);
  ctx.stroke();
  // Inner glass highlight at top
  ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.fillRect(x + 2, y + 1, w - 4, 1);
}

// ─── Text Overlay ───────────────────────────────────────────────

class TextOverlay {
  constructor() {
    this.visible = false;
    this.title = '';
    this.text = '';
    this.subtitle = '';
    this.findings = [];
    this.candidates = [];
    this.alpha = 0;
    this.targetAlpha = 0;
  }

  show(scene) {
    this.title = scene.title || '';
    this.text = scene.text || '';
    this.subtitle = scene.subtitle || '';
    this.findings = scene.findings || [];
    this.candidates = scene.candidates || [];
    this.visible = true;
    this.targetAlpha = 1;
  }

  hide() {
    this.targetAlpha = 0;
  }

  update(speed) {
    const step = 0.08 * (speed || 1);
    if (this.targetAlpha > this.alpha) {
      this.alpha = Math.min(1, this.alpha + step);
    } else if (this.targetAlpha < this.alpha) {
      this.alpha = Math.max(0, this.alpha - step);
      if (this.alpha === 0) this.visible = false;
    }
  }

  draw(ctx) {
    if (!this.visible || this.alpha <= 0) return;
    ctx.globalAlpha = this.alpha;

    const pad = 12;
    const x = 10;
    const maxW = 340;
    const controlsH = 32;
    const maxBottom = CANVAS_H - controlsH - 6;
    const lineH = 14;
    const titleLineH = 17;

    ctx.font = UI_FONT_TITLE;
    const titleLines = wrapText(ctx, this.title, maxW - pad * 2, 2);

    let contentH = pad * 2;
    contentH += titleLines.length * titleLineH;
    if (this.subtitle) contentH += lineH;
    ctx.font = UI_FONT;
    let textLines = [];
    if (this.text) {
      textLines = wrapText(ctx, this.text, maxW - pad * 2, 3);
      contentH += textLines.length * lineH + 4;
    }
    ctx.font = UI_FONT_SMALL;
    let findingsLines = [];
    for (const f of this.findings) {
      findingsLines.push(...wrapText(ctx, '• ' + f, maxW - pad * 2 - 4, 2));
    }
    if (findingsLines.length) contentH += findingsLines.length * 12 + 4;
    const candCount = Math.min(this.candidates.length, 4);
    if (candCount) contentH += candCount * 12 + 4;

    const maxH = maxBottom - 10;
    if (contentH > maxH) contentH = maxH;

    let y = maxBottom - contentH;
    if (y < 10) y = 10;

    _drawPanel(ctx, x, y, maxW, contentH, 5);

    // Clip to box
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, maxW, contentH);
    ctx.clip();

    let ty = y + pad + 12;

    // Title
    ctx.fillStyle = '#f1c40f';
    ctx.font = UI_FONT_TITLE;
    ctx.textAlign = 'left';
    for (const line of titleLines) {
      ctx.fillText(line, x + pad, ty);
      ty += titleLineH;
    }

    if (this.subtitle) {
      ctx.fillStyle = '#c8d0d8';
      ctx.font = UI_FONT_SMALL;
      ctx.fillText(this.subtitle, x + pad, ty);
      ty += lineH;
    }

    if (textLines.length) {
      ctx.fillStyle = '#ffffff';
      ctx.font = UI_FONT;
      for (const line of textLines) {
        ctx.fillText(line, x + pad, ty);
        ty += lineH;
      }
      ty += 4;
    }

    if (findingsLines.length) {
      ctx.fillStyle = '#3498db';
      ctx.font = UI_FONT_SMALL;
      for (const line of findingsLines) {
        ctx.fillText(line, x + pad + 2, ty);
        ty += 12;
      }
    }

    if (candCount) {
      ctx.fillStyle = '#2ecc71';
      ctx.font = UI_FONT_SMALL;
      for (let i = 0; i < candCount; i++) {
        ctx.fillText('→ ' + this.candidates[i], x + pad + 2, ty);
        ty += 12;
      }
      if (this.candidates.length > 4) {
        ctx.fillStyle = '#95a5a6';
        ctx.fillText(`  +${this.candidates.length - 4} more`, x + pad + 2, ty);
      }
    }

    ctx.restore();
    ctx.globalAlpha = 1;
  }
}

// ─── Leaderboard Panel ──────────────────────────────────────────

class LeaderboardPanel {
  constructor() {
    this.entries = [];
    this.visible = false;
    this.alpha = 0;
    this.flashTick = 0;
  }

  update(newLeaderboard, isNewBest) {
    if (!newLeaderboard || newLeaderboard.length === 0) return;
    const oldNames = new Set(this.entries.map(e => e.name));
    this.entries = newLeaderboard.map((e, i) => ({
      name: e.name,
      score: e.score,
      isNew: !oldNames.has(e.name),
      highlight: i === 0 && isNewBest,
    }));
    this.visible = true;
    this.flashTick = 40;
  }

  draw(ctx) {
    if (!this.visible) return;
    if (this.alpha < 1) this.alpha = Math.min(1, this.alpha + 0.05);
    if (this.flashTick > 0) this.flashTick--;

    ctx.globalAlpha = this.alpha;

    const w = 190;
    const x = CANVAS_W - w - 10;
    const y = 10;
    const entryH = 16;
    const headerH = 24;
    const shown = this.entries.slice(0, 8);
    const h = headerH + shown.length * entryH + 10;

    _drawPanel(ctx, x, y, w, h, 5);

    // Header
    ctx.fillStyle = '#f1c40f';
    ctx.font = UI_FONT_TITLE;
    ctx.textAlign = 'left';
    ctx.fillText('Leaderboard', x + 10, y + 18);

    for (let i = 0; i < shown.length; i++) {
      const e = shown[i];
      const ey = y + headerH + i * entryH + 10;

      // Highlight for new #1
      if (e.highlight) {
        const glowAlpha = 0.15 + (this.flashTick > 0 ? 0.1 * Math.sin(this.flashTick * 0.3) : 0);
        ctx.fillStyle = `rgba(243, 156, 18, ${glowAlpha})`;
        ctx.fillRect(x + 3, ey - 12, w - 6, entryH);
      }

      // New entry flash
      if (e.isNew && this.flashTick > 0) {
        const flash = 0.08 * (this.flashTick / 40);
        ctx.fillStyle = `rgba(46, 204, 113, ${flash})`;
        ctx.fillRect(x + 3, ey - 12, w - 6, entryH);
      }

      ctx.fillStyle = i === 0 ? '#f1c40f' : '#95a5a6';
      ctx.font = UI_FONT_SMALL;
      ctx.textAlign = 'left';
      ctx.fillText(`${i + 1}.`, x + 10, ey);

      ctx.fillStyle = e.isNew ? '#2ecc71' : '#ecf0f1';
      ctx.font = UI_FONT_SMALL;
      let name = e.name;
      if (name.length > 14) name = name.substring(0, 13) + '…';
      ctx.fillText(name, x + 28, ey);

      ctx.fillStyle = i === 0 ? '#f1c40f' : '#bdc3c7';
      ctx.textAlign = 'right';
      ctx.fillText(e.score.toFixed(3), x + w - 10, ey);
    }

    ctx.textAlign = 'left';
    ctx.globalAlpha = 1;
  }
}

// ─── Cycle Counter + Progress ───────────────────────────────────

class ProgressUI {
  constructor() {
    this.cycle = 0;
    this.totalCycles = 4;
    this.sceneIndex = 0;
    this.totalScenes = 1;
  }

  draw(ctx) {
    if (this.cycle > 0) {
      _drawPanel(ctx, 10, 10, 120, 20, 4);
      ctx.fillStyle = '#f1c40f';
      ctx.font = UI_FONT_SMALL;
      ctx.textAlign = 'left';
      ctx.fillText(`Round ${this.cycle} of ${this.totalCycles}`, 18, 24);
    }

    // Progress bar with shimmer
    const barY = CANVAS_H - 4;
    const barW = CANVAS_W;
    const progress = this.totalScenes > 1 ? this.sceneIndex / (this.totalScenes - 1) : 0;
    ctx.fillStyle = 'rgba(30, 30, 50, 0.8)';
    ctx.fillRect(0, barY, barW, 4);

    const fillW = barW * progress;
    // Base fill
    ctx.fillStyle = '#f39c12';
    ctx.fillRect(0, barY, fillW, 4);

    // Shimmer sweep
    if (fillW > 10) {
      const tick = window.globalTick || 0;
      const shimmerPos = ((tick % 90) / 90) * fillW;
      const shimmerW = 30;
      const grad = ctx.createLinearGradient(shimmerPos - shimmerW, barY, shimmerPos + shimmerW, barY);
      grad.addColorStop(0, 'rgba(255, 220, 100, 0)');
      grad.addColorStop(0.5, 'rgba(255, 220, 100, 0.4)');
      grad.addColorStop(1, 'rgba(255, 220, 100, 0)');
      ctx.fillStyle = grad;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, barY, fillW, 4);
      ctx.clip();
      ctx.fillRect(shimmerPos - shimmerW, barY, shimmerW * 2, 4);
      ctx.restore();
    }
  }
}

// ─── Playback Controls ──────────────────────────────────────────

class PlaybackControls {
  constructor() {
    this.playing = true;
    this.speed = 1;
    this.buttons = [
      { label: '⏮', action: 'prev', x: CANVAS_W / 2 - 70, y: CANVAS_H - 28, w: 28, h: 22 },
      { label: '⏸', action: 'toggle', x: CANVAS_W / 2 - 26, y: CANVAS_H - 28, w: 52, h: 22 },
      { label: '⏭', action: 'next', x: CANVAS_W / 2 + 42, y: CANVAS_H - 28, w: 28, h: 22 },
      { label: '1x', action: 'speed', x: CANVAS_W / 2 + 82, y: CANVAS_H - 28, w: 40, h: 22 },
    ];
  }

  draw(ctx) {
    this.buttons[1].label = this.playing ? '⏸' : '▶';
    this.buttons[3].label = this.speed + 'x';

    for (const btn of this.buttons) {
      const grad = ctx.createLinearGradient(btn.x, btn.y, btn.x, btn.y + btn.h);
      grad.addColorStop(0, 'rgba(40, 30, 70, 0.9)');
      grad.addColorStop(1, 'rgba(25, 20, 50, 0.9)');
      ctx.fillStyle = grad;
      roundRect(ctx, btn.x, btn.y, btn.w, btn.h, 3);
      ctx.fill();
      ctx.strokeStyle = 'rgba(80, 60, 140, 0.4)';
      ctx.lineWidth = 0.5;
      roundRect(ctx, btn.x, btn.y, btn.w, btn.h, 3);
      ctx.stroke();
      // Glass highlight
      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(btn.x + 1, btn.y + 1, btn.w - 2, 1);

      ctx.fillStyle = '#ecf0f1';
      ctx.font = UI_FONT_SMALL;
      ctx.textAlign = 'center';
      ctx.fillText(btn.label, btn.x + btn.w / 2, btn.y + 15);
    }

    // Mute indicator
    if (typeof SFX !== 'undefined' && SFX.muted) {
      ctx.fillStyle = '#e74c3c';
      ctx.font = UI_FONT_SMALL;
      ctx.textAlign = 'right';
      ctx.fillText('MUTED', CANVAS_W - 12, CANVAS_H - 14);
    }
    ctx.textAlign = 'left';
  }

  handleClick(cx, cy) {
    for (const btn of this.buttons) {
      if (cx >= btn.x && cx <= btn.x + btn.w && cy >= btn.y && cy <= btn.y + btn.h) {
        return btn.action;
      }
    }
    return null;
  }
}

// ─── Celebration Effect ─────────────────────────────────────────

class CelebrationEffect {
  constructor() {
    this.particles = [];
    this.active = false;
  }

  trigger() {
    this.active = true;
    this.particles = [];
    for (let i = 0; i < 60; i++) {
      this.particles.push({
        x: CANVAS_W / 2 + (Math.random() - 0.5) * 160,
        y: CANVAS_H / 2,
        px: 0, py: 0, // previous position for trails
        vx: (Math.random() - 0.5) * 5,
        vy: -Math.random() * 4 - 1.5,
        color: ['#f1c40f', '#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#22d3ee'][Math.floor(Math.random() * 6)],
        life: 70 + Math.random() * 50,
        size: 2 + Math.random() * 2.5,
        isCircle: Math.random() > 0.5,
      });
    }
  }

  update(speed) {
    if (!this.active) return;
    const spd = speed || 1;
    for (const p of this.particles) {
      p.px = p.x;
      p.py = p.y;
      p.x += p.vx * spd;
      p.y += p.vy * spd;
      p.vy += 0.06 * spd;
      p.life -= spd;
    }
    this.particles = this.particles.filter(p => p.life > 0);
    if (this.particles.length === 0) this.active = false;
  }

  draw(ctx) {
    if (!this.active) return;
    for (const p of this.particles) {
      // Trail
      if (p.px && p.py) {
        ctx.globalAlpha = Math.min(0.3, p.life / 30);
        ctx.fillStyle = p.color;
        ctx.fillRect(Math.round(p.px), Math.round(p.py), p.size * 0.7, p.size * 0.7);
      }
      // Main particle
      ctx.globalAlpha = Math.min(1, p.life / 20);
      ctx.fillStyle = p.color;
      if (p.isCircle) {
        ctx.beginPath();
        ctx.arc(Math.round(p.x), Math.round(p.y), p.size / 2, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillRect(Math.round(p.x), Math.round(p.y), p.size, p.size);
      }
    }
    ctx.globalAlpha = 1;
  }
}

// ─── Summary Overlay (finale results card) ──────────────────────

class SummaryOverlay {
  constructor() {
    this.visible = false;
    this.alpha = 0;
    this.data = null;
  }

  show(summary) {
    if (!summary) return;
    this.data = summary;
    this.visible = true;
  }

  hide() {
    this.visible = false;
    this.alpha = 0;
    this.data = null;
  }

  update(speed) {
    if (this.visible && this.alpha < 1) {
      this.alpha = Math.min(1, this.alpha + 0.03 * (speed || 1));
    }
  }

  draw(ctx) {
    if (!this.visible || !this.data || this.alpha <= 0) return;
    ctx.globalAlpha = this.alpha;

    const d = this.data;
    const w = 400;
    const h = 240;
    const x = (CANVAS_W - w) / 2;
    const y = (CANVAS_H - h) / 2 - 16;

    // Dim background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Card with animated border
    const tick = window.globalTick || 0;
    const borderPulse = 1.8 + 0.7 * Math.sin(tick * 0.06);

    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, 'rgba(20, 16, 45, 0.97)');
    grad.addColorStop(1, 'rgba(10, 8, 22, 0.98)');
    ctx.fillStyle = grad;
    roundRect(ctx, x, y, w, h, 8);
    ctx.fill();
    ctx.strokeStyle = `rgba(241, 196, 15, ${0.4 + 0.15 * Math.sin(tick * 0.06)})`;
    ctx.lineWidth = borderPulse;
    roundRect(ctx, x, y, w, h, 8);
    ctx.stroke();
    // Glass highlight
    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.fillRect(x + 4, y + 2, w - 8, 1);

    let ty = y + 28;

    // Title with subtle glow
    ctx.save();
    ctx.shadowColor = 'rgba(241, 196, 15, 0.3)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = '#f1c40f';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Experiment Results', CANVAS_W / 2, ty);
    ctx.restore();
    ty += 24;

    // Divider
    const divGrad = ctx.createLinearGradient(x + 20, 0, x + w - 20, 0);
    divGrad.addColorStop(0, 'rgba(241, 196, 15, 0)');
    divGrad.addColorStop(0.5, 'rgba(241, 196, 15, 0.4)');
    divGrad.addColorStop(1, 'rgba(241, 196, 15, 0)');
    ctx.fillStyle = divGrad;
    ctx.fillRect(x + 20, ty - 8, w - 40, 1);
    ty += 12;

    // Column headers
    ctx.font = '10px monospace';
    ctx.fillStyle = '#95a5a6';
    const colX = x + 20;
    const col1 = x + 170;
    const col2 = x + 260;
    const col3 = x + 350;
    ctx.textAlign = 'center';
    ctx.fillText('AUC', col1, ty);
    ctx.fillText('Top 20%', col2, ty);
    ctx.fillText('revenue captured', col3, ty);
    ty += 20;

    // Baseline row
    ctx.textAlign = 'left';
    ctx.fillStyle = '#95a5a6';
    ctx.font = '11px monospace';
    let bName = d.baseline.name;
    if (bName.length > 16) bName = bName.substring(0, 15) + '…';
    ctx.fillText(bName, colX, ty);
    ctx.textAlign = 'center';
    ctx.fillText(d.baseline.auc.toFixed(3), col1, ty);
    ctx.fillText((d.baseline.at_20 * 100).toFixed(1) + '%', col2, ty);
    ctx.fillText('$' + d.baseline.value_captured.toLocaleString(), col3, ty);
    ty += 10;
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('(baseline)', colX, ty);
    ty += 24;

    // Winner row with glow
    ctx.save();
    ctx.shadowColor = 'rgba(46, 204, 113, 0.2)';
    ctx.shadowBlur = 6;
    ctx.textAlign = 'left';
    ctx.fillStyle = '#2ecc71';
    ctx.font = 'bold 11px monospace';
    let wName = d.winner.name;
    if (wName.length > 16) wName = wName.substring(0, 15) + '…';
    ctx.fillText(wName, colX, ty);
    ctx.textAlign = 'center';
    ctx.fillText(d.winner.auc.toFixed(3), col1, ty);
    ctx.fillText((d.winner.at_20 * 100).toFixed(1) + '%', col2, ty);
    ctx.fillText('$' + d.winner.value_captured.toLocaleString(), col3, ty);
    ctx.restore();
    ty += 10;
    ctx.fillStyle = '#27ae60';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('(winner)', colX, ty);
    ty += 30;

    // Divider
    ctx.fillStyle = divGrad;
    ctx.fillRect(x + 20, ty - 10, w - 40, 1);

    // Improvement headline with glow
    ctx.save();
    ctx.shadowColor = 'rgba(241, 196, 15, 0.4)';
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#f1c40f';
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      '+$' + d.extra_revenue.toLocaleString() + ' additional quarterly revenue captured',
      CANVAS_W / 2, ty + 8
    );
    ctx.restore();
    ty += 22;

    ctx.fillStyle = '#bdc3c7';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      `+${d.improvement_pct}% improvement over baseline`,
      CANVAS_W / 2, ty + 4
    );

    ctx.textAlign = 'left';
    ctx.globalAlpha = 1;
  }
}

// ─── Robot Portrait Strip (Hall of Fame) ─────────────────────────

class RobotPortraitStrip {
  constructor() {
    this.robots = []; // array of { cycleIndex, palette }
    this.maxVisible = 6;
  }

  addRobot(cycleIndex) {
    // Don't add duplicates
    if (this.robots.some(r => r.cycleIndex === cycleIndex)) return;
    const p = ROBOT_PALETTES[cycleIndex % ROBOT_PALETTES.length];
    this.robots.push({
      cycleIndex,
      palette: {
        ...p,
        outline: BASE_PALETTE[1],
        screen: BASE_PALETTE[2],
      },
    });
  }

  clear() {
    this.robots = [];
  }

  draw(ctx) {
    if (this.robots.length === 0) return;

    const startX = 12;
    const startY = 34;
    const portraitW = 22;
    const portraitH = 22;
    const gap = 3;
    const shown = this.robots.slice(-this.maxVisible);

    for (let i = 0; i < shown.length; i++) {
      const r = shown[i];
      const x = startX + i * (portraitW + gap);
      const y = startY;
      const isActive = i === shown.length - 1;

      // Background circle
      ctx.fillStyle = isActive ? 'rgba(40, 30, 70, 0.95)' : 'rgba(20, 16, 40, 0.85)';
      ctx.beginPath();
      ctx.arc(x + portraitW / 2, y + portraitH / 2, portraitW / 2 + 1, 0, Math.PI * 2);
      ctx.fill();

      // Active glow border
      if (isActive) {
        const rgb = this._hexToRgb(r.palette.eyes);
        const tick = window.globalTick || 0;
        const pulse = 0.5 + 0.3 * Math.sin(tick * 0.06);
        ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${pulse})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x + portraitW / 2, y + portraitH / 2, portraitW / 2 + 1, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeStyle = 'rgba(80, 60, 140, 0.3)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.arc(x + portraitW / 2, y + portraitH / 2, portraitW / 2 + 1, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw mini robot head (rows 0-8 of sprite, scaled down)
      this._drawMiniHead(ctx, x + 3, y + 2, r.palette);

      // Past robots get slight dim
      if (!isActive) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.arc(x + portraitW / 2, y + portraitH / 2, portraitW / 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  _drawMiniHead(ctx, px, py, palette) {
    // Draw rows 0-8 of SPRITE_DOWN_0 at ~1x scale (16 pixels wide → 16px)
    const headRows = SPRITE_DOWN_0.slice(0, 9);
    const scale = 1.1;
    const headPalette = [
      null,
      palette.outline,
      palette.screen,
      palette.body,
      null,
      null,
      palette.eyes,
      palette.antenna,
      palette.accent || palette.body,
    ];

    for (let r = 0; r < headRows.length; r++) {
      for (let c = 0; c < 16; c++) {
        const val = headRows[r][c];
        if (val === 0 || !headPalette[val]) continue;
        ctx.fillStyle = headPalette[val];
        const sx = Math.round(px + c * scale);
        const sy = Math.round(py + r * scale);
        const sw = Math.max(1, Math.round(px + (c + 1) * scale) - sx);
        const sh = Math.max(1, Math.round(py + (r + 1) * scale) - sy);
        ctx.fillRect(sx, sy, sw, sh);
      }
    }
  }

  _hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    } : { r: 34, g: 211, b: 238 };
  }
}

// ─── Helpers ────────────────────────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function wrapText(ctx, text, maxWidth, maxLines) {
  maxLines = maxLines || 5;
  const words = text.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    const test = current ? current + ' ' + word : word;
    if (ctx.measureText(test).width > maxWidth && current) {
      lines.push(current);
      current = word;
      if (lines.length >= maxLines) break;
    } else {
      current = test;
    }
  }
  if (current && lines.length < maxLines) lines.push(current);
  if (lines.length >= maxLines && current && lines[lines.length - 1] !== current) {
    const last = lines[lines.length - 1];
    if (ctx.measureText(last + '…').width <= maxWidth) {
      lines[lines.length - 1] = last + '…';
    }
  }
  return lines;
}
