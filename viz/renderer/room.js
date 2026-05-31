// Room rendering — Deep Space Command theme

const TILE = 40;
const ROOM_W = 15;
const ROOM_H = 10;

const COLORS = {
  floor: '#2a2640',
  floorAlt: '#262338',
  wall: '#1a1535',
  wallTop: '#110e22',
  baseboard: '#2d2845',
  baseboardHighlight: '#3d3860',
  desk: '#3a3555',
  deskTop: '#332f4e',
  deskShadow: '#252040',
  deskHighlight: '#4a4570',
  monitor: '#0d0b18',
  monitorScreen: '#40e8a0',
  bookshelf: '#4a3860',
  bookshelfFrame: '#3a2850',
  bookshelfShadow: '#2d1f40',
  bookshelfHighlight: '#5a4870',
  bookRed: '#e74c3c',
  bookBlue: '#3498db',
  bookGreen: '#2ecc71',
  bookYellow: '#f1c40f',
  whiteboard: '#e8ecf0',
  whiteboardBorder: '#5a5475',
  labDesk: '#353050',
  labDeskShadow: '#2a2540',
  labDeskHighlight: '#454070',
  labScreen: '#40e8a0',
  labelBg: 'rgba(10, 8, 20, 0.7)',
  labelText: '#ecf0f1',
};

// Clockwise circuit: Desk(left) → Library(top) → Lab(right) → WB(bottom)
const STATIONS = {
  desk:       { x: 1,  y: 4, w: 2, h: 2, stand: [3, 5] },
  library:    { x: 6,  y: 0, w: 3, h: 2, stand: [7, 2] },
  lab:        { x: 11, y: 5, w: 2, h: 2, stand: [10, 6] },
  whiteboard: { x: 5,  y: 7, w: 4, h: 2, stand: [7, 6] },
  center:     { x: 7,  y: 4, w: 1, h: 1, stand: [7, 4] },
};

// Subtle per-tile noise
const _tileNoise = [];
(function () {
  let seed = 42;
  for (let i = 0; i < ROOM_W * ROOM_H; i++) {
    seed = (seed * 16807 + 0) % 2147483647;
    _tileNoise.push((seed / 2147483647 - 0.5) * 0.04);
  }
})();

function _adjustBrightness(hex, factor) {
  const r = Math.min(255, Math.max(0, Math.round(parseInt(hex.slice(1, 3), 16) * (1 + factor))));
  const g = Math.min(255, Math.max(0, Math.round(parseInt(hex.slice(3, 5), 16) * (1 + factor))));
  const b = Math.min(255, Math.max(0, Math.round(parseInt(hex.slice(5, 7), 16) * (1 + factor))));
  return `rgb(${r},${g},${b})`;
}

function isWall(tx, ty) {
  return tx === 0 || ty === 0 || tx === ROOM_W - 1 || ty === ROOM_H - 1;
}

function isWalkable(tx, ty) {
  if (tx < 0 || ty < 0 || tx >= ROOM_W || ty >= ROOM_H) return false;
  if (isWall(tx, ty)) return false;
  for (const s of Object.values(STATIONS)) {
    if (s.w === 0 && s.h === 0) continue;
    if (tx >= s.x && tx < s.x + s.w && ty >= s.y && ty < s.y + s.h) return false;
  }
  return true;
}

function drawFloor(ctx) {
  for (let y = 0; y < ROOM_H; y++) {
    for (let x = 0; x < ROOM_W; x++) {
      const noise = _tileNoise[y * ROOM_W + x];
      if (isWall(x, y)) {
        const wallBase = y === 0 ? COLORS.wallTop : COLORS.wall;
        ctx.fillStyle = _adjustBrightness(wallBase, noise);
      } else {
        const base = (x + y) % 2 === 0 ? COLORS.floor : COLORS.floorAlt;
        ctx.fillStyle = _adjustBrightness(base, noise);
      }
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
    }
  }
  // Baseboard
  ctx.fillStyle = COLORS.baseboard;
  ctx.fillRect(TILE, TILE, (ROOM_W - 2) * TILE, 2);
  ctx.fillRect(TILE, (ROOM_H - 1) * TILE - 2, (ROOM_W - 2) * TILE, 2);
  ctx.fillRect(TILE, TILE, 2, (ROOM_H - 2) * TILE);
  ctx.fillRect((ROOM_W - 1) * TILE - 2, TILE, 2, (ROOM_H - 2) * TILE);
  // Highlight
  ctx.fillStyle = COLORS.baseboardHighlight;
  ctx.fillRect(TILE, TILE, (ROOM_W - 2) * TILE, 1);
  ctx.fillRect(TILE, TILE, 1, (ROOM_H - 2) * TILE);
}

function _drawGlow(ctx, cx, cy, radius, color) {
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  grad.addColorStop(0, color);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
}

function _drawScanlines(ctx, x, y, w, h) {
  const tick = window.globalTick || 0;
  const offset = Math.floor(tick / 4) % 4;
  ctx.fillStyle = 'rgba(0,0,0,0.1)';
  for (let ly = offset; ly < h; ly += 4) {
    ctx.fillRect(x, y + ly, w, 1);
  }
}

// ─── Set Dressing ───────────────────────────────────────────────

function drawSetDressing(ctx) {
  const tick = window.globalTick || 0;

  // Command ring in center — visible but not distracting
  const cx = 7.5 * TILE, cy = 4.5 * TILE;
  ctx.strokeStyle = 'rgba(100, 80, 180, 0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, 55, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(100, 80, 180, 0.08)';
  ctx.beginPath();
  ctx.arc(cx, cy, 45, 0, Math.PI * 2);
  ctx.stroke();
  // Cross lines + diagonal ticks
  ctx.strokeStyle = 'rgba(100, 80, 180, 0.07)';
  ctx.beginPath();
  ctx.moveTo(cx - 55, cy); ctx.lineTo(cx + 55, cy);
  ctx.moveTo(cx, cy - 55); ctx.lineTo(cx, cy + 55);
  ctx.stroke();
  // Rotating tick on the ring
  const ringAngle = tick * 0.015;
  ctx.fillStyle = 'rgba(100, 80, 220, 0.2)';
  ctx.fillRect(
    cx + Math.cos(ringAngle) * 50 - 1,
    cy + Math.sin(ringAngle) * 50 - 1,
    3, 3
  );

  // Floor path circuit — dashed connecting stations
  ctx.strokeStyle = 'rgba(100, 80, 180, 0.1)';
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 8]);
  const pathPoints = [
    [3 * TILE + 20, 5 * TILE + 20],   // desk stand
    [7 * TILE + 20, 2 * TILE + 20],   // library stand
    [10 * TILE + 20, 6 * TILE + 20],  // lab stand
    [7 * TILE + 20, 6 * TILE + 20],   // whiteboard stand
  ];
  ctx.beginPath();
  ctx.moveTo(pathPoints[0][0], pathPoints[0][1]);
  for (let i = 1; i < pathPoints.length; i++) {
    ctx.lineTo(pathPoints[i][0], pathPoints[i][1]);
  }
  ctx.lineTo(pathPoints[0][0], pathPoints[0][1]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Small arrow markers along the path (clockwise direction indicators)
  for (let i = 0; i < pathPoints.length; i++) {
    const [ax, ay] = pathPoints[i];
    const [bx, by] = pathPoints[(i + 1) % pathPoints.length];
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    ctx.fillStyle = 'rgba(100, 80, 220, 0.15)';
    ctx.beginPath();
    ctx.arc(mx, my, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Wall constellation dots — top wall + side walls
  const starSeed = [
    [4.2, 0.3], [4.8, 0.6], [5.1, 0.2], [9.5, 0.4], [10.1, 0.7],
    [10.8, 0.3], [11.2, 0.5], [11.8, 0.2], [3.5, 0.5], [12.5, 0.6],
    [2.2, 0.4], [13.5, 0.3], [1.5, 0.7], [8.3, 0.15],
    // Side wall stars
    [0.3, 3.2], [0.6, 5.8], [0.2, 7.3],
    [14.3, 2.5], [14.7, 4.8], [14.4, 7.6],
  ];
  for (const [sx, sy] of starSeed) {
    const brightness = 0.2 + Math.sin(tick * 0.02 + sx * 3) * 0.1;
    ctx.fillStyle = `rgba(180, 200, 255, ${brightness})`;
    const size = sy < 1 ? 1.5 : 1; // top wall stars slightly bigger
    ctx.fillRect(Math.round(sx * TILE), Math.round(sy * TILE), size, size);
  }

  // Left wall — small clock
  const clockX = 1 * TILE + 20, clockY = 2 * TILE + 20;
  ctx.strokeStyle = 'rgba(140, 120, 200, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(clockX, clockY, 10, 0, Math.PI * 2);
  ctx.stroke();
  // Hour markers
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    ctx.fillStyle = 'rgba(140, 120, 200, 0.2)';
    ctx.fillRect(clockX + Math.sin(a) * 8 - 0.5, clockY - Math.cos(a) * 8 - 0.5, 1, 1);
  }
  // Clock hands
  const minuteAngle = (tick * 0.01) % (Math.PI * 2);
  const hourAngle = minuteAngle / 12;
  ctx.strokeStyle = 'rgba(180, 160, 230, 0.35)';
  ctx.beginPath();
  ctx.moveTo(clockX, clockY);
  ctx.lineTo(clockX + Math.sin(hourAngle) * 5, clockY - Math.cos(hourAngle) * 5);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(200, 180, 240, 0.4)';
  ctx.beginPath();
  ctx.moveTo(clockX, clockY);
  ctx.lineTo(clockX + Math.sin(minuteAngle) * 8, clockY - Math.cos(minuteAngle) * 8);
  ctx.stroke();

  // Right wall — small status display
  const dpX = 13 * TILE + 10, dpY = 2 * TILE + 10;
  ctx.fillStyle = 'rgba(40, 30, 70, 0.5)';
  ctx.fillRect(dpX, dpY, 22, 16);
  ctx.strokeStyle = 'rgba(100, 80, 160, 0.35)';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(dpX, dpY, 22, 16);
  // Blinking dot
  const blink = Math.sin(tick * 0.1) > 0;
  if (blink) {
    ctx.fillStyle = '#40e8a0';
    ctx.fillRect(dpX + 16, dpY + 3, 3, 3);
    _drawGlow(ctx, dpX + 17.5, dpY + 4.5, 8, 'rgba(64, 232, 160, 0.15)');
  }
  // Tiny bars
  ctx.fillStyle = 'rgba(64, 232, 160, 0.4)';
  ctx.fillRect(dpX + 3, dpY + 4, 9, 2);
  ctx.fillRect(dpX + 3, dpY + 8, 7, 2);
  ctx.fillRect(dpX + 3, dpY + 12, 5, 2);

  // Cable runs — connecting desk → center → lab
  ctx.strokeStyle = 'rgba(80, 60, 140, 0.18)';
  ctx.lineWidth = 1;
  // Desk → center
  ctx.beginPath();
  ctx.moveTo(3 * TILE, 5 * TILE + 30);
  ctx.lineTo(5.5 * TILE, 5 * TILE + 30);
  ctx.lineTo(6 * TILE, 4.5 * TILE);
  ctx.stroke();
  // Center → lab
  ctx.beginPath();
  ctx.moveTo(9 * TILE, 4.5 * TILE);
  ctx.lineTo(9.5 * TILE, 5 * TILE + 10);
  ctx.lineTo(11 * TILE, 5 * TILE + 10);
  ctx.stroke();
  // Desk → whiteboard (along bottom)
  ctx.strokeStyle = 'rgba(80, 60, 140, 0.12)';
  ctx.beginPath();
  ctx.moveTo(2 * TILE, 6 * TILE + 20);
  ctx.lineTo(2 * TILE, 7.5 * TILE);
  ctx.lineTo(5 * TILE, 7.5 * TILE);
  ctx.stroke();

  // Floor hex grid — very faint geometric pattern in center area
  ctx.strokeStyle = 'rgba(80, 60, 140, 0.04)';
  ctx.lineWidth = 0.5;
  for (let gx = 4; gx <= 10; gx++) {
    for (let gy = 3; gy <= 6; gy++) {
      const px = gx * TILE + 20, py = gy * TILE + 20;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * Math.PI * 2 - Math.PI / 6;
        const hx = px + Math.cos(a) * 12;
        const hy = py + Math.sin(a) * 12;
        if (i === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
}

// ─── Furniture ──────────────────────────────────────────────────

function drawDesk(ctx) {
  const s = STATIONS.desk;
  const x = s.x * TILE, y = s.y * TILE;
  const w = s.w * TILE, h = s.h * TILE;
  ctx.fillStyle = COLORS.deskShadow;
  ctx.fillRect(x + 3, y + 3, w, h);
  ctx.fillStyle = COLORS.deskTop;
  ctx.fillRect(x, y, w, 5);
  ctx.fillStyle = COLORS.desk;
  ctx.fillRect(x, y + 5, w, h - 5);
  ctx.fillStyle = COLORS.deskHighlight;
  ctx.fillRect(x, y, w, 1);
  ctx.fillRect(x, y, 1, h);
  // Monitor
  ctx.fillStyle = COLORS.monitor;
  ctx.fillRect(x + 10, y + 6, 40, 32);
  ctx.fillStyle = COLORS.monitorScreen;
  ctx.fillRect(x + 14, y + 9, 32, 26);
  _drawScanlines(ctx, x + 14, y + 9, 32, 26);
  _drawGlow(ctx, x + 30, y + 22, 50, 'rgba(64, 232, 160, 0.15)');
  // Stand
  ctx.fillStyle = '#2a2545';
  ctx.fillRect(x + 26, y + 38, 12, 4);
  ctx.fillRect(x + 22, y + 42, 20, 3);
  // Papers
  ctx.fillStyle = '#c8c0e0';
  ctx.fillRect(x + 58, y + 12, 16, 20);
  ctx.fillStyle = '#a098c0';
  ctx.fillRect(x + 60, y + 16, 10, 2);
  ctx.fillRect(x + 60, y + 20, 10, 2);
  ctx.fillRect(x + 60, y + 24, 8, 2);
  // Coffee mug (warm touch)
  ctx.fillStyle = '#c06030';
  ctx.fillRect(x + 58, y + 38, 10, 14);
  ctx.fillStyle = '#d07040';
  ctx.fillRect(x + 68, y + 40, 4, 8);
}

function drawWhiteboard(ctx) {
  const s = STATIONS.whiteboard;
  const x = s.x * TILE, y = s.y * TILE;
  const w = s.w * TILE, h = s.h * TILE;

  // Light pool on the floor around the whiteboard — it's a light source!
  _drawGlow(ctx, x + w / 2, y - 10, 100, 'rgba(220, 225, 240, 0.08)');
  _drawGlow(ctx, x + w / 2, y + h / 2, 80, 'rgba(200, 210, 240, 0.12)');

  ctx.fillStyle = COLORS.whiteboardBorder;
  ctx.fillRect(x - 3, y, w + 6, h + 3);
  ctx.fillStyle = COLORS.whiteboard;
  ctx.fillRect(x, y + 3, w, h - 3);

  // Bright edge highlight — this is the brightest object in the room
  ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.fillRect(x, y + 3, w, 1);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.fillRect(x, y + 3, 1, h - 3);

  // Marker lines
  ctx.fillStyle = '#e74c3c';
  ctx.fillRect(x + 10, y + 12, 60, 4);
  ctx.fillStyle = '#3498db';
  ctx.fillRect(x + 10, y + 24, 85, 4);
  ctx.fillStyle = '#2ecc71';
  ctx.fillRect(x + 10, y + 36, 50, 4);
  ctx.fillStyle = '#9b59b6';
  ctx.fillRect(x + 10, y + 48, 70, 4);
  // Diagram
  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 82, y + 10, 24, 18);
  ctx.strokeRect(x + 82, y + 36, 24, 18);
  ctx.beginPath();
  ctx.moveTo(x + 94, y + 28);
  ctx.lineTo(x + 94, y + 36);
  ctx.stroke();
  // Marker tray
  ctx.fillStyle = '#4a4470';
  ctx.fillRect(x + 12, y + h - 5, w - 24, 5);
  ctx.fillStyle = '#e74c3c';
  ctx.fillRect(x + 18, y + h - 8, 5, 8);
  ctx.fillStyle = '#3498db';
  ctx.fillRect(x + 26, y + h - 8, 5, 8);
  ctx.fillStyle = '#2ecc71';
  ctx.fillRect(x + 34, y + h - 8, 5, 8);
}

function drawLibrary(ctx) {
  const s = STATIONS.library;
  const x = s.x * TILE, y = s.y * TILE;
  const w = s.w * TILE, h = s.h * TILE;
  ctx.fillStyle = COLORS.bookshelfShadow;
  ctx.fillRect(x + 3, y + 3, w, h);
  ctx.fillStyle = COLORS.bookshelfFrame;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = COLORS.bookshelf;
  ctx.fillRect(x + 4, y + 4, w - 8, h - 8);
  ctx.fillStyle = COLORS.bookshelfHighlight;
  ctx.fillRect(x, y, w, 1);
  ctx.fillRect(x, y, 1, h);
  const bookColors = [COLORS.bookRed, COLORS.bookBlue, COLORS.bookGreen, COLORS.bookYellow];
  const shelfH = Math.floor((h - 8) / 3);
  for (let shelf = 0; shelf < 3; shelf++) {
    const sy = y + 4 + shelf * shelfH;
    ctx.fillStyle = COLORS.bookshelfFrame;
    ctx.fillRect(x + 4, sy + shelfH - 4, w - 8, 4);
    for (let b = 0; b < 6; b++) {
      ctx.fillStyle = bookColors[(shelf * 6 + b) % bookColors.length];
      const bx = x + 6 + b * 12;
      const bh = 16 + (b % 3) * 5;
      ctx.fillRect(bx, sy + shelfH - 4 - bh, 10, bh);
    }
  }
}

function drawLab(ctx) {
  const s = STATIONS.lab;
  const x = s.x * TILE, y = s.y * TILE;
  const w = s.w * TILE, h = s.h * TILE;
  ctx.fillStyle = COLORS.labDeskShadow;
  ctx.fillRect(x + 3, y + 3, w, h);
  ctx.fillStyle = COLORS.labDesk;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = COLORS.labDeskHighlight;
  ctx.fillRect(x, y, w, 1);
  ctx.fillRect(x, y, 1, h);
  for (let m = 0; m < 2; m++) {
    const mx = x + 6 + m * 38;
    ctx.fillStyle = '#0d0b18';
    ctx.fillRect(mx, y + 4, 32, 26);
    ctx.fillStyle = COLORS.labScreen;
    ctx.fillRect(mx + 3, y + 7, 26, 20);
    _drawScanlines(ctx, mx + 3, y + 7, 26, 20);
    _drawGlow(ctx, mx + 16, y + 17, 35, 'rgba(64, 232, 160, 0.12)');
    ctx.fillStyle = 'rgba(26, 122, 58, 0.8)';
    for (let l = 0; l < 4; l++) {
      ctx.fillRect(mx + 6, y + 10 + l * 5, 10 + (l % 3) * 4, 2);
    }
    ctx.fillStyle = '#2a2545';
    ctx.fillRect(mx + 10, y + 30, 12, 3);
    ctx.fillRect(mx + 7, y + 33, 18, 3);
  }
  ctx.fillStyle = '#2a2545';
  ctx.fillRect(x + 10, y + 42, 36, 14);
  ctx.fillStyle = '#353060';
  for (let k = 0; k < 6; k++) {
    ctx.fillRect(x + 13 + k * 5, y + 45, 3, 3);
    ctx.fillRect(x + 13 + k * 5, y + 50, 3, 3);
  }
  ctx.fillStyle = '#353060';
  ctx.fillRect(x + 54, y + 44, 10, 14);
  ctx.fillStyle = '#403870';
  ctx.fillRect(x + 55, y + 45, 8, 4);
}

// ─── Ambient Motes ──────────────────────────────────────────────

const _ambientMotes = [];
(function () {
  const moteColors = ['rgba(34,211,238,', 'rgba(180,140,255,', 'rgba(232,121,249,', 'rgba(251,191,36,', 'rgba(220,220,255,'];
  for (let i = 0; i < 18; i++) {
    _ambientMotes.push({
      x: Math.random() * ROOM_W * TILE,
      y: Math.random() * ROOM_H * TILE,
      vx: (Math.random() - 0.5) * 0.3,
      vy: -Math.random() * 0.2 - 0.1,
      size: 1 + Math.random(),
      colorBase: moteColors[Math.floor(Math.random() * moteColors.length)],
      alpha: 0.08 + Math.random() * 0.12,
      phase: Math.random() * Math.PI * 2,
    });
  }
})();

function drawAmbientMotes(ctx) {
  const tick = window.globalTick || 0;
  for (const m of _ambientMotes) {
    m.x += m.vx + Math.sin(tick * 0.02 + m.phase) * 0.15;
    m.y += m.vy;
    if (m.y < 0) { m.y = ROOM_H * TILE; m.x = Math.random() * ROOM_W * TILE; }
    if (m.x < 0) m.x = ROOM_W * TILE;
    if (m.x > ROOM_W * TILE) m.x = 0;
    const pulse = 0.7 + 0.3 * Math.sin(tick * 0.04 + m.phase);
    ctx.fillStyle = m.colorBase + (m.alpha * pulse).toFixed(3) + ')';
    ctx.fillRect(Math.round(m.x), Math.round(m.y), m.size, m.size);
  }
}

// ─── Labels & Room Draw ─────────────────────────────────────────

function drawStationLabels(ctx) {
  const labels = { desk: 'Desk', whiteboard: 'Scoreboard', library: 'Library', lab: 'Lab' };
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  for (const [key, label] of Object.entries(labels)) {
    const s = STATIONS[key];
    const cx = (s.x + s.w / 2) * TILE;
    const atTop = (key === 'library');
    const cy = atTop ? (s.y + s.h + 0.2) * TILE : (s.y - 0.3) * TILE;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = COLORS.labelBg;
    ctx.fillRect(cx - tw / 2 - 6, cy - 10, tw + 12, 17);
    ctx.fillStyle = COLORS.labelText;
    ctx.fillText(label, cx, cy + 3);
  }
}

function drawRoom(ctx) {
  drawFloor(ctx);
  drawSetDressing(ctx);
  drawDesk(ctx);
  drawWhiteboard(ctx);
  drawLibrary(ctx);
  drawLab(ctx);
  drawAmbientMotes(ctx);
  drawStationLabels(ctx);
}

// BFS pathfinding
function findPath(sx, sy, tx, ty) {
  if (sx === tx && sy === ty) return [];
  const visited = new Set();
  const queue = [[sx, sy, []]];
  visited.add(`${sx},${sy}`);
  const dirs = [[0,-1],[0,1],[-1,0],[1,0]];
  while (queue.length > 0) {
    const [cx, cy, path] = queue.shift();
    for (const [dx, dy] of dirs) {
      const nx = cx + dx, ny = cy + dy;
      const key = `${nx},${ny}`;
      if (visited.has(key)) continue;
      if (!isWalkable(nx, ny) && !(nx === tx && ny === ty)) continue;
      const newPath = [...path, [nx, ny]];
      if (nx === tx && ny === ty) return newPath;
      visited.add(key);
      queue.push([nx, ny, newPath]);
    }
  }
  return [[tx, ty]];
}
