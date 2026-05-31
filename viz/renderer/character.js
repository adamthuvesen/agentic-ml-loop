// Character — fun colorful robot with glow effects, breathing, dance, and teleport beam

const CHAR_SIZE = 40;
const WALK_SPEED = 5;
const ANIM_FRAME_TICKS = 8;

const DIR_DOWN = 0, DIR_UP = 1, DIR_LEFT = 2, DIR_RIGHT = 3;

const BASE_PALETTE = [
  null,          // 0 transparent
  '#78350f',     // 1 warm brown outline
  '#1e293b',     // 2 face screen dark
  '#f97316',     // 3 body bright orange
  '#6b7280',     // 4 legs gray
  '#4b5563',     // 5 feet dark
  '#22d3ee',     // 6 cyan eyes/glow
  '#fbbf24',     // 7 yellow antenna/highlights
  '#ea580c',     // 8 deep orange accent
];

// Each entry overrides indices [3, 6, 7, 8] — body, eyes, antenna, accent
const ROBOT_PALETTES = [
  { body: '#f97316', eyes: '#22d3ee', antenna: '#fbbf24', accent: '#ea580c' }, // 0 orange (default)
  { body: '#3b82f6', eyes: '#f472b6', antenna: '#a78bfa', accent: '#2563eb' }, // 1 blue
  { body: '#10b981', eyes: '#fbbf24', antenna: '#f97316', accent: '#059669' }, // 2 emerald
  { body: '#8b5cf6', eyes: '#34d399', antenna: '#f472b6', accent: '#6d28d9' }, // 3 violet
  { body: '#ef4444', eyes: '#38bdf8', antenna: '#fcd34d', accent: '#b91c1c' }, // 4 red
  { body: '#06b6d4', eyes: '#f97316', antenna: '#22d3ee', accent: '#0891b2' }, // 5 cyan
  { body: '#d946ef', eyes: '#4ade80', antenna: '#fbbf24', accent: '#a21caf' }, // 6 fuchsia
  { body: '#f59e0b', eyes: '#818cf8', antenna: '#34d399', accent: '#d97706' }, // 7 amber
];

// Keep a module-level reference for drawSprite to use
let PALETTE = [...BASE_PALETTE];

const SPRITE_DOWN_0 = [
  [0,0,0,0,0,0,0,7,7,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,2,2,2,2,2,2,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,2,2,6,6,2,2,1,0,0,0,0],
  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,3,8,3,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,0,3,3,8,3,3,8,3,3,0,0,0,0],
  [0,0,0,0,0,4,4,0,0,4,4,0,0,0,0,0],
  [0,0,0,0,0,4,4,0,0,4,4,0,0,0,0,0],
  [0,0,0,0,5,5,5,0,0,5,5,5,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
const SPRITE_DOWN_1 = [
  [0,0,0,0,0,0,0,7,7,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,2,2,2,2,2,2,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,2,2,6,6,2,2,1,0,0,0,0],
  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,3,8,3,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,0,3,3,8,3,3,8,3,3,0,0,0,0],
  [0,0,0,0,0,4,4,0,0,4,4,0,0,0,0,0],
  [0,0,0,0,4,4,0,0,0,0,4,4,0,0,0,0],
  [0,0,0,5,5,5,0,0,0,0,5,5,5,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
const SPRITE_DOWN_2 = [
  [0,0,0,0,0,0,0,7,7,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,2,2,2,2,2,2,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,6,6,2,2,6,6,1,0,0,0,0],
  [0,0,0,0,1,2,2,6,6,2,2,1,0,0,0,0],
  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,3,8,3,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,0,3,3,8,3,3,8,3,3,0,0,0,0],
  [0,0,0,0,0,4,4,4,4,4,4,0,0,0,0,0],
  [0,0,0,0,0,0,4,4,4,4,0,0,0,0,0,0],
  [0,0,0,0,0,5,5,5,5,5,5,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
const SPRITE_UP_0 = [
  [0,0,0,0,0,0,0,7,7,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,8,1,1,1,1,8,1,0,0,0,0],
  [0,0,0,0,1,1,7,1,1,7,1,1,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,8,1,1,1,1,8,1,0,0,0,0],
  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,3,8,3,0,0,0],
  [0,0,0,3,3,7,3,3,3,7,3,3,3,0,0,0],
  [0,0,0,0,3,3,8,3,3,8,3,3,0,0,0,0],
  [0,0,0,0,0,4,4,0,0,4,4,0,0,0,0,0],
  [0,0,0,0,0,4,4,0,0,4,4,0,0,0,0,0],
  [0,0,0,0,5,5,5,0,0,5,5,5,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
const SPRITE_LEFT_0 = [
  [0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
  [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0],
  [0,0,0,1,2,2,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,6,6,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,6,6,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,2,6,6,2,2,1,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,3,0,0,0,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,0,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,0,0,0,0,0,0,0],
  [0,0,0,0,3,3,8,3,3,0,0,0,0,0,0,0],
  [0,0,0,0,4,4,0,4,4,0,0,0,0,0,0,0],
  [0,0,0,0,4,4,0,4,4,0,0,0,0,0,0,0],
  [0,0,0,5,5,5,0,5,5,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
const SPRITE_LEFT_1 = [
  [0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
  [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0],
  [0,0,0,1,2,2,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,6,6,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,6,6,2,2,2,1,0,0,0,0,0,0],
  [0,0,0,1,2,6,6,2,2,1,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,3,0,0,0,0,0,0],
  [0,0,0,3,8,3,3,3,3,3,0,0,0,0,0,0],
  [0,0,0,3,3,7,3,3,3,0,0,0,0,0,0,0],
  [0,0,0,0,3,3,8,3,3,0,0,0,0,0,0,0],
  [0,0,0,0,4,4,0,4,4,0,0,0,0,0,0,0],
  [0,0,0,4,4,0,0,0,4,4,0,0,0,0,0,0],
  [0,0,5,5,5,0,0,0,5,5,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

const SPRITES = {
  [DIR_DOWN]:  [SPRITE_DOWN_0, SPRITE_DOWN_1, SPRITE_DOWN_0, SPRITE_DOWN_2],
  [DIR_UP]:    [SPRITE_UP_0, SPRITE_UP_0, SPRITE_UP_0, SPRITE_UP_0],
  [DIR_LEFT]:  [SPRITE_LEFT_0, SPRITE_LEFT_1, SPRITE_LEFT_0, SPRITE_LEFT_1],
  [DIR_RIGHT]: null,
};

const SPRITE_SCALE = 2.5;

function drawSprite(ctx, sprite, px, py, mirror) {
  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const val = sprite[r][mirror ? 15 - c : c];
      if (val === 0) continue;
      ctx.fillStyle = PALETTE[val];
      const sx = Math.round(px + c * SPRITE_SCALE);
      const sy = Math.round(py + r * SPRITE_SCALE);
      const sw = Math.round(px + (c + 1) * SPRITE_SCALE) - sx;
      const sh = Math.round(py + (r + 1) * SPRITE_SCALE) - sy;
      ctx.fillRect(sx, sy, sw, sh);
    }
  }
}

const CharState = {
  IDLE: 'idle',
  WALK: 'walk',
  READ: 'read',
  TYPE: 'type',
  WRITE: 'write',
  THINK: 'think',
  DANCE: 'dance',
};

class Character {
  constructor(tileX, tileY) {
    this.tileX = tileX;
    this.tileY = tileY;
    this.pixelX = tileX * TILE;
    this.pixelY = tileY * TILE;
    this.dir = DIR_DOWN;
    this.state = CharState.IDLE;
    this.frame = 0;
    this.frameTick = 0;
    this.path = [];
    this.targetTile = null;
    this.onArrive = null;
    this.speechText = '';

    // Dance state
    this.danceTick = 0;
    this.danceBaseX = 0;
    this.danceBaseY = 0;
    this.danceCallback = null;
    this.danceSparkles = [];

    // Breathing
    this.breathTick = 0;

    // Teleport beam state
    this.isSpawning = false;
    this.isDeparting = false;
    this.teleportTick = 0;
    this.teleportDuration = 60;
    this.teleportCallback = null;
    this.teleportParticles = [];
    this.cycleIndex = 0;
    this.palette = [...BASE_PALETTE];
  }

  setPalette(cycleIndex) {
    this.cycleIndex = cycleIndex;
    const p = ROBOT_PALETTES[cycleIndex % ROBOT_PALETTES.length];
    this.palette = [...BASE_PALETTE];
    this.palette[3] = p.body;
    this.palette[6] = p.eyes;
    this.palette[7] = p.antenna;
    this.palette[8] = p.accent;
    PALETTE = this.palette;
  }

  startDepart(callback) {
    this.isDeparting = true;
    this.isSpawning = false;
    this.teleportTick = 0;
    this.teleportDuration = 60;
    this.teleportCallback = callback || null;
    this.teleportParticles = [];
    this.state = CharState.IDLE;
    this.speechText = 'My work here is done!';
    this.dir = DIR_DOWN;
  }

  startSpawn(callback) {
    this.isSpawning = true;
    this.isDeparting = false;
    this.teleportTick = 0;
    this.teleportDuration = 60;
    this.teleportCallback = callback || null;
    this.teleportParticles = [];
  }

  startDance(callback) {
    this.state = CharState.DANCE;
    this.danceTick = 0;
    this.danceBaseX = this.pixelX;
    this.danceBaseY = this.pixelY;
    this.danceCallback = callback || null;
    this.danceSparkles = [];
  }

  walkTo(tx, ty, callback) {
    this.path = findPath(this.tileX, this.tileY, tx, ty);
    this.targetTile = null;
    this.onArrive = callback || null;
    if (this.path.length === 0) {
      this.state = CharState.IDLE;
      if (this.onArrive) this.onArrive();
      return;
    }
    this.state = CharState.WALK;
    this._nextTileTarget();
  }

  _nextTileTarget() {
    if (this.path.length === 0) {
      this.state = CharState.IDLE;
      if (this.onArrive) {
        const cb = this.onArrive;
        this.onArrive = null;
        cb();
      }
      return;
    }
    this.targetTile = this.path.shift();
    const dx = this.targetTile[0] - this.tileX;
    const dy = this.targetTile[1] - this.tileY;
    if (dx > 0) this.dir = DIR_RIGHT;
    else if (dx < 0) this.dir = DIR_LEFT;
    else if (dy > 0) this.dir = DIR_DOWN;
    else if (dy < 0) this.dir = DIR_UP;
  }

  update(speed) {
    const spd = speed || 1;
    this.frameTick += spd;
    this.breathTick += spd;
    if (this.frameTick >= ANIM_FRAME_TICKS) {
      this.frameTick -= ANIM_FRAME_TICKS;
      this.frame = (this.frame + 1) % 4;
    }

    if (this.state === CharState.WALK && this.targetTile) {
      const tx = this.targetTile[0] * TILE;
      const ty = this.targetTile[1] * TILE;
      const dx = tx - this.pixelX;
      const dy = ty - this.pixelY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const walkSpeed = WALK_SPEED * spd;
      if (dist <= walkSpeed) {
        this.pixelX = tx;
        this.pixelY = ty;
        this.tileX = this.targetTile[0];
        this.tileY = this.targetTile[1];
        this._nextTileTarget();
      } else {
        this.pixelX += (dx / dist) * walkSpeed;
        this.pixelY += (dy / dist) * walkSpeed;
      }
    }

    if (this.state === CharState.DANCE) {
      this._updateDance(spd);
    }

    // Update teleport animation
    if (this.isDeparting || this.isSpawning) {
      this._updateTeleport(spd);
    }

    // Update sparkles
    for (const s of this.danceSparkles) {
      s.x += s.vx * spd;
      s.y += s.vy * spd;
      s.life -= spd;
    }
    this.danceSparkles = this.danceSparkles.filter(s => s.life > 0);

    // Update teleport particles
    for (const p of this.teleportParticles) {
      p.x += p.vx * spd;
      p.y += p.vy * spd;
      p.life -= spd;
    }
    this.teleportParticles = this.teleportParticles.filter(p => p.life > 0);
  }

  _updateTeleport(spd) {
    this.teleportTick += spd;
    const t = this.teleportTick;
    const dur = this.teleportDuration;
    const cx = this.pixelX + 20;
    const cy = this.pixelY + 20;

    if (this.isDeparting) {
      // Emit upward particles during dissolve phase (ticks 15-50)
      if (t >= 15 && t <= 50 && t % 2 === 0) {
        const colors = [this.palette[3], this.palette[6], this.palette[7], this.palette[8], '#fff'];
        for (let i = 0; i < 3; i++) {
          this.teleportParticles.push({
            x: cx + (Math.random() - 0.5) * 30,
            y: cy + 20 - (t - 15) / 35 * 40,
            vx: (Math.random() - 0.5) * 1.5,
            vy: -Math.random() * 2.5 - 1,
            life: 25 + Math.random() * 15,
            size: 1 + Math.random() * 2,
            color: colors[Math.floor(Math.random() * colors.length)],
          });
        }
      }

      if (t >= dur) {
        this.isDeparting = false;
        this.teleportTick = 0;
        if (this.teleportCallback) {
          const cb = this.teleportCallback;
          this.teleportCallback = null;
          cb();
        }
      }
    }

    if (this.isSpawning) {
      // Emit downward particles during materialize phase (ticks 10-45)
      if (t >= 10 && t <= 45 && t % 2 === 0) {
        const colors = [this.palette[3], this.palette[6], this.palette[7], this.palette[8], '#fff'];
        for (let i = 0; i < 3; i++) {
          this.teleportParticles.push({
            x: cx + (Math.random() - 0.5) * 30,
            y: cy - 20 + (t - 10) / 35 * 40,
            vx: (Math.random() - 0.5) * 1.5,
            vy: Math.random() * 1.5 + 0.5,
            life: 20 + Math.random() * 15,
            size: 1 + Math.random() * 2,
            color: colors[Math.floor(Math.random() * colors.length)],
          });
        }
      }

      // Arrival burst at tick 48
      if (t === 48) {
        const colors = [this.palette[3], this.palette[6], this.palette[7], this.palette[8]];
        for (let i = 0; i < 20; i++) {
          const angle = (Math.PI * 2 * i) / 20;
          const speed = 1.5 + Math.random() * 2;
          this.teleportParticles.push({
            x: cx,
            y: cy,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 25 + Math.random() * 15,
            size: 1.5 + Math.random() * 2,
            color: colors[Math.floor(Math.random() * colors.length)],
          });
        }
      }

      if (t >= dur) {
        this.isSpawning = false;
        this.teleportTick = 0;
        if (this.teleportCallback) {
          const cb = this.teleportCallback;
          this.teleportCallback = null;
          cb();
        }
      }
    }
  }

  _updateDance(spd) {
    this.danceTick += spd;
    const t = this.danceTick;

    if (t <= 15) {
      // Wiggle left-right
      const wiggle = (Math.floor(t / 4) % 2 === 0) ? -3 : 3;
      this.pixelX = this.danceBaseX + wiggle;
      this.dir = wiggle > 0 ? DIR_RIGHT : DIR_LEFT;
    } else if (t <= 30) {
      // Jump!
      this.pixelX = this.danceBaseX;
      const jumpPhase = t - 15;
      if (jumpPhase <= 7) {
        this.pixelY = this.danceBaseY - (jumpPhase / 7) * 8;
      } else {
        this.pixelY = this.danceBaseY - 8 + ((jumpPhase - 7) / 8) * 8;
      }
      this.dir = DIR_DOWN;
    } else if (t <= 60) {
      // Spin! Cycle directions
      this.pixelX = this.danceBaseX;
      this.pixelY = this.danceBaseY;
      const spinDirs = [DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT];
      this.dir = spinDirs[Math.floor((t - 31) / 4) % 4];
    } else if (t <= 90) {
      // Settle with a little bounce
      this.dir = DIR_DOWN;
      const bounce = t - 60;
      if (bounce <= 8) {
        this.pixelY = this.danceBaseY - (bounce / 8) * 4;
      } else {
        this.pixelY = this.danceBaseY - 4 + ((bounce - 8) / 22) * 4;
      }
      this.pixelX = this.danceBaseX;
    }

    // Emit sparkles during dance
    if (t % 3 === 0) {
      const colors = [this.palette[7], this.palette[6], this.palette[3], '#fff'];
      this.danceSparkles.push({
        x: this.pixelX + 20 + (Math.random() - 0.5) * 30,
        y: this.pixelY + 10 + (Math.random() - 0.5) * 20,
        vx: (Math.random() - 0.5) * 2,
        vy: -Math.random() * 1.5 - 0.5,
        life: 20 + Math.random() * 15,
        size: 1 + Math.random() * 1.5,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }

    // Dance done
    if (t >= 90) {
      this.pixelX = this.danceBaseX;
      this.pixelY = this.danceBaseY;
      this.dir = DIR_DOWN;
      this.state = CharState.IDLE;
      if (this.danceCallback) {
        const cb = this.danceCallback;
        this.danceCallback = null;
        cb();
      }
    }
  }

  draw(ctx) {
    const mirror = this.dir === DIR_RIGHT;
    const dirKey = mirror ? DIR_LEFT : this.dir;
    const frames = SPRITES[dirKey];
    if (!frames) return;

    const isDancing = this.state === CharState.DANCE;
    const isWalking = this.state === CharState.WALK;
    const animFrame = (isWalking || isDancing) ? this.frame % frames.length : 0;
    const sprite = frames[animFrame];

    let px = Math.round(this.pixelX);
    let py = Math.round(this.pixelY);

    // Breathing bob when idle at a station (not walking, not dancing, not teleporting)
    if (!isWalking && !isDancing && !this.isDeparting && !this.isSpawning) {
      const breathOffset = Math.sin(this.breathTick * Math.PI / 30) * 1.5;
      py += Math.round(breathOffset);
    }

    // Draw teleport beam behind the robot
    if (this.isDeparting || this.isSpawning) {
      this._drawTeleportBeam(ctx, px, py);
    }

    // Shadow oval (fade during teleport)
    const shadowAlpha = this.isDeparting
      ? Math.max(0, 1 - this.teleportTick / 40) * 0.2
      : this.isSpawning
        ? Math.min(1, this.teleportTick / 40) * 0.2
        : 0.2;
    ctx.fillStyle = `rgba(0, 0, 0, ${shadowAlpha})`;
    ctx.beginPath();
    ctx.ellipse(px + 20, Math.round(this.pixelY) + 38, 16, 5, 0, 0, Math.PI * 2);
    ctx.fill();

    // Draw sprite with row masking during teleport
    if (this.isDeparting) {
      const progress = Math.max(0, (this.teleportTick - 15) / 35); // dissolve from tick 15-50
      this._drawSpriteWithDissolve(ctx, sprite, px, py, mirror, progress, 'up');
    } else if (this.isSpawning) {
      const progress = Math.max(0, Math.min(1, (this.teleportTick - 8) / 37)); // materialize from tick 8-45
      this._drawSpriteWithDissolve(ctx, sprite, px, py, mirror, progress, 'down');
    } else {
      drawSprite(ctx, sprite, px, py, mirror);
    }

    // Eye glow effect (bloom) — skip when fully dissolved
    if (!(this.isDeparting && this.teleportTick > 45) && !(this.isSpawning && this.teleportTick < 15)) {
      this._drawEyeGlow(ctx, px, py, mirror);
    }

    // Antenna pulse — skip when fully dissolved
    if (!(this.isDeparting && this.teleportTick > 45) && !(this.isSpawning && this.teleportTick < 15)) {
      this._drawAntennaPulse(ctx, px, py, mirror);
    }

    // Dance sparkles
    this._drawSparkles(ctx);

    // Teleport particles (drawn on top)
    this._drawTeleportParticles(ctx);

    // Speech bubble
    if (this.speechText && !isWalking) {
      this._drawSpeechBubble(ctx, px, py);
    }
  }

  _drawSpriteWithDissolve(ctx, sprite, px, py, mirror, progress, direction) {
    // progress: 0 = fully visible, 1 = fully dissolved
    const clampedProgress = Math.max(0, Math.min(1, progress));
    for (let r = 0; r < 16; r++) {
      // 'up' = dissolve bottom-to-top: bottom rows vanish first
      // 'down' = materialize top-to-bottom: top rows appear first
      let rowVisible;
      if (direction === 'up') {
        // Row 15 disappears first, row 0 last
        const rowThreshold = (15 - r) / 16;
        rowVisible = clampedProgress < rowThreshold + 0.0625;
      } else {
        // Row 0 appears first, row 15 last
        const rowThreshold = r / 16;
        rowVisible = clampedProgress > rowThreshold - 0.0625;
      }
      if (!rowVisible) continue;

      // Edge rows get partial alpha for smooth transition
      let rowAlpha = 1;
      if (direction === 'up') {
        const rowNorm = (15 - r) / 16;
        const edge = clampedProgress - rowNorm;
        if (edge > -0.1 && edge < 0.0625) rowAlpha = Math.max(0.2, 1 - (edge + 0.1) * 5);
      } else {
        const rowNorm = r / 16;
        const edge = rowNorm - clampedProgress;
        if (edge > -0.1 && edge < 0.0625) rowAlpha = Math.max(0.2, 1 - (edge + 0.1) * 5);
      }

      if (rowAlpha < 1) ctx.globalAlpha = rowAlpha;
      for (let c = 0; c < 16; c++) {
        const val = sprite[r][mirror ? 15 - c : c];
        if (val === 0) continue;
        ctx.fillStyle = this.palette[val];
        const sx = Math.round(px + c * SPRITE_SCALE);
        const sy = Math.round(py + r * SPRITE_SCALE);
        const sw = Math.round(px + (c + 1) * SPRITE_SCALE) - sx;
        const sh = Math.round(py + (r + 1) * SPRITE_SCALE) - sy;
        ctx.fillRect(sx, sy, sw, sh);
      }
      if (rowAlpha < 1) ctx.globalAlpha = 1;
    }
  }

  _drawTeleportBeam(ctx, px, py) {
    const t = this.teleportTick;
    const cx = px + 20;
    const eyeColor = this.palette[6];

    // Parse eye color for rgba
    const rgb = this._hexToRgb(eyeColor);

    let beamAlpha, beamWidth;
    if (this.isDeparting) {
      // Beam: fade in ticks 8-18, full 18-48, fade out 48-60
      if (t < 8) { beamAlpha = 0; beamWidth = 0; }
      else if (t < 18) { beamAlpha = (t - 8) / 10 * 0.7; beamWidth = 4 + (t - 8) / 10 * 20; }
      else if (t < 48) { beamAlpha = 0.7; beamWidth = 24 + Math.sin(t * 0.15) * 4; }
      else { beamAlpha = (1 - (t - 48) / 12) * 0.7; beamWidth = 24 * (1 - (t - 48) / 12); }
    } else {
      // Spawn beam: fade in ticks 0-10, full 10-48, fade out 48-60
      if (t < 10) { beamAlpha = t / 10 * 0.7; beamWidth = 4 + t / 10 * 20; }
      else if (t < 48) { beamAlpha = 0.7; beamWidth = 24 + Math.sin(t * 0.15) * 4; }
      else { beamAlpha = (1 - (t - 48) / 12) * 0.7; beamWidth = 24 * (1 - (t - 48) / 12); }
    }

    if (beamAlpha <= 0 || beamWidth <= 0) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Main beam gradient (vertical column)
    const beamGrad = ctx.createLinearGradient(cx - beamWidth / 2, 0, cx + beamWidth / 2, 0);
    beamGrad.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
    beamGrad.addColorStop(0.3, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${beamAlpha * 0.4})`);
    beamGrad.addColorStop(0.5, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${beamAlpha})`);
    beamGrad.addColorStop(0.7, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${beamAlpha * 0.4})`);
    beamGrad.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);

    ctx.fillStyle = beamGrad;
    ctx.fillRect(cx - beamWidth / 2, 0, beamWidth, CANVAS_H);

    // Inner bright core
    const coreW = beamWidth * 0.3;
    ctx.fillStyle = `rgba(255, 255, 255, ${beamAlpha * 0.5})`;
    ctx.fillRect(cx - coreW / 2, 0, coreW, CANVAS_H);

    // Scanning ring effect at robot position
    const ringY = this.isDeparting
      ? py + 40 - (Math.max(0, t - 15) / 35) * 50
      : py + 40 - 50 + (Math.max(0, t - 8) / 37) * 50;
    const ringAlpha = beamAlpha * 0.6;
    ctx.strokeStyle = `rgba(255, 255, 255, ${ringAlpha})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, ringY, beamWidth * 0.6, 3, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  _drawTeleportParticles(ctx) {
    for (const p of this.teleportParticles) {
      ctx.globalAlpha = Math.min(1, p.life / 12);
      ctx.fillStyle = p.color;
      ctx.fillRect(Math.round(p.x), Math.round(p.y), p.size, p.size);
    }
    ctx.globalAlpha = 1;
  }

  _hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    } : { r: 34, g: 211, b: 238 }; // fallback cyan
  }

  _drawEyeGlow(ctx, px, py, mirror) {
    const tick = window.globalTick || 0;
    const pulse = 0.6 + 0.4 * Math.sin(tick * 0.08);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    const eyeScale = SPRITE_SCALE;
    const leftEyeX = px + (mirror ? 9 : 5) * eyeScale + eyeScale;
    const rightEyeX = px + (mirror ? 5 : 9) * eyeScale + eyeScale;
    const eyeY = py + 4.5 * eyeScale + eyeScale;

    const glowRadius = 8;
    const alpha = 0.12 * pulse;
    const rgb = this._hexToRgb(this.palette[6]);

    for (const ex of [leftEyeX, rightEyeX]) {
      const grad = ctx.createRadialGradient(ex, eyeY, 0, ex, eyeY, glowRadius);
      grad.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`);
      grad.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(ex - glowRadius, eyeY - glowRadius, glowRadius * 2, glowRadius * 2);
    }

    ctx.restore();
  }

  _drawAntennaPulse(ctx, px, py, mirror) {
    const tick = window.globalTick || 0;
    const pulse = 0.5 + 0.5 * Math.sin(tick * 0.05);
    const ax = px + (mirror ? 8 : 7) * SPRITE_SCALE + SPRITE_SCALE / 2;
    const ay = py + 0.5 * SPRITE_SCALE;

    const rgb = this._hexToRgb(this.palette[7]);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const grad = ctx.createRadialGradient(ax, ay, 0, ax, ay, 6);
    grad.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.2 * pulse})`);
    grad.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(ax - 6, ay - 6, 12, 12);
    ctx.restore();
  }

  _drawSparkles(ctx) {
    for (const s of this.danceSparkles) {
      ctx.globalAlpha = Math.min(1, s.life / 10);
      ctx.fillStyle = s.color;
      ctx.fillRect(Math.round(s.x), Math.round(s.y), s.size, s.size);
    }
    ctx.globalAlpha = 1;
  }

  _drawSpeechBubble(ctx, px, py) {
    const cx = px + 20;
    const by = py - 6;

    ctx.font = '9px monospace';
    const tw = ctx.measureText(this.speechText).width;
    const padX = 6, padY = 4;
    const bw = tw + padX * 2;
    const bh = 14 + padY * 2;
    const bx = cx - bw / 2;
    const bubbleY = by - bh;

    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    roundRect(ctx, bx, bubbleY, bw, bh, 4);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 0.5;
    roundRect(ctx, bx, bubbleY, bw, bh, 4);
    ctx.stroke();

    // Triangle pointer
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    ctx.beginPath();
    ctx.moveTo(cx - 4, by);
    ctx.lineTo(cx + 4, by);
    ctx.lineTo(cx, by + 5);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(this.speechText, cx, bubbleY + bh / 2 + 3);
    ctx.textAlign = 'left';
  }
}
