// Game engine — main loop, canvas setup, input handling

(function () {
  'use strict';

  const W = 600;
  const H = 400;
  const SCALE = 2;

  let canvas, ctx;
  let character, overlay, leaderboard, progressUI, controls, celebration, summary, portraits;
  let player;
  let scriptData = null;

  function init() {
    canvas = document.getElementById('game');
    canvas.width = W;
    canvas.height = H;
    canvas.style.width = W * SCALE + 'px';
    canvas.style.height = H * SCALE + 'px';
    canvas.style.imageRendering = 'pixelated';
    ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    // Load script data from embedded element or window
    const scriptEl = document.getElementById('script-data');
    if (scriptEl) {
      scriptData = JSON.parse(scriptEl.textContent);
    } else if (window.SCRIPT_DATA) {
      scriptData = window.SCRIPT_DATA;
    }

    if (!scriptData) {
      ctx.fillStyle = '#ecf0f1';
      ctx.font = '14px monospace';
      ctx.fillText('No script data loaded.', 20, H / 2);
      ctx.fillText('Run: python viz/generate.py experiments/<id>', 20, H / 2 + 20);
      return;
    }

    // Initialize game objects
    const startStation = STATIONS['desk'] || STATIONS['center'];
    const [sx, sy] = startStation ? startStation.stand : [7, 4];
    character = new Character(sx, sy);
    overlay = new TextOverlay();
    leaderboard = new LeaderboardPanel();
    progressUI = new ProgressUI();
    controls = new PlaybackControls();
    celebration = new CelebrationEffect();
    summary = new SummaryOverlay();
    portraits = new RobotPortraitStrip();

    player = new ScenePlayer(scriptData, character, overlay, leaderboard, progressUI, celebration, summary, portraits);

    // Title
    const titleEl = document.getElementById('title');
    if (titleEl && scriptData.experiment) {
      titleEl.textContent = scriptData.experiment.name || 'Experiment Replay';
    }

    // Input
    canvas.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleKey);

    // Start loop
    requestAnimationFrame(gameLoop);
  }

  let lastTime = 0;
  const FPS = 30;
  const FRAME_MS = 1000 / FPS;

  function gameLoop(timestamp) {
    requestAnimationFrame(gameLoop);

    const delta = timestamp - lastTime;
    if (delta < FRAME_MS) return;
    lastTime = timestamp - (delta % FRAME_MS);

    update();
    draw();
  }

  function update() {
    if (!player) return;
    const spd = player.speed;
    window.globalTick = (window.globalTick || 0) + spd;
    character.update(spd);
    overlay.update(spd);
    celebration.update(spd);
    summary.update(spd);
    player.update();
    controls.playing = !player.paused;
    controls.speed = spd;
  }

  function draw() {
    if (!player) return;
    ctx.clearRect(0, 0, W, H);

    drawRoom(ctx);
    character.draw(ctx);

    overlay.draw(ctx);
    leaderboard.draw(ctx);
    progressUI.draw(ctx);
    portraits.draw(ctx);
    controls.draw(ctx);
    celebration.draw(ctx);
    summary.draw(ctx);
  }

  function handleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) / SCALE;
    const cy = (e.clientY - rect.top) / SCALE;

    const action = controls.handleClick(cx, cy);
    if (action === 'toggle') player.togglePause();
    else if (action === 'next') player.nextScene();
    else if (action === 'prev') player.prevScene();
    else if (action === 'speed') player.cycleSpeed();
  }

  function handleKey(e) {
    if (!player) return;
    if (e.code === 'Space') { e.preventDefault(); player.togglePause(); }
    else if (e.code === 'ArrowRight') player.nextScene();
    else if (e.code === 'ArrowLeft') player.prevScene();
    else if (e.code === 'KeyS') player.cycleSpeed();
    else if (e.code === 'KeyM') SFX.toggleMute();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
