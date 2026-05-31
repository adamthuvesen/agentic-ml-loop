// Scene sequencer — drives the animation from script.json data

const SCENE_STATE = {
  WALKING: 'walking',
  ACTIVE: 'active',
  TRANSITION: 'transition',
  ROBOT_SWAP: 'robot_swap',
};

const STATION_ACTIONS = {
  desk: CharState.READ,
  library: CharState.READ,
  lab: CharState.TYPE,
  whiteboard: CharState.WRITE,
  center: CharState.THINK,
};

const SPEECH_LINES = {
  intro:      ["Let's do this!", "Ooh, a new challenge!", "Time to science!"],
  hypothesis: ["I have a plan!", "What if we try...", "Hmm, let me think...", "Big brain time!"],
  research:   ["So many books!", "Fascinating...", "Ooh, interesting!", "Knowledge is power!"],
  training:   ["Beep boop!", "Crunching numbers...", "Go, models, go!", "*intense computing*"],
  evaluation: ["Drumroll please!", "And the winner is...", "Let's see the scores!", "The moment of truth!"],
  dance:      ["WOOHOO!", "New record!", "I'm the best!", "Dance time!", "Yesss!"],
  finale:     ["We did it!", "Mission complete!", "GG!", "That was fun!"],
  journal:    ["Writing it down!", "Dear diary...", "For the next agent!", "Noting my findings..."],
  arrive:     ["Reporting for duty!", "Let's pick up where they left off!", "New agent online!", "I've read the journal!"],
  depart:     ["My work here is done!", "Tag, you're it!", "Good luck, next agent!", "Signing off!"],
};

class ScenePlayer {
  constructor(scriptData, character, textOverlay, leaderboard, progressUI, celebration, summary, portraits) {
    this.scenes = scriptData.scenes || [];
    this.character = character;
    this.overlay = textOverlay;
    this.leaderboard = leaderboard;
    this.progress = progressUI;
    this.celebration = celebration;
    this.summary = summary;
    this.portraits = portraits || null;

    this.sceneIndex = 0;
    this.state = SCENE_STATE.TRANSITION;
    this.timer = 0;
    this.paused = false;
    this.speed = 1;
    this.currentCycle = 0;

    this.progress.totalScenes = this.scenes.length;
    this.progress.totalCycles = scriptData.total_cycles || 4;

    // Start first scene after a beat
    this.timer = 30;
  }

  get currentScene() {
    return this.scenes[this.sceneIndex] || null;
  }

  sceneDuration(scene) {
    if (!scene) return 150;
    // Proportional to text: ~1.25s per 15 words, min ~3.75s
    // (0.8x speed = 1.25x duration, walking stays normal)
    const words = (scene.text || '').split(' ').length
      + (scene.findings || []).join(' ').split(' ').length
      + (scene.candidates || []).length * 2;
    const baseTicks = Math.max(129, Math.ceil(words / 15 * 43));
    if (scene.type === 'intro') return Math.max(baseTicks, 259);
    if (scene.type === 'finale') return Math.max(baseTicks, 301);
    if (scene.type === 'evaluation') return Math.max(baseTicks, 215);
    return baseTicks;
  }

  update() {
    if (this.paused) return;

    const speedTicks = this.speed;

    if (this.state === SCENE_STATE.TRANSITION) {
      this.timer -= speedTicks;
      if (this.timer <= 0) {
        this._startScene();
      }
      return;
    }

    if (this.state === SCENE_STATE.WALKING) {
      return;
    }

    if (this.state === SCENE_STATE.ROBOT_SWAP) {
      // Animation is driven by character's teleport state — just wait
      return;
    }

    if (this.state === SCENE_STATE.ACTIVE) {
      this.timer -= speedTicks;
      if (this.timer <= 0) {
        this._endScene();
      }
    }
  }

  _startScene() {
    const scene = this.currentScene;
    if (!scene) return;

    this.progress.sceneIndex = this.sceneIndex;
    if (scene.cycle) this.progress.cycle = scene.cycle;

    // Cycle boundary detection — trigger robot swap animation
    if (scene.cycle && scene.cycle !== this.currentCycle && this.currentCycle > 0) {
      this.state = SCENE_STATE.ROBOT_SWAP;
      this.overlay.hide();
      this.summary.hide();

      const departLines = SPEECH_LINES.depart || [];
      this.character.speechText = departLines[Math.floor(Math.random() * departLines.length)] || '';

      SFX.depart();
      this.character.startDepart(() => {
        // Switch palette to new cycle
        this.character.setPalette(scene.cycle - 1); // 0-indexed palette

        // Teleport to desk for dramatic entrance
        const desk = STATIONS['desk'];
        if (desk) {
          this.character.tileX = desk.stand[0];
          this.character.tileY = desk.stand[1];
          this.character.pixelX = desk.stand[0] * TILE;
          this.character.pixelY = desk.stand[1] * TILE;
        }

        SFX.spawn();
        this.character.startSpawn(() => {
          this.currentCycle = scene.cycle;

          // Add portrait for new agent
          if (this.portraits) {
            this.portraits.addRobot(scene.cycle - 1);
          }

          const arriveLines = SPEECH_LINES.arrive || [];
          this.character.speechText = arriveLines[Math.floor(Math.random() * arriveLines.length)] || '';

          // Resume normal scene flow
          this._startScene();
        });
      });
      return;
    }

    // First cycle — set palette silently
    if (scene.cycle && this.currentCycle === 0) {
      this.currentCycle = scene.cycle;
      this.character.setPalette(scene.cycle - 1);
      if (this.portraits) {
        this.portraits.addRobot(scene.cycle - 1);
      }
    }

    // Play scene-type sound effect
    if (scene.type === 'intro') SFX.intro();
    else if (scene.type === 'hypothesis') SFX.hypothesis();
    else if (scene.type === 'training') SFX.training();
    else if (scene.type === 'evaluation') SFX.evaluation();
    else if (scene.type === 'finale') SFX.finale();
    else SFX.sceneTransition();

    const station = STATIONS[scene.station];
    if (!station) {
      this._onArrived();
      return;
    }

    const [tx, ty] = station.stand;
    if (this.character.tileX === tx && this.character.tileY === ty) {
      this._onArrived();
    } else {
      this.state = SCENE_STATE.WALKING;
      this.overlay.hide();
      this.summary.hide();
      this.character.speechText = '';
      this.character.walkTo(tx, ty, () => this._onArrived());
    }
  }

  _onArrived() {
    const scene = this.currentScene;
    if (!scene) return;

    this.state = SCENE_STATE.ACTIVE;
    this.timer = this.sceneDuration(scene);

    // Set character action + speech (journal scenes override to TYPE for writing)
    this.character.state = scene.type === 'journal'
      ? CharState.TYPE
      : (STATION_ACTIONS[scene.station] || CharState.IDLE);
    const lines = SPEECH_LINES[scene.type] || [];
    this.character.speechText = lines.length ? lines[Math.floor(Math.random() * lines.length)] : '';

    // Show text overlay
    this.overlay.show(scene);

    // Update leaderboard on evaluation scenes
    if (scene.type === 'evaluation' && scene.leaderboard) {
      this.leaderboard.update(scene.leaderboard, scene.is_new_best);
      if (scene.is_new_best) {
        SFX.newBest();
        this.celebration.trigger();
        // Dance! Then return to normal action
        const danceLines = SPEECH_LINES.dance || [];
        this.character.speechText = danceLines.length ? danceLines[Math.floor(Math.random() * danceLines.length)] : 'WOOHOO!';
        const normalAction = STATION_ACTIONS[scene.station] || CharState.IDLE;
        this.character.startDance(() => {
          this.character.state = normalAction;
        });
      }
    }

    // Finale — show full leaderboard + summary
    if (scene.type === 'finale' && scene.leaderboard) {
      this.leaderboard.update(scene.leaderboard, false);
    }
    if (scene.type === 'finale' && scene.summary) {
      this.summary.show(scene.summary);
    }
  }

  _endScene() {
    this.overlay.hide();
    this.character.state = CharState.IDLE;
    this.character.speechText = '';
    this.sceneIndex++;

    if (this.sceneIndex >= this.scenes.length) {
      this.sceneIndex = this.scenes.length - 1;
      this.state = SCENE_STATE.ACTIVE;
      this.paused = true;
      return;
    }

    this.state = SCENE_STATE.TRANSITION;
    this.timer = 20;
  }

  nextScene() {
    if (this.sceneIndex < this.scenes.length - 1) {
      this.overlay.hide();
      this.character.state = CharState.IDLE;
      this.sceneIndex++;
      this._rebuildState();
      this.state = SCENE_STATE.TRANSITION;
      this.timer = 5;
      this.paused = false;
    }
  }

  prevScene() {
    if (this.sceneIndex > 0) {
      this.overlay.hide();
      this.character.state = CharState.IDLE;

      this.sceneIndex--;
      this._rebuildState();
      this.state = SCENE_STATE.TRANSITION;
      this.timer = 5;
      this.paused = false;
    }
  }

  _rebuildState() {
    let lastLeaderboard = null;
    let lastCycle = 0;

    // Also rebuild portrait strip
    if (this.portraits) this.portraits.clear();

    for (let i = 0; i <= this.sceneIndex; i++) {
      const s = this.scenes[i];
      if (s.type === 'evaluation' && s.leaderboard) lastLeaderboard = s.leaderboard;
      if (s.type === 'finale' && s.leaderboard) lastLeaderboard = s.leaderboard;
      if (s.cycle && s.cycle !== lastCycle) {
        lastCycle = s.cycle;
        if (this.portraits) this.portraits.addRobot(s.cycle - 1);
      }
    }

    if (lastLeaderboard) {
      this.leaderboard.update(lastLeaderboard, false);
    } else {
      this.leaderboard.entries = [];
      this.leaderboard.visible = false;
    }

    // Restore palette for current cycle (instant, no animation)
    this.currentCycle = lastCycle;
    if (lastCycle > 0) {
      this.character.setPalette(lastCycle - 1);
    }
    // Cancel any in-progress teleport
    this.character.isDeparting = false;
    this.character.isSpawning = false;
    this.character.teleportParticles = [];
  }

  togglePause() {
    this.paused = !this.paused;
  }

  cycleSpeed() {
    const speeds = [1, 1.25, 1.5, 2, 4];
    const idx = speeds.indexOf(this.speed);
    this.speed = speeds[(idx + 1) % speeds.length];
  }
}
