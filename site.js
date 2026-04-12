const canvas = document.getElementById("chemotaxisCanvas");
const ctx = canvas.getContext("2d");
const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");

const controls = {
  foodStrength: document.getElementById("foodStrength"),
  antibioticStrength: document.getElementById("antibioticStrength"),
  deathStrength: document.getElementById("deathStrength"),
  micLevel: document.getElementById("micLevel"),
  particleCount: document.getElementById("particleCount"),
};

const labels = {
  foodStrength: document.getElementById("foodStrengthValue"),
  antibioticStrength: document.getElementById("antibioticStrengthValue"),
  deathStrength: document.getElementById("deathStrengthValue"),
  micLevel: document.getElementById("micLevelValue"),
  particleCount: document.getElementById("particleCountValue"),
};

const resetButton = document.getElementById("resetButton");
const pauseButton = document.getElementById("pauseButton");
const densitySlider = document.getElementById("density");
const densityValue = document.getElementById("densityValue");

const world = {
  width: 180,
  height: 116,
  dt: 0.35,
  diffusion: 0.42,
  growthRate: 0.0024,
  foodDecay: 0.0016,
  antibioticDecay: 0.0013,
  running: true,
  particles: [],
  foodField: [],
  antibioticField: [],
  foodGradX: [],
  foodGradY: [],
  antibioticGradX: [],
  antibioticGradY: [],
  driftSamples: [],
};

function createGrid(fillValue = 0) {
  return Array.from({ length: world.height }, () => new Float32Array(world.width).fill(fillValue));
}

function gaussian(field, cx, cy, amplitude, sigma) {
  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      field[y][x] += amplitude * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    }
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function randomNormal() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function buildInitialFields() {
  world.foodField = createGrid(0.02);
  world.antibioticField = createGrid(0);

  gaussian(world.foodField, 34, 28, 1.9, 12);
  gaussian(world.foodField, 88, 54, 1.25, 10);
  gaussian(world.foodField, 144, 82, 1.7, 15);

  gaussian(world.antibioticField, 46, 88, 1.45, 13);
  gaussian(world.antibioticField, 126, 34, 1.75, 16);
  gaussian(world.antibioticField, 158, 92, 1.1, 10);
}

function buildParticles() {
  const count = Number(controls.particleCount.value);
  world.particles = [];
  const seeds = [
    { x: 26, y: 22 },
    { x: 58, y: 39 },
    { x: 101, y: 71 },
    { x: 128, y: 48 },
  ];
  for (let i = 0; i < count; i += 1) {
    const seed = seeds[i % seeds.length];
    world.particles.push({
      x: clamp(seed.x + randomNormal() * 7, 0, world.width - 1),
      y: clamp(seed.y + randomNormal() * 7, 0, world.height - 1),
    });
  }
}

function updateLabels() {
  Object.keys(controls).forEach((key) => {
    labels[key].textContent = controls[key].value;
  });
}

function getFieldValue(field, x, y) {
  const x0 = clamp(Math.floor(x), 0, world.width - 1);
  const y0 = clamp(Math.floor(y), 0, world.height - 1);
  const x1 = clamp(x0 + 1, 0, world.width - 1);
  const y1 = clamp(y0 + 1, 0, world.height - 1);
  const tx = x - x0;
  const ty = y - y0;
  const v00 = field[y0][x0];
  const v10 = field[y0][x1];
  const v01 = field[y1][x0];
  const v11 = field[y1][x1];
  return (
    (1 - tx) * (1 - ty) * v00 +
    tx * (1 - ty) * v10 +
    (1 - tx) * ty * v01 +
    tx * ty * v11
  );
}

function computeGradient(field) {
  const gx = createGrid(0);
  const gy = createGrid(0);
  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      const left = field[y][clamp(x - 1, 0, world.width - 1)];
      const right = field[y][clamp(x + 1, 0, world.width - 1)];
      const down = field[clamp(y - 1, 0, world.height - 1)][x];
      const up = field[clamp(y + 1, 0, world.height - 1)][x];
      gx[y][x] = (right - left) * 0.5;
      gy[y][x] = (up - down) * 0.5;
    }
  }
  return { gx, gy };
}

function computeLaplacian(field, x, y) {
  const c = field[y][x];
  const left = field[y][clamp(x - 1, 0, world.width - 1)];
  const right = field[y][clamp(x + 1, 0, world.width - 1)];
  const down = field[clamp(y - 1, 0, world.height - 1)][x];
  const up = field[clamp(y + 1, 0, world.height - 1)][x];
  return left + right + down + up - 4 * c;
}

function densityGrid() {
  const grid = createGrid(0);
  world.particles.forEach((particle) => {
    const x = clamp(Math.floor(particle.x), 0, world.width - 1);
    const y = clamp(Math.floor(particle.y), 0, world.height - 1);
    grid[y][x] += 1;
  });
  return grid;
}

function antibioticActivation(c) {
  const mic = Number(controls.micLevel.value);
  const numerator = c ** 4;
  return numerator / (numerator + mic ** 4 + 1e-6);
}

function resetSimulation() {
  buildInitialFields();
  buildParticles();
  updateLabels();
}

function updateFields() {
  const density = densityGrid();
  const nextFood = createGrid(0);
  const nextAntibiotic = createGrid(0);

  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      const localDensity = density[y][x];
      const food = world.foodField[y][x];
      const antibiotic = world.antibioticField[y][x];
      const foodConsume = world.foodDecay * localDensity * food / (food + 0.2);
      const antibioticConsume = world.antibioticDecay * localDensity * antibiotic;

      nextFood[y][x] = Math.max(0, food + world.dt * (0.18 * computeLaplacian(world.foodField, x, y) - foodConsume));
      nextAntibiotic[y][x] = Math.max(
        0,
        antibiotic + world.dt * (0.15 * computeLaplacian(world.antibioticField, x, y) - antibioticConsume)
      );
    }
  }

  world.foodField = nextFood;
  world.antibioticField = nextAntibiotic;

  const phiFood = world.foodField.map((row) => Float32Array.from(row, (value) => Math.log1p(value / 0.2)));
  const phiAntibiotic = world.antibioticField.map((row) => Float32Array.from(row, (value) => Math.log1p(value / 0.25)));
  const foodGradient = computeGradient(phiFood);
  const antibioticGradient = computeGradient(phiAntibiotic);

  world.foodGradX = foodGradient.gx;
  world.foodGradY = foodGradient.gy;
  world.antibioticGradX = antibioticGradient.gx;
  world.antibioticGradY = antibioticGradient.gy;
}

function updateParticles() {
  const nextParticles = [];
  const births = [];
  const foodStrength = Number(controls.foodStrength.value);
  const antibioticStrength = Number(controls.antibioticStrength.value);
  const deathStrength = Number(controls.deathStrength.value);
  const density = densityGrid();

  world.particles.forEach((particle) => {
    const localFood = getFieldValue(world.foodField, particle.x, particle.y);
    const localAntibiotic = getFieldValue(world.antibioticField, particle.x, particle.y);
    const localDensity = getFieldValue(density, particle.x, particle.y);

    const foodGradX = getFieldValue(world.foodGradX, particle.x, particle.y);
    const foodGradY = getFieldValue(world.foodGradY, particle.x, particle.y);
    const antibioticGradX = getFieldValue(world.antibioticGradX, particle.x, particle.y);
    const antibioticGradY = getFieldValue(world.antibioticGradY, particle.x, particle.y);

    const chiA = 0.8 * foodStrength / (1 + localDensity * 0.06);
    const chiC = 0.55 * antibioticStrength * antibioticActivation(localAntibiotic) / (1 + localDensity * 0.06);

    const driftX = chiA * foodGradX + chiC * antibioticGradX;
    const driftY = chiA * foodGradY + chiC * antibioticGradY;
    const randomScale = Math.sqrt(2 * world.diffusion * world.dt) * 0.34;

    particle.x = clamp(particle.x + driftX * world.dt * 12 + randomNormal() * randomScale, 0, world.width - 1);
    particle.y = clamp(particle.y + driftY * world.dt * 12 + randomNormal() * randomScale, 0, world.height - 1);

    const deathRate = deathStrength * 0.035 * antibioticActivation(localAntibiotic) * (localAntibiotic / (localAntibiotic + Number(controls.micLevel.value) + 1e-3));
    if (Math.random() < deathRate) {
      return;
    }

    nextParticles.push(particle);

    const birthRate = world.growthRate * (0.25 + localFood);
    if (Math.random() < birthRate && nextParticles.length + births.length < 800) {
      births.push({
        x: clamp(particle.x + randomNormal() * 1.2, 0, world.width - 1),
        y: clamp(particle.y + randomNormal() * 1.2, 0, world.height - 1),
      });
    }
  });

  world.particles = nextParticles.concat(births);
  buildDriftSamples();
}

function buildDriftSamples() {
  world.driftSamples = [];
  for (let y = 10; y < world.height; y += 18) {
    for (let x = 10; x < world.width; x += 22) {
      const foodGradX = world.foodGradX[y][x];
      const foodGradY = world.foodGradY[y][x];
      const antibioticGradX = world.antibioticGradX[y][x];
      const antibioticGradY = world.antibioticGradY[y][x];
      world.driftSamples.push({
        x,
        y,
        dx: 1.6 * foodGradX + 1.3 * antibioticGradX,
        dy: 1.6 * foodGradY + 1.3 * antibioticGradY,
      });
    }
  }
}

function drawField(field, colorFn, alpha) {
  const image = ctx.createImageData(world.width, world.height);
  let maxValue = 0;
  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      maxValue = Math.max(maxValue, field[y][x]);
    }
  }
  const safeMax = Math.max(maxValue, 1e-6);

  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      const i = (y * world.width + x) * 4;
      const value = field[y][x] / safeMax;
      const [r, g, b] = colorFn(value);
      image.data[i] = r;
      image.data[i + 1] = g;
      image.data[i + 2] = b;
      image.data[i + 3] = Math.floor(alpha * 255 * value);
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = world.width;
  offscreen.height = world.height;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

function drawMicContour() {
  const mic = Number(controls.micLevel.value);
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let y = 1; y < world.height - 1; y += 1) {
    for (let x = 1; x < world.width - 1; x += 1) {
      const c = world.antibioticField[y][x];
      if (c >= mic && (
        world.antibioticField[y][x - 1] < mic ||
        world.antibioticField[y][x + 1] < mic ||
        world.antibioticField[y - 1][x] < mic ||
        world.antibioticField[y + 1][x] < mic
      )) {
        const px = (x / world.width) * canvas.width;
        const py = (y / world.height) * canvas.height;
        ctx.rect(px, py, canvas.width / world.width, canvas.height / world.height);
      }
    }
  }
  ctx.stroke();
}

function drawDriftArrows() {
  ctx.strokeStyle = "rgba(122, 231, 255, 0.75)";
  ctx.lineWidth = 1.1;
  world.driftSamples.forEach((sample) => {
    const startX = (sample.x / world.width) * canvas.width;
    const startY = (sample.y / world.height) * canvas.height;
    const endX = startX + sample.dx * 26;
    const endY = startY + sample.dy * 26;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  });
}

function drawParticles() {
  world.particles.forEach((particle) => {
    const px = (particle.x / world.width) * canvas.width;
    const py = (particle.y / world.height) * canvas.height;
    ctx.beginPath();
    ctx.fillStyle = "#ffe46b";
    ctx.arc(px, py, 3.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawStats() {
  ctx.fillStyle = "rgba(255,250,240,0.82)";
  ctx.fillRect(16, 16, 240, 88);
  ctx.fillStyle = "#14211b";
  ctx.font = "600 16px Space Grotesk, sans-serif";
  ctx.fillText("Live KS Playground", 28, 40);
  ctx.font = "14px Space Grotesk, sans-serif";
  ctx.fillText(`bacteria: ${world.particles.length}`, 28, 62);
  ctx.fillText(`MIC contour: ${Number(controls.micLevel.value).toFixed(2)}`, 28, 82);
  ctx.fillText("cyan arrows show instantaneous net drift", 28, 102);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#efe8d8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawField(world.foodField, (value) => [58, 200 - value * 32, 84 + value * 100], 0.78);
  drawField(world.antibioticField, (value) => [124 + value * 58, 76, 220], 0.62);
  drawMicContour();
  drawDriftArrows();
  drawParticles();
  drawStats();
}

function animate() {
  if (world.running) {
    updateFields();
    updateParticles();
  }
  draw();
  requestAnimationFrame(animate);
}

Object.keys(controls).forEach((key) => {
  controls[key].addEventListener("input", () => {
    updateLabels();
  });
});

resetButton.addEventListener("click", resetSimulation);
pauseButton.addEventListener("click", () => {
  world.running = !world.running;
  pauseButton.textContent = world.running ? "Pause" : "Resume";
});

resetSimulation();
updateFields();
buildDriftSamples();
animate();

// ----------------------------
// Bacteria mini-game
// ----------------------------
let bacteria = [];
let food = [];

function createBacteria() {
  bacteria = [];
  const count = parseInt(densitySlider.value, 10);
  densityValue.textContent = String(count);

  for (let i = 0; i < count; i += 1) {
    bacteria.push({
      x: Math.random() * gameCanvas.width,
      y: Math.random() * gameCanvas.height,
      vx: 0,
      vy: 0,
      biasX: 0,
      biasY: 0,
    });
  }
}

gameCanvas.addEventListener("click", (e) => {
  const rect = gameCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (gameCanvas.width / rect.width);
  const y = (e.clientY - rect.top) * (gameCanvas.height / rect.height);
  food.push({ x, y });
});

densitySlider.addEventListener("input", createBacteria);

function getGradient(x, y) {
  let gx = 0;
  let gy = 0;

  food.forEach((f) => {
    const dx = f.x - x;
    const dy = f.y - y;
    const dist = Math.sqrt(dx * dx + dy * dy) + 1;
    const strength = 1 / dist;
    gx += (dx / dist) * strength;
    gy += (dy / dist) * strength;
  });

  return { gx, gy };
}

function updateGame() {
  bacteria.forEach((b) => {
    b.vx += (Math.random() - 0.5) * 0.8;
    b.vy += (Math.random() - 0.5) * 0.8;

    const { gx, gy } = getGradient(b.x, b.y);
    const chi = 0.08 * (1 / (1 + bacteria.length / 40));

    b.biasX = 0.9 * b.biasX + 0.1 * gx;
    b.biasY = 0.9 * b.biasY + 0.1 * gy;

    b.vx += b.biasX * chi;
    b.vy += b.biasY * chi;

    b.x += b.vx;
    b.y += b.vy;

    b.vx *= 0.85;
    b.vy *= 0.85;

    b.x = Math.max(0, Math.min(gameCanvas.width, b.x));
    b.y = Math.max(0, Math.min(gameCanvas.height, b.y));
  });

  food = food.filter((f) => {
    let eaten = false;

    bacteria.forEach((b) => {
      const dx = f.x - b.x;
      const dy = f.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 10) eaten = true;
    });

    return !eaten;
  });
}

function drawGame() {
  gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);

  const gradient = gameCtx.createLinearGradient(0, 0, gameCanvas.width, gameCanvas.height);
  gradient.addColorStop(0, "#020617");
  gradient.addColorStop(1, "#0f172a");
  gameCtx.fillStyle = gradient;
  gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

  food.forEach((f) => {
    gameCtx.font = "26px Arial";
    gameCtx.fillText("🍬", f.x, f.y);
  });

  bacteria.forEach((b) => {
    gameCtx.font = "18px Arial";
    gameCtx.fillText("🦠", b.x, b.y);
  });

  gameCtx.fillStyle = "rgba(255,255,255,0.85)";
  gameCtx.font = "15px Space Grotesk, sans-serif";
  gameCtx.fillText("Click to drop food. Bacteria diffuse first, then chemotactically bias toward it.", 16, 26);
}

function loopGame() {
  updateGame();
  drawGame();
  requestAnimationFrame(loopGame);
}

createBacteria();
loopGame();
