/* Plate I: the papers as a landscape (home page).
 *
 * Every paper listed in _data/plate.yml is a hill on a round island. Newer papers are
 * taller: a hill's height halves for every HALF_LIFE months it lies behind the newest
 * paper, and is then scaled by the paper's `weight`. A few red dots climb the hills
 * while pushing each other apart, as in the RePPO paper.
 *
 * The island turns slowly. Drag to turn it, hover a hill (or its entry in the list of
 * publications) to light it, click a hill to jump to its entry, click empty ground to
 * release new climbers.
 *
 * How it is drawn: the island is cut into ridgelines that run left to right and are
 * drawn from the back to the front. Each one first paints the strip below itself in
 * the page colour, which hides whatever lies behind it, and is then stroked.
 */
(() => {
  const plate = document.getElementById("plate");
  const dataElement = document.getElementById("papers-data");
  if (!plate || !dataElement) return;
  const canvas = plate.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  /* ---------- Settings ---------- */

  const HALF_LIFE = 24; // months for a hill to lose half its height
  const MIN_HEIGHT = 0.11; // the lowest a hill can be, relative to the tallest
  const TURN_SPEED = 0.03; // radians per second
  const REST_AFTER = 60; // seconds without interaction before the island stops turning
  const CLIMBERS = 13;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Papers ---------- */

  const papers = JSON.parse(dataElement.textContent);
  const monthsOf = (date) => {
    const [year, month = 6] = String(date).split("-").map(Number);
    return year * 12 + month;
  };
  const newest = Math.max(...papers.map((paper) => monthsOf(paper.date) || 0));
  const oldestFirst = [...papers].sort((a, b) => monthsOf(a.date) - monthsOf(b.date));

  papers.forEach((paper) => {
    const age = newest - monthsOf(paper.date) || 0; // a paper without a date counts as new
    const recency = 0.5 ** (age / HALF_LIFE);
    paper.height = MIN_HEIGHT + (1 - MIN_HEIGHT) * Math.min(1, recency * (paper.weight ?? 1));
    paper.spread = 0.13 + 0.07 * paper.height; // taller hills are a little wider
    paper.riseDelay = 0.4 + 0.25 * oldestFirst.indexOf(paper); // hills rise oldest first
    paper.rise = 0; // 0 to 1 while the hill grows at the start
    paper.glow = 0; // 0 to 1 while the hill is lit
    paper.entry = document.getElementById("p-" + paper.key); // its entry in the publication list

    // The label is a real link, so the plate also works from the keyboard.
    paper.flag = document.createElement("a");
    paper.flag.className = "flag";
    paper.flag.href = "#p-" + paper.key;
    paper.flag.innerHTML = '<span class="l"></span>';
    paper.flag.firstChild.append(
      paper.short,
      Object.assign(document.createElement("span"), { className: "c", textContent: String(paper.date).slice(0, 4) })
    );
    plate.append(paper.flag);
  });

  /* ---------- The view ---------- */

  let width, height, pixelRatio; // size of the canvas
  let scale, centerX, centerY; // radius and centre of the island on screen
  let tilt, relief, lineCount, stepsAcross; // how flat it is seen, how tall the hills are, level of detail
  let angle = -0.35; // how far the island has turned
  let cosAngle, sinAngle;
  let colors = {};

  function resize() {
    width = plate.clientWidth;
    height = plate.clientHeight;
    if (!width) return;
    const narrow = width < 600;
    tilt = narrow ? 0.44 : 0.36;
    relief = narrow ? 0.58 : 0.4;
    lineCount = narrow ? 58 : 86;
    stepsAcross = narrow ? 140 : 240;
    // The island must fit the box in both directions, with room above for the tallest hill and its label.
    scale = Math.min(width * (narrow ? 0.455 : 0.43), (height - 62) / (1.6 * tilt + relief));
    centerX = width / 2;
    centerY = height - 22 - scale * tilt;
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    papers.forEach((paper) => {
      paper.flagWidth = paper.flag.firstChild.offsetWidth;
      paper.flagHeight = paper.flag.firstChild.offsetHeight;
    });
    draw();
  }

  // The canvas cannot use CSS variables, so the three colours are read from the page (tokens are #RRGGBB).
  function readColors() {
    const style = getComputedStyle(document.documentElement);
    for (const name of ["ink", "paper", "rubric"]) {
      const hex = style.getPropertyValue("--" + name).trim();
      colors[name] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(",");
    }
  }
  const ink = (opacity) => `rgba(${colors.ink},${opacity})`;
  const rubric = (opacity) => `rgba(${colors.rubric},${opacity})`;

  /* ---------- The ground ---------- */

  const smoothstep = (t) => {
    t = Math.min(1, Math.max(0, t));
    return t * t * (3 - 2 * t);
  };
  const easeOut = (t) => 1 - (1 - Math.min(1, Math.max(0, t))) ** 3;

  // Height of the ground at (x, y). The island is the disc of radius 1 around (0, 0).
  function ground(x, y) {
    const fromCenter = Math.hypot(x, y);
    if (fromCenter >= 1) return 0;
    // gently rolling ground
    let z =
      0.026 +
      0.02 * Math.sin(2.3 * x + 1.1) * Math.cos(1.9 * y - 0.6) +
      0.014 * Math.sin(3.7 * x - 2.4 * y + 0.5) +
      0.01 * Math.cos(5.1 * y + 2.2 * x);
    // one bell-shaped hill per paper
    for (const paper of papers) {
      z += paper.height * relief * paper.rise * Math.exp(-((x - paper.x) ** 2 + (y - paper.y) ** 2) / (2 * paper.spread ** 2));
    }
    return Math.max(0, z) * smoothstep((1 - fromCenter) / 0.3); // slope down to nothing at the shore
  }

  // Where a point of the island lands on screen. `depth` runs from -1 (far side) to 1 (near side).
  function toScreen(x, y) {
    const across = x * cosAngle - y * sinAngle;
    const depth = x * sinAngle + y * cosAngle;
    return { x: centerX + across * scale, y: centerY + depth * scale * tilt - ground(x, y) * scale, across, depth };
  }

  /* ---------- The climbers ---------- */

  let climbers = [];

  function releaseClimbers() {
    climbers = Array.from({ length: CLIMBERS }, () => {
      const direction = Math.random() * 2 * Math.PI;
      const distance = Math.sqrt(Math.random()) * 0.8;
      return { x: distance * Math.cos(direction), y: distance * Math.sin(direction) };
    });
  }

  // Each climber walks uphill and is pushed away from the others, most strongly by those standing high.
  function moveClimbers(seconds) {
    const heights = climbers.map((climber) => ground(climber.x, climber.y));
    const moves = climbers.map((climber, i) => {
      const d = 0.01;
      const slopeX = (ground(climber.x + d, climber.y) - ground(climber.x - d, climber.y)) / (2 * d);
      const slopeY = (ground(climber.x, climber.y + d) - ground(climber.x, climber.y - d)) / (2 * d);
      const steepness = Math.hypot(slopeX, slopeY);
      let moveX = (0.5 * slopeX) / (steepness + 0.25);
      let moveY = (0.5 * slopeY) / (steepness + 0.25);

      climbers.forEach((other, j) => {
        if (other === climber) return;
        const dx = climber.x - other.x;
        const dy = climber.y - other.y;
        const distance = Math.hypot(dx, dy) + 1e-6;
        const push = 0.55 * (0.25 + heights[j] / relief) * Math.exp(-(distance ** 2) / (2 * 0.22 ** 2));
        moveX += (dx / distance) * push;
        moveY += (dy / distance) * push;
      });

      const fromCenter = Math.hypot(climber.x, climber.y);
      if (fromCenter > 0.8) {
        // keep away from the shore
        moveX -= (climber.x / fromCenter) * (fromCenter - 0.8) * 6;
        moveY -= (climber.y / fromCenter) * (fromCenter - 0.8) * 6;
      }
      return [moveX * seconds, moveY * seconds];
    });
    climbers.forEach((climber, i) => {
      climber.x += moves[i][0];
      climber.y += moves[i][1];
    });
  }

  /* ---------- Drawing ---------- */

  // Screen and island coordinates of the points along the ridgeline being drawn.
  const screenX = [];
  const screenY = [];
  const islandX = [];
  const islandY = [];

  function traceRidge(from, to) {
    ctx.beginPath();
    ctx.moveTo(screenX[from], screenY[from]);
    for (let k = from + 1; k <= to; k++) ctx.lineTo(screenX[k], screenY[k]);
  }

  function drawShore(opacity, endAngle) {
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, scale, scale * tilt, 0, 0, endAngle);
    ctx.strokeStyle = ink(opacity);
    ctx.stroke();
  }

  // Stroke the part of the current ridgeline that lies near a lit hill again, in the accent colour, fading with distance.
  function drawGlow(paper, last, nearness) {
    const distanceTo = (k) => Math.hypot(islandX[k] - paper.x, islandY[k] - paper.y);
    const reach = 2.1 * paper.spread;
    let from = -1;
    let to = -1;
    for (let k = 0; k <= last; k++) {
      if (distanceTo(k) < reach) {
        if (from < 0) from = k;
        to = k;
      }
    }
    if (to - from < 2) return;
    const fade = ctx.createLinearGradient(screenX[from], 0, screenX[to], 0);
    for (let stop = 0; stop <= 8; stop++) {
      const k = Math.round(from + ((to - from) * stop) / 8);
      const strength = Math.exp(-(distanceTo(k) ** 2) / (1.8 * paper.spread ** 2));
      fade.addColorStop(stop / 8, rubric((paper.glow * (0.5 + 0.5 * nearness) * strength).toFixed(3)));
    }
    traceRidge(from, to);
    ctx.strokeStyle = fade;
    ctx.lineWidth = 1.1;
    ctx.stroke();
    ctx.lineWidth = 0.8;
  }

  function draw() {
    if (!width) return;
    cosAngle = Math.cos(angle);
    sinAngle = Math.sin(angle);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.lineJoin = "round";
    ctx.lineWidth = 0.8;

    drawShore(0.34, 2 * Math.PI); // the whole shoreline, faintly; hills in front of it will cover parts

    const climberOpacity = Math.min(1, Math.max(0, (time - 2.4) / 0.6)); // climbers fade in once the hills are up
    for (let line = 0; line < lineCount; line++) {
      const depth = -1 + (2 * (line + 0.5)) / lineCount;
      const nearness = line / (lineCount - 1);
      const halfWidth = Math.sqrt(1 - depth * depth); // of the island at this depth
      const last = Math.max(6, Math.round(halfWidth * stepsAcross));
      for (let k = 0; k <= last; k++) {
        const across = -halfWidth + (2 * halfWidth * k) / last;
        islandX[k] = across * cosAngle + depth * sinAngle;
        islandY[k] = -across * sinAngle + depth * cosAngle;
        screenX[k] = centerX + across * scale;
        screenY[k] = centerY + depth * scale * tilt - ground(islandX[k], islandY[k]) * scale;
      }

      // hide what is behind this ridge, then draw it (nearer lines are darker)
      const seaLevel = centerY + depth * scale * tilt + 1;
      traceRidge(0, last);
      ctx.lineTo(screenX[last], seaLevel);
      ctx.lineTo(screenX[0], seaLevel);
      ctx.fillStyle = `rgb(${colors.paper})`;
      ctx.fill();
      traceRidge(0, last);
      ctx.strokeStyle = ink(0.24 + 0.6 * nearness);
      ctx.stroke();

      for (const paper of papers) if (paper.glow > 0.01) drawGlow(paper, last, nearness);

      // climbers standing between this ridge and the next, so that nearer ridges can hide them
      for (const climber of climbers) {
        const at = toScreen(climber.x, climber.y);
        if (Math.floor(((at.depth + 1) / 2) * lineCount) !== line) continue;
        ctx.beginPath();
        ctx.arc(at.x, at.y - 2.2, 2.7, 0, 2 * Math.PI);
        ctx.fillStyle = rubric(climberOpacity);
        ctx.fill();
      }
    }

    drawShore(0.85, Math.PI); // the near half of the shoreline, on top
    placeFlags();
  }

  // Put each label above its summit. A label steps up when it would cover the label of a nearer hill,
  // and fades when its summit is hidden behind nearer ground.
  function placeFlags() {
    const placed = []; // boxes of the labels placed so far: [left, right, top, bottom]
    const nearestFirst = [...papers].sort((a, b) => toScreen(b.x, b.y).depth - toScreen(a.x, a.y).depth);
    for (const paper of nearestFirst) {
      const summit = toScreen(paper.x, paper.y);

      // walk from the summit towards the viewer: is any ground there higher on screen than the summit?
      let hidden = false;
      for (let depth = summit.depth + 0.05; depth < 1 && !hidden; depth += 0.05) {
        if (summit.across ** 2 + depth ** 2 >= 1) break;
        const x = summit.across * cosAngle + depth * sinAngle;
        const y = -summit.across * sinAngle + depth * cosAngle;
        hidden = toScreen(x, y).y < summit.y - 3;
      }

      const left = summit.x - paper.flagWidth / 2 - 3;
      const right = summit.x + paper.flagWidth / 2 + 3;
      let bottom = summit.y - 17;
      for (const [otherLeft, otherRight, otherTop, otherBottom] of placed) {
        const overlaps = right > otherLeft && left < otherRight && bottom > otherTop - 2 && bottom - paper.flagHeight < otherBottom + 2;
        if (overlaps) bottom = otherTop - 3;
      }
      placed.push([left, right, bottom - paper.flagHeight, bottom]);

      const style = paper.flag.style;
      style.transform = `translate(${summit.x.toFixed(1)}px, ${summit.y.toFixed(1)}px) translate(-50%, -100%)`;
      style.setProperty("--lift", `${Math.max(0, summit.y - 17 - bottom).toFixed(1)}px`);
      style.opacity = Math.min(1, paper.rise * 1.4) * (hidden && paper !== lit ? 0.34 : 1);
      style.zIndex = 10 + Math.round(summit.depth * 10);
    }
  }

  /* ---------- Animation ---------- */

  let time = 0; // seconds since the plate started
  let lastFrame = 0;
  let lastTouched = 0;
  let running = false;
  let onScreen = true;
  let lit = null; // the paper whose hill is lit
  let drag = null;

  function frame(now) {
    const seconds = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;
    time += seconds;
    for (const paper of papers) {
      paper.rise = easeOut((time - paper.riseDelay) / 1.5);
      paper.glow += ((paper === lit ? 1 : 0) - paper.glow) * Math.min(1, seconds * 9);
    }
    moveClimbers(seconds * 1.15);
    if (!drag && !lit) angle += TURN_SPEED * seconds;
    draw();

    const resting = now - lastTouched > REST_AFTER * 1000;
    running = onScreen && !document.hidden && !resting;
    if (running) requestAnimationFrame(frame);
  }

  // Called whenever something happens: (re)starts the animation, or just redraws when motion is reduced.
  function wake() {
    lastTouched = performance.now();
    if (reducedMotion) {
      papers.forEach((paper) => (paper.glow = paper === lit ? 1 : 0));
      draw();
    } else if (!running) {
      running = true;
      lastFrame = performance.now();
      requestAnimationFrame(frame);
    }
  }

  /* ---------- Interaction ---------- */

  function light(paper) {
    if (paper === lit) return;
    lit = paper;
    for (const each of papers) {
      each.flag.classList.toggle("lit", each === paper);
      if (each.entry) each.entry.classList.toggle("is-lit", each === paper);
    }
    plate.classList.toggle("aim", Boolean(paper));
    wake();
  }

  // The paper whose summit is close to the pointer, if any.
  function paperAt(event) {
    const box = canvas.getBoundingClientRect();
    return papers.find((paper) => {
      const summit = toScreen(paper.x, paper.y);
      return Math.hypot(event.clientX - box.left - summit.x, event.clientY - box.top - summit.y - 6) < 40;
    });
  }

  for (const paper of papers) {
    for (const element of [paper.flag, paper.entry]) {
      if (!element) continue;
      element.addEventListener("mouseenter", () => light(paper));
      element.addEventListener("mouseleave", () => light(null));
      element.addEventListener("focusin", () => light(paper));
      element.addEventListener("focusout", () => light(null));
    }
  }

  canvas.addEventListener("pointerdown", (event) => {
    drag = { startX: event.clientX, startY: event.clientY, startAngle: angle, moved: false };
    canvas.setPointerCapture(event.pointerId);
    wake();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (drag) {
      const dx = event.clientX - drag.startX;
      if (Math.hypot(dx, event.clientY - drag.startY) > 5) drag.moved = true;
      if (drag.moved) angle = drag.startAngle + dx * 0.006;
      plate.classList.toggle("dragging", drag.moved);
      wake();
    } else if (event.pointerType === "mouse") {
      light(paperAt(event) || null);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (drag && !drag.moved) {
      // a click: on a hill it follows the hill's link, elsewhere it releases new climbers
      const paper = paperAt(event);
      if (paper) paper.flag.click();
      else releaseClimbers();
    }
    drag = null;
    plate.classList.remove("dragging");
    wake();
  });
  canvas.addEventListener("pointercancel", () => {
    // the browser took over the gesture (a vertical swipe scrolls the page)
    if (drag) angle = drag.startAngle;
    drag = null;
    plate.classList.remove("dragging");
  });
  canvas.addEventListener("pointerleave", () => light(null));

  /* ---------- Start ---------- */

  readColors();
  releaseClimbers();
  if (reducedMotion) {
    // no animation: show the finished landscape with the climbers already settled
    time = 60;
    papers.forEach((paper) => (paper.rise = 1));
    resize();
    for (let i = 0; i < 600; i++) moveClimbers(0.016);
  }
  resize();
  wake();

  new ResizeObserver(resize).observe(plate);
  new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (onScreen) wake();
  }).observe(plate);
  document.addEventListener("visibilitychange", wake);
  document.fonts.ready.then(resize); // label sizes change once the font has loaded

  // redraw in the new colours when the theme changes
  const recolor = () => {
    readColors();
    wake();
  };
  new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
})();
