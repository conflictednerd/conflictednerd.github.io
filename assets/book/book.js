/* Small behaviours used on every page. Each block below is independent of the others:
 *   1. the light/dark switch
 *   2. the running head (current section and reading progress)
 *   3. the notes list (summary of the hovered note in the margin)
 *   4. the busy beaver machine beside the guest lecture
 */

/* ---------- 1. Light / dark ---------- */
// The page follows the system setting until the button is pressed; the choice is then remembered.
// (A saved choice is applied in the <head>, before the page paints: see _layouts/book.liquid.)
(() => {
  const root = document.documentElement;
  const button = document.getElementById("theme");
  const system = matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : system.matches);

  function refresh() {
    root.classList.toggle("is-dark", isDark());
    button.setAttribute("aria-label", isDark() ? "Switch to the light theme" : "Switch to the dark theme");
  }

  button.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    root.classList.add("theming"); // lets the colours cross-fade instead of snapping
    setTimeout(() => root.classList.remove("theming"), 600);
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // storage can be unavailable (private windows); the switch still works for this visit
    }
    refresh();
  });
  system.addEventListener("change", refresh);
  refresh();
})();

/* ---------- 2. Running head ---------- */
// Elements with a data-title attribute are the "sections" of a page. The head shows the one being
// read (with "§ n" if it has data-no), and a thin line across it shows how far down the page we are.
(() => {
  const head = document.getElementById("head");
  const where = document.getElementById("where");
  const sections = [...document.querySelectorAll("[data-title]")];

  function update() {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    head.style.setProperty("--p", scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0);

    // the current section is the last one whose top has passed under the head
    let current = null;
    for (const section of sections) {
      if (section.getBoundingClientRect().top < 110) current = section;
    }
    // at the very bottom, the last section counts even if it is too short to reach the head
    if (sections.length && scrollable > 0 && window.scrollY >= scrollable - 2) current = sections[sections.length - 1];

    where.classList.toggle("on", Boolean(current));
    if (current) {
      where.firstElementChild.textContent = current.dataset.no ? `§ ${current.dataset.no}` : "";
      where.lastElementChild.textContent = current.dataset.title;
    }
  }

  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
})();

/* ---------- 3. Notes list ---------- */
// On wide screens each note's summary is hidden; pointing at a note shows its summary in the margin,
// level with that row. Leaving the list brings back the margin's usual text.
(() => {
  const list = document.getElementById("toc");
  const note = document.getElementById("toc-note");
  if (!list || !note) return;
  const usualText = note.textContent;

  function show(link) {
    if (getComputedStyle(note).float === "none") return; // narrow screens: summaries are shown in the list itself
    const summary = link && link.querySelector(".d");
    note.textContent = summary ? summary.textContent : usualText;
    note.style.transform = summary ? `translateY(${link.offsetTop - list.offsetTop}px)` : "";
  }

  for (const link of list.querySelectorAll("a")) {
    link.addEventListener("mouseenter", () => show(link));
    link.addEventListener("focus", () => show(link));
    link.addEventListener("blur", () => show(null));
  }
  list.addEventListener("mouseleave", () => show(null));
})();

/* ---------- 4. Busy beaver ---------- */
// The 4-state busy beaver: started on a blank tape it runs for 107 steps and halts with 13 ones.
// It is shown as a row of tape cells and its rule table, with the rule in use highlighted.
(() => {
  const tape = document.getElementById("bb-tape");
  const table = document.getElementById("bb-rules");
  const status = document.getElementById("bb-status");
  if (!tape || !table || !status) return;

  // RULES[state][symbol under the head] = [symbol to write, where the head moves, next state]
  const RULES = {
    A: [
      [1, +1, "B"],
      [1, -1, "B"],
    ],
    B: [
      [1, -1, "A"],
      [0, -1, "C"],
    ],
    C: [
      [1, +1, "H"],
      [1, -1, "D"],
    ],
    D: [
      [1, +1, "D"],
      [0, +1, "A"],
    ],
  };
  const STATES = Object.keys(RULES);
  const CELLS = 14; // started on cell START, the machine never leaves these 14 cells
  const START = 10;

  // build the tape and the rule table
  const cells = Array.from({ length: CELLS }, () => tape.appendChild(document.createElement("i")));
  const addToTable = (text, className = "") => {
    const cell = table.appendChild(document.createElement("span"));
    cell.textContent = text;
    cell.className = className;
    return cell;
  };
  const ruleCells = {};
  addToTable("", "k");
  STATES.forEach((state) => addToTable(state, "st"));
  for (const symbol of [0, 1]) {
    addToTable(symbol, "k");
    for (const state of STATES) {
      const [write, move, next] = RULES[state][symbol];
      ruleCells[state + symbol] = addToTable(`${write}${move > 0 ? "R" : "L"}${next}`);
    }
  }

  let bits, head, state, steps;

  function reset() {
    bits = Array(CELLS).fill(0);
    head = START;
    state = "A";
    steps = 0;
  }

  function step() {
    const [write, move, next] = RULES[state][bits[head]];
    bits[head] = write;
    head += move;
    state = next;
    steps += 1;
  }

  function show() {
    const halted = state === "H";
    cells.forEach((cell, i) => {
      cell.className = (bits[i] ? "on" : "") + (i === head ? " at" : "");
    });
    for (const [key, cell] of Object.entries(ruleCells)) {
      cell.classList.toggle("on", !halted && key === state + bits[head]);
    }
    const ones = bits.filter(Boolean).length;
    status.textContent = halted ? `Halted after ${steps} steps, leaving ${ones} ones.` : `State ${state}, step ${steps} of 107.`;
  }

  reset();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    while (state !== "H") step(); // no animation: show the halted machine
    show();
    return;
  }
  show();

  // One step every 230 ms. After halting it rests for about six seconds, then starts over.
  let rest = 0;
  setInterval(() => {
    if (document.hidden) return;
    if (state !== "H") step();
    else if (++rest > 26) {
      rest = 0;
      reset();
    }
    show();
  }, 230);

  tape.addEventListener("click", () => {
    rest = 0;
    reset();
    show();
  });
})();
