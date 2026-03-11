// Doom Assumptions - Interactive Visual Tree

(function () {
  "use strict";

  // --- State ---
  let currentTreeIndex = 0;
  let currentWorldview = null;
  let probabilities = {};      // node id -> probability (leaves + pinned nodes)
  let pinnedNodes = {};         // node id -> true if user has pinned this node's probability
  let selectedNodeId = null;
  let collapsedNodes = {};
  let variableValues = {};
  let rangeMode = false;
  let probRanges = {};          // node id -> {lo, hi} for leaves in range mode

  // --- DOM refs ---
  const treeSelect = document.getElementById("tree-select");
  const worldviewSelect = document.getElementById("worldview-select");
  const resetBtn = document.getElementById("reset-btn");
  const treeRoot = document.getElementById("tree-root");
  const treeSvg = document.getElementById("tree-svg");
  const treeGraph = document.getElementById("tree-graph");
  const infoPanel = document.getElementById("info-panel");
  const variablesContainer = document.getElementById("variables-container");
  const rangeToggle = document.getElementById("range-toggle");
  const saveWorldviewBtn = document.getElementById("save-worldview-btn");
  const shareBtn = document.getElementById("share-btn");

  // --- Helpers ---

  function getTree() {
    return TREES[currentTreeIndex];
  }

  function findNode(node, id) {
    if (node.id === id) return node;
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        var found = findNode(node.children[i], id);
        if (found) return found;
      }
    }
    return null;
  }

  function getLeaves(node, leaves) {
    leaves = leaves || [];
    if (node.type === "leaf") {
      leaves.push(node);
    } else if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        getLeaves(node.children[i], leaves);
      }
    }
    return leaves;
  }

  // Find all complement pairs: returns array of {source, complement}
  function getComplementPairs(node, pairs) {
    pairs = pairs || [];
    if (node.complement_of) {
      pairs.push({ sourceId: node.complement_of, complementId: node.id });
    }
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        getComplementPairs(node.children[i], pairs);
      }
    }
    return pairs;
  }

  function computeProb(node) {
    // If this node is pinned by the user, return the pinned value
    if (pinnedNodes[node.id] && probabilities[node.id] != null) {
      return probabilities[node.id];
    }

    if (node.type === "leaf") {
      if (node.complement_of) {
        var src = findNode(getTree().tree, node.complement_of);
        return 1 - computeProb(src);
      }
      return probabilities[node.id] != null ? probabilities[node.id] : 0.5;
    }
    if (!node.children || node.children.length === 0) return 0;

    if (node.type === "and") {
      var p = 1;
      for (var i = 0; i < node.children.length; i++) p *= computeProb(node.children[i]);
      return p;
    }
    if (node.type === "or") {
      // Sum for mutually exclusive branches (law of total probability).
      // If non-exclusive OR nodes are added, this should use 1-(1-a)(1-b) instead.
      var p = 0;
      for (var i = 0; i < node.children.length; i++) p += computeProb(node.children[i]);
      return Math.min(p, 1);
    }
    return 0;
  }

  function computeProbRange(node) {
    if (pinnedNodes[node.id] && probabilities[node.id] != null) {
      var p = probabilities[node.id];
      return { lo: p, hi: p };
    }

    if (node.type === "leaf") {
      if (node.complement_of) {
        var src = findNode(getTree().tree, node.complement_of);
        var srcR = computeProbRange(src);
        return { lo: 1 - srcR.hi, hi: 1 - srcR.lo };
      }
      if (rangeMode && probRanges[node.id]) {
        return probRanges[node.id];
      }
      var p = probabilities[node.id] != null ? probabilities[node.id] : 0.5;
      return { lo: p, hi: p };
    }
    if (!node.children || node.children.length === 0) return { lo: 0, hi: 0 };

    if (node.type === "and") {
      var lo = 1, hi = 1;
      for (var i = 0; i < node.children.length; i++) {
        var r = computeProbRange(node.children[i]);
        lo *= r.lo;
        hi *= r.hi;
      }
      return { lo: lo, hi: hi };
    }
    if (node.type === "or") {
      var lo = 0, hi = 0;
      for (var i = 0; i < node.children.length; i++) {
        var r = computeProbRange(node.children[i]);
        lo += r.lo;
        hi += r.hi;
      }
      return { lo: Math.min(lo, 1), hi: Math.min(hi, 1) };
    }
    return { lo: 0, hi: 0 };
  }

  function formatRange(range) {
    return formatProb(range.lo) + " – " + formatProb(range.hi);
  }

  function formatProb(p) {
    if (p < 0.001 && p > 0) return "<0.1%";
    if (p < 0.01 && p > 0) return (p * 100).toFixed(2) + "%";
    if (p > 0.99 && p < 1) return ">99%";
    return (p * 100).toFixed(1) + "%";
  }

  function probColor(p) {
    if (p > 0.5) return "var(--accent-safety)";
    if (p > 0.2) return "var(--accent-universe)";
    return "var(--accent-life)";
  }

  // --- Variable substitution ---

  function subVars(text, capitalize) {
    if (!text) return text;
    var tree = getTree();
    if (!tree.variables) return text;
    var result = text;
    Object.keys(tree.variables).forEach(function (key) {
      var val = variableValues[key] || tree.variables[key].default || key;
      var regex = new RegExp("\\b" + escapeRegex(key) + "\\b", "g");
      result = result.replace(regex, val);
    });
    // Capitalise first letter (#2 fix)
    if (result.length > 0) {
      result = result.charAt(0).toUpperCase() + result.slice(1);
    }
    return result;
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function initVariables() {
    variableValues = {};
    var tree = getTree();
    if (!tree.variables) return;
    Object.keys(tree.variables).forEach(function (key) {
      variableValues[key] = tree.variables[key].default || "";
    });
  }

  // Build variable editor UI — show label, not raw key (#7)
  function renderVariables() {
    variablesContainer.innerHTML = "";
    var tree = getTree();
    if (!tree.variables) return;

    Object.keys(tree.variables).forEach(function (key) {
      var varDef = tree.variables[key];

      var group = document.createElement("div");
      group.className = "control-group var-group";

      var label = document.createElement("label");
      label.className = "control-label";
      label.textContent = key;
      label.setAttribute("for", "var-" + key);
      label.title = varDef.label || key;
      group.appendChild(label);

      var input = document.createElement("input");
      input.type = "text";
      input.id = "var-" + key;
      input.className = "control-input";
      input.value = variableValues[key] || varDef.default || "";
      input.placeholder = varDef.label || key;

      input.addEventListener("change", function () {
        variableValues[key] = input.value;
        renderTree();
        updateInfoPanel();
      });

      group.appendChild(input);
      variablesContainer.appendChild(group);
    });
  }

  // --- Save / Load / Share ---

  var STORAGE_KEY = "doom-assumptions-worldviews";

  function getSavedWorldviews() {
    try {
      var data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : {};
    } catch (e) { return {}; }
  }

  function saveWorldview(name) {
    var treeId = getTree().id;
    var all = getSavedWorldviews();
    if (!all[treeId]) all[treeId] = {};
    var entry = { probabilities: {} };
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && probabilities[leaf.id] != null) {
        entry.probabilities[leaf.id] = probabilities[leaf.id];
      }
    });
    if (rangeMode) {
      entry.ranges = {};
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of && probRanges[leaf.id]) {
          entry.ranges[leaf.id] = probRanges[leaf.id];
        }
      });
    }
    all[treeId][name] = entry;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }

  function deleteSavedWorldview(name) {
    var treeId = getTree().id;
    var all = getSavedWorldviews();
    if (all[treeId]) {
      delete all[treeId][name];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    }
  }

  function applySavedWorldview(name) {
    var treeId = getTree().id;
    var all = getSavedWorldviews();
    var entry = all[treeId] && all[treeId][name];
    if (!entry) return;
    pinnedNodes = {};
    Object.keys(entry.probabilities).forEach(function (id) {
      probabilities[id] = entry.probabilities[id];
    });
    if (entry.ranges) {
      probRanges = {};
      Object.keys(entry.ranges).forEach(function (id) {
        probRanges[id] = entry.ranges[id];
      });
    }
  }

  function encodeStateToHash() {
    var state = {
      t: currentTreeIndex,
      p: {}
    };
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && probabilities[leaf.id] != null) {
        state.p[leaf.id] = Math.round(probabilities[leaf.id] * 1000) / 1000;
      }
    });
    if (rangeMode) {
      state.r = {};
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of && probRanges[leaf.id]) {
          state.r[leaf.id] = [
            Math.round(probRanges[leaf.id].lo * 1000) / 1000,
            Math.round(probRanges[leaf.id].hi * 1000) / 1000
          ];
        }
      });
    }
    return btoa(JSON.stringify(state));
  }

  function loadStateFromHash() {
    var hash = window.location.hash.slice(1);
    if (!hash) return false;
    try {
      var state = JSON.parse(atob(hash));
      if (state.t != null && TREES[state.t]) {
        currentTreeIndex = state.t;
        treeSelect.value = state.t;
      }
      initProbabilities();
      initVariables();
      populateWorldviewSelect();
      renderVariables();
      if (state.p) {
        Object.keys(state.p).forEach(function (id) {
          probabilities[id] = state.p[id];
        });
      }
      if (state.r) {
        rangeMode = true;
        rangeToggle.textContent = "On";
        rangeToggle.classList.add("active");
        Object.keys(state.r).forEach(function (id) {
          probRanges[id] = { lo: state.r[id][0], hi: state.r[id][1] };
        });
      }
      worldviewSelect.value = "custom";
      currentWorldview = null;
      return true;
    } catch (e) { return false; }
  }

  // --- Populate selectors ---

  function populateTreeSelect() {
    treeSelect.innerHTML = "";
    TREES.forEach(function (tree, i) {
      var opt = document.createElement("option");
      opt.value = i;
      opt.textContent = tree.title;
      treeSelect.appendChild(opt);
    });
  }

  function populateWorldviewSelect() {
    var tree = getTree();
    worldviewSelect.innerHTML = "";
    Object.keys(tree.worldviews).forEach(function (key) {
      var opt = document.createElement("option");
      opt.value = key;
      opt.textContent = tree.worldviews[key].name;
      worldviewSelect.appendChild(opt);
    });
    // Saved worldviews from localStorage
    var saved = getSavedWorldviews();
    var treeId = tree.id;
    if (saved[treeId]) {
      var names = Object.keys(saved[treeId]);
      if (names.length > 0) {
        var sep = document.createElement("option");
        sep.disabled = true;
        sep.textContent = "── Saved ──";
        worldviewSelect.appendChild(sep);
        names.forEach(function (name) {
          var opt = document.createElement("option");
          opt.value = "saved:" + name;
          opt.textContent = name;
          worldviewSelect.appendChild(opt);
        });
      }
    }
    var custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "Custom";
    worldviewSelect.appendChild(custom);
  }

  function applyWorldview(key) {
    var tree = getTree();
    if (key === "custom") {
      currentWorldview = null;
      return;
    }
    // Handle saved worldviews
    if (key.indexOf("saved:") === 0) {
      var name = key.slice(6);
      currentWorldview = key;
      applySavedWorldview(name);
      return;
    }
    var wv = tree.worldviews[key];
    if (!wv) return;
    currentWorldview = key;
    // Clear pins when switching worldview
    pinnedNodes = {};
    var leaves = getLeaves(tree.tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && wv.probabilities[leaf.id] != null) {
        probabilities[leaf.id] = wv.probabilities[leaf.id];
      }
    });
  }

  // --- Render tree as visual graph ---

  function renderTree() {
    treeRoot.innerHTML = "";

    // Apply tree-level CSS class (e.g. "formal" for wider cards)
    treeGraph.className = "tg-tree";
    if (getTree().cssClass) {
      treeGraph.classList.add(getTree().cssClass);
    }

    var el = buildNodeEl(getTree().tree);
    treeRoot.appendChild(el);

    // Draw connectors after layout settles
    requestAnimationFrame(function () {
      requestAnimationFrame(drawConnectors);
    });

    renderSensitivity();
  }

  function buildNodeEl(node) {
    var hasChildren = node.children && node.children.length > 0;
    var isCollapsed = !!collapsedNodes[node.id];
    var isPinned = !!pinnedNodes[node.id];
    var prob = computeProb(node);

    // Wrapper
    var wrapper = document.createElement("div");
    wrapper.className = "tg-node";
    wrapper.dataset.id = node.id;

    // Card
    var card = document.createElement("div");
    card.className = "tg-card";
    if (node.id === selectedNodeId) card.classList.add("selected");
    if (isPinned) card.classList.add("pinned");

    // Card header: type + prob + collapse toggle
    var header = document.createElement("div");
    header.className = "tg-card-header";

    var typeBadge = document.createElement("span");
    typeBadge.className = "tg-type " + node.type;
    typeBadge.textContent = node.type;
    header.appendChild(typeBadge);

    var probEl = document.createElement("span");
    probEl.className = "tg-prob";
    if (rangeMode) {
      var range = computeProbRange(node);
      probEl.textContent = (range.lo === range.hi) ? formatProb(prob) : formatRange(range);
    } else {
      probEl.textContent = formatProb(prob);
    }
    header.appendChild(probEl);

    var collapseBtn = document.createElement("span");
    collapseBtn.className = "tg-collapse";
    if (!hasChildren) collapseBtn.classList.add("hidden-toggle");
    collapseBtn.textContent = isCollapsed ? "+" : "\u2212";
    collapseBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleCollapse(node.id);
    });
    header.appendChild(collapseBtn);

    card.appendChild(header);

    // Name (short display label)
    // Formal trees keep symbolic names on cards; casual trees substitute variables
    var rawName = node.name || node.label || node.id;
    var label = document.createElement("span");
    label.className = "tg-label";
    label.textContent = getTree().substituteNames === false ? rawName : subVars(rawName);
    card.appendChild(label);

    // Probability bar
    var bar = document.createElement("div");
    bar.className = "tg-bar";
    bar.style.position = "relative";
    var barFill = document.createElement("div");
    barFill.className = "tg-bar-fill";
    barFill.style.width = (prob * 100) + "%";
    barFill.style.backgroundColor = probColor(prob);
    bar.appendChild(barFill);
    if (rangeMode) {
      var rng = computeProbRange(node);
      if (rng.lo !== rng.hi) {
        var barRange = document.createElement("div");
        barRange.className = "tg-bar-range";
        barRange.style.left = (rng.lo * 100) + "%";
        barRange.style.width = ((rng.hi - rng.lo) * 100) + "%";
        bar.appendChild(barRange);
      }
    }
    card.appendChild(bar);

    // Click to select
    card.addEventListener("click", function () {
      selectNode(node);
    });

    wrapper.appendChild(card);

    // Slider — available on ALL nodes (#1), not just leaves
    var showSlider = (node.type === "leaf") || hasChildren;
    var isComplement = !!node.complement_of;

    if (showSlider && rangeMode && node.type === "leaf" && !isComplement) {
      // Dual range slider for leaves in range mode
      var rng = probRanges[node.id] || { lo: prob, hi: prob };
      var rangeWrap = document.createElement("div");
      rangeWrap.className = "tg-range-wrap";

      var track = document.createElement("div");
      track.className = "tg-range-track";
      var fill = document.createElement("div");
      fill.className = "tg-range-fill";
      fill.style.left = (rng.lo * 100) + "%";
      fill.style.width = ((rng.hi - rng.lo) * 100) + "%";
      track.appendChild(fill);
      rangeWrap.appendChild(track);

      var loInput = document.createElement("input");
      loInput.type = "range";
      loInput.className = "tg-range-lo";
      loInput.min = "0";
      loInput.max = "100";
      loInput.step = "1";
      loInput.value = Math.round(rng.lo * 100);

      var hiInput = document.createElement("input");
      hiInput.type = "range";
      hiInput.className = "tg-range-hi";
      hiInput.min = "0";
      hiInput.max = "100";
      hiInput.step = "1";
      hiInput.value = Math.round(rng.hi * 100);

      function onRangeInput() {
        var lo = parseInt(loInput.value) / 100;
        var hi = parseInt(hiInput.value) / 100;
        if (lo > hi) { var tmp = lo; lo = hi; hi = tmp; }
        probRanges[node.id] = { lo: lo, hi: hi };
        probabilities[node.id] = (lo + hi) / 2;
        currentWorldview = null;
        worldviewSelect.value = "custom";
        fill.style.left = (lo * 100) + "%";
        fill.style.width = ((hi - lo) * 100) + "%";
        rangeVal.textContent = formatProb(lo) + " – " + formatProb(hi);
        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
      }

      loInput.addEventListener("input", onRangeInput);
      hiInput.addEventListener("input", onRangeInput);

      rangeWrap.appendChild(loInput);
      rangeWrap.appendChild(hiInput);

      var rangeVal = document.createElement("span");
      rangeVal.className = "tg-range-val";
      rangeVal.textContent = formatProb(rng.lo) + " – " + formatProb(rng.hi);
      rangeWrap.appendChild(rangeVal);

      wrapper.appendChild(rangeWrap);
    } else if (showSlider) {
      // Single slider (point estimate mode, or branch/complement nodes)
      var sliderWrap = document.createElement("div");
      sliderWrap.className = "tg-slider-wrap";
      if (isComplement) sliderWrap.classList.add("complement");
      // Dim slider on branch nodes when not pinned (children are driving the value)
      if (hasChildren && !isPinned) sliderWrap.classList.add("unpinned");

      var slider = document.createElement("input");
      slider.type = "range";
      slider.className = "tg-slider";
      slider.min = "0";
      slider.max = "100";
      slider.step = "1";
      slider.value = Math.round(prob * 100);
      slider.disabled = isComplement;

      slider.addEventListener("input", function () {
        var newVal = parseInt(slider.value) / 100;
        probabilities[node.id] = newVal;
        currentWorldview = null;
        worldviewSelect.value = "custom";

        if (hasChildren && !pinnedNodes[node.id]) {
          // First move on a branch node — pin it and re-render to show unpin button
          pinnedNodes[node.id] = true;
          renderTree();
          updateInfoPanel();
          return;
        }

        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
      });

      var sliderVal = document.createElement("span");
      sliderVal.className = "tg-slider-val";
      sliderVal.textContent = formatProb(prob);

      // Unpin button for branch nodes
      if (hasChildren && isPinned) {
        var unpinBtn = document.createElement("span");
        unpinBtn.className = "tg-unpin";
        unpinBtn.textContent = "\u00d7";
        unpinBtn.title = "Unpin — use children's values";
        unpinBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          delete pinnedNodes[node.id];
          delete probabilities[node.id];
          renderTree();
          updateInfoPanel();
        });
        sliderWrap.appendChild(unpinBtn);
      }

      sliderWrap.appendChild(slider);
      sliderWrap.appendChild(sliderVal);
      wrapper.appendChild(sliderWrap);
    }

    // Children
    if (hasChildren) {
      var childrenEl = document.createElement("div");
      childrenEl.className = "tg-children";
      if (isCollapsed) childrenEl.classList.add("collapsed");
      // Dim children when parent is pinned
      if (isPinned) childrenEl.classList.add("dimmed");

      node.children.forEach(function (child) {
        childrenEl.appendChild(buildNodeEl(child));
      });

      wrapper.appendChild(childrenEl);
    }

    return wrapper;
  }

  // --- Collapse / Expand ---

  function toggleCollapse(nodeId) {
    collapsedNodes[nodeId] = !collapsedNodes[nodeId];
    renderTree();
  }

  // --- SVG Connectors (#5 fix: use offset-based positioning) ---

  function getOffsetPos(el, ancestor) {
    var x = 0, y = 0;
    var current = el;
    while (current && current !== ancestor) {
      x += current.offsetLeft;
      y += current.offsetTop;
      current = current.offsetParent;
    }
    return { x: x, y: y };
  }

  function drawConnectors() {
    var w = treeGraph.scrollWidth;
    var h = treeGraph.scrollHeight;

    treeSvg.setAttribute("width", w);
    treeSvg.setAttribute("height", h);
    treeSvg.innerHTML = "";

    drawNodeConnectors(getTree().tree);
    drawComplementLinks();
  }

  function drawNodeConnectors(node) {
    if (!node.children || node.children.length === 0) return;
    if (collapsedNodes[node.id]) return;

    var parentCard = treeRoot.querySelector('[data-id="' + node.id + '"] > .tg-card');
    if (!parentCard) return;

    var parentPos = getOffsetPos(parentCard, treeGraph);
    var px = parentPos.x + parentCard.offsetWidth / 2;
    var py = parentPos.y + parentCard.offsetHeight;

    for (var i = 0; i < node.children.length; i++) {
      var child = node.children[i];
      var childCard = treeRoot.querySelector('[data-id="' + child.id + '"] > .tg-card');
      if (!childCard) continue;

      var childPos = getOffsetPos(childCard, treeGraph);
      var cx = childPos.x + childCard.offsetWidth / 2;
      var cy = childPos.y;

      var midY = py + (cy - py) * 0.4;

      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d",
        "M" + px + "," + py +
        " C" + px + "," + midY +
        " " + cx + "," + midY +
        " " + cx + "," + cy
      );
      path.setAttribute("stroke", "rgba(255,255,255,0.1)");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("fill", "none");
      treeSvg.appendChild(path);

      drawNodeConnectors(child);
    }
  }

  // --- Complement visual links (#4) ---

  function drawComplementLinks() {
    var pairs = getComplementPairs(getTree().tree);

    pairs.forEach(function (pair) {
      var sourceCard = treeRoot.querySelector('[data-id="' + pair.sourceId + '"] > .tg-card');
      var compCard = treeRoot.querySelector('[data-id="' + pair.complementId + '"] > .tg-card');
      if (!sourceCard || !compCard) return;

      var sPos = getOffsetPos(sourceCard, treeGraph);
      var cPos = getOffsetPos(compCard, treeGraph);

      var sx = sPos.x + sourceCard.offsetWidth / 2;
      var sy = sPos.y + sourceCard.offsetHeight / 2;
      var cx = cPos.x + compCard.offsetWidth / 2;
      var cy = cPos.y + compCard.offsetHeight / 2;

      // Draw a dashed curved line between complement pairs
      var midX = (sx + cx) / 2;
      var bulge = Math.min(40, Math.abs(cy - sy) * 0.3);

      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d",
        "M" + sx + "," + sy +
        " Q" + midX + "," + (Math.max(sy, cy) + bulge) +
        " " + cx + "," + cy
      );
      path.setAttribute("stroke", "rgba(168,85,247,0.3)");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("stroke-dasharray", "6,4");
      path.setAttribute("fill", "none");
      treeSvg.appendChild(path);

      // Label at midpoint
      var labelX = midX;
      var labelY = Math.max(sy, cy) + bulge * 0.6;
      var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", labelX);
      text.setAttribute("y", labelY + 14);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "rgba(168,85,247,0.5)");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-family", "var(--font-mono)");
      text.textContent = "1 \u2212 P";
      treeSvg.appendChild(text);
    });
  }

  // --- Update probabilities in-place ---

  function updateAllProbabilities() {
    updateNodeProbs(getTree().tree);
    requestAnimationFrame(drawConnectors);
  }

  function updateNodeProbs(node) {
    var prob = computeProb(node);
    var wrapper = treeRoot.querySelector('[data-id="' + node.id + '"]');
    if (!wrapper) return;

    var probEl = wrapper.querySelector(":scope > .tg-card .tg-prob");
    if (probEl) {
      if (rangeMode) {
        var rng = computeProbRange(node);
        probEl.textContent = (rng.lo === rng.hi) ? formatProb(prob) : formatRange(rng);
      } else {
        probEl.textContent = formatProb(prob);
      }
    }

    var barFill = wrapper.querySelector(":scope > .tg-card .tg-bar-fill");
    if (barFill) {
      barFill.style.width = (prob * 100) + "%";
      barFill.style.backgroundColor = probColor(prob);
    }

    // Update range bar if present
    if (rangeMode) {
      var barRange = wrapper.querySelector(":scope > .tg-card .tg-bar-range");
      if (barRange) {
        var rng = computeProbRange(node);
        barRange.style.left = (rng.lo * 100) + "%";
        barRange.style.width = ((rng.hi - rng.lo) * 100) + "%";
      }
    }

    // Update slider + value
    var sliderWrap = wrapper.querySelector(":scope > .tg-slider-wrap");
    if (sliderWrap) {
      var slider = sliderWrap.querySelector(".tg-slider");
      var sliderVal = sliderWrap.querySelector(".tg-slider-val");
      if (slider) slider.value = Math.round(prob * 100);
      if (sliderVal) sliderVal.textContent = formatProb(prob);
    }

    if (node.children) {
      node.children.forEach(function (child) {
        updateNodeProbs(child);
      });
    }
  }

  // --- Sensitivity Analysis ---

  function computeSensitivity() {
    var tree = getTree();
    var root = tree.tree;
    var leaves = getLeaves(root).filter(function (l) { return !l.complement_of; });
    var bump = 0.10; // 10 percentage points
    var results = [];

    var baseProb = computeProb(root);

    leaves.forEach(function (leaf) {
      var original = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;

      // Bump up by 10pp (or down if at ceiling)
      var bumped = original + bump;
      var direction = "+";
      if (bumped > 1) {
        bumped = original - bump;
        direction = "\u2212";
      }
      bumped = Math.max(0, Math.min(1, bumped));

      probabilities[leaf.id] = bumped;
      var newRoot = computeProb(root);
      probabilities[leaf.id] = original;

      var deltaPP = (newRoot - baseProb) * 100;
      results.push({ id: leaf.id, name: leaf.name, deltaPP: deltaPP, absDelta: Math.abs(deltaPP), direction: direction });
    });

    results.sort(function (a, b) { return b.absDelta - a.absDelta; });
    return results;
  }

  function renderSensitivity() {
    var chart = document.getElementById("sensitivity-chart");
    if (!chart) return;
    chart.innerHTML = "";

    var results = computeSensitivity();
    if (results.length === 0) return;

    var maxDelta = results[0].absDelta || 1;

    results.forEach(function (item) {
      var row = document.createElement("div");
      row.className = "sensitivity-row";
      row.dataset.nodeId = item.id;
      if (item.id === selectedNodeId) row.classList.add("active");

      var name = document.createElement("span");
      name.className = "sensitivity-name";
      var rawName = item.name || item.id;
      name.textContent = getTree().substituteNames === false ? rawName : subVars(rawName);
      name.title = name.textContent;
      row.appendChild(name);

      var track = document.createElement("div");
      track.className = "sensitivity-bar-track";
      var fill = document.createElement("div");
      fill.className = "sensitivity-bar-fill";
      var pct = maxDelta > 0 ? (item.absDelta / maxDelta) * 100 : 0;
      fill.style.width = pct + "%";
      fill.style.backgroundColor = sensitivityColor(item.absDelta, maxDelta);
      track.appendChild(fill);
      row.appendChild(track);

      var val = document.createElement("span");
      val.className = "sensitivity-value";
      var sign = item.deltaPP >= 0 ? "+" : "\u2212";
      val.textContent = sign + item.absDelta.toFixed(1) + "pp";
      row.appendChild(val);

      row.addEventListener("click", function () {
        var node = findNode(getTree().tree, item.id);
        if (node) {
          selectNode(node);
          var card = treeRoot.querySelector('[data-id="' + item.id + '"] > .tg-card');
          if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
        }
      });

      chart.appendChild(row);
    });
  }

  function sensitivityColor(sens, maxSens) {
    var ratio = maxSens > 0 ? sens / maxSens : 0;
    if (ratio > 0.66) return "var(--accent-safety)";
    if (ratio > 0.33) return "var(--accent-universe)";
    return "var(--accent-life)";
  }

  // --- Crux Analysis ---

  var cruxA = document.getElementById("crux-a");
  var cruxB = document.getElementById("crux-b");

  function getWorldviewProbs(key) {
    var tree = getTree();
    var probs = {};
    var leaves = getLeaves(tree.tree);

    if (key === "custom") {
      // Use current probabilities
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of) {
          probs[leaf.id] = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;
        }
      });
      return probs;
    }

    if (key.indexOf("saved:") === 0) {
      var name = key.slice(6);
      var all = getSavedWorldviews();
      var entry = all[tree.id] && all[tree.id][name];
      if (entry) return entry.probabilities;
    }

    var wv = tree.worldviews[key];
    if (wv) return wv.probabilities;

    return {};
  }

  function computeRootWithProbs(leafProbs) {
    // Temporarily set probabilities, compute root, restore
    var tree = getTree();
    var backup = {};
    var leaves = getLeaves(tree.tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of) {
        backup[leaf.id] = probabilities[leaf.id];
        if (leafProbs[leaf.id] != null) {
          probabilities[leaf.id] = leafProbs[leaf.id];
        }
      }
    });
    var savedPins = pinnedNodes;
    pinnedNodes = {};
    var result = computeProb(tree.tree);
    pinnedNodes = savedPins;
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of) {
        probabilities[leaf.id] = backup[leaf.id];
      }
    });
    return result;
  }

  function computeCrux() {
    var keyA = cruxA.value;
    var keyB = cruxB.value;
    if (!keyA || !keyB || keyA === keyB) return [];

    var probsA = getWorldviewProbs(keyA);
    var probsB = getWorldviewProbs(keyB);
    var rootA = computeRootWithProbs(probsA);
    var rootB = computeRootWithProbs(probsB);

    var leaves = getLeaves(getTree().tree).filter(function (l) { return !l.complement_of; });
    var results = [];

    leaves.forEach(function (leaf) {
      var id = leaf.id;
      var valA = probsA[id] != null ? probsA[id] : 0.5;
      var valB = probsB[id] != null ? probsB[id] : 0.5;

      // Swap: give A this leaf from B, see how A's root changes
      var swapped = {};
      Object.keys(probsA).forEach(function (k) { swapped[k] = probsA[k]; });
      swapped[id] = valB;
      var rootSwapped = computeRootWithProbs(swapped);
      var impact = Math.abs(rootSwapped - rootA);

      results.push({
        id: id,
        name: leaf.name,
        valA: valA,
        valB: valB,
        impactPP: impact * 100
      });
    });

    results.sort(function (a, b) { return b.impactPP - a.impactPP; });
    return { results: results, rootA: rootA, rootB: rootB };
  }

  function populateCruxSelectors() {
    [cruxA, cruxB].forEach(function (sel) {
      sel.innerHTML = "";
      var tree = getTree();
      Object.keys(tree.worldviews).forEach(function (key) {
        var opt = document.createElement("option");
        opt.value = key;
        opt.textContent = tree.worldviews[key].name;
        sel.appendChild(opt);
      });
      var saved = getSavedWorldviews();
      if (saved[tree.id]) {
        var names = Object.keys(saved[tree.id]);
        if (names.length > 0) {
          var sep = document.createElement("option");
          sep.disabled = true;
          sep.textContent = "── Saved ──";
          sel.appendChild(sep);
          names.forEach(function (name) {
            var opt = document.createElement("option");
            opt.value = "saved:" + name;
            opt.textContent = name;
            sel.appendChild(opt);
          });
        }
      }
      var custom = document.createElement("option");
      custom.value = "custom";
      custom.textContent = "Custom (current)";
      sel.appendChild(custom);
    });

    // Default: first and second preset
    var keys = Object.keys(getTree().worldviews);
    if (keys.length >= 2) {
      cruxA.value = keys[0];
      cruxB.value = keys[keys.length - 1];
    }
  }

  function getWorldviewName(key) {
    if (key === "custom") return "Current";
    if (key.indexOf("saved:") === 0) return key.slice(6);
    var wv = getTree().worldviews[key];
    return wv ? wv.name : key;
  }

  function renderCrux() {
    var chart = document.getElementById("crux-chart");
    var summary = document.getElementById("crux-summary");
    if (!chart || !summary) return;
    chart.innerHTML = "";
    summary.innerHTML = "";

    var data = computeCrux();
    if (!data || !data.results || data.results.length === 0) return;

    var nameA = getWorldviewName(cruxA.value);
    var nameB = getWorldviewName(cruxB.value);

    // Summary: clear statement of disagreement
    summary.innerHTML =
      '<div class="crux-summary-item">' +
        '<span class="crux-summary-label">' + nameA + '</span>' +
        '<span class="crux-summary-value crux-val-a">' + formatProb(data.rootA) + '</span>' +
      '</div>' +
      '<div class="crux-summary-item">' +
        '<span class="crux-summary-label">' + nameB + '</span>' +
        '<span class="crux-summary-value crux-val-b">' + formatProb(data.rootB) + '</span>' +
      '</div>' +
      '<div class="crux-summary-item">' +
        '<span class="crux-summary-label">Gap</span>' +
        '<span class="crux-summary-value">' + (Math.abs(data.rootA - data.rootB) * 100).toFixed(1) + 'pp</span>' +
      '</div>';

    // Column header row
    var headerRow = document.createElement("div");
    headerRow.className = "sensitivity-row crux-header";
    headerRow.innerHTML =
      '<span class="sensitivity-name crux-col-label">Assumption</span>' +
      '<span class="sensitivity-bar-track crux-col-label" style="background:transparent">Impact on disagreement</span>' +
      '<span class="crux-row-values crux-col-label">' +
        '<span class="crux-val-a">A</span>' +
        '<span class="crux-arrow">&nbsp;</span>' +
        '<span class="crux-val-b">B</span>' +
      '</span>' +
      '<span class="sensitivity-value crux-col-label">Closes</span>';
    chart.appendChild(headerRow);

    var maxImpact = data.results[0].impactPP || 1;

    data.results.forEach(function (item) {
      var row = document.createElement("div");
      row.className = "sensitivity-row";
      row.dataset.nodeId = item.id;

      var name = document.createElement("span");
      name.className = "sensitivity-name";
      var rawName = item.name || item.id;
      name.textContent = getTree().substituteNames === false ? rawName : subVars(rawName);
      name.title = name.textContent;
      row.appendChild(name);

      var track = document.createElement("div");
      track.className = "sensitivity-bar-track";
      var fill = document.createElement("div");
      fill.className = "sensitivity-bar-fill";
      var pct = maxImpact > 0 ? (item.impactPP / maxImpact) * 100 : 0;
      fill.style.width = pct + "%";
      fill.style.backgroundColor = sensitivityColor(item.impactPP, maxImpact);
      track.appendChild(fill);
      row.appendChild(track);

      var vals = document.createElement("span");
      vals.className = "crux-row-values";
      vals.innerHTML =
        '<span class="crux-val-a">' + formatProb(item.valA) + '</span>' +
        '<span class="crux-arrow">\u2192</span>' +
        '<span class="crux-val-b">' + formatProb(item.valB) + '</span>';
      row.appendChild(vals);

      var impact = document.createElement("span");
      impact.className = "sensitivity-value";
      impact.textContent = item.impactPP.toFixed(1) + "pp";
      row.appendChild(impact);

      row.addEventListener("click", function () {
        var node = findNode(getTree().tree, item.id);
        if (node) {
          selectNode(node);
          var card = treeRoot.querySelector('[data-id="' + item.id + '"] > .tg-card');
          if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
        }
      });

      chart.appendChild(row);
    });
  }

  cruxA.addEventListener("change", renderCrux);
  cruxB.addEventListener("change", renderCrux);

  // --- Info panel ---

  function selectNode(node) {
    selectedNodeId = node.id;

    document.querySelectorAll(".tg-card.selected").forEach(function (el) {
      el.classList.remove("selected");
    });
    var wrapper = treeRoot.querySelector('[data-id="' + node.id + '"]');
    if (wrapper) {
      wrapper.querySelector(":scope > .tg-card").classList.add("selected");
    }

    updateInfoPanel();
    updateSensitivityHighlight();
  }

  function updateSensitivityHighlight() {
    document.querySelectorAll(".sensitivity-row").forEach(function (row) {
      row.classList.toggle("active", row.dataset.nodeId === selectedNodeId);
    });
  }

  function updateInfoPanel() {
    if (!selectedNodeId) return;
    var node = findNode(getTree().tree, selectedNodeId);
    if (!node) return;

    infoPanel.querySelector(".info-panel-empty").style.display = "none";
    infoPanel.querySelector(".info-panel-content").style.display = "block";

    var rawName = node.name || node.label || node.id;
    document.getElementById("info-title").textContent = getTree().substituteNames === false ? rawName : subVars(rawName);

    var typeEl = document.getElementById("info-type");
    var typeText = node.type === "and" ? "AND node" : node.type === "or" ? "OR node" : "Leaf assumption";
    if (pinnedNodes[node.id]) typeText += " (pinned)";
    typeEl.textContent = typeText;
    typeEl.className = "info-type " + node.type;

    var desc = node.description || "No description available.";
    document.getElementById("info-description").textContent = getTree().substituteNames === false ? desc : subVars(desc);
    if (rangeMode) {
      var rng = computeProbRange(node);
      document.getElementById("info-probability").textContent =
        (rng.lo === rng.hi) ? formatProb(computeProb(node)) : formatRange(rng);
    } else {
      document.getElementById("info-probability").textContent = formatProb(computeProb(node));
    }

    var complementEl = document.getElementById("info-complement");
    if (node.complement_of) {
      complementEl.style.display = "block";
      var src = findNode(getTree().tree, node.complement_of);
      var srcName = src ? (src.name || src.label) : node.complement_of;
      document.getElementById("info-complement-target").textContent = getTree().substituteNames === false ? srcName : subVars(srcName);
    } else {
      complementEl.style.display = "none";
    }
  }

  // --- Init ---

  function initProbabilities() {
    probabilities = {};
    pinnedNodes = {};
    probRanges = {};
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of) {
        probabilities[leaf.id] = 0.5;
      }
    });
  }

  function init() {
    populateTreeSelect();

    // Try loading state from URL hash first
    if (!loadStateFromHash()) {
      initProbabilities();
      initVariables();
      populateWorldviewSelect();
      renderVariables();

      var firstKey = Object.keys(getTree().worldviews)[0];
      if (firstKey) {
        worldviewSelect.value = firstKey;
        applyWorldview(firstKey);
      }
    }

    populateCruxSelectors();
    renderTree();
    renderCrux();
  }

  // --- Events ---

  treeSelect.addEventListener("change", function () {
    currentTreeIndex = parseInt(treeSelect.value);
    selectedNodeId = null;
    collapsedNodes = {};
    initProbabilities();
    initVariables();
    populateWorldviewSelect();
    renderVariables();

    var firstKey = Object.keys(getTree().worldviews)[0];
    if (firstKey) {
      worldviewSelect.value = firstKey;
      applyWorldview(firstKey);
    }

    populateCruxSelectors();
    renderTree();
    renderCrux();
    infoPanel.querySelector(".info-panel-empty").style.display = "block";
    infoPanel.querySelector(".info-panel-content").style.display = "none";
  });

  worldviewSelect.addEventListener("change", function () {
    applyWorldview(worldviewSelect.value);
    renderTree();
    updateInfoPanel();
  });

  rangeToggle.addEventListener("click", function (e) {
    e.preventDefault();
    rangeMode = !rangeMode;
    rangeToggle.textContent = rangeMode ? "On" : "Off";
    rangeToggle.classList.toggle("active", rangeMode);
    if (rangeMode) {
      // Initialize ranges from current point estimates with ±10pp spread
      var leaves = getLeaves(getTree().tree);
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of && !probRanges[leaf.id]) {
          var p = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;
          probRanges[leaf.id] = {
            lo: Math.max(0, p - 0.1),
            hi: Math.min(1, p + 0.1)
          };
        }
      });
    }
    renderTree();
    updateInfoPanel();
  });

  resetBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    var firstKey = Object.keys(getTree().worldviews)[0];
    if (firstKey) {
      worldviewSelect.value = firstKey;
      applyWorldview(firstKey);
    } else {
      initProbabilities();
    }
    collapsedNodes = {};
    renderTree();
    updateInfoPanel();
  });

  saveWorldviewBtn.addEventListener("click", function (e) {
    e.preventDefault();
    var name = prompt("Name for this worldview:");
    if (!name || !name.trim()) return;
    name = name.trim();
    saveWorldview(name);
    populateWorldviewSelect();
    populateCruxSelectors();
    worldviewSelect.value = "saved:" + name;
    currentWorldview = "saved:" + name;
  });

  shareBtn.addEventListener("click", function (e) {
    e.preventDefault();
    var hash = encodeStateToHash();
    var url = window.location.origin + window.location.pathname + "#" + hash;
    navigator.clipboard.writeText(url).then(function () {
      var orig = shareBtn.textContent;
      shareBtn.textContent = "Copied!";
      setTimeout(function () { shareBtn.textContent = orig; }, 1500);
    }, function () {
      prompt("Copy this link:", url);
    });
  });

  // Redraw connectors on resize (debounced)
  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      requestAnimationFrame(function () {
        requestAnimationFrame(drawConnectors);
      });
    }, 100);
  });

  // Also observe tree container size changes
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(function () {
      requestAnimationFrame(function () {
        requestAnimationFrame(drawConnectors);
      });
    }).observe(treeGraph);
  }

  // --- Go ---
  init();
})();
