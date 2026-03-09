// Doom Assumptions - Interactive Visual Tree

(function () {
  "use strict";

  // --- State ---
  let currentTreeIndex = 0;
  let currentWorldview = null;
  let probabilities = {};
  let selectedNodeId = null;
  let collapsedNodes = {};
  let variableValues = {}; // variable key -> current value string

  // --- DOM refs ---
  const treeSelect = document.getElementById("tree-select");
  const worldviewSelect = document.getElementById("worldview-select");
  const resetBtn = document.getElementById("reset-btn");
  const treeRoot = document.getElementById("tree-root");
  const treeSvg = document.getElementById("tree-svg");
  const treeGraph = document.getElementById("tree-graph");
  const infoPanel = document.getElementById("info-panel");
  const variablesContainer = document.getElementById("variables-container");

  // --- Helpers ---

  function getTree() {
    return TREES[currentTreeIndex];
  }

  function findNode(node, id) {
    if (node.id === id) return node;
    if (node.children) {
      for (const child of node.children) {
        const found = findNode(child, id);
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
      for (const child of node.children) {
        getLeaves(child, leaves);
      }
    }
    return leaves;
  }

  function computeProb(node) {
    if (node.type === "leaf") {
      if (node.complement_of) {
        const src = findNode(getTree().tree, node.complement_of);
        return 1 - computeProb(src);
      }
      return probabilities[node.id] != null ? probabilities[node.id] : 0.5;
    }
    if (!node.children || node.children.length === 0) return 0;

    if (node.type === "and") {
      let p = 1;
      for (const child of node.children) p *= computeProb(child);
      return p;
    }
    if (node.type === "or") {
      let p = 0;
      for (const child of node.children) p += computeProb(child);
      return Math.min(p, 1);
    }
    return 0;
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

  // Replace variable keys in text with their current values.
  // Matches whole-word occurrences of each variable key.
  function subVars(text) {
    if (!text) return text;
    var tree = getTree();
    if (!tree.variables) return text;
    var result = text;
    Object.keys(tree.variables).forEach(function (key) {
      var val = variableValues[key] || tree.variables[key].default || key;
      // Replace the variable key as a whole word (e.g. "D" but not "DAI")
      var regex = new RegExp("\\b" + key + "\\b", "g");
      result = result.replace(regex, val);
    });
    return result;
  }

  // Initialise variable values from tree defaults
  function initVariables() {
    variableValues = {};
    var tree = getTree();
    if (!tree.variables) return;
    Object.keys(tree.variables).forEach(function (key) {
      variableValues[key] = tree.variables[key].default || "";
    });
  }

  // Build variable editor UI
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

  // --- Populate selectors ---

  function populateTreeSelect() {
    treeSelect.innerHTML = "";
    TREES.forEach(function (tree, i) {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = tree.title;
      treeSelect.appendChild(opt);
    });
  }

  function populateWorldviewSelect() {
    const tree = getTree();
    worldviewSelect.innerHTML = "";
    Object.keys(tree.worldviews).forEach(function (key) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = tree.worldviews[key].name;
      worldviewSelect.appendChild(opt);
    });
    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "Custom";
    worldviewSelect.appendChild(custom);
  }

  function applyWorldview(key) {
    const tree = getTree();
    if (key === "custom") {
      currentWorldview = null;
      return;
    }
    const wv = tree.worldviews[key];
    if (!wv) return;
    currentWorldview = key;
    const leaves = getLeaves(tree.tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && wv.probabilities[leaf.id] != null) {
        probabilities[leaf.id] = wv.probabilities[leaf.id];
      }
    });
  }

  // --- Render tree as visual graph ---

  function renderTree() {
    treeRoot.innerHTML = "";
    var el = buildNodeEl(getTree().tree);
    treeRoot.appendChild(el);

    // Draw connectors after layout settles
    requestAnimationFrame(function () {
      requestAnimationFrame(drawConnectors);
    });
  }

  function buildNodeEl(node) {
    var hasChildren = node.children && node.children.length > 0;
    var isCollapsed = !!collapsedNodes[node.id];
    var prob = computeProb(node);

    // Wrapper
    var wrapper = document.createElement("div");
    wrapper.className = "tg-node";
    wrapper.dataset.id = node.id;

    // Card
    var card = document.createElement("div");
    card.className = "tg-card";
    if (node.id === selectedNodeId) card.classList.add("selected");

    // Card header: type + prob + collapse toggle
    var header = document.createElement("div");
    header.className = "tg-card-header";

    var typeBadge = document.createElement("span");
    typeBadge.className = "tg-type " + node.type;
    typeBadge.textContent = node.type;
    header.appendChild(typeBadge);

    var probEl = document.createElement("span");
    probEl.className = "tg-prob";
    probEl.textContent = formatProb(prob);
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
    var label = document.createElement("span");
    label.className = "tg-label";
    label.textContent = subVars(node.name || node.label || node.id);
    card.appendChild(label);

    // Probability bar
    var bar = document.createElement("div");
    bar.className = "tg-bar";
    var barFill = document.createElement("div");
    barFill.className = "tg-bar-fill";
    barFill.style.width = (prob * 100) + "%";
    barFill.style.backgroundColor = probColor(prob);
    bar.appendChild(barFill);
    card.appendChild(bar);

    // Click to select
    card.addEventListener("click", function () {
      selectNode(node);
    });

    wrapper.appendChild(card);

    // Slider for leaf nodes
    if (node.type === "leaf") {
      var sliderWrap = document.createElement("div");
      sliderWrap.className = "tg-slider-wrap";
      if (node.complement_of) sliderWrap.classList.add("complement");

      var slider = document.createElement("input");
      slider.type = "range";
      slider.className = "tg-slider";
      slider.min = "0";
      slider.max = "100";
      slider.step = "1";
      slider.value = Math.round(prob * 100);
      slider.disabled = !!node.complement_of;

      slider.addEventListener("input", function () {
        probabilities[node.id] = parseInt(slider.value) / 100;
        currentWorldview = null;
        worldviewSelect.value = "custom";
        updateAllProbabilities();
        updateInfoPanel();
      });

      var sliderVal = document.createElement("span");
      sliderVal.className = "tg-slider-val";
      sliderVal.textContent = formatProb(prob);

      sliderWrap.appendChild(slider);
      sliderWrap.appendChild(sliderVal);
      wrapper.appendChild(sliderWrap);
    }

    // Children
    if (hasChildren) {
      var childrenEl = document.createElement("div");
      childrenEl.className = "tg-children";
      if (isCollapsed) childrenEl.classList.add("collapsed");

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

  // --- SVG Connectors ---

  function drawConnectors() {
    var graphRect = treeGraph.getBoundingClientRect();
    var w = treeGraph.scrollWidth;
    var h = treeGraph.scrollHeight;

    treeSvg.setAttribute("width", w);
    treeSvg.setAttribute("height", h);
    treeSvg.innerHTML = "";

    drawNodeConnectors(getTree().tree, graphRect);
  }

  function drawNodeConnectors(node, graphRect) {
    if (!node.children || node.children.length === 0) return;
    if (collapsedNodes[node.id]) return;

    var parentCard = treeRoot.querySelector('[data-id="' + node.id + '"] > .tg-card');
    if (!parentCard) return;

    var parentRect = parentCard.getBoundingClientRect();
    var px = parentRect.left + parentRect.width / 2 - graphRect.left;
    var py = parentRect.bottom - graphRect.top;

    for (var i = 0; i < node.children.length; i++) {
      var child = node.children[i];
      var childCard = treeRoot.querySelector('[data-id="' + child.id + '"] > .tg-card');
      if (!childCard) continue;

      var childRect = childCard.getBoundingClientRect();
      var cx = childRect.left + childRect.width / 2 - graphRect.left;
      var cy = childRect.top - graphRect.top;

      // Account for slider space on leaf nodes
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

      // Recurse into children
      drawNodeConnectors(child, graphRect);
    }
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

    // Update card probability text
    var probEl = wrapper.querySelector(":scope > .tg-card .tg-prob");
    if (probEl) probEl.textContent = formatProb(prob);

    // Update bar
    var barFill = wrapper.querySelector(":scope > .tg-card .tg-bar-fill");
    if (barFill) {
      barFill.style.width = (prob * 100) + "%";
      barFill.style.backgroundColor = probColor(prob);
    }

    // Update slider + value for leaves
    if (node.type === "leaf") {
      var sliderWrap = wrapper.querySelector(":scope > .tg-slider-wrap");
      if (sliderWrap) {
        var slider = sliderWrap.querySelector(".tg-slider");
        var sliderVal = sliderWrap.querySelector(".tg-slider-val");
        if (slider) slider.value = Math.round(prob * 100);
        if (sliderVal) sliderVal.textContent = formatProb(prob);
      }
    }

    if (node.children) {
      node.children.forEach(function (child) {
        updateNodeProbs(child);
      });
    }
  }

  // --- Info panel ---

  function selectNode(node) {
    selectedNodeId = node.id;

    // Update selection styling
    document.querySelectorAll(".tg-card.selected").forEach(function (el) {
      el.classList.remove("selected");
    });
    var wrapper = treeRoot.querySelector('[data-id="' + node.id + '"]');
    if (wrapper) {
      wrapper.querySelector(":scope > .tg-card").classList.add("selected");
    }

    updateInfoPanel();
  }

  function updateInfoPanel() {
    if (!selectedNodeId) return;
    var node = findNode(getTree().tree, selectedNodeId);
    if (!node) return;

    infoPanel.querySelector(".info-panel-empty").style.display = "none";
    infoPanel.querySelector(".info-panel-content").style.display = "block";

    document.getElementById("info-title").textContent = subVars(node.name || node.label || node.id);

    var typeEl = document.getElementById("info-type");
    typeEl.textContent = node.type === "and" ? "AND node" : node.type === "or" ? "OR node" : "Leaf assumption";
    typeEl.className = "info-type " + node.type;

    document.getElementById("info-description").textContent = subVars(node.description) || "No description available.";
    document.getElementById("info-probability").textContent = formatProb(computeProb(node));

    var complementEl = document.getElementById("info-complement");
    if (node.complement_of) {
      complementEl.style.display = "block";
      var src = findNode(getTree().tree, node.complement_of);
      document.getElementById("info-complement-target").textContent = src ? subVars(src.name || src.label) : node.complement_of;
    } else {
      complementEl.style.display = "none";
    }
  }

  // --- Init ---

  function initProbabilities() {
    probabilities = {};
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of) {
        probabilities[leaf.id] = 0.5;
      }
    });
  }

  function init() {
    populateTreeSelect();
    initProbabilities();
    initVariables();
    populateWorldviewSelect();
    renderVariables();

    var firstKey = Object.keys(getTree().worldviews)[0];
    if (firstKey) {
      worldviewSelect.value = firstKey;
      applyWorldview(firstKey);
    }

    renderTree();
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

    renderTree();
    infoPanel.querySelector(".info-panel-empty").style.display = "block";
    infoPanel.querySelector(".info-panel-content").style.display = "none";
  });

  worldviewSelect.addEventListener("change", function () {
    applyWorldview(worldviewSelect.value);
    renderTree();
    updateInfoPanel();
  });

  resetBtn.addEventListener("click", function () {
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

  // Redraw connectors on resize
  window.addEventListener("resize", function () {
    requestAnimationFrame(drawConnectors);
  });

  // --- Go ---
  init();
})();
