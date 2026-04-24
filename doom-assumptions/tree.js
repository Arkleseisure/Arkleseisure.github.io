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
  let worldviewAuthor = "";
  let worldviewPerspective = "";  // "", "inside", or "outside"
  let rangeMode = false;
  let rangeIndependent = false;  // false = worst-case bounds, true = independent error propagation
  let probRanges = {};          // node id -> {lo, hi} for leaves in range mode

  // --- DOM refs ---
  const treeSelect = document.getElementById("tree-select");
  const worldviewSelect = document.getElementById("worldview-select");
  const resetBtn = document.getElementById("reset-btn");
  const treeRoot = document.getElementById("tree-root");
  const treeSvg = document.getElementById("tree-svg");
  const treeGraph = document.getElementById("tree-graph");
  const tgScroll = document.getElementById("tg-scroll");
  const infoPanel = document.getElementById("info-panel");
  const variablesContainer = document.getElementById("variables-container");
  const rangeToggle = document.getElementById("range-toggle");
  const rangeModeGroup = document.getElementById("range-mode-group");
  const rangeModeToggle = document.getElementById("range-mode-toggle");
  const saveWorldviewBtn = document.getElementById("save-worldview-btn");
  const exportBtn = document.getElementById("export-btn");
  const importBtn = document.getElementById("import-btn");
  const importFile = document.getElementById("import-file");
  const shareBtn = document.getElementById("share-btn");
  const worldviewTitle = document.getElementById("worldview-title");

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

  // Get all free parameters: non-complement leaves + pinned branch nodes
  function getFreeParams(node, params) {
    params = params || [];
    if (pinnedNodes[node.id] && node.children && node.children.length > 0) {
      // Pinned branch node is a free parameter; don't recurse into children
      params.push(node);
      return params;
    }
    if (node.type === "leaf" && !node.complement_of) {
      params.push(node);
    } else if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        getFreeParams(node.children[i], params);
      }
    }
    return params;
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
      if (rangeMode && probRanges[node.id]) {
        return probRanges[node.id];
      }
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

    if (rangeIndependent) {
      return computeProbRangeIndependent(node);
    }

    // Worst-case: enumerate free parameter corners (exact for multilinear
    // expressions), which correctly accounts for complement anti-correlation.
    return computeProbRangeWorstCase(node);
  }

  // --- Beta distribution utilities for Monte Carlo ---
  // Attempt to fit alpha, beta such that quantile(0.1) ≈ lo, quantile(0.9) ≈ hi
  // Uses a simple search; caches results for performance

  // Sample from Beta(a, b) using Jöhnk's algorithm for small a,b
  // and the gamma method for larger values
  function sampleBeta(a, b) {
    var ga = sampleGamma(a);
    var gb = sampleGamma(b);
    if (ga + gb === 0) return 0.5;
    return ga / (ga + gb);
  }

  // Sample from Gamma(shape) using Marsaglia & Tsang's method
  function sampleGamma(shape) {
    if (shape < 1) {
      return sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
    }
    var d = shape - 1/3;
    var c = 1 / Math.sqrt(9 * d);
    while (true) {
      var x, v;
      do {
        x = sampleNormal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      var u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  // Standard normal via Box-Muller
  var _normalSpare = null;
  function sampleNormal() {
    if (_normalSpare !== null) {
      var s = _normalSpare;
      _normalSpare = null;
      return s;
    }
    var u, v, s;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    s = Math.sqrt(-2 * Math.log(s) / s);
    _normalSpare = v * s;
    return u * s;
  }

  // Regularized incomplete beta function I_x(a, b) via continued fraction (Lentz)
  function betaCDF(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    // Use symmetry relation for better convergence
    if (x > (a + 1) / (a + b + 2)) {
      return 1 - betaCDF(1 - x, b, a);
    }
    var logBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    var front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - logBeta) / a;
    // Continued fraction (Lentz's method)
    var f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    f = d;
    for (var m = 1; m <= 200; m++) {
      // Even step
      var num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
      d = 1 + num * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + num / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      f *= c * d;
      // Odd step
      num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
      d = 1 + num * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + num / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      var delta = c * d;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-10) break;
    }
    return Math.min(1, Math.max(0, front * f));
  }

  // Log-gamma via Stirling approximation (Lanczos)
  function lgamma(z) {
    var g = 7;
    var c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    }
    z -= 1;
    var x = c[0];
    for (var i = 1; i < g + 2; i++) x += c[i] / (z + i);
    var t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  // Fit Beta(a, b) so that 10th percentile ≈ lo, 90th percentile ≈ hi
  // Uses binary search on concentration parameter nu = a + b, with mean pinned
  var _betaFitCache = {};
  function fitBeta(lo, hi) {
    var key = lo.toFixed(4) + "," + hi.toFixed(4);
    if (_betaFitCache[key]) return _betaFitCache[key];

    if (lo >= hi) {
      var result = { a: 100, b: 100 * (1 - lo) / Math.max(lo, 0.001) };
      _betaFitCache[key] = result;
      return result;
    }

    // 2D binary search: outer on mu (mean), inner on nu (concentration)
    var muLo = Math.max(0.01, lo), muHi = Math.min(0.99, hi);
    var bestA = 2, bestB = 2;

    for (var muIter = 0; muIter < 30; muIter++) {
      var mu = (muLo + muHi) / 2;

      // Inner: binary search on nu so CDF(hi) - CDF(lo) ≈ 0.8
      var nuLo = 2, nuHi = 10000;
      for (var nuIter = 0; nuIter < 30; nuIter++) {
        var nuMid = (nuLo + nuHi) / 2;
        var a = mu * nuMid;
        var b = (1 - mu) * nuMid;
        var spread = betaCDF(hi, a, b) - betaCDF(lo, a, b);
        if (spread < 0.8) {
          nuLo = nuMid;
        } else {
          nuHi = nuMid;
        }
      }
      var nu = (nuLo + nuHi) / 2;
      bestA = mu * nu;
      bestB = (1 - mu) * nu;

      var cdfLo = betaCDF(lo, bestA, bestB);
      if (Math.abs(cdfLo - 0.1) < 0.001) break;
      if (cdfLo > 0.1) {
        muLo = mu; // mean too low, shift right
      } else {
        muHi = mu; // mean too high, shift left
      }
    }

    var result = { a: bestA, b: bestB };
    _betaFitCache[key] = result;
    return result;
  }

  // --- Worst-case range propagation via corner enumeration ---
  // The tree expression is multilinear in each free parameter, so extrema
  // occur at corners (each param at its lo or hi). Enumerating all 2^N
  // corners gives the exact worst-case bounds, correctly accounting for
  // complement anti-correlation (since source and complement are derived
  // from the same sampled value at each corner).
  var WC_EXACT_THRESHOLD = 15; // up to 2^15 = 32768 corners enumerated exactly
  var WC_SAMPLE_COUNT = 32768; // random corner samples when N exceeds threshold
  var _wcCache = null;       // { nodeId -> { lo, hi } }

  function invalidateWCCache() {
    _wcCache = null;
  }

  function ensureWCCache() {
    if (_wcCache) return;
    _wcCache = {};

    var tree = getTree();
    var root = tree.tree;

    // Collect free parameters (non-complement leaves + pinned branch nodes)
    // with their ranges. Reuses collectLeafParams to match MC semantics.
    var rawParams = [];
    collectLeafParams(root, rawParams);

    // Map raw params to their ranges (lo, hi, original value)
    var params = rawParams.map(function (p) {
      var r = probRanges[p.id];
      return { id: p.id, lo: r.lo, hi: r.hi, original: probabilities[p.id] };
    });

    // Collect all node IDs to track (same traversal as MC)
    var allNodes = [];
    collectAllNodes(root, allNodes);

    // Initialize min/max trackers
    var mins = {}, maxs = {};
    allNodes.forEach(function (n) {
      mins[n.id] = Infinity;
      maxs[n.id] = -Infinity;
    });

    // If no free parameters, just evaluate once
    if (params.length === 0) {
      allNodes.forEach(function (n) {
        var v = computeProb(n);
        mins[n.id] = v;
        maxs[n.id] = v;
      });
      allNodes.forEach(function (n) {
        _wcCache[n.id] = { lo: mins[n.id], hi: maxs[n.id] };
      });
      return;
    }

    var N = params.length;
    var useExact = N <= WC_EXACT_THRESHOLD;
    var numCorners = useExact ? (1 << N) : WC_SAMPLE_COUNT;

    for (var c = 0; c < numCorners; c++) {
      // Assign each parameter to its lo or hi for this corner
      for (var i = 0; i < N; i++) {
        var useHi;
        if (useExact) {
          useHi = (c >> i) & 1;
        } else {
          useHi = Math.random() < 0.5 ? 1 : 0;
        }
        var p = params[i];
        probabilities[p.id] = useHi ? p.hi : p.lo;
      }
      // Evaluate every node at this corner and update min/max
      for (var j = 0; j < allNodes.length; j++) {
        var node = allNodes[j];
        var v = computeProb(node);
        if (v < mins[node.id]) mins[node.id] = v;
        if (v > maxs[node.id]) maxs[node.id] = v;
      }
    }

    // Restore original probabilities
    for (var i = 0; i < N; i++) {
      probabilities[params[i].id] = params[i].original;
    }

    allNodes.forEach(function (n) {
      _wcCache[n.id] = { lo: mins[n.id], hi: maxs[n.id] };
    });
  }

  function computeProbRangeWorstCase(node) {
    ensureWCCache();
    if (_wcCache && _wcCache[node.id]) {
      return _wcCache[node.id];
    }
    // Shouldn't happen, but fall back to point estimate
    var p = computeProb(node);
    return { lo: p, hi: p };
  }

  // --- Monte Carlo range propagation ---
  var MC_SAMPLES = 20000;
  var _mcCache = null;       // { nodeId -> { lo, hi } }
  var _mcLeafSamples = null; // { leafId -> [sample0, sample1, ...] }
  var _mcLeafParams = null;  // cached leaf params for replay

  function invalidateMCCache() {
    _mcCache = null;
    _mcLeafSamples = null;
    _mcLeafParams = null;
  }

  function ensureMCCache() {
    if (_mcCache) return;
    _mcCache = {};

    var tree = getTree();
    var root = tree.tree;

    // Collect all leaf/pinned params to sample
    var leafParams = [];
    collectLeafParams(root, leafParams);
    _mcLeafParams = leafParams;

    if (leafParams.length === 0) {
      _mcLeafSamples = {};
      return;
    }

    // Collect all node IDs to track
    var allNodes = [];
    collectAllNodes(root, allNodes);

    // Initialize sample arrays
    var nodeSamples = {};
    allNodes.forEach(function (n) { nodeSamples[n.id] = []; });

    // Store per-sample leaf values for replay
    _mcLeafSamples = {};
    for (var i = 0; i < leafParams.length; i++) {
      _mcLeafSamples[leafParams[i].id] = [];
    }

    // Run Monte Carlo once for entire tree
    for (var s = 0; s < MC_SAMPLES; s++) {
      // Sample each leaf from its beta distribution
      for (var i = 0; i < leafParams.length; i++) {
        var lp = leafParams[i];
        var val = sampleBeta(lp.a, lp.b);
        probabilities[lp.id] = val;
        _mcLeafSamples[lp.id].push(val);
      }
      // Compute all node values for this sample
      collectNodeValues(root, nodeSamples);
    }

    // Restore original probabilities
    for (var i = 0; i < leafParams.length; i++) {
      probabilities[leafParams[i].id] = leafParams[i].original;
    }

    // Compute 10th/90th percentile for each node
    allNodes.forEach(function (n) {
      var arr = nodeSamples[n.id];
      if (arr.length === 0) return;
      arr.sort(function (a, b) { return a - b; });
      var idx10 = Math.floor(arr.length * 0.1);
      var idx90 = Math.floor(arr.length * 0.9);
      _mcCache[n.id] = { lo: arr[idx10], hi: arr[idx90] };
    });
  }

  function collectAllNodes(node, list) {
    list.push(node);
    if (pinnedNodes[node.id]) return; // Don't recurse into pinned children
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        collectAllNodes(node.children[i], list);
      }
    }
  }

  function collectNodeValues(node, nodeSamples) {
    var val = computeProb(node);
    if (nodeSamples[node.id]) nodeSamples[node.id].push(val);
    if (pinnedNodes[node.id]) return;
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        collectNodeValues(node.children[i], nodeSamples);
      }
    }
  }

  function computeProbRangeIndependent(node) {
    ensureMCCache();
    if (_mcCache && _mcCache[node.id]) {
      return _mcCache[node.id];
    }
    var p = computeProb(node);
    return { lo: p, hi: p };
  }

  function collectLeafParams(node, params) {
    if (pinnedNodes[node.id] && probabilities[node.id] != null) {
      // Pinned node with a range — treat as a leaf for MC
      if (probRanges[node.id]) {
        var r = probRanges[node.id];
        var fit = fitBeta(r.lo, r.hi);
        params.push({ id: node.id, a: fit.a, b: fit.b, original: probabilities[node.id] });
      }
      return; // Don't recurse into pinned node's children
    }
    if (node.type === "leaf") {
      if (node.complement_of) return; // Handled via source
      var r = (rangeMode && probRanges[node.id]) ? probRanges[node.id] : null;
      if (r && r.lo !== r.hi) {
        var fit = fitBeta(r.lo, r.hi);
        params.push({ id: node.id, a: fit.a, b: fit.b, original: probabilities[node.id] });
      }
      return;
    }
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) {
        collectLeafParams(node.children[i], params);
      }
    }
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

  // Compose the worldview title based on author + perspective
  // Called whenever author/perspective change or worldview is switched.
  function updateWorldviewTitle() {
    var text = "";
    if (worldviewAuthor) {
      var possessive = worldviewAuthor + "\u2019s";
      if (worldviewPerspective === "inside") {
        text = possessive + " inside view";
      } else if (worldviewPerspective === "outside") {
        text = possessive + " outside view";
      } else {
        text = possessive + " worldview";
      }
    } else if (currentWorldview && currentWorldview !== "custom") {
      // Built-in preset: show its display name
      if (currentWorldview.indexOf("saved:") === 0) {
        text = currentWorldview.slice(6);
      } else {
        var tree = getTree();
        var wv = tree.worldviews[currentWorldview];
        if (wv) text = wv.name;
      }
    }
    if (worldviewTitle) worldviewTitle.textContent = text;
  }

  // Sync sidebar author/perspective inputs to current state. Safe to call
  // after worldview changes to reflect loaded metadata without rebuilding
  // the whole sidebar.
  function syncWorldviewMetaInputs() {
    var authorInput = document.getElementById("var-author");
    if (authorInput) authorInput.value = worldviewAuthor;
    var perspSelect = document.getElementById("var-perspective");
    if (perspSelect) perspSelect.value = worldviewPerspective;
  }

  // Build variable editor UI — show label, not raw key (#7)
  function renderVariables() {
    variablesContainer.innerHTML = "";
    var tree = getTree();

    if (tree.variables) {
      Object.keys(tree.variables).forEach(function (key) {
        var varDef = tree.variables[key];

        var group = document.createElement("div");
        group.className = "control-group var-group";

        var label = document.createElement("label");
        label.className = "control-label";
        label.textContent = varDef.displayName || key;
        label.setAttribute("for", "var-" + key);
        label.title = varDef.label || key;
        group.appendChild(label);

        var input = document.createElement("input");
        input.type = "text";
        input.id = "var-" + key;
        input.className = "control-input";
        input.value = variableValues[key] || varDef.default || "";
        input.placeholder = varDef.label || key;

        input.addEventListener("input", function () {
          variableValues[key] = input.value;
          renderTree();
          updateInfoPanel();
        });

        group.appendChild(input);
        variablesContainer.appendChild(group);
      });
    }

    // --- Worldview metadata: Author + Perspective ---
    // Separator above metadata fields
    var sep = document.createElement("div");
    sep.className = "sidebar-sep";
    variablesContainer.appendChild(sep);

    // Author input
    var authorGroup = document.createElement("div");
    authorGroup.className = "control-group var-group";
    var authorLabel = document.createElement("label");
    authorLabel.className = "control-label";
    authorLabel.textContent = "Author";
    authorLabel.setAttribute("for", "var-author");
    authorGroup.appendChild(authorLabel);
    var authorInput = document.createElement("input");
    authorInput.type = "text";
    authorInput.id = "var-author";
    authorInput.className = "control-input";
    authorInput.value = worldviewAuthor;
    authorInput.placeholder = "Your name";
    authorInput.addEventListener("input", function () {
      worldviewAuthor = authorInput.value;
      updateWorldviewTitle();
    });
    authorGroup.appendChild(authorInput);
    variablesContainer.appendChild(authorGroup);

    // Perspective dropdown
    var perspGroup = document.createElement("div");
    perspGroup.className = "control-group var-group";
    var perspLabelRow = document.createElement("div");
    perspLabelRow.className = "label-with-help";
    var perspLabel = document.createElement("label");
    perspLabel.className = "control-label";
    perspLabel.textContent = "Perspective";
    perspLabel.setAttribute("for", "var-perspective");
    perspLabelRow.appendChild(perspLabel);
    var helpIcon = document.createElement("span");
    helpIcon.className = "help-icon";
    helpIcon.textContent = "?";
    helpIcon.setAttribute("tabindex", "0");
    helpIcon.setAttribute("data-tip",
      "Inside view: your own first-principles reasoning from " +
      "the mechanisms, models, and evidence you know about this " +
      "problem.\n\n" +
      "Outside view: a view that takes into account what other " +
      "people think — aggregating opinions, deferring to experts, " +
      "or calibrating against community consensus.");
    perspLabelRow.appendChild(helpIcon);
    perspGroup.appendChild(perspLabelRow);
    var perspSelect = document.createElement("select");
    perspSelect.id = "var-perspective";
    perspSelect.className = "control-input";
    [
      { v: "", l: "None" },
      { v: "inside", l: "Inside view" },
      { v: "outside", l: "Outside view" }
    ].forEach(function (o) {
      var opt = document.createElement("option");
      opt.value = o.v;
      opt.textContent = o.l;
      perspSelect.appendChild(opt);
    });
    perspSelect.value = worldviewPerspective;
    perspSelect.addEventListener("change", function () {
      worldviewPerspective = perspSelect.value;
      updateWorldviewTitle();
    });
    perspGroup.appendChild(perspSelect);
    variablesContainer.appendChild(perspGroup);

    updateWorldviewTitle();
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
    var entry = {
      probabilities: {},
      author: worldviewAuthor,
      perspective: worldviewPerspective
    };
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
    worldviewAuthor = entry.author || "";
    worldviewPerspective = entry.perspective || "";
  }

  function encodeStateToHash() {
    var state = {
      t: currentTreeIndex,
      p: {}
    };
    if (worldviewAuthor) state.a = worldviewAuthor;
    if (worldviewPerspective) state.v = worldviewPerspective;
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
      worldviewAuthor = state.a || "";
      worldviewPerspective = state.v || "";
      worldviewSelect.value = "custom";
      currentWorldview = null;
      syncWorldviewMetaInputs();
      updateWorldviewTitle();
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
          var entry = saved[treeId][name];
          var label = name;
          if (entry && (entry.author || entry.perspective)) {
            var parts = [];
            if (entry.author) parts.push(entry.author);
            if (entry.perspective) parts.push(entry.perspective);
            label += " \u2014 " + parts.join(", ");
          }
          opt.textContent = label;
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
    // Built-in presets don't carry author/perspective
    worldviewAuthor = "";
    worldviewPerspective = "";
  }

  // --- Render tree as visual graph ---

  function renderTree() {
    invalidateMCCache();
    invalidateWCCache();
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
    renderUncertaintyReduction();
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

    var showDualRange = showSlider && rangeMode &&
      (node.type === "leaf" || (hasChildren && isPinned));

    if (showDualRange) {
      // Dual range slider for leaves and pinned branch nodes in range mode
      var rng;
      if (isComplement) {
        rng = computeProbRange(node);
      } else {
        rng = probRanges[node.id] || { lo: prob, hi: prob };
      }
      var rangeWrap = document.createElement("div");
      rangeWrap.className = "tg-range-wrap";
      if (isComplement) rangeWrap.classList.add("complement");

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
        if (isComplement) {
          // Update source node's range (flipped)
          var sourceId = node.complement_of;
          var sourceNode = findNode(getTree().tree, sourceId);
          probRanges[sourceId] = { lo: 1 - hi, hi: 1 - lo };
          probabilities[sourceId] = 1 - (lo + hi) / 2;
          // Pin the source if it's a branch node
          if (sourceNode && sourceNode.children && sourceNode.children.length > 0 && !pinnedNodes[sourceId]) {
            pinnedNodes[sourceId] = true;
            currentWorldview = null;
            worldviewSelect.value = "custom";
            renderTree();
            updateInfoPanel();
            return;
          }
        } else {
          probRanges[node.id] = { lo: lo, hi: hi };
          probabilities[node.id] = (lo + hi) / 2;
        }
        currentWorldview = null;
        worldviewSelect.value = "custom";
        fill.style.left = (lo * 100) + "%";
        fill.style.width = ((hi - lo) * 100) + "%";
        rangeVal.textContent = formatProb(lo) + " – " + formatProb(hi);
        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
        renderUncertaintyReduction();
      }

      loInput.addEventListener("input", onRangeInput);
      hiInput.addEventListener("input", onRangeInput);

      // Unpin button for pinned branch nodes
      if (hasChildren && isPinned) {
        var unpinBtn = document.createElement("span");
        unpinBtn.className = "tg-unpin tg-range-unpin";
        unpinBtn.textContent = "\u00d7";
        unpinBtn.title = "Unpin — use children's values";
        unpinBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          delete pinnedNodes[node.id];
          delete probabilities[node.id];
          delete probRanges[node.id];
          renderTree();
          updateInfoPanel();
        });
        rangeWrap.appendChild(unpinBtn);
      }

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

      slider.addEventListener("input", function () {
        var newVal = parseInt(slider.value) / 100;
        if (isComplement) {
          // Update the source node with 1 - value
          var sourceId = node.complement_of;
          var sourceNode = findNode(getTree().tree, sourceId);
          probabilities[sourceId] = 1 - newVal;
          // Pin the source if it's a branch node (otherwise computeProb ignores stored value)
          if (sourceNode && sourceNode.children && sourceNode.children.length > 0) {
            if (!pinnedNodes[sourceId]) {
              pinnedNodes[sourceId] = true;
              // Need full re-render to show pin state on source
              currentWorldview = null;
              worldviewSelect.value = "custom";
              renderTree();
              updateInfoPanel();
              return;
            }
          }
          currentWorldview = null;
          worldviewSelect.value = "custom";
          updateAllProbabilities();
          updateInfoPanel();
          renderSensitivity();
          renderUncertaintyReduction();
          return;
        }
        probabilities[node.id] = newVal;
        currentWorldview = null;
        worldviewSelect.value = "custom";

        if (hasChildren && !pinnedNodes[node.id]) {
          // First move on a branch node — pin it and re-render to show unpin button/dual slider
          pinnedNodes[node.id] = true;
          if (rangeMode) {
            probRanges[node.id] = { lo: newVal, hi: newVal };
          }
          renderTree();
          updateInfoPanel();
          return;
        }

        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
    renderUncertaintyReduction();
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
          delete probRanges[node.id];
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
    // Measure / draw at natural scale so offset-based positioning is correct.
    treeGraph.style.zoom = "";
    equalizeRowHeights();
    var w = treeGraph.scrollWidth;
    var h = treeGraph.scrollHeight;

    treeSvg.setAttribute("width", w);
    treeSvg.setAttribute("height", h);
    treeSvg.innerHTML = "";

    drawNodeConnectors(getTree().tree);
    drawComplementLinks();

    applyAutoFit(w);
  }

  function equalizeRowHeights() {
    // Within each .tg-children row, force sibling cards to the max card height
    // so their nested rows line up — keeps complement links from going diagonal.
    var rows = treeGraph.querySelectorAll(".tg-children");
    rows.forEach(function (row) {
      var wrappers = row.children;
      var cards = [];
      for (var i = 0; i < wrappers.length; i++) {
        var card = wrappers[i].firstElementChild;
        if (card && card.classList.contains("tg-card")) cards.push(card);
      }
      if (cards.length < 2) {
        cards.forEach(function (c) { c.style.minHeight = ""; });
        return;
      }
      cards.forEach(function (c) { c.style.minHeight = ""; });
      var max = 0;
      cards.forEach(function (c) {
        var h = c.getBoundingClientRect().height;
        if (h > max) max = h;
      });
      if (max > 0) {
        cards.forEach(function (c) { c.style.minHeight = max + "px"; });
      }
    });
  }

  function applyAutoFit(naturalWidth) {
    if (!tgScroll) return;
    var avail = tgScroll.clientWidth;
    if (!avail || !naturalWidth) return;
    var MIN_SCALE = 0.55;
    var scale = Math.min(1, avail / naturalWidth);
    if (scale < MIN_SCALE) scale = MIN_SCALE;
    treeGraph.style.zoom = scale === 1 ? "" : String(scale);
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
      // Draw child → parent so that conceptually probability flows upward.
      path.setAttribute("d",
        "M" + cx + "," + cy +
        " C" + cx + "," + midY +
        " " + px + "," + midY +
        " " + px + "," + py
      );
      path.setAttribute("stroke", "rgba(255,255,255,0.18)");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("fill", "none");
      treeSvg.appendChild(path);

      // Small arrow halfway along the curve, oriented along the tangent,
      // same stroke colour as the line so it reads as a directional hint.
      // Cubic Bézier points: P0=(cx,cy), P1=(cx,midY), P2=(px,midY), P3=(px,py)
      var t = 0.5, mt = 1 - t;
      var bx = mt*mt*mt*cx + 3*mt*mt*t*cx + 3*mt*t*t*px + t*t*t*px;
      var by = mt*mt*mt*cy + 3*mt*mt*t*midY + 3*mt*t*t*midY + t*t*t*py;
      var tangX = 3*mt*mt*(cx-cx) + 6*mt*t*(px-cx) + 3*t*t*(px-px);
      var tangY = 3*mt*mt*(midY-cy) + 6*mt*t*(midY-midY) + 3*t*t*(py-midY);
      var angle = Math.atan2(tangY, tangX) * 180 / Math.PI;

      var arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      arrow.setAttribute("points", "-4,-3 4,0 -4,3");
      arrow.setAttribute("fill", "rgba(255,255,255,0.35)");
      arrow.setAttribute("transform", "translate(" + bx + " " + by + ") rotate(" + angle + ")");
      treeSvg.appendChild(arrow);

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
      // Skip if either node is inside a collapsed parent
      if (sourceCard.closest(".tg-children.collapsed") || compCard.closest(".tg-children.collapsed")) return;

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
    invalidateMCCache();
    invalidateWCCache();
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

    // Update range sliders (for complement nodes whose source changed)
    var rangeWrap = wrapper.querySelector(":scope > .tg-range-wrap");
    if (rangeWrap && rangeMode) {
      var rng = computeProbRange(node);
      var loSlider = rangeWrap.querySelector(".tg-range-lo");
      var hiSlider = rangeWrap.querySelector(".tg-range-hi");
      var rangeFill = rangeWrap.querySelector(".tg-range-fill");
      var rangeVal = rangeWrap.querySelector(".tg-range-val");
      if (loSlider) loSlider.value = Math.round(rng.lo * 100);
      if (hiSlider) hiSlider.value = Math.round(rng.hi * 100);
      if (rangeFill) {
        rangeFill.style.left = (rng.lo * 100) + "%";
        rangeFill.style.width = ((rng.hi - rng.lo) * 100) + "%";
      }
      if (rangeVal) rangeVal.textContent = formatProb(rng.lo) + " – " + formatProb(rng.hi);
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
    var leaves = getFreeParams(root);
    var epsilon = 0.005; // small epsilon for numerical derivative
    var results = [];

    leaves.forEach(function (leaf) {
      var original = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;

      // Two-sided finite difference: dRoot/dLeaf
      var lo = Math.max(0, original - epsilon);
      var hi = Math.min(1, original + epsilon);
      var delta = hi - lo;
      if (delta === 0) {
        results.push({ id: leaf.id, name: leaf.name, derivative: 0 });
        return;
      }

      probabilities[leaf.id] = hi;
      var rootHi = computeProb(root);
      probabilities[leaf.id] = lo;
      var rootLo = computeProb(root);
      probabilities[leaf.id] = original;

      var derivative = (rootHi - rootLo) / delta;
      results.push({ id: leaf.id, name: leaf.name, derivative: derivative, absDerivative: Math.abs(derivative) });
    });

    results.sort(function (a, b) { return b.absDerivative - a.absDerivative; });
    return results;
  }

  function renderSensitivity() {
    var chart = document.getElementById("sensitivity-chart");
    if (!chart) return;
    chart.innerHTML = "";

    var results = computeSensitivity();
    if (results.length === 0) return;

    var maxDeriv = results[0].absDerivative || 1;

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
      var pct = maxDeriv > 0 ? (item.absDerivative / maxDeriv) * 100 : 0;
      fill.style.width = pct + "%";
      fill.style.backgroundColor = sensitivityColor(item.absDerivative, maxDeriv);
      track.appendChild(fill);
      row.appendChild(track);

      var val = document.createElement("span");
      val.className = "sensitivity-value";
      val.textContent = item.absDerivative.toFixed(2);
      val.title = "dP(root)/dP(leaf): 1pp change in this leaf changes the root by " + (item.absDerivative * 1).toFixed(2) + "pp";
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

  // --- Uncertainty Reduction ---

  function computeUncertaintyReduction() {
    if (!rangeMode) return [];

    // Ensure MC has run so we have stored leaf samples
    if (rangeIndependent) {
      ensureMCCache();
    }

    var tree = getTree();
    var root = tree.tree;
    var leaves = getFreeParams(root);

    var baseRange = computeProbRange(root);
    var baseWidth = baseRange.hi - baseRange.lo;
    if (baseWidth === 0) return [];

    var results = [];

    if (rangeIndependent && _mcLeafSamples && _mcLeafParams && _mcLeafParams.length > 0) {
      // Replay stored MC samples with each leaf fixed at midpoint
      leaves.forEach(function (leaf) {
        var origRange = probRanges[leaf.id];
        if (!origRange || origRange.lo === origRange.hi) return;
        if (!_mcLeafSamples[leaf.id]) return;

        var mid = (origRange.lo + origRange.hi) / 2;
        var rootSamples = [];

        for (var s = 0; s < MC_SAMPLES; s++) {
          // Restore all leaf values from stored samples
          for (var i = 0; i < _mcLeafParams.length; i++) {
            var lp = _mcLeafParams[i];
            probabilities[lp.id] = _mcLeafSamples[lp.id][s];
          }
          // Override this leaf with midpoint
          probabilities[leaf.id] = mid;
          rootSamples.push(computeProb(root));
        }

        // Restore original probabilities
        for (var i = 0; i < _mcLeafParams.length; i++) {
          probabilities[_mcLeafParams[i].id] = _mcLeafParams[i].original;
        }

        rootSamples.sort(function (a, b) { return a - b; });
        var idx10 = Math.floor(MC_SAMPLES * 0.1);
        var idx90 = Math.floor(MC_SAMPLES * 0.9);
        var pinnedWidth = rootSamples[idx90] - rootSamples[idx10];
        var reduction = (baseWidth - pinnedWidth) * 100;

        results.push({
          id: leaf.id,
          name: leaf.name,
          reduction: reduction,
          leafWidth: (origRange.hi - origRange.lo) * 100
        });
      });
    } else {
      // Worst-case mode: analytical
      leaves.forEach(function (leaf) {
        var origRange = probRanges[leaf.id];
        if (!origRange) return;

        var mid = (origRange.lo + origRange.hi) / 2;
        probRanges[leaf.id] = { lo: mid, hi: mid };
        var pinnedRange = computeProbRange(root);
        probRanges[leaf.id] = origRange;

        var pinnedWidth = pinnedRange.hi - pinnedRange.lo;
        var reduction = (baseWidth - pinnedWidth) * 100;

        results.push({
          id: leaf.id,
          name: leaf.name,
          reduction: reduction,
          leafWidth: (origRange.hi - origRange.lo) * 100
        });
      });
    }

    results.sort(function (a, b) { return b.reduction - a.reduction; });
    return results;
  }

  function renderUncertaintyReduction() {
    var panel = document.getElementById("uncertainty-panel");
    var chart = document.getElementById("uncertainty-chart");
    if (!panel || !chart) return;

    if (!rangeMode) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";
    chart.innerHTML = "";

    var results = computeUncertaintyReduction();
    if (results.length === 0) return;

    var maxReduction = results[0].reduction || 1;

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
      var pct = maxReduction > 0 ? (item.reduction / maxReduction) * 100 : 0;
      fill.style.width = pct + "%";
      fill.style.backgroundColor = sensitivityColor(item.reduction, maxReduction);
      track.appendChild(fill);
      row.appendChild(track);

      var val = document.createElement("span");
      val.className = "sensitivity-value";
      val.textContent = item.reduction.toFixed(1) + "pp";
      val.title = "Resolving this narrows the final range by " + item.reduction.toFixed(1) + "pp (leaf range: " + item.leafWidth.toFixed(0) + "pp)";
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

    var leaves = getFreeParams(getTree().tree);
    var results = [];

    leaves.forEach(function (leaf) {
      var id = leaf.id;
      var valA = probsA[id] != null ? probsA[id] : 0.5;
      var valB = probsB[id] != null ? probsB[id] : 0.5;

      // Impact on A: if A adopts B's value for this leaf
      var swappedA = {};
      Object.keys(probsA).forEach(function (k) { swappedA[k] = probsA[k]; });
      swappedA[id] = valB;
      var impactA = Math.abs(computeRootWithProbs(swappedA) - rootA) * 100;

      // Impact on B: if B adopts A's value for this leaf
      var swappedB = {};
      Object.keys(probsB).forEach(function (k) { swappedB[k] = probsB[k]; });
      swappedB[id] = valA;
      var impactB = Math.abs(computeRootWithProbs(swappedB) - rootB) * 100;

      results.push({
        id: id,
        name: leaf.name,
        valA: valA,
        valB: valB,
        impactA: impactA,
        impactB: impactB,
        maxImpact: Math.max(impactA, impactB)
      });
    });

    results.sort(function (a, b) { return b.maxImpact - a.maxImpact; });
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
            var entry = saved[tree.id][name];
            var label = name;
            if (entry && (entry.author || entry.perspective)) {
              var parts = [];
              if (entry.author) parts.push(entry.author);
              if (entry.perspective) parts.push(entry.perspective);
              label += " \u2014 " + parts.join(", ");
            }
            opt.textContent = label;
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
      '<span class="crux-bar-pair crux-col-label">' +
        '<span class="crux-bar-label crux-val-a">Moves A</span>' +
        '<span class="crux-bar-label crux-val-b">Moves B</span>' +
      '</span>' +
      '<span class="crux-row-values crux-col-label">' +
        '<span class="crux-val-a">A</span>' +
        '<span class="crux-val-b">B</span>' +
      '</span>';
    chart.appendChild(headerRow);

    var maxImpact = data.results.length > 0 ? data.results[0].maxImpact || 1 : 1;

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

      // Two bars: impact on A and impact on B
      var barPair = document.createElement("div");
      barPair.className = "crux-bar-pair";

      var rowA = document.createElement("div");
      rowA.className = "crux-bar-row";
      var trackA = document.createElement("div");
      trackA.className = "sensitivity-bar-track";
      var fillA = document.createElement("div");
      fillA.className = "sensitivity-bar-fill";
      var pctA = maxImpact > 0 ? (item.impactA / maxImpact) * 100 : 0;
      fillA.style.width = pctA + "%";
      fillA.style.backgroundColor = "var(--accent-life)";
      trackA.appendChild(fillA);
      rowA.appendChild(trackA);
      var ppA = document.createElement("span");
      ppA.className = "crux-bar-pp crux-val-a";
      ppA.textContent = item.impactA.toFixed(1) + "pp";
      rowA.appendChild(ppA);
      barPair.appendChild(rowA);

      var rowB = document.createElement("div");
      rowB.className = "crux-bar-row";
      var trackB = document.createElement("div");
      trackB.className = "sensitivity-bar-track";
      var fillB = document.createElement("div");
      fillB.className = "sensitivity-bar-fill";
      var pctB = maxImpact > 0 ? (item.impactB / maxImpact) * 100 : 0;
      fillB.style.width = pctB + "%";
      fillB.style.backgroundColor = "var(--accent-safety)";
      trackB.appendChild(fillB);
      rowB.appendChild(trackB);
      var ppB = document.createElement("span");
      ppB.className = "crux-bar-pp crux-val-b";
      ppB.textContent = item.impactB.toFixed(1) + "pp";
      rowB.appendChild(ppB);
      barPair.appendChild(rowB);

      row.appendChild(barPair);

      var vals = document.createElement("span");
      vals.className = "crux-row-values";
      vals.innerHTML =
        '<span class="crux-val-a">' + formatProb(item.valA) + '</span>' +
        '<span class="crux-val-b">' + formatProb(item.valB) + '</span>';
      row.appendChild(vals);

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
    _betaFitCache = {};
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of) {
        probabilities[leaf.id] = 0.5;
      }
    });
  }

  function seedDefaultCollapsed() {
    collapsedNodes = {};
    (function walk(node) {
      if (!node) return;
      if (node.defaultCollapsed) collapsedNodes[node.id] = true;
      if (node.children) node.children.forEach(walk);
    })(getTree().tree);
  }

  function init() {
    populateTreeSelect();

    // Try loading state from URL hash first
    if (!loadStateFromHash()) {
      initProbabilities();
      initVariables();
      populateWorldviewSelect();

      var firstKey = Object.keys(getTree().worldviews)[0];
      if (firstKey) {
        worldviewSelect.value = firstKey;
        applyWorldview(firstKey);
      }
      renderVariables();
    }

    populateCruxSelectors();
    seedDefaultCollapsed();
    renderTree();
    renderCrux();
  }

  // --- Events ---

  treeSelect.addEventListener("change", function () {
    currentTreeIndex = parseInt(treeSelect.value);
    selectedNodeId = null;
    initProbabilities();
    initVariables();
    populateWorldviewSelect();

    var firstKey = Object.keys(getTree().worldviews)[0];
    if (firstKey) {
      worldviewSelect.value = firstKey;
      applyWorldview(firstKey);
    }
    renderVariables();

    populateCruxSelectors();
    seedDefaultCollapsed();
    renderTree();
    renderCrux();
    infoPanel.querySelector(".info-panel-empty").style.display = "block";
    infoPanel.querySelector(".info-panel-content").style.display = "none";
  });

  worldviewSelect.addEventListener("change", function () {
    applyWorldview(worldviewSelect.value);
    syncWorldviewMetaInputs();
    updateWorldviewTitle();
    renderTree();
    updateInfoPanel();
  });

  rangeToggle.addEventListener("click", function (e) {
    e.preventDefault();
    rangeMode = !rangeMode;
    rangeToggle.textContent = rangeMode ? "On" : "Off";
    rangeToggle.classList.toggle("active", rangeMode);
    rangeModeGroup.style.display = rangeMode ? "" : "none";
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

  rangeModeToggle.addEventListener("click", function (e) {
    e.preventDefault();
    rangeIndependent = !rangeIndependent;
    rangeModeToggle.textContent = rangeIndependent ? "Independent" : "Worst case";
    rangeModeToggle.title = rangeIndependent
      ? "Monte Carlo with beta distributions. Ranges are 10th–90th percentiles."
      : "Worst-case bounds: all lows together, all highs together.";
    rangeModeToggle.classList.toggle("active", rangeIndependent);
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
      worldviewAuthor = "";
      worldviewPerspective = "";
    }
    syncWorldviewMetaInputs();
    updateWorldviewTitle();
    seedDefaultCollapsed();
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
    updateWorldviewTitle();
  });

  exportBtn.addEventListener("click", function (e) {
    e.preventDefault();
    var name = prompt("Name for this worldview file:", "My worldview");
    if (!name || !name.trim()) return;
    name = name.trim();
    var tree = getTree();
    var data = {
      name: name,
      treeId: tree.id,
      treeTitle: tree.title,
      author: worldviewAuthor,
      perspective: worldviewPerspective,
      probabilities: {},
      ranges: null
    };
    var leaves = getLeaves(tree.tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && probabilities[leaf.id] != null) {
        data.probabilities[leaf.id] = probabilities[leaf.id];
      }
    });
    if (rangeMode) {
      data.ranges = {};
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of && probRanges[leaf.id]) {
          data.ranges[leaf.id] = probRanges[leaf.id];
        }
      });
    }
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name.replace(/[^a-zA-Z0-9_-]/g, "_") + ".json";
    a.click();
    URL.revokeObjectURL(url);
  });

  importBtn.addEventListener("click", function (e) {
    e.preventDefault();
    importFile.click();
  });

  importFile.addEventListener("change", function () {
    var file = importFile.files[0];
    if (!file) return;
    var fileName = file.name.replace(/\.json$/i, "");
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        var data = JSON.parse(e.target.result);
        if (data.treeId && data.treeId !== getTree().id) {
          // Switch to the right tree
          for (var i = 0; i < TREES.length; i++) {
            if (TREES[i].id === data.treeId) {
              currentTreeIndex = i;
              treeSelect.value = i;
              initProbabilities();
              initVariables();
              populateWorldviewSelect();
              renderVariables();
              populateCruxSelectors();
              break;
            }
          }
        }
        if (data.probabilities) {
          Object.keys(data.probabilities).forEach(function (id) {
            probabilities[id] = data.probabilities[id];
          });
        }
        if (data.ranges) {
          rangeMode = true;
          rangeToggle.textContent = "On";
          rangeToggle.classList.add("active");
          Object.keys(data.ranges).forEach(function (id) {
            probRanges[id] = data.ranges[id];
          });
        }
        // Set author/perspective from file before saving
        worldviewAuthor = data.author || "";
        worldviewPerspective = data.perspective || "";
        // Save as named worldview (use name from file, then filename as fallback)
        var wvName = data.name || fileName;
        saveWorldview(wvName);
        populateWorldviewSelect();
        populateCruxSelectors();
        worldviewSelect.value = "saved:" + wvName;
        currentWorldview = "saved:" + wvName;
        syncWorldviewMetaInputs();
        updateWorldviewTitle();
        renderTree();
        renderCrux();
        updateInfoPanel();
      } catch (err) {
        alert("Invalid worldview file.");
      }
    };
    reader.readAsText(file);
    importFile.value = "";
  });

  // --- Share: generate image + link, use Web Share API, fallback to modal ---

  var _html2canvasPromise = null;
  function loadHtml2Canvas() {
    if (_html2canvasPromise) return _html2canvasPromise;
    _html2canvasPromise = new Promise(function (resolve, reject) {
      if (window.html2canvas) return resolve(window.html2canvas);
      var s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      s.onload = function () { resolve(window.html2canvas); };
      s.onerror = function () { reject(new Error("Failed to load html2canvas")); };
      document.head.appendChild(s);
    });
    return _html2canvasPromise;
  }

  function imageFilename() {
    var parts = [];
    if (worldviewAuthor) parts.push(worldviewAuthor);
    if (worldviewPerspective) parts.push(worldviewPerspective + "-view");
    else if (worldviewAuthor) parts.push("worldview");
    if (parts.length === 0) parts.push(getTree().id || "tree");
    return parts.join("-").replace(/[^a-zA-Z0-9_-]/g, "_") + ".png";
  }

  // html2canvas doesn't reliably render absolutely-positioned SVG paths, so
  // we pre-rasterize the connector SVG and swap in a plain <img> during capture.
  function svgToImageDataUrl(svgEl) {
    var w = parseInt(svgEl.getAttribute("width") || "0", 10);
    var h = parseInt(svgEl.getAttribute("height") || "0", 10);
    if (!w || !h) return Promise.resolve(null);
    var clone = svgEl.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    var svgStr = new XMLSerializer().serializeToString(clone);
    var svgUrl = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgStr);
    return new Promise(function (resolve) {
      var img = new Image();
      img.onload = function () {
        var canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(img, 0, 0, w, h);
        resolve({ dataUrl: canvas.toDataURL("image/png"), w: w, h: h });
      };
      img.onerror = function () { resolve(null); };
      img.src = svgUrl;
    });
  }

  // Generate a PNG blob of the tree-main area with a site footer appended.
  function generateShareImage() {
    var svgImg = null;
    return svgToImageDataUrl(treeSvg).then(function (result) {
      svgImg = result;
      return loadHtml2Canvas();
    }).then(function (h2c) {
      var target = document.querySelector(".tree-main");
      var bg = getComputedStyle(document.body).backgroundColor;
      return h2c(target, {
        backgroundColor: bg,
        scale: 2,
        useCORS: true,
        logging: false,
        onclone: function (doc) {
          // Swap pre-rasterized SVG in for the connector paths
          if (svgImg) {
            var clonedSvg = doc.getElementById("tree-svg");
            if (clonedSvg) {
              var img = doc.createElement("img");
              img.src = svgImg.dataUrl;
              img.width = svgImg.w;
              img.height = svgImg.h;
              img.style.cssText =
                "position:absolute;top:0;left:0;pointer-events:none;" +
                "width:" + svgImg.w + "px;height:" + svgImg.h + "px;";
              clonedSvg.parentNode.replaceChild(img, clonedSvg);
            }
          }
          // Append a site-attribution footer inside the captured area
          var treeMain = doc.querySelector(".tree-main");
          if (treeMain) {
            var footer = doc.createElement("div");
            footer.className = "tree-image-footer";
            footer.textContent = "lifeuniversesafety.com/doom-assumptions";
            treeMain.appendChild(footer);
          }
        }
      });
    }).then(function (canvas) {
      return new Promise(function (resolve) {
        canvas.toBlob(function (blob) { resolve(blob); }, "image/png");
      });
    });
  }

  function buildShareUrl() {
    var hash = encodeStateToHash();
    return window.location.origin + window.location.pathname + "#" + hash;
  }

  function buildShareTitle() {
    if (worldviewAuthor) {
      var possessive = worldviewAuthor + "\u2019s";
      if (worldviewPerspective === "inside") return possessive + " inside view";
      if (worldviewPerspective === "outside") return possessive + " outside view";
      return possessive + " worldview";
    }
    return "Doom Assumptions worldview";
  }

  function openShareModal(blob, url, filename) {
    var modal = document.getElementById("share-modal");
    var img = document.getElementById("share-modal-img");
    var urlInput = document.getElementById("share-modal-url-input");
    var imgUrl = URL.createObjectURL(blob);
    img.src = imgUrl;
    urlInput.value = url;
    modal.style.display = "flex";

    function close() {
      modal.style.display = "none";
      URL.revokeObjectURL(imgUrl);
    }

    document.getElementById("share-modal-close").onclick = close;
    document.getElementById("share-modal-backdrop").onclick = close;

    document.getElementById("share-copy-link").onclick = function () {
      navigator.clipboard.writeText(url).then(function () {
        var btn = document.getElementById("share-copy-link");
        var orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(function () { btn.textContent = orig; }, 1500);
      }, function () {
        urlInput.select();
      });
    };

    document.getElementById("share-download-img").onclick = function () {
      var a = document.createElement("a");
      a.href = imgUrl;
      a.download = filename;
      a.click();
    };
  }

  shareBtn.addEventListener("click", function (e) {
    e.preventDefault();
    var orig = shareBtn.textContent;
    shareBtn.textContent = "...";
    shareBtn.disabled = true;

    var url = buildShareUrl();
    var title = buildShareTitle();
    var filename = imageFilename();

    generateShareImage().then(function (blob) {
      shareBtn.textContent = orig;
      shareBtn.disabled = false;

      var file = new File([blob], filename, { type: "image/png" });
      var shareData = {
        files: [file],
        title: "Doom Assumptions",
        text: title + " \u2014 " + url
      };

      // Try Web Share API with files (mobile mainly)
      if (navigator.canShare && navigator.canShare(shareData) && navigator.share) {
        navigator.share(shareData).catch(function () {
          // User cancelled or share failed; fall through to modal
          openShareModal(blob, url, filename);
        });
      } else {
        openShareModal(blob, url, filename);
      }
    }).catch(function (err) {
      shareBtn.textContent = orig;
      shareBtn.disabled = false;
      alert("Could not generate share image: " + err.message);
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
