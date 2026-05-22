// Doom Assumptions - Interactive Visual Tree

(function () {
  "use strict";

  // --- State ---
  let currentTreeIndex = 0;
  let currentWorldview = null;
  let probabilities = {};      // node id -> probability (leaves + pinned nodes)
  let pinnedNodes = {};         // node id -> true if user has pinned this node's probability
  let overrideValues = {};      // node id -> user's override value, preserved across pin toggles
  let selectedNodeId = null;
  let collapsedNodes = {};
  let variableValues = {};
  let worldviewAuthor = "";
  let worldviewPerspective = "";  // "", "inside", or "outside"
  let rangeMode = false;
  let rangeIndependent = false;  // false = worst-case bounds, true = independent error propagation
  let probRanges = {};          // node id -> {lo, hi} for leaves in range mode
  let editMode = false;         // when true, every card shows edit-action chrome

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
  const deleteWorldviewBtn = document.getElementById("delete-worldview-btn");
  const importBtn = document.getElementById("import-btn");
  const importFile = document.getElementById("import-file");
  const shareBtn = document.getElementById("share-btn");
  const worldviewTitle = document.getElementById("worldview-title");
  const editToggle = document.getElementById("edit-toggle");
  const editModal = document.getElementById("edit-modal");
  const editModalTitle = document.getElementById("edit-modal-title");
  const editModalName = document.getElementById("edit-modal-name");
  const editModalDesc = document.getElementById("edit-modal-desc");
  const editModalSave = document.getElementById("edit-modal-save");
  const editModalCancel = document.getElementById("edit-modal-cancel");

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

  // --- Unconditional / "mass" probability ---
  //
  // The headline value on every card is conditional on the chain of ancestor
  // assumptions. The "unconditional" value is what fraction of all worlds
  // actually flow through this branch — i.e. multiplied by the marginal
  // probability of the conditioning events upstream.
  //
  // Within an AND, one child supplies the marginal conditioning event and the
  // other supplies the conditional event. Heuristic for picking which is which:
  //   - mixed AND (leaf + branch): the leaf is the marginal, the branch is
  //     conditional on it.
  //   - joint AND (two leaves): per Sean's reorder, child 0 is the marginal
  //     ("X happens") and child 1 is the conditional ("D | X").
  function isMarginalSibling(child, parent, idx) {
    if (!parent || parent.type !== "and") return false;
    var allLeaves = parent.children.every(function (c) { return c.type === "leaf"; });
    if (allLeaves && parent.children.length === 2) {
      return idx === 0;
    }
    return child.type === "leaf";
  }

  function annotateUnconditional(node, contextP, store) {
    store[node.id] = computeProb(node) * contextP;
    if (!node.children || node.children.length === 0) return;
    if (node.type === "or") {
      for (var i = 0; i < node.children.length; i++) {
        annotateUnconditional(node.children[i], contextP, store);
      }
    } else if (node.type === "and") {
      var marginalProduct = 1;
      for (var i = 0; i < node.children.length; i++) {
        if (isMarginalSibling(node.children[i], node, i)) {
          marginalProduct *= computeProb(node.children[i]);
        }
      }
      for (var i = 0; i < node.children.length; i++) {
        if (isMarginalSibling(node.children[i], node, i)) {
          annotateUnconditional(node.children[i], contextP, store);
        } else {
          annotateUnconditional(node.children[i], contextP * marginalProduct, store);
        }
      }
    }
  }

  function computeUnconditional(node) {
    var store = {};
    annotateUnconditional(getTree().tree, 1, store);
    return store[node.id] != null ? store[node.id] : computeProb(node);
  }

  // --- World-only flow (for connector thickness) ---
  //
  // P(reaching this node's world configuration), ignoring all P(D|...) factors.
  // OR-children inherit the parent's world; their AND siblings split the
  // parent's world by the marginal world-defining leaf inside each AND.
  // AND-children share their parent AND's world (no refinement at this step).
  function computeWorldFlowMap() {
    var map = {};
    function walk(node, parentFlow, parent) {
      var flow;
      if (parent == null) {
        flow = 1;
      } else if (parent.type === "or") {
        // Each OR child is a sub-world refinement. For an AND child, the
        // world marginal is the AND's marginal (world-defining) leaf value.
        if (node.type === "and") {
          var marginal = null;
          for (var i = 0; i < node.children.length; i++) {
            if (isMarginalSibling(node.children[i], node, i)) {
              marginal = node.children[i];
              break;
            }
          }
          flow = parentFlow * (marginal ? computeProb(marginal) : 1);
        } else {
          // OR → leaf (rare in our trees): treat leaf value as the world refinement.
          flow = parentFlow * computeProb(node);
        }
      } else {
        // parent.type === "and" — no world refinement, child stays in same world.
        flow = parentFlow;
      }
      map[node.id] = flow;
      if (node.children) {
        for (var k = 0; k < node.children.length; k++) {
          walk(node.children[k], flow, node);
        }
      }
    }
    walk(getTree().tree, 1, null);
    return map;
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

  // Quantile of Beta(a, b): find x with betaCDF(x, a, b) = q. Bisection.
  function betaQuantile(q, a, b) {
    if (q <= 0) return 0;
    if (q >= 1) return 1;
    var lo = 0, hi = 1;
    for (var i = 0; i < 60; i++) {
      var mid = (lo + hi) / 2;
      if (betaCDF(mid, a, b) < q) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // Generate a default [lo, hi] range from a point estimate p using a Beta
  // with effective sample size n. Mean-parameterised (alpha = n*p, beta =
  // n*(1-p)), with alpha/beta floored at 1 to keep the distribution
  // unimodal at endpoints. The displayed range is the 10/90 quantiles —
  // the same percentiles fitBeta() inverts back to (a, b) for MC sampling,
  // so generation and propagation use a coherent Beta family throughout.
  //
  // n=20 gives roughly: p=0.5→[0.32,0.68], p=0.9→[0.74,0.98], p=0.99→
  // [0.95,0.999]. Higher n → tighter ranges; not yet user-tunable.
  var RANGE_DEFAULT_N = 20;
  function rangeFromPoint(p, n) {
    if (n == null) n = RANGE_DEFAULT_N;
    if (p <= 0) return { lo: 0, hi: 0 };
    if (p >= 1) return { lo: 1, hi: 1 };
    var a = Math.max(1, n * p);
    var b = Math.max(1, n * (1 - p));
    return {
      lo: betaQuantile(0.1, a, b),
      hi: betaQuantile(0.9, a, b)
    };
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
  var _wcCache = null;       // { nodeId -> { lo, hi } } conditional
  var _wcCacheU = null;      // { nodeId -> { lo, hi } } unconditional / mass

  function invalidateWCCache() {
    _wcCache = null;
    _wcCacheU = null;
  }

  function ensureWCCache() {
    if (_wcCache) return;
    _wcCache = {};
    _wcCacheU = {};

    var tree = getTree();
    var root = tree.tree;

    var rawParams = [];
    collectLeafParams(root, rawParams);

    var params = rawParams.map(function (p) {
      var r = probRanges[p.id];
      return { id: p.id, lo: r.lo, hi: r.hi, original: probabilities[p.id] };
    });

    var allNodes = [];
    collectAllNodes(root, allNodes);

    var mins = {}, maxs = {}, minsU = {}, maxsU = {};
    allNodes.forEach(function (n) {
      mins[n.id] = Infinity;  maxs[n.id] = -Infinity;
      minsU[n.id] = Infinity; maxsU[n.id] = -Infinity;
    });

    if (params.length === 0) {
      var uncondStore0 = {};
      annotateUnconditional(root, 1, uncondStore0);
      allNodes.forEach(function (n) {
        var v = computeProb(n);
        mins[n.id] = v; maxs[n.id] = v;
        var u = uncondStore0[n.id] != null ? uncondStore0[n.id] : v;
        minsU[n.id] = u; maxsU[n.id] = u;
      });
      allNodes.forEach(function (n) {
        _wcCache[n.id] = { lo: mins[n.id], hi: maxs[n.id] };
        _wcCacheU[n.id] = { lo: minsU[n.id], hi: maxsU[n.id] };
      });
      return;
    }

    var N = params.length;
    var useExact = N <= WC_EXACT_THRESHOLD;
    var numCorners = useExact ? (1 << N) : WC_SAMPLE_COUNT;

    for (var c = 0; c < numCorners; c++) {
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
      var uncondStore = {};
      annotateUnconditional(root, 1, uncondStore);
      for (var j = 0; j < allNodes.length; j++) {
        var node = allNodes[j];
        var v = computeProb(node);
        if (v < mins[node.id]) mins[node.id] = v;
        if (v > maxs[node.id]) maxs[node.id] = v;
        var u = uncondStore[node.id];
        if (u != null) {
          if (u < minsU[node.id]) minsU[node.id] = u;
          if (u > maxsU[node.id]) maxsU[node.id] = u;
        }
      }
    }

    for (var i = 0; i < N; i++) {
      probabilities[params[i].id] = params[i].original;
    }

    allNodes.forEach(function (n) {
      _wcCache[n.id] = { lo: mins[n.id], hi: maxs[n.id] };
      _wcCacheU[n.id] = { lo: minsU[n.id], hi: maxsU[n.id] };
    });
  }

  function computeProbRangeWorstCase(node) {
    ensureWCCache();
    if (_wcCache && _wcCache[node.id]) {
      return _wcCache[node.id];
    }
    var p = computeProb(node);
    return { lo: p, hi: p };
  }

  // --- Monte Carlo range propagation ---
  var MC_SAMPLES = 20000;
  var _mcCache = null;       // { nodeId -> { lo, hi } } conditional
  var _mcCacheU = null;      // { nodeId -> { lo, hi } } unconditional / mass
  var _mcLeafSamples = null; // { leafId -> [sample0, sample1, ...] }
  var _mcLeafParams = null;  // cached leaf params for replay

  function invalidateMCCache() {
    _mcCache = null;
    _mcCacheU = null;
    _mcLeafSamples = null;
    _mcLeafParams = null;
  }

  function ensureMCCache() {
    if (_mcCache) return;
    _mcCache = {};
    _mcCacheU = {};

    var tree = getTree();
    var root = tree.tree;

    var leafParams = [];
    collectLeafParams(root, leafParams);
    _mcLeafParams = leafParams;

    if (leafParams.length === 0) {
      _mcLeafSamples = {};
      return;
    }

    var allNodes = [];
    collectAllNodes(root, allNodes);

    var nodeSamples = {};
    var nodeSamplesU = {};
    allNodes.forEach(function (n) {
      nodeSamples[n.id] = [];
      nodeSamplesU[n.id] = [];
    });

    _mcLeafSamples = {};
    for (var i = 0; i < leafParams.length; i++) {
      _mcLeafSamples[leafParams[i].id] = [];
    }

    for (var s = 0; s < MC_SAMPLES; s++) {
      for (var i = 0; i < leafParams.length; i++) {
        var lp = leafParams[i];
        var val = sampleBeta(lp.a, lp.b);
        probabilities[lp.id] = val;
        _mcLeafSamples[lp.id].push(val);
      }
      collectNodeValues(root, nodeSamples);
      var uncondStoreMC = {};
      annotateUnconditional(root, 1, uncondStoreMC);
      for (var k = 0; k < allNodes.length; k++) {
        var nid = allNodes[k].id;
        if (uncondStoreMC[nid] != null) nodeSamplesU[nid].push(uncondStoreMC[nid]);
      }
    }

    for (var i = 0; i < leafParams.length; i++) {
      probabilities[leafParams[i].id] = leafParams[i].original;
    }

    allNodes.forEach(function (n) {
      var arr = nodeSamples[n.id];
      if (arr.length > 0) {
        arr.sort(function (a, b) { return a - b; });
        var idx10 = Math.floor(arr.length * 0.1);
        var idx90 = Math.floor(arr.length * 0.9);
        _mcCache[n.id] = { lo: arr[idx10], hi: arr[idx90] };
      }
      var arrU = nodeSamplesU[n.id];
      if (arrU.length > 0) {
        arrU.sort(function (a, b) { return a - b; });
        var idx10U = Math.floor(arrU.length * 0.1);
        var idx90U = Math.floor(arrU.length * 0.9);
        _mcCacheU[n.id] = { lo: arrU[idx10U], hi: arrU[idx90U] };
      }
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

  function computeUnconditionalRange(node) {
    if (!rangeMode) {
      var u = computeUnconditional(node);
      return { lo: u, hi: u };
    }
    if (rangeIndependent) {
      ensureMCCache();
      if (_mcCacheU && _mcCacheU[node.id]) return _mcCacheU[node.id];
    } else {
      ensureWCCache();
      if (_wcCacheU && _wcCacheU[node.id]) return _wcCacheU[node.id];
    }
    var u2 = computeUnconditional(node);
    return { lo: u2, hi: u2 };
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

  // --- Parent map for ancestor lookup (used for conditioning labels) ---
  var _parentMapByTreeId = {};
  function getParentMap() {
    var t = getTree();
    if (_parentMapByTreeId[t.id]) return _parentMapByTreeId[t.id];
    var map = {};
    var walk = function (n, p) {
      if (p) map[n.id] = p;
      if (n.children) {
        for (var i = 0; i < n.children.length; i++) walk(n.children[i], n);
      }
    };
    walk(t.tree, null);
    _parentMapByTreeId[t.id] = map;
    return map;
  }

  // Conditioning label = the title of the AND-branch this node sits within.
  // Rules:
  //   - For an AND node: walk past parent OR to the next AND up.
  //   - For an OR node: use its parent AND (which represents the joint
  //     context the OR is conditional on).
  //   - For a leaf: depends on its role in its AND parent.
  //       Marginal leaf (e.g. "X happens") — skip the parent AND, use the
  //         AND above that, because the leaf itself defines this branch.
  //       Conditional leaf (e.g. "D | X") — use the parent AND, since the
  //         AND represents the joint X-context the leaf is conditional on.
  function findAncestorAnd(node) {
    var map = getParentMap();
    var current = map[node.id];
    while (current) {
      if (current.type === "and") return current;
      current = map[current.id];
    }
    return null;
  }

  function getConditioningLabel(node) {
    var map = getParentMap();
    var ancestorAnd = null;
    if (node.type === "and" || node.type === "or") {
      ancestorAnd = findAncestorAnd(node);
    } else if (node.type === "leaf") {
      var parent = map[node.id];
      if (parent && parent.type === "and") {
        var idx = parent.children.indexOf(node);
        if (isMarginalSibling(node, parent, idx)) {
          ancestorAnd = findAncestorAnd(parent);
        } else {
          ancestorAnd = parent;
        }
      }
    }
    if (!ancestorAnd) return null;
    var name = ancestorAnd.name || ancestorAnd.label || ancestorAnd.id;
    return (getTree().substituteNames === false) ? name : subVars(name, false);
  }

  function formatHeadlineText(node) {
    // Headline is the unconditional probability mass.
    if (rangeMode) {
      var ur = computeUnconditionalRange(node);
      return (ur.lo === ur.hi) ? formatProb(ur.lo) : formatRange(ur);
    }
    return formatProb(computeUnconditional(node));
  }

  function formatConditionalCaption(node) {
    // Secondary line shows the local conditional value. Skip when it equals
    // the unconditional (e.g. the root and top-level branches) — no info.
    var condText, uncondText;
    if (rangeMode) {
      var cr = computeProbRange(node);
      var ur = computeUnconditionalRange(node);
      condText = (cr.lo === cr.hi) ? formatProb(cr.lo) : formatRange(cr);
      uncondText = (ur.lo === ur.hi) ? formatProb(ur.lo) : formatRange(ur);
    } else {
      condText = formatProb(computeProb(node));
      uncondText = formatProb(computeUnconditional(node));
    }
    if (condText === uncondText) return "";
    var conditioning = getConditioningLabel(node);
    if (conditioning) {
      return condText + " given " + conditioning;
    }
    return condText + " in branch";
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
    // Capitalise first letter unless caller opted out (mid-sentence use).
    if (capitalize !== false && result.length > 0) {
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
      // For "X-like" labels we don't actually know whether the underlying
      // estimate is the person's inside or outside view, so drop the
      // perspective qualifier and call it a "worldview".
      if (/-like$/i.test(worldviewAuthor)) {
        text = worldviewAuthor + " worldview";
      } else {
        var perspText = worldviewPerspective
          ? (worldviewPerspective + " view")
          : "worldview";
        text = worldviewAuthor + "\u2019s " + perspText;
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

    // Delete button only makes sense for user-saved worldviews.
    if (deleteWorldviewBtn) {
      var isSaved = !!(currentWorldview && currentWorldview.indexOf("saved:") === 0);
      deleteWorldviewBtn.style.display = isSaved ? "" : "none";
    }
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
          // Changing a variable (Danger, Timeframe, etc.) breaks the
          // assumption that the probs reflect the named worldview, so drop
          // back to Custom — we shouldn't claim e.g. Yudkowsky thinks 95%
          // about a different danger than the one his estimate was for.
          if (currentWorldview) {
            currentWorldview = null;
            worldviewSelect.value = "custom";
            worldviewAuthor = "";
            worldviewPerspective = "";
            syncWorldviewMetaInputs();
            updateWorldviewTitle();
          }
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

    overrideValues = {};
    // Always reset probRanges — otherwise stale ranges from the previously
    // active worldview leak in when this saved worldview has none.
    probRanges = {};
    Object.keys(entry.probabilities).forEach(function (id) {
      probabilities[id] = entry.probabilities[id];
    });
    if (entry.ranges) {
      Object.keys(entry.ranges).forEach(function (id) {
        probRanges[id] = entry.ranges[id];
      });
    }
    worldviewAuthor = entry.author || "";
    worldviewPerspective = entry.perspective || "";
  }

  // After loading any worldview, fill in default ±10pp ranges for any leaf
  // that doesn't have an explicit range supplied (only matters if range
  // mode is currently on).
  function ensureLeafRanges() {
    if (!rangeMode) return;
    var leaves = getLeaves(getTree().tree);
    leaves.forEach(function (leaf) {
      if (leaf.complement_of) return;
      if (probRanges[leaf.id]) return;
      var p = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;
      probRanges[leaf.id] = rangeFromPoint(p);
    });
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
    // Embed the tree structure when the user has edited it (#57). On load,
    // loadStateFromHash applies state.tr in place of the bundled tree.
    if (treeModified) {
      state.tr = serializeTree(getTree().tree);
    }
    // UTF-8-encode before btoa — node descriptions contain em-dashes and
    // other non-Latin-1 characters that raw btoa rejects.
    return btoa(unescape(encodeURIComponent(JSON.stringify(state))));
  }

  // Persist the current state into the URL hash. Used after every edit.
  function saveStateToHash() {
    try {
      var h = encodeStateToHash();
      // Use replaceState so we don't pile up history entries on every edit.
      history.replaceState(null, "", "#" + h);
    } catch (e) { /* ignore — likely too-large hash */ }
  }

  function loadStateFromHash() {
    var hash = window.location.hash.slice(1);
    if (!hash) return false;
    try {
      // Mirror of encodeStateToHash: undo UTF-8 wrap first, then JSON parse.
      // Old hashes without a UTF-8 wrap still decode cleanly because
      // decodeURIComponent(escape(...)) is a no-op for pure-ASCII.
      var raw;
      try { raw = decodeURIComponent(escape(atob(hash))); }
      catch (e) { raw = atob(hash); }
      var state = JSON.parse(raw);
      if (state.t != null && TREES[state.t]) {
        currentTreeIndex = state.t;
        treeSelect.value = state.t;
      }
      // Tree-structure override from a previously-edited URL (#57).
      // Mutates TREES[currentTreeIndex].tree in place so getTree() returns
      // the edited shape. Marks the tree as modified so subsequent saves
      // re-emit `tr`.
      if (state.tr) {
        TREES[currentTreeIndex].tree = state.tr;
        treeModified = true;
        _parentMapByTreeId = {};
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
    var visibleCount = 0;
    TREES.forEach(function (tree, i) {
      if (tree.hidden) return; // kept in code but not exposed in the UI
      var opt = document.createElement("option");
      opt.value = i;
      opt.textContent = tree.title;
      treeSelect.appendChild(opt);
      visibleCount++;
    });
    // Hide the Tree picker when there's only one tree to choose from.
    var group = treeSelect.closest(".control-group");
    if (group) group.style.display = visibleCount > 1 ? "" : "none";
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

    overrideValues = {};
    // Reset ranges; worldview may supply new ones below.
    probRanges = {};
    var leaves = getLeaves(tree.tree);
    leaves.forEach(function (leaf) {
      if (!leaf.complement_of && wv.probabilities[leaf.id] != null) {
        probabilities[leaf.id] = wv.probabilities[leaf.id];
      }
      if (wv.ranges && wv.ranges[leaf.id] && !leaf.complement_of) {
        probRanges[leaf.id] = {
          lo: wv.ranges[leaf.id].lo,
          hi: wv.ranges[leaf.id].hi
        };
      }
    });
    // Built-in presets don't carry author/perspective
    worldviewAuthor = wv.author || "";
    worldviewPerspective = wv.perspective || "";
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

    // Compute layout, then draw connectors. Two rAFs so the browser has
    // settled into one paint frame before we measure card sizes.
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        layoutTreePositions();
        drawConnectors();
      });
    });

    renderSensitivity();
    renderUncertaintyReduction();
  }

  // A "joint" AND has 2 children with at least one leaf — we compress it:
  //   - "double": both children are leaves, both shown inline in one card
  //   - "subtree": one leaf (inline) + one subtree (absorbed; its children
  //     surface as the joint's visual children below the card)
  // Build the ▲/▼ pin-toggle button bound to `node`. Reused by both the
  // initial render in buildNodeEl and the in-place auto-pin path so a slider
  // drag doesn't require a full DOM rebuild to surface the toggle.
  function createPinToggle(node, isRangeContext) {
    var isPinned = !!pinnedNodes[node.id];
    var override = overrideValues[node.id];
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tg-pin-toggle" + (isRangeContext ? " tg-range-toggle" : "") + (isPinned ? " active" : "");
    btn.textContent = isPinned ? "▲" : "▼";
    if (isRangeContext) {
      btn.title = isPinned
        ? "Your override is driving this node. Click to use children's computed value."
        : "Children's computed value is driving this node. Click to restore your override.";
    } else {
      btn.title = isPinned
        ? "Your override (" + formatProb(override) + ") is driving this node. Click to switch to children's computed value."
        : "Children's computed value is driving this node. Click to restore your override (" + formatProb(override) + ").";
    }
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var nowPinned;
      if (pinnedNodes[node.id]) {
        delete pinnedNodes[node.id];
        delete probRanges[node.id];
        nowPinned = false;
      } else {
        pinnedNodes[node.id] = true;
        probabilities[node.id] = overrideValues[node.id];
        if (rangeMode) {
          probRanges[node.id] = { lo: overrideValues[node.id], hi: overrideValues[node.id] };
        }
        nowPinned = true;
      }
      currentWorldview = null;
      worldviewSelect.value = "custom";
      // In-place update — toggle visual state on this button and its card
      // without rebuilding the DOM.
      btn.classList.toggle("active", nowPinned);
      btn.textContent = nowPinned ? "▲" : "▼";
      btn.title = isRangeContext
        ? (nowPinned
            ? "Your override is driving this node. Click to use children's computed value."
            : "Children's computed value is driving this node. Click to restore your override.")
        : (nowPinned
            ? "Your override (" + formatProb(overrideValues[node.id]) + ") is driving this node. Click to switch to children's computed value."
            : "Children's computed value is driving this node. Click to restore your override (" + formatProb(overrideValues[node.id]) + ").");
      var card = document.querySelector('[data-id="' + node.id + '"] > .tg-card');
      if (card) card.classList.toggle("pinned", nowPinned);
      updateAllProbabilities();
      updateInfoPanel();
      renderSensitivity();
      renderUncertaintyReduction();
    });
    return btn;
  }

  function isJointAnd(node) {
    if (node.type !== "and") return null;
    if (!node.children || node.children.length !== 2) return null;
    var leaves = node.children.filter(function (c) { return c.type === "leaf"; });
    if (leaves.length === 2) {
      return { kind: "double" };
    }
    if (leaves.length === 1) {
      var leaf = leaves[0];
      var subtree = node.children[0] === leaf ? node.children[1] : node.children[0];
      return { kind: "subtree", leaf: leaf, subtree: subtree };
    }
    return null;
  }

  function buildNodeEl(node) {
    var hasChildren = node.children && node.children.length > 0;
    var isCollapsed = !!collapsedNodes[node.id];
    var isPinned = !!pinnedNodes[node.id];
    var jointInfo = isJointAnd(node);
    var isJoint = !!jointInfo;
    var prob = computeProb(node);

    // Wrapper
    var wrapper = document.createElement("div");
    wrapper.className = "tg-node";
    if (isJoint) wrapper.classList.add("tg-node-joint");
    wrapper.dataset.id = node.id;

    // Card
    var card = document.createElement("div");
    card.className = "tg-card";
    if (isJoint) card.classList.add("tg-card-joint");
    if (jointInfo && jointInfo.kind === "subtree") card.classList.add("tg-card-joint-subtree");
    if (node.id === selectedNodeId) card.classList.add("selected");
    if (isPinned) card.classList.add("pinned");

    // Card header: prob + collapse toggle. (OR/AND/leaf type badge removed
    // — structure is clear from layout and node names.)
    var header = document.createElement("div");
    header.className = "tg-card-header";

    var probColumn = document.createElement("span");
    probColumn.className = "tg-prob-column";

    // Headline = unconditional probability ("mass" — fraction of all worlds
    // that flow through this branch). The conditional ("local") value sits
    // below in smaller text since it's the more contextual reading.
    var probEl = document.createElement("span");
    probEl.className = "tg-prob";
    probEl.textContent = formatHeadlineText(node);
    probColumn.appendChild(probEl);

    var probMassEl = document.createElement("span");
    probMassEl.className = "tg-prob-mass";
    probMassEl.textContent = formatConditionalCaption(node);
    if (!probMassEl.textContent) probMassEl.style.display = "none";
    probColumn.appendChild(probMassEl);

    header.appendChild(probColumn);

    // For "subtree" joints, the collapse toggle drives the absorbed subtree's
    // collapse state — that's what controls the visual children below the card.
    var collapseTargetId = (jointInfo && jointInfo.kind === "subtree")
      ? jointInfo.subtree.id : node.id;
    var collapseTargetCollapsed = !!collapsedNodes[collapseTargetId];

    var collapseBtn = document.createElement("span");
    collapseBtn.className = "tg-collapse";
    // Hide collapse for leaves (no children) and for double-joint ANDs
    // (children are baked into the card; nothing below to collapse).
    var nothingToCollapse = !hasChildren ||
      (jointInfo && jointInfo.kind === "double");
    if (nothingToCollapse) collapseBtn.classList.add("hidden-toggle");
    collapseBtn.textContent = collapseTargetCollapsed ? "+" : "\u2212";
    collapseBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleCollapse(collapseTargetId);
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

    // Click to select. Stop propagation so a click on an absorbed leaf
    // inside a joint AND selects the leaf rather than bubbling up and
    // selecting the joint AND parent.
    card.addEventListener("click", function (e) {
      e.stopPropagation();
      selectNode(node);
    });

    // Edit-mode action chrome. Always built; CSS gates visibility on
    // `.tg-tree.tg-edit-mode`. Cards inside `.tg-joint-inner` have the row
    // hidden via the more-specific selector in tree.css.
    card.appendChild(buildEditActions(node));

    wrapper.appendChild(card);

    // Slider — available on ALL nodes (#1), not just leaves.
    // Joint AND nodes don't get an external slider; their leaves' sliders sit inside the card.
    // No slider on the root — overriding the root is conceptually meaningless
    // (just discards the whole tree below).
    var isRoot = node.id === getTree().tree.id;
    var showSlider = ((node.type === "leaf") || hasChildren) && !isJoint && !isRoot;
    var isComplement = !!node.complement_of;

    var showDualRange = showSlider && rangeMode;

    if (showDualRange) {
      // Dual range slider for all sliderable nodes in range mode.
      // Unpinned branches initialise from the propagated computed range so
      // the slider style stays consistent with leaves.
      var rng;
      if (isComplement || (hasChildren && !isPinned)) {
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
          var wasUnpinnedBranch = hasChildren && !pinnedNodes[node.id];
          probRanges[node.id] = { lo: lo, hi: hi };
          probabilities[node.id] = (lo + hi) / 2;
          if (hasChildren) overrideValues[node.id] = (lo + hi) / 2;
          if (wasUnpinnedBranch) {
            // In-place pin — attach the ▲ toggle and .pinned class without
            // tearing down the slider, so the drag continues smoothly.
            pinnedNodes[node.id] = true;
            card.classList.add("pinned");
            if (rangeWrap && !rangeWrap.querySelector(".tg-pin-toggle")) {
              rangeWrap.appendChild(createPinToggle(node, true));
            }
          }
        }
        currentWorldview = null;
        worldviewSelect.value = "custom";
        fill.style.left = (lo * 100) + "%";
        fill.style.width = ((hi - lo) * 100) + "%";
        if (rvLo && document.activeElement !== rvLo) rvLo.value = (lo * 100).toFixed(1);
        if (rvHi && document.activeElement !== rvHi) rvHi.value = (hi * 100).toFixed(1);
        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
        renderUncertaintyReduction();
      }

      loInput.addEventListener("input", onRangeInput);
      hiInput.addEventListener("input", onRangeInput);

      // Two-state pin toggle (range mode): preserves override across toggles.
      if (hasChildren && overrideValues[node.id] != null) {
        var toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "tg-pin-toggle tg-range-toggle" + (isPinned ? " active" : "");
        toggleBtn.textContent = isPinned ? "▲" : "▼";
        toggleBtn.title = isPinned
          ? "Your override is driving this node. Click to use children's computed value."
          : "Children's computed value is driving this node. Click to restore your override.";
        toggleBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          if (pinnedNodes[node.id]) {
            delete pinnedNodes[node.id];
            delete probRanges[node.id];
          } else {
            pinnedNodes[node.id] = true;
            probabilities[node.id] = overrideValues[node.id];
            if (rangeMode) {
              probRanges[node.id] = { lo: overrideValues[node.id], hi: overrideValues[node.id] };
            }
          }
          currentWorldview = null;
          worldviewSelect.value = "custom";
          renderTree();
          updateInfoPanel();
          renderSensitivity();
          renderUncertaintyReduction();
        });
        rangeWrap.appendChild(toggleBtn);
      }

      rangeWrap.appendChild(loInput);
      rangeWrap.appendChild(hiInput);

      // Editable lo/hi number inputs (with an em-dash separator) — same
      // visual position as the old "X% – Y%" text, but typeable.
      var rangeVal = document.createElement("span");
      rangeVal.className = "tg-range-val";
      var rvLo = document.createElement("input");
      rvLo.type = "number"; rvLo.min = "0"; rvLo.max = "100"; rvLo.step = "0.1";
      rvLo.className = "tg-range-val-input";
      rvLo.value = (rng.lo * 100).toFixed(1);
      var rvDash = document.createElement("span");
      rvDash.textContent = " – ";
      var rvHi = document.createElement("input");
      rvHi.type = "number"; rvHi.min = "0"; rvHi.max = "100"; rvHi.step = "0.1";
      rvHi.className = "tg-range-val-input";
      rvHi.value = (rng.hi * 100).toFixed(1);
      rangeVal.appendChild(rvLo);
      rangeVal.appendChild(rvDash);
      rangeVal.appendChild(rvHi);
      function syncFromRangeVal() {
        var loV = parseFloat(rvLo.value);
        var hiV = parseFloat(rvHi.value);
        if (isNaN(loV) || isNaN(hiV)) return;
        loV = Math.max(0, Math.min(100, loV));
        hiV = Math.max(0, Math.min(100, hiV));
        loInput.value = Math.round(loV);
        hiInput.value = Math.round(hiV);
        onRangeInput();
      }
      rvLo.addEventListener("input", syncFromRangeVal);
      rvHi.addEventListener("input", syncFromRangeVal);
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
        if (hasChildren) overrideValues[node.id] = newVal;
        currentWorldview = null;
        worldviewSelect.value = "custom";

        if (hasChildren && !pinnedNodes[node.id]) {
          // First move on a branch node — pin in-place so the drag isn't
          // interrupted by a full DOM rebuild. Attach the ▲ toggle button
          // and mark the card pinned; updateAllProbabilities() handles
          // the propagated value updates.
          pinnedNodes[node.id] = true;
          if (rangeMode) {
            probRanges[node.id] = { lo: newVal, hi: newVal };
          }
          card.classList.add("pinned");
          if (sliderWrap && !sliderWrap.querySelector(".tg-pin-toggle")) {
            sliderWrap.appendChild(createPinToggle(node, false));
          }
        }

        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
        renderUncertaintyReduction();
      });

      // Editable number input — lets users type a precise probability
      // instead of fiddling with the slider. Two-way bound: typing updates
      // the slider, dragging updates the input.
      var sliderVal = document.createElement("input");
      sliderVal.type = "number";
      sliderVal.min = "0";
      sliderVal.max = "100";
      sliderVal.step = "0.1";
      sliderVal.className = "tg-slider-val tg-slider-input";
      sliderVal.value = (prob * 100).toFixed(1);
      sliderVal.addEventListener("input", function () {
        var raw = parseFloat(sliderVal.value);
        if (isNaN(raw)) return;
        var clamped = Math.max(0, Math.min(100, raw));
        var v = clamped / 100;
        slider.value = Math.round(clamped);
        if (isComplement) {
          var sourceId = node.complement_of;
          probabilities[sourceId] = 1 - v;
          currentWorldview = null;
          worldviewSelect.value = "custom";
          updateAllProbabilities();
          updateInfoPanel();
          renderSensitivity();
          renderUncertaintyReduction();
          return;
        }
        probabilities[node.id] = v;
        if (hasChildren) overrideValues[node.id] = v;
        currentWorldview = null;
        worldviewSelect.value = "custom";
        if (hasChildren && !pinnedNodes[node.id]) {
          pinnedNodes[node.id] = true;
          if (rangeMode) {
            probRanges[node.id] = { lo: v, hi: v };
          }
          card.classList.add("pinned");
          if (sliderWrap && !sliderWrap.querySelector(".tg-pin-toggle")) {
            sliderWrap.appendChild(createPinToggle(node, false));
          }
        }
        updateAllProbabilities();
        updateInfoPanel();
        renderSensitivity();
        renderUncertaintyReduction();
      });

      // Two-state pin toggle: appears once a branch node has an override.
      // Active = user's override drives this node (top-down).
      // Inactive = children's computed value drives it (bottom-up); override preserved.
      if (hasChildren && overrideValues[node.id] != null) {
        var toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "tg-pin-toggle" + (isPinned ? " active" : "");
        toggleBtn.textContent = isPinned ? "▲" : "▼";
        toggleBtn.title = isPinned
          ? "Your override (" + formatProb(overrideValues[node.id]) + ") is driving this node. Click to switch to children's computed value."
          : "Children's computed value is driving this node. Click to restore your override (" + formatProb(overrideValues[node.id]) + ").";
        toggleBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          if (pinnedNodes[node.id]) {
            delete pinnedNodes[node.id];
            delete probRanges[node.id];
          } else {
            pinnedNodes[node.id] = true;
            probabilities[node.id] = overrideValues[node.id];
            if (rangeMode) {
              probRanges[node.id] = { lo: overrideValues[node.id], hi: overrideValues[node.id] };
            }
          }
          currentWorldview = null;
          worldviewSelect.value = "custom";
          renderTree();
          updateInfoPanel();
          renderSensitivity();
          renderUncertaintyReduction();
        });
        sliderWrap.appendChild(toggleBtn);
      }

      sliderWrap.appendChild(slider);
      sliderWrap.appendChild(sliderVal);
      wrapper.appendChild(sliderWrap);
    }

    // Children
    if (hasChildren) {
      if (jointInfo && jointInfo.kind === "double") {
        // Both children are leaves — render them inside the joint card, compactly.
        var inner = document.createElement("div");
        inner.className = "tg-joint-inner";
        if (isCollapsed) inner.classList.add("collapsed");
        node.children.forEach(function (child) {
          inner.appendChild(buildNodeEl(child));
        });
        card.appendChild(inner);
      } else if (jointInfo && jointInfo.kind === "subtree") {
        // Leaf + subtree: build both children fully, then put the subtree's
        // card+slider inside the joint and lift its children section out so
        // it renders below the joint (as the joint's "visual children").
        var inner = document.createElement("div");
        inner.className = "tg-joint-inner";
        inner.appendChild(buildNodeEl(jointInfo.leaf));

        var subtreeWrapper = buildNodeEl(jointInfo.subtree);
        var liftedChildren = subtreeWrapper.querySelector(":scope > .tg-children");
        if (liftedChildren && liftedChildren.parentNode === subtreeWrapper) {
          subtreeWrapper.removeChild(liftedChildren);
        }
        inner.appendChild(subtreeWrapper);
        card.appendChild(inner);

        if (liftedChildren) {
          if (isPinned) liftedChildren.classList.add("dimmed");
          wrapper.appendChild(liftedChildren);
        }
      } else {
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
    }

    return wrapper;
  }

  // --- Collapse / Expand ---

  function toggleCollapse(nodeId) {
    collapsedNodes[nodeId] = !collapsedNodes[nodeId];
    renderTree();
  }

  // --- Tree editing (#57) ---
  //
  // Phase 1: edit-mode toggle, plus an action row on every card with an
  // "Edit" button that opens a modal for renaming + describing the node.
  // The tree object in TREES[currentTreeIndex] is mutated in place; the
  // worldview/probability maps follow because they key by node id.

  function buildEditActions(node) {
    var row = document.createElement("div");
    row.className = "tg-edit-actions";
    var isLeaf = node.type === "leaf";
    var isComplement = !!node.complement_of;
    var isRoot = node.id === getTree().tree.id;
    var jointInfo = isJointAnd(node);

    // A split-able leaf is any non-complement leaf. In the doom tree this
    // is meant for P(D|world) leaves but world-leaves work too. Leaves
    // absorbed inside a joint card don't get their own edit row (CSS
    // hides them) so this check is intentionally permissive.
    var canSplit = isLeaf && !isComplement;

    // Mergeable nodes are the inverse of split: either an OR with
    // AND-supernode children, OR a joint-subtree AND whose absorbed OR
    // has AND-supernode children. In the latter case the merge target is
    // the absorbed OR — collapsing it turns the joint-subtree into a
    // joint-double.
    var mergeTargetId = null;
    if (node.type === "or" && node.children && node.children.length >= 1 &&
        node.children.every(function (c) { return c.type === "and"; })) {
      mergeTargetId = node.id;
    } else if (jointInfo && jointInfo.kind === "subtree") {
      var inner = jointInfo.subtree;
      if (inner && inner.children && inner.children.length >= 1 &&
          inner.children.every(function (c) { return c.type === "and"; })) {
        mergeTargetId = inner.id;
      }
    }

    function addBtn(cls, glyph, title, handler) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = cls;
      b.title = title;
      b.textContent = glyph;
      b.addEventListener("click", function (e) {
        e.stopPropagation();
        handler();
      });
      row.appendChild(b);
    }

    addBtn("tg-edit-edit", "✎", "Edit name and description", function () {
      openEditModal(node);
    });

    if (canSplit) {
      addBtn("tg-edit-split", "⎇", "Split this leaf into two supernodes", function () {
        openSplitModal(node);
      });
    }

    if (mergeTargetId && !isRoot) {
      addBtn("tg-edit-merge", "⊟", "Collapse this split back into a single P(D | world) leaf", function () {
        var target = findNode(getTree().tree, mergeTargetId);
        if (!target) return;
        if (!confirm("Collapse “" + (target.name || target.id) + "” back into a single leaf? Its " + target.children.length + " supernode children and their subtrees will be discarded; the leaf inherits the current computed value.")) return;
        mergeSplit(mergeTargetId);
      });
    }

    return row;
  }

  var _editModalTargetId = null;
  function openEditModal(node) {
    _editModalTargetId = node.id;
    editModalTitle.textContent = "Edit “" + (node.name || node.id) + "”";
    editModalName.value = node.name || "";
    editModalDesc.value = node.description || "";
    editModal.classList.add("open");
    setTimeout(function () { editModalName.focus(); editModalName.select(); }, 0);
  }
  function closeEditModal() {
    editModal.classList.remove("open");
    _editModalTargetId = null;
  }
  function commitEditModal() {
    if (!_editModalTargetId) return;
    var node = findNode(getTree().tree, _editModalTargetId);
    if (!node) { closeEditModal(); return; }
    var newName = editModalName.value.trim();
    var newDesc = editModalDesc.value;
    if (newName) node.name = newName;
    node.description = newDesc;
    closeEditModal();
    postTreeEdit();
  }

  // Tree-structure mutation helpers — used by buildEditActions handlers.

  var treeModified = false;
  function generateNodeId() {
    return "usr-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 6);
  }

  // Find { parent, index } of a node, or null for root / not found.
  function findParentLocation(root, nodeId) {
    if (!root.children) return null;
    for (var i = 0; i < root.children.length; i++) {
      if (root.children[i].id === nodeId) return { parent: root, index: i };
      var rec = findParentLocation(root.children[i], nodeId);
      if (rec) return rec;
    }
    return null;
  }

  function walkSubtreeIds(node, cb) {
    cb(node);
    if (node.children) {
      for (var i = 0; i < node.children.length; i++) walkSubtreeIds(node.children[i], cb);
    }
  }

  // Split a P(D|world) leaf into two mutually-exclusive supernodes —
  // the structural primitive that produces the 4 existing splits in the
  // bundled tree. Replaces the leaf with an OR (keeping the leaf's id and
  // name as the "D | parent" label) that contains 2 AND-supernode children:
  //
  //   OR  "<original name>"           (was the leaf)
  //   ├── AND "<yes> worlds"          (yes supernode)
  //   │   ├── leaf "<yes>"            (world-defining; gets a slider)
  //   │   └── leaf "D | <yes>"        (conditional; gets a slider)
  //   └── AND "<no> worlds"           (no supernode)
  //       ├── leaf "<no>"             (complement_of the yes-world leaf)
  //       └── leaf "D | <no>"
  //
  // The original leaf's probability is preserved as the default value for
  // both new "D | <branch>" leaves so the computed P(D|parent) doesn't
  // jump on first split.
  function splitLeafIntoSupernodes(leafId, yesName, noName) {
    var leaf = findNode(getTree().tree, leafId);
    if (!leaf || leaf.type !== "leaf" || leaf.complement_of) return;
    if (!yesName) yesName = "yes";
    if (!noName) noName = "no";

    var origName = leaf.name || leaf.id;
    var origDesc = leaf.description || "";
    var origProb = probabilities[leafId];

    var yesAndId = generateNodeId();
    var noAndId = generateNodeId();
    var yesWorldId = generateNodeId();
    var noWorldId = generateNodeId();
    var yesDId = generateNodeId();
    var noDId = generateNodeId();

    // Mutate the leaf into an OR (keep id so existing references stick).
    leaf.type = "or";
    leaf.name = origName;
    leaf.description = origDesc;
    leaf.children = [
      {
        id: yesAndId, type: "and",
        name: yesName + " worlds",
        description: "Worlds where " + yesName + ".",
        children: [
          {
            id: yesWorldId, type: "leaf",
            name: yesName,
            description: "Your credence that we're in a world where " + yesName + "."
          },
          {
            id: yesDId, type: "leaf",
            name: "D | " + yesName,
            description: "Among worlds where " + yesName + ", your credence that D occurs within T."
          }
        ]
      },
      {
        id: noAndId, type: "and",
        name: noName + " worlds",
        description: "Worlds where " + noName + ".",
        children: [
          {
            id: noWorldId, type: "leaf",
            name: noName,
            description: "Your credence that we're in a world where " + noName + ". Computed as 1 − P(" + yesName + ").",
            complement_of: yesWorldId
          },
          {
            id: noDId, type: "leaf",
            name: "D | " + noName,
            description: "Among worlds where " + noName + ", your credence that D occurs within T."
          }
        ]
      }
    ];

    // Drop the old leaf's slider state, seed defaults for the new leaves.
    delete probabilities[leafId];
    delete probRanges[leafId];
    delete overrideValues[leafId];
    delete pinnedNodes[leafId];

    probabilities[yesWorldId] = 0.5;
    // noWorldId is complement — auto-computed, no entry needed.
    probabilities[yesDId] = origProb != null ? origProb : 0.5;
    probabilities[noDId]  = origProb != null ? origProb : 0.5;

    postTreeEdit();
  }

  // Inverse of splitLeafIntoSupernodes: collapse an OR (and its subtree)
  // back into a single leaf carrying the OR's current computed value.
  function mergeSplit(orId) {
    var orNode = findNode(getTree().tree, orId);
    if (!orNode || orNode.type !== "or") return;

    // Preserve the value the user is currently seeing for this branch.
    var currentP = computeProb(orNode);

    // Collect ids in the subtree (excluding the OR itself, whose slot
    // we'll reuse for the new leaf). Then drop their slider state and
    // clean up any complement references *outside* the subtree that
    // pointed *into* it.
    var deletedIds = {};
    walkSubtreeIds(orNode, function (n) {
      if (n.id !== orNode.id) deletedIds[n.id] = true;
    });
    function dropState(id) {
      delete probabilities[id];
      delete probRanges[id];
      delete pinnedNodes[id];
      delete overrideValues[id];
      delete collapsedNodes[id];
    }
    Object.keys(deletedIds).forEach(dropState);

    // Any leaf outside the subtree whose complement_of pointed into the
    // subtree becomes a standalone leaf with a sensible default.
    var rootTree = getTree().tree;
    walkSubtreeIds(rootTree, function (n) {
      if (n === orNode || deletedIds[n.id]) return;
      if (n.complement_of && deletedIds[n.complement_of]) {
        delete n.complement_of;
        if (probabilities[n.id] == null) probabilities[n.id] = 0.5;
      }
    });

    // Convert the OR into a leaf in place.
    orNode.type = "leaf";
    delete orNode.children;
    probabilities[orId] = currentP != null ? currentP : 0.5;

    postTreeEdit();
  }

  // Called after any tree-structure or text edit: invalidate caches, mark
  // dirty, re-render, sync URL hash.
  function postTreeEdit() {
    treeModified = true;
    _parentMapByTreeId = {};
    invalidateMCCache();
    invalidateWCCache();
    renderTree();
    updateInfoPanel();
    saveStateToHash();
  }

  // Serialize a tree to a compact JSON-friendly shape (drops undefined keys).
  function serializeTree(node) {
    var s = { id: node.id, type: node.type };
    if (node.name) s.name = node.name;
    if (node.description) s.description = node.description;
    if (node.complement_of) s.complement_of = node.complement_of;
    if (node.children && node.children.length) s.children = node.children.map(serializeTree);
    return s;
  }

  // --- Split modal — collect names for the two new mutually-exclusive sub-worlds ---

  var _splitTargetId = null;
  function openSplitModal(leafNode) {
    _splitTargetId = leafNode.id;
    splitModalLeaf.textContent = leafNode.name || leafNode.id;
    splitModalYes.value = "";
    splitModalNo.value = "";
    splitModal.classList.add("open");
    setTimeout(function () { splitModalYes.focus(); }, 0);
  }
  function closeSplitModal() {
    splitModal.classList.remove("open");
    _splitTargetId = null;
  }
  function commitSplitModal() {
    if (!_splitTargetId) return;
    var yes = splitModalYes.value.trim();
    var no = splitModalNo.value.trim();
    if (!yes || !no) {
      // Highlight whichever's missing; don't close.
      (yes ? splitModalNo : splitModalYes).focus();
      return;
    }
    var id = _splitTargetId;
    closeSplitModal();
    splitLeafIntoSupernodes(id, yes, no);
  }

  var splitModal, splitModalLeaf, splitModalYes, splitModalNo, splitModalSave, splitModalCancel;
  function ensureSplitModal() {
    if (splitModal) return;
    splitModal = document.getElementById("split-modal");
    if (!splitModal) return;
    splitModalLeaf = document.getElementById("split-modal-leaf");
    splitModalYes = document.getElementById("split-modal-yes");
    splitModalNo = document.getElementById("split-modal-no");
    splitModalSave = document.getElementById("split-modal-save");
    splitModalCancel = document.getElementById("split-modal-cancel");
    splitModalSave.addEventListener("click", function (e) { e.preventDefault(); commitSplitModal(); });
    splitModalCancel.addEventListener("click", function (e) { e.preventDefault(); closeSplitModal(); });
    splitModal.addEventListener("click", function (e) { if (e.target === splitModal) closeSplitModal(); });
    function handleKey(e) {
      if (e.key === "Enter") { e.preventDefault(); commitSplitModal(); }
      if (e.key === "Escape") { e.preventDefault(); closeSplitModal(); }
    }
    splitModalYes.addEventListener("keydown", handleKey);
    splitModalNo.addEventListener("keydown", handleKey);
  }

  // --- JS-driven tree layout (Reingold-Tilford-style contour packing) ---
  //
  // After buildNodeEl creates the DOM with regular flex flow, we measure
  // each card's natural size, then compute (x, y) per visible card and
  // position them absolutely within .tg-tree. The .tg-children containers
  // stay in the DOM for collapse handling but don't lay anything out
  // themselves (.tg-rt-layout overrides in tree.css). Subtrees are packed
  // against each other using left/right contour comparison, so a small
  // sibling next to a wide subtree nudges right up against it instead of
  // sitting out at its subtree's right edge.

  var LAYOUT_SIBLING_GAP = 20;   // horizontal gap between adjacent subtrees
  var LAYOUT_ROW_GAP = 50;       // vertical gap between depth levels
  var LAYOUT_X_PADDING = 12;     // horizontal padding inside .tg-tree
  var LAYOUT_Y_PADDING = 6;      // vertical padding inside .tg-tree

  // Build a "visual node" for layout from a data node. Handles:
  //   - joint-double: single combined card, no visual children
  //   - joint-subtree: single combined card, absorbed subtree's children
  //     become this node's visual children
  //   - collapsed branches: no visual children
  function buildVisualNode(dataNode) {
    var card = treeRoot.querySelector('[data-id="' + dataNode.id + '"] > .tg-card');
    if (!card) return null;
    var visual = {
      id: dataNode.id,
      card: card,
      cardW: card.offsetWidth,
      cardH: card.offsetHeight,
      children: [],
      offsetX: 0,
      leftContour: null,
      rightContour: null,
      absX: 0,
      absY: 0,
    };

    var jointInfo = isJointAnd(dataNode);
    // For joint-subtree, the absorbed OR's id controls collapse of the
    // lifted children below.
    var collapseGuardId = (jointInfo && jointInfo.kind === "subtree")
      ? jointInfo.subtree.id : dataNode.id;

    var visualChildren;
    if (jointInfo && jointInfo.kind === "double") {
      visualChildren = [];                            // no children below
    } else if (jointInfo && jointInfo.kind === "subtree") {
      visualChildren = jointInfo.subtree.children || [];
    } else {
      visualChildren = dataNode.children || [];
    }

    if (collapsedNodes[collapseGuardId]) visualChildren = [];

    for (var i = 0; i < visualChildren.length; i++) {
      var childVisual = buildVisualNode(visualChildren[i]);
      if (childVisual) visual.children.push(childVisual);
    }
    return visual;
  }

  // Walk visual tree, recording max card height per depth.
  function collectLevelHeights(visual, depth, out) {
    if (!out[depth] || visual.cardH > out[depth]) out[depth] = visual.cardH;
    for (var i = 0; i < visual.children.length; i++) {
      collectLevelHeights(visual.children[i], depth + 1, out);
    }
  }

  // First pass (postorder): compute each child's offsetX relative to its
  // parent and build left/right contour arrays for each subtree.
  function layoutSubtree(visual) {
    if (visual.children.length === 0) {
      visual.leftContour = [-visual.cardW / 2];
      visual.rightContour = [visual.cardW / 2];
      return;
    }

    // Recurse first so each child has contours.
    for (var i = 0; i < visual.children.length; i++) layoutSubtree(visual.children[i]);

    // Pack children left-to-right. First child anchored at 0; each
    // subsequent child shifted right just enough to clear the previous
    // sibling's rightContour at every overlapping depth.
    var offsets = [];
    offsets[0] = 0;
    visual.children[0].offsetX = 0;

    for (var i = 1; i < visual.children.length; i++) {
      var prev = visual.children[i - 1];
      var curr = visual.children[i];
      var prevRight = prev.rightContour;
      var currLeft = curr.leftContour;
      var maxDepth = Math.min(prevRight.length, currLeft.length);
      var minOffset = -Infinity;
      for (var d = 0; d < maxDepth; d++) {
        // We need (offsets[i] + currLeft[d]) >= (offsets[i-1] + prevRight[d]) + GAP
        var needed = offsets[i - 1] + prevRight[d] + LAYOUT_SIBLING_GAP - currLeft[d];
        if (needed > minOffset) minOffset = needed;
      }
      if (minOffset === -Infinity) minOffset = 0;
      offsets[i] = minOffset;
      curr.offsetX = minOffset;
    }

    // Center children under the parent: shift all children so the midpoint
    // of the first and last is at x=0 (the parent's local origin).
    var firstX = offsets[0];
    var lastX = offsets[visual.children.length - 1];
    var rowCenter = (firstX + lastX) / 2;
    for (var i = 0; i < visual.children.length; i++) {
      visual.children[i].offsetX -= rowCenter;
    }

    // Build this subtree's contours. Level 0 is just this card.
    visual.leftContour = [-visual.cardW / 2];
    visual.rightContour = [visual.cardW / 2];

    var maxChildDepth = 0;
    for (var i = 0; i < visual.children.length; i++) {
      var len = visual.children[i].leftContour.length;
      if (len > maxChildDepth) maxChildDepth = len;
    }
    for (var d = 0; d < maxChildDepth; d++) {
      var minLeft = Infinity, maxRight = -Infinity;
      for (var i = 0; i < visual.children.length; i++) {
        var c = visual.children[i];
        if (d < c.leftContour.length) {
          var l = c.offsetX + c.leftContour[d];
          var r = c.offsetX + c.rightContour[d];
          if (l < minLeft) minLeft = l;
          if (r > maxRight) maxRight = r;
        }
      }
      if (minLeft !== Infinity) {
        visual.leftContour[d + 1] = minLeft;
        visual.rightContour[d + 1] = maxRight;
      }
    }
  }

  // Second pass (preorder): convert relative offsets into absolute x.
  // Y is assigned from precomputed per-level y positions.
  function applyAbsolutePositions(visual, parentX, depth, levelYs) {
    visual.absX = parentX + visual.offsetX;
    visual.absY = levelYs[depth];
    for (var i = 0; i < visual.children.length; i++) {
      applyAbsolutePositions(visual.children[i], visual.absX, depth + 1, levelYs);
    }
  }

  // Walk visual tree and emit each node to a callback.
  function walkVisual(visual, cb) {
    cb(visual);
    for (var i = 0; i < visual.children.length; i++) walkVisual(visual.children[i], cb);
  }

  function layoutTreePositions() {
    // Clear any prior absolute positioning before measuring so we read
    // each card's natural size, not a stale absolute layout.
    treeGraph.classList.remove("tg-rt-layout");
    var allCards = treeRoot.querySelectorAll(".tg-card");
    for (var i = 0; i < allCards.length; i++) {
      var c = allCards[i];
      c.style.position = "";
      c.style.left = "";
      c.style.top = "";
      c.style.transform = "";
      c.style.margin = "";
    }
    treeGraph.style.width = "";
    treeGraph.style.height = "";

    // Build visual tree (queries DOM for current card sizes).
    var visual = buildVisualNode(getTree().tree);
    if (!visual) return;

    // Compute per-depth y positions (max card height at each level + gap).
    var levelHeights = [];
    collectLevelHeights(visual, 0, levelHeights);
    var levelYs = [];
    var y = LAYOUT_Y_PADDING;
    for (var i = 0; i < levelHeights.length; i++) {
      levelYs.push(y);
      y += (levelHeights[i] || 0) + LAYOUT_ROW_GAP;
    }
    var totalH = y - LAYOUT_ROW_GAP + LAYOUT_Y_PADDING;

    // Layout: contour-pack subtrees, then assign absolute positions.
    layoutSubtree(visual);
    applyAbsolutePositions(visual, 0, 0, levelYs);

    // Find horizontal bounds and shift so leftmost = padding.
    var bounds = { minX: Infinity, maxX: -Infinity };
    walkVisual(visual, function (n) {
      var l = n.absX - n.cardW / 2;
      var r = n.absX + n.cardW / 2;
      if (l < bounds.minX) bounds.minX = l;
      if (r > bounds.maxX) bounds.maxX = r;
    });
    var shift = LAYOUT_X_PADDING - bounds.minX;
    var totalW = (bounds.maxX - bounds.minX) + LAYOUT_X_PADDING * 2;

    // Apply positions to the DOM. We set `left` to the card's LEFT edge
    // (not its center) so drawConnectors' existing `offsetLeft +
    // offsetWidth/2 → center` arithmetic keeps working.
    walkVisual(visual, function (n) {
      n.card.style.position = "absolute";
      n.card.style.left = (n.absX + shift - n.cardW / 2) + "px";
      n.card.style.top = n.absY + "px";
      n.card.style.transform = "";
      n.card.style.margin = "0";
    });

    treeGraph.style.width = totalW + "px";
    treeGraph.style.height = totalH + "px";
    treeGraph.classList.add("tg-rt-layout");
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
    // equalizeRowHeights was needed to keep complement links from going
    // diagonal under the old flex layout. With JS layout (tg-rt-layout),
    // siblings share a y and growing cards via min-height would push them
    // past the per-level row spacing we just computed, so we skip it.
    if (!treeGraph.classList.contains("tg-rt-layout")) {
      equalizeRowHeights();
    }
    var w = treeGraph.scrollWidth;
    var h = treeGraph.scrollHeight;

    treeSvg.setAttribute("width", w);
    treeSvg.setAttribute("height", h);
    treeSvg.innerHTML = "";

    // World-flow map drives connector thickness.
    _currentWorldFlow = computeWorldFlowMap();

    drawNodeConnectors(getTree().tree);
    drawComplementLinks();

    applyAutoFit(w);
  }
  var _currentWorldFlow = {};

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
    // Leave a few pixels of slack so subpixel rounding doesn't trigger a scrollbar.
    var SLACK = 4;
    var scale = Math.min(1, (avail - SLACK) / naturalWidth);
    if (scale < MIN_SCALE) scale = MIN_SCALE;
    treeGraph.style.zoom = scale === 1 ? "" : String(scale);
  }

  function drawNodeConnectors(node) {
    if (!node.children || node.children.length === 0) return;

    var jointInfo = isJointAnd(node);
    // "double" joints have no visible children below the card.
    if (jointInfo && jointInfo.kind === "double") return;

    // The id whose collapse state controls visibility of children below this card.
    var collapseGuardId = (jointInfo && jointInfo.kind === "subtree")
      ? jointInfo.subtree.id : node.id;
    if (collapsedNodes[collapseGuardId]) return;

    // Visual children: the cards that actually appear below this node's card.
    var visualChildren = (jointInfo && jointInfo.kind === "subtree")
      ? (jointInfo.subtree.children || [])
      : node.children;

    var parentCard = treeRoot.querySelector('[data-id="' + node.id + '"] > .tg-card');
    if (!parentCard) return;

    var parentPos = getOffsetPos(parentCard, treeGraph);
    var px = parentPos.x + parentCard.offsetWidth / 2;
    var py = parentPos.y + parentCard.offsetHeight;

    for (var i = 0; i < visualChildren.length; i++) {
      var child = visualChildren[i];
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
      // Stroke width scales with world-only flow — the unconditional
      // probability of being in this child's world configuration, ignoring
      // all P(D|...) factors (sqrt so tiny flows stay visible).
      var flowP = _currentWorldFlow[child.id];
      if (flowP == null) flowP = computeProb(child);
      var strokeW = 0.5 + Math.sqrt(Math.max(0, Math.min(1, flowP))) * 3.5;
      path.setAttribute("stroke", "rgba(255,255,255,0.18)");
      path.setAttribute("stroke-width", strokeW.toFixed(2));
      path.setAttribute("fill", "none");
      // If the effective parent is pinned, the children's contributions
      // get discarded in favour of the override — dash just the connectors
      // going into the pinned node. For joint-subtree ANDs, the absorbed
      // subtree (an OR) is the structural parent of these lifted children,
      // so we check its pin state rather than the joint AND card's.
      var effectivePinnedId = (jointInfo && jointInfo.kind === "subtree")
        ? jointInfo.subtree.id : node.id;
      if (pinnedNodes[effectivePinnedId]) {
        path.setAttribute("stroke-dasharray", "6,4");
        path.setAttribute("stroke", "rgba(255,255,255,0.12)");
      }
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
      probEl.textContent = formatHeadlineText(node);
    }

    var probMassEl = wrapper.querySelector(":scope > .tg-card .tg-prob-mass");
    if (probMassEl) {
      var caption = formatConditionalCaption(node);
      probMassEl.textContent = caption;
      probMassEl.style.display = caption ? "" : "none";
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
      if (sliderVal) {
        // sliderVal is now an <input type="number"> so we update .value
        // (and only when not focused so we don't fight a user mid-typing).
        if (document.activeElement !== sliderVal) {
          sliderVal.value = (prob * 100).toFixed(1);
        }
      }
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
      // Range-val now contains two number inputs flanking an em-dash —
      // update each input's .value unless it's currently focused (don't
      // fight a typing user).
      if (rangeVal) {
        var rvInputs = rangeVal.querySelectorAll("input.tg-range-val-input");
        if (rvInputs.length === 2) {
          if (document.activeElement !== rvInputs[0]) {
            rvInputs[0].value = (rng.lo * 100).toFixed(1);
          }
          if (document.activeElement !== rvInputs[1]) {
            rvInputs[1].value = (rng.hi * 100).toFixed(1);
          }
        } else {
          // Fallback for any legacy span-based range-val (shouldn't happen).
          rangeVal.textContent = formatProb(rng.lo) + " – " + formatProb(rng.hi);
        }
      }
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
      results.push({
        id: leaf.id,
        name: leaf.name,
        derivative: derivative,
        absDerivative: Math.abs(derivative)
      });
    });

    // Signed sort: biggest doom-increasing leaves at top, biggest
    // doom-decreasing at bottom.
    results.sort(function (a, b) { return b.derivative - a.derivative; });
    return results;
  }

  function renderSensitivity() {
    var chart = document.getElementById("sensitivity-chart");
    if (!chart) return;
    chart.innerHTML = "";

    var results = computeSensitivity();
    if (results.length === 0) return;

    var maxDeriv = 0;
    results.forEach(function (r) {
      if (r.absDerivative > maxDeriv) maxDeriv = r.absDerivative;
    });
    if (maxDeriv === 0) maxDeriv = 1;

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

      // Centre-anchored signed bar (same treatment as crux). Positive →
      // right (raising leaf raises P(D)); negative → left (raising leaf
      // lowers P(D)).
      var track = document.createElement("div");
      track.className = "sensitivity-bar-track crux-bar-track-signed";
      var fill = document.createElement("div");
      fill.className = "sensitivity-bar-fill";
      var halfPct = (Math.abs(item.derivative) / maxDeriv) * 50;
      fill.style.width = halfPct + "%";
      if (item.derivative >= 0) {
        fill.style.left = "50%";
        fill.style.backgroundColor = sensitivityColor(item.absDerivative, maxDeriv);
      } else {
        fill.style.left = (50 - halfPct) + "%";
        fill.style.backgroundColor = "#e76e7a"; // same anti-direction red used by crux
      }
      track.appendChild(fill);
      row.appendChild(track);

      var val = document.createElement("span");
      val.className = "sensitivity-value";
      val.textContent = (item.derivative >= 0 ? "+" : "") + item.derivative.toFixed(2);
      val.title = "dP(root)/dP(leaf): 1pp change in this leaf changes the root by " + item.derivative.toFixed(2) + "pp";
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

    // Structurally uncertainty reduction is non-negative — pinning a leaf
    // can only narrow (or leave unchanged) the root's range. Tiny negatives
    // are Monte Carlo noise; clamp them to 0.
    results.forEach(function (r) {
      if (r.reduction < 0) r.reduction = 0;
    });

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

    overrideValues = {};
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

    var oldGap = Math.abs(rootA - rootB);

    leaves.forEach(function (leaf) {
      var id = leaf.id;
      var valA = probsA[id] != null ? probsA[id] : 0.5;
      var valB = probsB[id] != null ? probsB[id] : 0.5;

      // Impact on A: A adopts B's value for this leaf. Positive = gap closes
      // (real crux); negative = gap widens (anti-crux — convincing A here
      // actually pulls them further from B at the root level).
      var swappedA = {};
      Object.keys(probsA).forEach(function (k) { swappedA[k] = probsA[k]; });
      swappedA[id] = valB;
      var newRootA = computeRootWithProbs(swappedA);
      var impactA = (oldGap - Math.abs(newRootA - rootB)) * 100;

      // Impact on B: same logic in the other direction.
      var swappedB = {};
      Object.keys(probsB).forEach(function (k) { swappedB[k] = probsB[k]; });
      swappedB[id] = valA;
      var newRootB = computeRootWithProbs(swappedB);
      var impactB = (oldGap - Math.abs(rootA - newRootB)) * 100;

      results.push({
        id: id,
        name: leaf.name,
        valA: valA,
        valB: valB,
        impactA: impactA,
        impactB: impactB,
        maxImpact: Math.max(Math.abs(impactA), Math.abs(impactB))
      });
    });

    // Signed sort: biggest real cruxes at the top, anti-cruxes at the bottom.
    // (maxImpact is still |impact|, used for bar scaling.)
    results.sort(function (a, b) {
      var sa = (a.impactA + a.impactB) / 2;
      var sb = (b.impactA + b.impactB) / 2;
      return sb - sa;
    });
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

    var maxImpact = 0;
    data.results.forEach(function (r) {
      if (r.maxImpact > maxImpact) maxImpact = r.maxImpact;
    });
    if (maxImpact === 0) maxImpact = 1;

    function makeBar(impact, accentVar) {
      // Center-anchored bar. Positive → right half (closes gap, real crux).
      // Negative → left half (widens gap, anti-crux). Width relative to the
      // largest |impact| in the chart so the scale is shared across rows.
      var track = document.createElement("div");
      track.className = "sensitivity-bar-track crux-bar-track-signed";
      var fill = document.createElement("div");
      fill.className = "sensitivity-bar-fill";
      var halfPct = maxImpact > 0 ? (Math.abs(impact) / maxImpact) * 50 : 0;
      fill.style.width = halfPct + "%";
      if (impact >= 0) {
        fill.style.left = "50%";
        fill.style.backgroundColor = "var(" + accentVar + ")";
      } else {
        fill.style.left = (50 - halfPct) + "%";
        fill.style.backgroundColor = "#e76e7a"; // shared "anti-crux" red
      }
      track.appendChild(fill);
      return track;
    }

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

      var barPair = document.createElement("div");
      barPair.className = "crux-bar-pair";

      var rowA = document.createElement("div");
      rowA.className = "crux-bar-row";
      rowA.appendChild(makeBar(item.impactA, "--accent-life"));
      var ppA = document.createElement("span");
      ppA.className = "crux-bar-pp crux-val-a";
      ppA.textContent = (item.impactA >= 0 ? "+" : "") + item.impactA.toFixed(1) + "pp";
      rowA.appendChild(ppA);
      barPair.appendChild(rowA);

      var rowB = document.createElement("div");
      rowB.className = "crux-bar-row";
      rowB.appendChild(makeBar(item.impactB, "--accent-safety"));
      var ppB = document.createElement("span");
      ppB.className = "crux-bar-pp crux-val-b";
      ppB.textContent = (item.impactB >= 0 ? "+" : "") + item.impactB.toFixed(1) + "pp";
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

    document.querySelectorAll(".tg-card.selected, .tg-joint-summary.selected").forEach(function (el) {
      el.classList.remove("selected");
    });
    var wrapper = treeRoot.querySelector('[data-id="' + node.id + '"]');
    if (wrapper) {
      // Normal node wrapper: highlight its direct .tg-card child.
      // Joint summary row: it IS the wrapper for the absorbed subtree node.
      var card = wrapper.querySelector(":scope > .tg-card");
      if (card) {
        card.classList.add("selected");
      } else if (wrapper.classList.contains("tg-joint-summary")) {
        wrapper.classList.add("selected");
      }
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
    if (pinnedNodes[node.id]) {
      typeEl.textContent = "pinned";
      typeEl.className = "info-type " + node.type;
      typeEl.style.display = "";
    } else {
      typeEl.textContent = "";
      typeEl.style.display = "none";
    }

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

    // If a named worldview with per-leaf reasoning is active, show its
    // rationale for this node (or for the complement source if this is a
    // complement leaf).
    var reasoningEl = document.getElementById("info-worldview-reasoning");
    var reasoningText = "";
    var reasoningLeafId = node.complement_of || node.id;
    if (currentWorldview) {
      var tree = getTree();
      var wv = tree.worldviews && tree.worldviews[currentWorldview];
      if (wv && wv.reasoning && wv.reasoning[reasoningLeafId]) {
        reasoningText = wv.reasoning[reasoningLeafId];
        document.getElementById("info-reasoning-label").textContent =
          (wv.name || currentWorldview) + (node.complement_of ? " rationale (for source leaf)" : " rationale");
      }
    }
    if (reasoningText) {
      reasoningEl.style.display = "block";
      document.getElementById("info-reasoning-text").textContent = reasoningText;
    } else {
      reasoningEl.style.display = "none";
    }
  }

  // --- Init ---

  function initProbabilities() {
    probabilities = {};
    pinnedNodes = {};

    overrideValues = {};
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

  // Keep --sticky-offset in sync with the actual site-header + tree-controls
  // height so sticky sidebars don't overlap them (the controls bar wraps to
  // multiple lines at narrower widths, changing its height).
  function updateStickyOffset() {
    var header = document.querySelector(".site-header");
    var controls = document.querySelector(".tree-controls");
    var total = (header ? header.offsetHeight : 80)
              + (controls ? controls.offsetHeight : 50)
              + 8; // small breathing space
    document.documentElement.style.setProperty("--sticky-offset", total + "px");
  }
  window.addEventListener("resize", updateStickyOffset);

  function init() {
    populateTreeSelect();
    updateStickyOffset();
    ensureSplitModal();

    // Try loading state from URL hash first
    if (!loadStateFromHash()) {
      initProbabilities();
      initVariables();
      populateWorldviewSelect();
      // Default to neutral 50% priors (Custom) instead of any preset.
      worldviewSelect.value = "custom";
      currentWorldview = null;
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
    // Switching trees lands you at neutral 50% priors.
    worldviewSelect.value = "custom";
    currentWorldview = null;
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
    ensureLeafRanges();
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
      // Wrap each leaf's current point estimate with a Beta-derived 90% CI
      // (n=20). See rangeFromPoint() — same Beta family used for MC sampling.
      var leaves = getLeaves(getTree().tree);
      leaves.forEach(function (leaf) {
        if (!leaf.complement_of && !probRanges[leaf.id]) {
          var p = probabilities[leaf.id] != null ? probabilities[leaf.id] : 0.5;
          probRanges[leaf.id] = rangeFromPoint(p);
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

  // Edit-mode toggle (#57 phase 1): flip the .tg-edit-mode class on the tree;
  // CSS surfaces the per-card action chrome. No data changes here — edits
  // happen via the modal opened from each card's edit button.
  editToggle.addEventListener("click", function (e) {
    e.preventDefault();
    editMode = !editMode;
    editToggle.textContent = editMode ? "On" : "Off";
    editToggle.classList.toggle("active", editMode);
    treeGraph.classList.toggle("tg-edit-mode", editMode);
  });

  editModalSave.addEventListener("click", function (e) {
    e.preventDefault();
    commitEditModal();
  });
  editModalCancel.addEventListener("click", function (e) {
    e.preventDefault();
    closeEditModal();
  });
  // Close on backdrop click (but not on clicks inside the card)
  editModal.addEventListener("click", function (e) {
    if (e.target === editModal) closeEditModal();
  });
  // Enter to save when focus is in the name field
  editModalName.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); commitEditModal(); }
    if (e.key === "Escape") { e.preventDefault(); closeEditModal(); }
  });
  editModalDesc.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { e.preventDefault(); closeEditModal(); }
  });

  resetBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Reset all probabilities to 50% and clear the current worldview? This cannot be undone.")) return;
    // Reset to neutral 50% priors (Custom), regardless of which presets exist.
    initProbabilities();
    worldviewAuthor = "";
    worldviewPerspective = "";
    currentWorldview = null;
    worldviewSelect.value = "custom";
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

    // 1) Save to localStorage so it shows up in the Worldview dropdown later.
    saveWorldview(name);
    populateWorldviewSelect();
    populateCruxSelectors();
    worldviewSelect.value = "saved:" + name;
    currentWorldview = "saved:" + name;
    updateWorldviewTitle();

    // 2) Download a JSON copy so the user has a portable backup / can share
    //    by file. (Merges the old "Export" button into Save.)
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

  deleteWorldviewBtn.addEventListener("click", function (e) {
    e.preventDefault();
    if (!currentWorldview || currentWorldview.indexOf("saved:") !== 0) return;
    var name = currentWorldview.slice(6);
    if (!confirm("Delete saved worldview \"" + name + "\"? This cannot be undone.")) return;
    deleteSavedWorldview(name);
    populateWorldviewSelect();
    populateCruxSelectors();
    // Drop back to neutral 50% / Custom after deletion.
    initProbabilities();
    worldviewAuthor = "";
    worldviewPerspective = "";
    currentWorldview = null;
    worldviewSelect.value = "custom";
    syncWorldviewMetaInputs();
    updateWorldviewTitle();
    seedDefaultCollapsed();
    renderTree();
    updateInfoPanel();
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

    // On narrow viewports applyAutoFit shrinks the tree via CSS `zoom`, and
    // tg-scroll clips overflow. html2canvas 1.4.1 mishandles `zoom` and
    // captures the visible (clipped) bounding rect, so the output ends up
    // cramped or truncated. Temporarily un-zoom and force containers to the
    // natural tree width so capture happens at 100% scale, then restore.
    var treeMainEl = document.querySelector(".tree-main");
    var saved = {
      zoom: treeGraph.style.zoom,
      scrollOverflow: tgScroll.style.overflow,
      scrollWidth: tgScroll.style.width,
      mainWidth: treeMainEl ? treeMainEl.style.width : null
    };
    treeGraph.style.zoom = "";
    var naturalW = treeGraph.scrollWidth;
    tgScroll.style.overflow = "visible";
    tgScroll.style.width = naturalW + "px";
    if (treeMainEl) treeMainEl.style.width = naturalW + "px";

    // Card positions may have shifted under the overridden widths (centered
    // flex, etc.), so redraw connectors so they line up with where the cards
    // actually are now. drawConnectors() calls applyAutoFit() at the end,
    // which can sneak a tiny fractional zoom back on; clear it once more.
    void treeGraph.offsetWidth;
    drawConnectors();
    treeGraph.style.zoom = "";

    function restoreLayout() {
      treeGraph.style.zoom = saved.zoom;
      tgScroll.style.overflow = saved.scrollOverflow;
      tgScroll.style.width = saved.scrollWidth;
      if (treeMainEl) treeMainEl.style.width = saved.mainWidth;
      // Redraw against the restored layout so the live page is correct again.
      drawConnectors();
    }

    return svgToImageDataUrl(treeSvg).then(function (result) {
      svgImg = result;
      return loadHtml2Canvas();
    }).then(function (h2c) {
      var target = treeMainEl || document.querySelector(".tree-main");
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
      restoreLayout();
      return new Promise(function (resolve) {
        canvas.toBlob(function (blob) { resolve(blob); }, "image/png");
      });
    }).catch(function (err) {
      restoreLayout();
      throw err;
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
