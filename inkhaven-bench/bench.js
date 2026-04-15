(function () {
  'use strict';

  const DATA_URL = 'inkhaven_bench.json';

  const state = {
    data: null,
    sort: {
      posts: { key: 'score', dir: 'desc' },
      authors: { key: 'score', dir: 'desc' },
    },
    filter: { posts: '', authors: '' },
    activeDate: null,
  };

  function fmtScore(s) {
    return (s >= 0 ? '+' : '') + s.toFixed(2) + 'σ';
  }

  function fmtMeanSE(mean, se) {
    const sign = mean >= 0 ? '+' : '';
    return `${sign}${mean.toFixed(2)} ± ${se.toFixed(2)}σ`;
  }

  function escapeHTML(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function scoreClass(s) {
    return s >= 0 ? 'num score-pos' : 'num score-neg';
  }

  function linkCell(text, url) {
    if (!url) return escapeHTML(text);
    return `<a href="${escapeHTML(url)}" target="_blank" rel="noopener">${escapeHTML(text)}</a>`;
  }

  // ---------- Sorting ----------

  function applySort(rows, key, dir) {
    const sign = dir === 'asc' ? 1 : -1;
    const sorted = rows.slice();
    sorted.sort((a, b) => {
      const va = a[key];
      const vb = b[key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') {
        return (va - vb) * sign;
      }
      return String(va).localeCompare(String(vb)) * sign;
    });
    return sorted;
  }

  function bindSortHeaders(tableId, panelKey, render) {
    const table = document.getElementById(tableId);
    const ths = table.querySelectorAll('th[data-sort]');
    ths.forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (key === 'rank') return;
        const cur = state.sort[panelKey];
        if (cur.key === key) {
          cur.dir = cur.dir === 'desc' ? 'asc' : 'desc';
        } else {
          cur.key = key;
          cur.dir = key === 'author' || key === 'title' || key === 'date' ? 'asc' : 'desc';
        }
        render();
      });
    });
  }

  function updateSortIndicators(tableId, panelKey) {
    const table = document.getElementById(tableId);
    const ths = table.querySelectorAll('th[data-sort]');
    const cur = state.sort[panelKey];
    ths.forEach((th) => {
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.dataset.sort === cur.key) {
        th.classList.add(cur.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
      }
    });
  }

  // ---------- Top Posts ----------

  function renderPosts() {
    const tbody = document.querySelector('#posts-table tbody');
    const filter = state.filter.posts.toLowerCase().trim();
    let rows = state.data.posts;
    if (filter) {
      rows = rows.filter((r) =>
        (r.title || '').toLowerCase().includes(filter) ||
        (r.author || '').toLowerCase().includes(filter)
      );
    }
    const cur = state.sort.posts;
    rows = applySort(rows, cur.key, cur.dir);

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="bench-empty">No posts match.</td></tr>';
    } else {
      const html = rows.map((r, i) => `
        <tr>
          <td class="num muted">${i + 1}</td>
          <td class="${scoreClass(r.score)}">${fmtMeanSE(r.score, r.se)}</td>
          <td>${escapeHTML(r.author)}</td>
          <td>${linkCell(r.title, r.url)}</td>
          <td class="num muted">${escapeHTML(r.date)}</td>
        </tr>`).join('');
      tbody.innerHTML = html;
    }
    updateSortIndicators('posts-table', 'posts');
  }

  // ---------- Top Authors ----------

  function renderAuthors() {
    const tbody = document.querySelector('#authors-table tbody');
    const filter = state.filter.authors.toLowerCase().trim();
    let rows = state.data.authors;
    if (filter) {
      rows = rows.filter((r) => (r.author || '').toLowerCase().includes(filter));
    }
    const cur = state.sort.authors;
    rows = applySort(rows, cur.key, cur.dir);

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="bench-empty">No authors match.</td></tr>';
    } else {
      const html = rows.map((r, i) => `
        <tr>
          <td class="num muted">${i + 1}</td>
          <td class="${scoreClass(r.score)}">${fmtScore(r.score)}</td>
          <td class="num">${r.n}</td>
          <td>${escapeHTML(r.author)}</td>
          <td>${linkCell(r.title, r.url)}</td>
        </tr>`).join('');
      tbody.innerHTML = html;
    }
    updateSortIndicators('authors-table', 'authors');
  }

  // ---------- By Day ----------

  function renderByDay() {
    const tbody = document.querySelector('#byday-table tbody');
    const date = state.activeDate;
    const rows = state.data.posts
      .filter((r) => r.date === date)
      .slice()
      .sort((a, b) => b.score - a.score);

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="bench-empty">No posts on this date.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map((r, i) => `
      <tr>
        <td class="num muted">${i + 1}</td>
        <td class="${scoreClass(r.score)}">${fmtMeanSE(r.score, r.se)}</td>
        <td>${escapeHTML(r.author)}</td>
        <td>${linkCell(r.title, r.url)}</td>
      </tr>`).join('');
  }

  function setActiveDate(d) {
    const dates = state.data.dates;
    if (!dates.includes(d)) return;
    state.activeDate = d;
    document.getElementById('date-picker').value = d;
    document.getElementById('date-prev').disabled = dates.indexOf(d) === 0;
    document.getElementById('date-next').disabled = dates.indexOf(d) === dates.length - 1;
    renderByDay();
  }

  // ---------- Tabs ----------

  function activateTab(name) {
    document.querySelectorAll('.bench-tab').forEach((t) => {
      t.classList.toggle('active', t.dataset.tab === name);
    });
    document.querySelectorAll('.bench-panel').forEach((p) => {
      p.classList.toggle('active', p.id === 'panel-' + name);
    });
  }

  // ---------- Init ----------

  function renderMeta() {
    const d = state.data;
    const ts = new Date(d.generated_at);
    const tsStr = isNaN(ts.getTime()) ? d.generated_at : ts.toLocaleString();
    document.getElementById('bench-meta').textContent =
      `${d.n_posts} posts • ${d.n_authors} authors • ${d.dates.length} days • bootstrap B=${d.bootstrap_B} • last built ${tsStr}`;
    const note = document.getElementById('author-filter-note');
    if (note && d.author_min_posts) {
      note.textContent =
        `Ranked by mean post score. Limited to authors with at least ${d.author_min_posts} posts ` +
        `(⅔ of the ${d.n_days} residency days so far) to avoid small-sample flukes.`;
    }
  }

  function init(data) {
    state.data = data;
    renderMeta();

    bindSortHeaders('posts-table', 'posts', renderPosts);
    bindSortHeaders('authors-table', 'authors', renderAuthors);

    document.getElementById('post-search').addEventListener('input', (e) => {
      state.filter.posts = e.target.value;
      renderPosts();
    });
    document.getElementById('author-search').addEventListener('input', (e) => {
      state.filter.authors = e.target.value;
      renderAuthors();
    });

    document.querySelectorAll('.bench-tab').forEach((t) => {
      t.addEventListener('click', () => activateTab(t.dataset.tab));
    });

    const dates = data.dates;
    const picker = document.getElementById('date-picker');
    if (dates.length) {
      picker.min = dates[0];
      picker.max = dates[dates.length - 1];
      setActiveDate(dates[dates.length - 1]);
    }
    picker.addEventListener('change', (e) => setActiveDate(e.target.value));
    document.getElementById('date-prev').addEventListener('click', () => {
      const idx = dates.indexOf(state.activeDate);
      if (idx > 0) setActiveDate(dates[idx - 1]);
    });
    document.getElementById('date-next').addEventListener('click', () => {
      const idx = dates.indexOf(state.activeDate);
      if (idx >= 0 && idx < dates.length - 1) setActiveDate(dates[idx + 1]);
    });

    renderPosts();
    renderAuthors();
  }

  fetch(DATA_URL, { cache: 'no-cache' })
    .then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(init)
    .catch((err) => {
      document.getElementById('bench-meta').textContent =
        'Failed to load leaderboard data: ' + err.message;
    });
})();
