/* ===== 辞書管理画面 JS ===== */

/* ─── State ─────────────────────────────────────────────── */
let _currentPage = 1;
let _pageSize = 20;
let _totalItems = 0;
let _searchQuery = '';
let _searchTimer = null;

/* ─── Utility ────────────────────────────────────────────── */
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showMsg(elId, text, isOk) {
  const el = document.getElementById(elId);
  el.textContent = text;
  el.className = 'result-msg ' + (isOk ? 'ok' : 'err');
}

function showError(elId, text) {
  const el = document.getElementById(elId);
  el.textContent = text;
  el.classList.remove('hidden');
}

function hideError(elId) {
  document.getElementById(elId).classList.add('hidden');
}

/* ─── List / Pagination ──────────────────────────────────── */

async function loadList(page = 1) {
  _currentPage = page;
  hideError('listError');
  const params = new URLSearchParams({
    search: _searchQuery,
    page: page,
    page_size: _pageSize,
  });
  try {
    const r = await fetch('/api/dict?' + params);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    _totalItems = d.total;
    renderTable(d.items);
    renderPagination(d.page, d.page_size, d.total);
  } catch (e) {
    showError('listError', 'エラー: ' + e.message);
  }
}

function renderTable(items) {
  const tbody = document.getElementById('dictBody');
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#a0aec0;padding:24px">データがありません</td></tr>';
    return;
  }
  tbody.innerHTML = items.map(it => `
    <tr id="row-${it.id}">
      <td><strong>${escHtml(it.surface)}</strong></td>
      <td>${escHtml(it.reading)}</td>
      <td>${escHtml(it.normalized || '')}</td>
      <td><span style="font-size:.78rem;color:#718096">${escHtml(it.pos)}</span></td>
      <td>${it.cost}</td>
      <td><span class="badge-enabled ${it.enabled ? 'on' : 'off'}">${it.enabled ? '有効' : '無効'}</span></td>
      <td>
        <button class="btn-icon" title="編集" onclick="openEditModal(${it.id})">✏️</button>
        <button class="btn-icon" title="削除" onclick="deleteEntry(${it.id}, '${escHtml(it.surface)}')">🗑</button>
      </td>
    </tr>
  `).join('');
}

function renderPagination(page, size, total) {
  const pages = Math.ceil(total / size) || 1;
  document.getElementById('pageInfo').textContent =
    `${page} / ${pages} ページ（全 ${total.toLocaleString()} 件）`;
  document.getElementById('btnPrev').disabled = page <= 1;
  document.getElementById('btnNext').disabled = page >= pages;
}

function changePage(delta) {
  loadList(_currentPage + delta);
}

function onSearchInput() {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    _searchQuery = document.getElementById('searchInput').value;
    loadList(1);
  }, 500);
}

/* ─── Add / Edit Modal ───────────────────────────────────── */

function openAddModal() {
  document.getElementById('modalTitle').textContent = '用語を追加';
  document.getElementById('editId').value = '';
  document.getElementById('fSurface').value = '';
  document.getElementById('fReading').value = '';
  document.getElementById('fNormalized').value = '';
  document.getElementById('fPos').value = '名詞,固有名詞,一般';
  document.getElementById('fCost').value = '5000';
  document.getElementById('fEnabled').value = '1';
  hideError('modalError');
  hideDupWarn();
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById('fSurface').focus();
}

function openEditModal(id) {
  const row = document.getElementById('row-' + id);
  if (!row) return;
  const cells = row.querySelectorAll('td');
  document.getElementById('modalTitle').textContent = '用語を編集';
  document.getElementById('editId').value = id;
  document.getElementById('fSurface').value = cells[0].querySelector('strong').textContent;
  document.getElementById('fReading').value = cells[1].textContent;
  document.getElementById('fNormalized').value = cells[2].textContent;
  document.getElementById('fPos').value = cells[3].querySelector('span').textContent;
  document.getElementById('fCost').value = cells[4].textContent;
  document.getElementById('fEnabled').value =
    cells[5].querySelector('span').classList.contains('on') ? '1' : '0';
  hideError('modalError');
  hideDupWarn();
  document.getElementById('modalBackdrop').classList.add('open');
  document.getElementById('fSurface').focus();
}

function closeModal(e) {
  if (e.target === document.getElementById('modalBackdrop')) closeModalDirect();
}
function closeModalDirect() {
  document.getElementById('modalBackdrop').classList.remove('open');
}

function hideDupWarn() {
  document.getElementById('dupWarn').classList.add('hidden');
}

function showDupWarn(entries) {
  const el = document.getElementById('dupWarn');
  const lines = entries.map(e =>
    `・読み: ${escHtml(e.reading)}　正規化: ${escHtml(e.normalized || '（なし）')}　品詞: ${escHtml(e.pos)}`
  ).join('<br>');
  el.innerHTML = `⚠ この表記はすでに ${entries.length} 件登録されています。内容を確認してください：<br>${lines}`;
  el.classList.remove('hidden');
}

async function checkDuplicateSurface() {
  const surface = document.getElementById('fSurface').value.trim();
  const editId = document.getElementById('editId').value;
  if (!surface) { hideDupWarn(); return; }
  try {
    const r = await fetch('/api/dict/check?' + new URLSearchParams({ surface }));
    const d = await r.json();
    const conflicts = editId
      ? d.entries.filter(e => String(e.id) !== String(editId))
      : d.entries;
    if (conflicts.length > 0) showDupWarn(conflicts);
    else hideDupWarn();
  } catch { /* ネットワークエラーは無視 */ }
}

async function saveEntry() {
  const surface = document.getElementById('fSurface').value.trim();
  const reading = document.getElementById('fReading').value.trim();
  if (!surface) { showError('modalError', '表記は必須です'); return; }
  if (!reading) { showError('modalError', '読みは必須です'); return; }

  const body = {
    surface,
    reading,
    normalized: document.getElementById('fNormalized').value.trim(),
    pos: document.getElementById('fPos').value.trim() || '名詞,固有名詞,一般',
    cost: parseInt(document.getElementById('fCost').value) || 5000,
    enabled: parseInt(document.getElementById('fEnabled').value),
  };

  const editId = document.getElementById('editId').value;
  const isEdit = !!editId;

  // 重複チェック（新規追加時、または表記が変更された場合）
  if (!document.getElementById('dupWarn').classList.contains('hidden')) {
    if (!confirm('同じ表記がすでに登録されています。このまま続けて登録しますか？')) return;
  }

  const url = isEdit ? `/api/dict/${editId}` : '/api/dict';
  const method = isEdit ? 'PUT' : 'POST';

  const btn = document.getElementById('btnModalSave');
  btn.disabled = true;
  hideError('modalError');
  try {
    const r = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const d = await r.json();
      throw new Error(d.detail || r.statusText);
    }
    closeModalDirect();
    loadList(_currentPage);
  } catch (e) {
    showError('modalError', 'エラー: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

/* ─── Delete ─────────────────────────────────────────────── */

async function deleteEntry(id, surface) {
  if (!confirm(`「${surface}」を削除してよろしいですか？`)) return;
  try {
    const r = await fetch(`/api/dict/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    loadList(_currentPage);
  } catch (e) {
    alert('削除エラー: ' + e.message);
  }
}

/* ─── CSV Import / Export ────────────────────────────────── */

async function handleImport(input) {
  const file = input.files[0];
  if (!file) return;
  const label = document.getElementById('importLabel');
  label.classList.add('has-file');
  label.textContent = file.name;

  const formData = new FormData();
  formData.append('file', file);
  try {
    const r = await fetch('/api/dict/import', { method: 'POST', body: formData });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    alert(`${d.imported} 件インポートしました`);
    loadList(1);
  } catch (e) {
    alert('インポートエラー: ' + e.message);
  } finally {
    input.value = '';
    label.classList.remove('has-file');
    label.textContent = 'CSVインポート';
  }
}

async function doExport() {
  try {
    const r = await fetch('/api/dict/export');
    if (!r.ok) throw new Error(r.statusText);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'user_dict.csv';
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('エクスポートエラー: ' + e.message);
  }
}

/* ─── Rebuild / Mode ─────────────────────────────────────── */

async function loadMode() {
  try {
    const r = await fetch('/api/dict/mode');
    const d = await r.json();
    document.getElementById('modeSelect').value = d.mode;
    document.getElementById('modeBadge').textContent = `分割モード: ${d.mode}`;
  } catch { /* ignore */ }
}

async function onModeChange() {
  const mode = document.getElementById('modeSelect').value;
  try {
    const r = await fetch('/api/dict/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    const d = await r.json();
    document.getElementById('modeBadge').textContent = `分割モード: ${d.mode}`;
    showMsg('rebuildMsg', `分割モードを ${d.mode} に変更しました`, true);
  } catch (e) {
    showMsg('rebuildMsg', 'エラー: ' + e.message, false);
  }
}

async function doRebuild() {
  const btn = document.getElementById('btnRebuild');
  const spin = document.getElementById('spinRebuild');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.childNodes[btn.childNodes.length - 1].textContent = ' 構築中...';
  spin.style.display = 'inline-block';
  showMsg('rebuildMsg', '辞書を構築・リロードしています...', true);

  try {
    const r = await fetch('/api/dict/rebuild', { method: 'POST' });
    const d = await r.json();
    if (d.status === 'ok') {
      showMsg('rebuildMsg', `✓ ${d.message}`, true);
    } else {
      showMsg('rebuildMsg', '✗ ' + d.message, false);
    }
  } catch (e) {
    showMsg('rebuildMsg', 'エラー: ' + e.message, false);
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    spin.style.display = 'none';
    // ボタンテキストを復元（最後のテキストノードを設定）
    const nodes = Array.from(btn.childNodes);
    for (const n of nodes) {
      if (n.nodeType === Node.TEXT_NODE) { n.textContent = '辞書を反映'; break; }
    }
  }
}

/* ─── Test ───────────────────────────────────────────────── */

async function doTest() {
  const text = document.getElementById('testInput').value.trim();
  if (!text) return;
  const area = document.getElementById('tokenArea');
  area.textContent = '解析中...';
  try {
    const r = await fetch('/api/dict/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    area.innerHTML = d.tokens
      .map(t => `<span class="token-chip">${escHtml(t)}</span>`)
      .join('');
  } catch (e) {
    area.textContent = 'エラー: ' + e.message;
  }
}

document.getElementById('testInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doTest(); }
});

/* ─── AI候補抽出 ─────────────────────────────────────────── */

let _suggestCandidates = [];

async function doSuggest() {
  const btn = document.getElementById('btnSuggest');
  const spin = document.getElementById('spinSuggest');
  const status = document.getElementById('suggestStatus');
  hideError('suggestError');
  btn.disabled = true;
  spin.style.display = 'inline-block';
  status.textContent = 'AIが引き継ぎノートを分析中です...（数秒〜数十秒かかります）';
  document.getElementById('suggestTableArea').classList.add('hidden');

  try {
    const r = await fetch('/api/dict/suggest', { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);

    _suggestCandidates = d.candidates || [];
    status.textContent =
      `${_suggestCandidates.length} 件の候補を抽出しました（サンプル: ${d.sampled} 件、除外済み: ${d.excluded_existing} 件）`;

    if (_suggestCandidates.length > 0) {
      renderSuggestTable(_suggestCandidates);
      document.getElementById('suggestTableArea').classList.remove('hidden');
      document.getElementById('suggestRegResult').textContent = '';
    }
  } catch (e) {
    showError('suggestError', 'エラー: ' + e.message);
    status.textContent = '';
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
  }
}

function renderSuggestTable(candidates) {
  const tbody = document.getElementById('suggestBody');
  tbody.innerHTML = candidates.map((c, i) => `
    <tr id="sr-${i}">
      <td><input type="checkbox" class="sr-check" data-idx="${i}" checked></td>
      <td><strong>${escHtml(c.surface)}</strong></td>
      <td><input type="text" class="sr-norm" data-idx="${i}" value="${escHtml(c.normalized || '')}" placeholder="${escHtml(c.surface)}"></td>
      <td><input type="text" class="sr-read" data-idx="${i}" value="" placeholder="カタカナで入力（必須）"></td>
      <td style="font-size:.78rem;color:#718096">${escHtml(c.reason || '')}</td>
      <td><button class="btn-icon" title="除外" onclick="removeSuggestRow(${i})">✕</button></td>
    </tr>
  `).join('');
  document.getElementById('checkAll').checked = true;
}

function removeSuggestRow(idx) {
  const row = document.getElementById('sr-' + idx);
  if (row) row.remove();
}

function toggleAllSuggest(cb) {
  document.querySelectorAll('.sr-check').forEach(c => { c.checked = cb.checked; });
}

async function doRegisterSuggest() {
  const resultEl = document.getElementById('suggestRegResult');
  resultEl.textContent = '';

  // 読みが空かチェック
  const checks = document.querySelectorAll('.sr-check:checked');
  const emptyReadings = [];
  checks.forEach(cb => {
    const idx = cb.dataset.idx;
    const reading = document.querySelector(`.sr-read[data-idx="${idx}"]`).value.trim();
    if (!reading) emptyReadings.push(idx);
  });
  if (emptyReadings.length > 0) {
    resultEl.textContent = `⚠ 読みが未入力の行があります（${emptyReadings.length} 件）。読みを入力してから登録してください。`;
    resultEl.style.color = '#b7791f';
    return;
  }

  const entries = [];
  checks.forEach(cb => {
    const idx = cb.dataset.idx;
    const cand = _suggestCandidates[parseInt(idx)] || {};
    const reading = document.querySelector(`.sr-read[data-idx="${idx}"]`).value.trim();
    const normalized = document.querySelector(`.sr-norm[data-idx="${idx}"]`).value.trim();
    entries.push({
      surface: cand.surface || '',
      reading,
      normalized,
      pos: '名詞,固有名詞,一般',
      cost: 5000,
    });
  });

  if (!entries.length) {
    resultEl.textContent = '登録する候補が選択されていません';
    resultEl.style.color = '#718096';
    return;
  }

  const btn = document.getElementById('btnRegisterSuggest');
  btn.disabled = true;
  resultEl.textContent = '登録中...';
  resultEl.style.color = '#718096';

  try {
    const r = await fetch('/api/dict/suggest/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entries }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);

    resultEl.textContent =
      `✓ ${d.registered} 件登録しました（スキップ: ${d.skipped} 件）`;
    resultEl.style.color = '#276749';

    document.getElementById('suggestTableArea').classList.add('hidden');
    document.getElementById('suggestBody').innerHTML = '';
    _suggestCandidates = [];
    loadList(1);
  } catch (e) {
    resultEl.textContent = 'エラー: ' + e.message;
    resultEl.style.color = '#c53030';
  } finally {
    btn.disabled = false;
  }
}

/* ─── Init ───────────────────────────────────────────────── */
loadList(1);
loadMode();
