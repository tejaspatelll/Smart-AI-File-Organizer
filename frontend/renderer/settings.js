window.addEventListener('DOMContentLoaded', async () => {
  const apiKeyInput = document.getElementById('apiKeyInput');
  const saveBtn = document.getElementById('saveBtn');
  const deleteBtn = document.getElementById('deleteBtn');
  const closeBtn = document.getElementById('closeBtn');
  const themeSelect = document.getElementById('themeSelect');

  // Load current key
  try {
    const existingKey = await window.electronAPI.settings.getApiKey();
    if (existingKey) apiKeyInput.value = existingKey;
  } catch (_) {}

  const notify = (msg, type = 'info') => {
    try {
      const toast = document.createElement('div');
      toast.className = `toast align-items-center text-white bg-${type} border-0`;
      toast.innerHTML = `<div class="d-flex"><div class="toast-body">${msg}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
      document.body.appendChild(toast);
      new bootstrap.Toast(toast).show();
      toast.addEventListener('hidden.bs.toast', () => toast.remove());
    } catch (_) { alert(msg); }
  };

  saveBtn.addEventListener('click', async () => {
    const newKey = apiKeyInput.value.trim();
    if (!newKey) { notify('Please enter a valid API key.', 'danger'); return; }
    await window.electronAPI.settings.setApiKey(newKey);
    notify('API key saved successfully.', 'success');
  });

  deleteBtn.addEventListener('click', async () => {
    const confirmDelete = confirm('Delete the stored API key?');
    if (!confirmDelete) return;
    await window.electronAPI.settings.deleteApiKey();
    apiKeyInput.value = '';
    notify('API key removed.', 'warning');
  });

  closeBtn.addEventListener('click', () => window.close());

  // Reflect theme to this window immediately
  const applyThemeToDocument = (value) => {
    const html = document.documentElement;
    if (value === 'dark') html.setAttribute('data-theme', 'dark');
    else if (value === 'light') html.removeAttribute('data-theme');
    else {
      const preferDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (preferDark) html.setAttribute('data-theme', 'dark'); else html.removeAttribute('data-theme');
    }
  };

  try {
    const savedTheme = localStorage.getItem('sfo-theme-v2') || 'auto';
    if (themeSelect) themeSelect.value = savedTheme;
    applyThemeToDocument(savedTheme);
  } catch (_) {}

  themeSelect?.addEventListener('change', (e) => {
    const value = e.target.value;
    try { localStorage.setItem('sfo-theme-v2', value); } catch (_) {}
    applyThemeToDocument(value);
    if (window.opener && window.opener.document) {
      const html = window.opener.document.documentElement;
      if (value === 'dark') html.setAttribute('data-theme', 'dark');
      else if (value === 'light') html.removeAttribute('data-theme');
      else {
        const preferDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (preferDark) html.setAttribute('data-theme', 'dark'); else html.removeAttribute('data-theme');
      }
    }
  });
});