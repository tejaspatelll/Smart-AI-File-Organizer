/**
 * Smart File Organizer - Apple-Inspired UI
 * Advanced file organization with custom prompts and templates
 */

class SmartOrganizer {
  constructor() {
    this.currentView = 'welcome';
    this.selectedFolder = null;
    this.organizationPlan = [];
    this.selectedMethod = 'quick';
    this.customPrompt = '';
    this.selectedTemplate = '';
    this.selectedFiles = new Set();
    this.allFiles = [];
    this.isProcessing = false;
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.setupAnimations();
    this.applyPersistedTheme();
    this.initTooltips();
    this.showView('welcome');
  }

  setupEventListeners() {
    // Welcome view interactions
    document.querySelectorAll('.quick-start-card').forEach(card => {
      card.addEventListener('click', (e) => {
        this.selectQuickStartOption(e.currentTarget.dataset.action);
      });
    });

    document.getElementById('get-started-btn').addEventListener('click', () => {
      this.showView('directory');
    });

    // Directory selection
    document.getElementById('select-folder-btn').addEventListener('click', () => {
      this.selectFolder();
    });

    document.getElementById('change-folder-btn').addEventListener('click', () => {
      this.selectFolder();
    });

    // Navigation buttons
    document.getElementById('back-to-welcome-btn').addEventListener('click', () => {
      this.showView('welcome');
    });

    document.getElementById('continue-to-method-btn').addEventListener('click', () => {
      this.showView('method');
    });

    document.getElementById('back-to-directory-btn').addEventListener('click', () => {
      this.showView('directory');
    });

    document.getElementById('back-to-method-btn').addEventListener('click', () => {
      this.showView('method');
    });

    // Method selection
    document.querySelectorAll('#method-tab .nav-link').forEach(tab => {
      tab.addEventListener('click', (e) => {
        this.selectMethod(e.target.dataset.bsTarget.replace('#', '').replace('-method', ''));
      });
    });

    // Custom prompt handling
    document.getElementById('custom-prompt').addEventListener('input', (e) => {
      this.customPrompt = e.target.value;
      this.validateMethodSelection();
    });

    document.querySelectorAll('.prompt-suggestion').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const prompt = e.currentTarget.dataset.prompt;
        document.getElementById('custom-prompt').value = prompt;
        this.customPrompt = prompt;
        this.validateMethodSelection();
      });
    });

    // Template selection
    document.querySelectorAll('.template-card').forEach(card => {
      card.addEventListener('click', (e) => {
        this.selectTemplate(e.currentTarget.dataset.template);
      });
    });

    // Advanced options
    document.getElementById('min-confidence').addEventListener('input', (e) => {
      document.getElementById('confidence-value').textContent = e.target.value;
    });

    // Organization controls
    document.getElementById('start-organization-btn').addEventListener('click', () => {
      this.startOrganization();
    });

    // Results view interactions
    this.setupResultsInteractions();

    // Success view actions
    document.getElementById('view-folder-btn').addEventListener('click', () => {
      this.viewOrganizedFolder();
    });

    document.getElementById('undo-organization-btn').addEventListener('click', () => {
      this.undoOrganization();
    });

    document.getElementById('organize-another-btn').addEventListener('click', () => {
      this.reset();
    });

    // Help and settings
    document.getElementById('help-btn').addEventListener('click', () => {
      this.showHelp();
    });

    document.getElementById('settings-btn').addEventListener('click', () => {
      this.showSettings();
    });

    // Theme toggle
    document.getElementById('theme-toggle-btn')?.addEventListener('click', () => {
      this.toggleTheme();
    });

    // Command palette trigger
    document.getElementById('command-palette-btn')?.addEventListener('click', () => {
      this.openCommandPalette();
    });

    // Sidebar toggle
    const sidebar = document.getElementById('side-drawer');
    const toggleBtn = document.getElementById('sidebar-toggle-btn');
    const closeBtn = document.getElementById('sidebar-close-btn');
    const updateExpanded = () => toggleBtn?.setAttribute('aria-expanded', String(sidebar?.classList.contains('open')));
    toggleBtn?.addEventListener('click', () => {
      sidebar?.classList.toggle('open');
      if (sidebar) sidebar.setAttribute('aria-hidden', sidebar.classList.contains('open') ? 'false' : 'true');
      updateExpanded();
    });
    closeBtn?.addEventListener('click', () => {
      sidebar?.classList.remove('open');
      if (sidebar) sidebar.setAttribute('aria-hidden', 'true');
      updateExpanded();
      toggleBtn?.focus();
    });
  }

  setupResultsInteractions() {
    // Search functionality
    const searchInput = document.getElementById('file-search');
    const clearSearchBtn = document.getElementById('clear-search');
    
    searchInput.addEventListener('input', () => this.applySearch());
    clearSearchBtn.addEventListener('click', () => {
      searchInput.value = '';
      this.applySearch();
    });
    
    // View mode toggle
    document.querySelectorAll('input[name="view-mode"]').forEach(radio => {
      radio.addEventListener('change', (e) => this.switchViewMode(e.target.value));
    });
    
    // Filter controls
    document.querySelectorAll('input[name="priority-filter"]').forEach(radio => {
      radio.addEventListener('change', () => this.applyFilters());
    });

    document.querySelectorAll('input[name="confidence-filter"]').forEach(radio => {
      radio.addEventListener('change', () => this.applyFilters());
    });

    document.getElementById('show-duplicates-new').addEventListener('change', () => {
      this.applyFilters();
    });

    // Quick action handlers
    document.getElementById('approve-high-confidence').addEventListener('click', () => {
      this.approveHighConfidence();
    });
    
    document.getElementById('select-by-type').addEventListener('click', () => {
      this.showFileTypeSelector();
    });
    
    document.getElementById('group-similar').addEventListener('click', () => {
      this.groupSimilarFiles();
    });
    
    document.getElementById('export-plan').addEventListener('click', () => {
      this.exportPlan();
    });

    // Selection controls
    document.getElementById('select-all-new-btn').addEventListener('click', () => {
      this.selectAllFiles();
    });

    document.getElementById('deselect-all-new-btn').addEventListener('click', () => {
      this.deselectAllFiles();
    });

    document.getElementById('select-all-checkbox-new').addEventListener('change', (e) => {
      if (e.target.checked) {
        this.selectAllFiles();
    } else {
        this.deselectAllFiles();
      }
    });

    // Apply organization
    document.getElementById('apply-organization-btn').addEventListener('click', () => {
      this.applyOrganization();
    });
  }

  setupAnimations() {
    // Add smooth transitions between views
    const style = document.createElement('style');
    style.textContent = `
      .view-container {
        transition: opacity 0.3s ease-out, transform 0.3s ease-out;
      }
      
      .view-container.entering {
        opacity: 0;
        transform: translateY(20px);
      }
      
      .view-container.active {
        opacity: 1;
        transform: translateY(0);
      }
      
      .card, .method-card, .template-card, .quick-start-card {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
      }
    `;
    document.head.appendChild(style);
  }

  selectQuickStartOption(action) {
    // Visual feedback
    document.querySelectorAll('.quick-start-card').forEach(card => {
      card.classList.remove('selected');
    });
    
    event.currentTarget.classList.add('selected');
    
    // Set default method based on selection
    this.selectedMethod = action;
    
    // Add smooth transition effect
    setTimeout(() => {
      event.currentTarget.classList.remove('selected');
    }, 1000);
  }

  async selectFolder() {
    try {
      const result = await window.electronAPI.selectFolder();
      if (result && !result.canceled && result.filePaths.length > 0) {
        this.selectedFolder = result.filePaths[0];
        this.displaySelectedFolder();
        document.getElementById('continue-to-method-btn').disabled = false;
      }
    } catch (error) {
      this.showError('Failed to select folder', error.message);
    }
  }

  displaySelectedFolder() {
    if (!this.selectedFolder) return;

    const folderName = this.selectedFolder.split('/').pop() || this.selectedFolder;
    document.getElementById('folder-name').textContent = folderName;
    document.getElementById('folder-path').textContent = this.selectedFolder;
    
    document.getElementById('folder-drop-zone').classList.add('d-none');
    document.getElementById('selected-folder-info').classList.remove('d-none');
  }

  selectMethod(method) {
    this.selectedMethod = method;
    this.validateMethodSelection();
  }

  selectTemplate(template) {
    // Visual feedback
    document.querySelectorAll('.template-card').forEach(card => {
      card.classList.remove('selected');
    });
    
    event.currentTarget.classList.add('selected');
    this.selectedTemplate = template;
    this.validateMethodSelection();
  }

  validateMethodSelection() {
    const startBtn = document.getElementById('start-organization-btn');
    let isValid = false;

    switch (this.selectedMethod) {
      case 'quick':
        isValid = true;
        break;
      case 'custom':
        isValid = this.customPrompt.length > 10;
        break;
      case 'template':
        isValid = this.selectedTemplate !== '';
        break;
      case 'advanced':
        isValid = true;
        break;
    }

    startBtn.disabled = !isValid;
  }

  async startOrganization() {
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    this.showView('processing');
    
    try {
      await this.runOrganizationProcess();
    } catch (error) {
      this.showError('Organization failed', error.message);
      this.showView('method');
    } finally {
      this.isProcessing = false;
    }
  }

  async runOrganizationProcess() {
    try {
      console.log('Starting organization process...');
      console.log('Selected folder:', this.selectedFolder);
      console.log('Selected method:', this.selectedMethod);
      
      // Get organization options based on selected method
      const options = {
        includeDuplicates: true // Always include duplicate detection
      };

      // Add method-specific options
      if (this.selectedMethod === 'custom' && this.customPrompt) {
        options.customPrompt = this.customPrompt.trim();
        console.log('Using custom prompt:', options.customPrompt);
      } else if (this.selectedMethod === 'template' && this.selectedTemplate) {
        options.template = this.selectedTemplate;
        console.log('Using template:', options.template);
      }

      console.log('Organization options:', options);

      this.updateProcessingStatus('Starting organization analysis...', 0);
      
      console.log('Calling electronAPI.organizePlan...');
      const plan = await window.electronAPI.organizePlan(this.selectedFolder, options);
      console.log('Received plan:', plan);
      
      if (!plan || !Array.isArray(plan)) {
        console.error('Invalid plan received:', plan);
        throw new Error('Invalid response from organization service');
      }

      if (plan.length === 0) {
        console.log('Empty plan received - no files to organize');
        this.showNotification('No files need to be organized. Your folder is already well-organized!', 'info');
        this.showView('welcome');
        return;
      }

      console.log(`Organization plan generated with ${plan.length} items`);
      // Attach stable IDs to avoid expensive lookups later
      this.organizationPlan = plan.map((f, i) => ({ ...f, _id: i }));
      this.allFiles = [...this.organizationPlan];
      this.selectedFiles.clear();
      this.allFiles.forEach((_, index) => this.selectedFiles.add(index));

      this.displayResults();
    } catch (error) {
      console.error('Organization failed with error:', error);
      console.error('Error stack:', error.stack);
      
      // Provide user-friendly error messages
      let errorMessage = 'Organization process failed. ';
      if (error.message.includes('GEMINI_API_KEY')) {
        errorMessage += 'AI service is not configured. Please check your API key settings.';
      } else if (error.message.includes('Permission denied')) {
        errorMessage += 'Unable to access the selected folder. Please check folder permissions.';
      } else if (error.message.includes('No such file or directory')) {
        errorMessage += 'The selected folder no longer exists or cannot be accessed.';
      } else if (error.message.includes('network') || error.message.includes('timeout')) {
        errorMessage += 'Network error. Please check your internet connection and try again.';
    } else {
        errorMessage += `Error details: ${error.message}. Please try again or contact support if the problem persists.`;
      }
      
      this.showError('Organization Failed', errorMessage);
      this.showView('method'); // Go back to method selection
    } finally {
      this.isProcessing = false;
    }
  }

  updateProcessingStatus(message, percentage) {
    const progressBar = document.getElementById('processing-progress');
    const statusText = document.getElementById('processing-status');
    const percentageText = document.getElementById('processing-percentage');
    
    progressBar.style.width = `${percentage}%`;
    progressBar.setAttribute('aria-valuenow', percentage);
    statusText.textContent = message;
    percentageText.textContent = `${percentage}%`;
    this.announce(`${message} ${percentage}%`);
  }

  displayResults() {
    // Show loading indicator for large datasets
    if (this.allFiles.length > 1000) {
      this.showTableLoading(true);
    }
    
    // Initialize pagination if needed
    this.currentPage = 1;
    this.itemsPerPage = this.allFiles.length > 5000 ? 100 : 1000;
    this.filteredFiles = [...this.allFiles];
    
    // Render new fast list view
    this.renderFastList();
    // Defer other views until the user switches via view toggle
    // this.populateCardView();
    // this.populateGroupedView();
    
    // Update summary and filters
    this.updatePlanSummary();
    this.recomputeFiltersAndRender();
    this.updateSelectionSummary();
    this.setupPagination();
    
    // Hide loading indicator
    this.showTableLoading(false);
    
    this.showView('results');
  }

  showTableLoading(show) {
    const container = document.querySelector('.results-table-container');
    if (!container) return;
    
    if (show) {
      const loadingDiv = document.createElement('div');
      loadingDiv.id = 'table-loading';
      loadingDiv.className = 'table-loading-overlay';
      loadingDiv.innerHTML = `
        <div class="loading-content">
          <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
          <p class="mt-3 mb-0">Rendering ${this.allFiles.length} files...</p>
        </div>
      `;
      container.appendChild(loadingDiv);
    } else {
      const loadingDiv = document.getElementById('table-loading');
      if (loadingDiv) {
        loadingDiv.remove();
      }
    }
  }

  setupPagination() {
    if (this.filteredFiles.length <= this.itemsPerPage) return;
    
    const totalPages = Math.ceil(this.filteredFiles.length / this.itemsPerPage);
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'pagination-container d-flex justify-content-center align-items-center mt-3';
    paginationContainer.innerHTML = `
      <nav aria-label="Results pagination">
        <ul class="pagination pagination-sm">
          <li class="page-item" id="prev-page">
            <a class="page-link" href="#" aria-label="Previous">
              <span aria-hidden="true">&laquo;</span>
            </a>
          </li>
          <li class="page-item active">
            <span class="page-link" id="current-page">1</span>
          </li>
          <li class="page-item">
            <span class="page-link">of ${totalPages}</span>
          </li>
          <li class="page-item" id="next-page">
            <a class="page-link" href="#" aria-label="Next">
              <span aria-hidden="true">&raquo;</span>
            </a>
          </li>
        </ul>
      </nav>
      <div class="ms-3">
        <select class="form-select form-select-sm" id="items-per-page" style="width: auto;">
          <option value="50" ${this.itemsPerPage === 50 ? 'selected' : ''}>50 per page</option>
          <option value="100" ${this.itemsPerPage === 100 ? 'selected' : ''}>100 per page</option>
          <option value="250" ${this.itemsPerPage === 250 ? 'selected' : ''}>250 per page</option>
          <option value="500" ${this.itemsPerPage === 500 ? 'selected' : ''}>500 per page</option>
          <option value="1000" ${this.itemsPerPage === 1000 ? 'selected' : ''}>1000 per page</option>
        </select>
      </div>
    `;
    
    // Add pagination to results footer
    const resultsFooter = document.querySelector('.results-footer');
    resultsFooter.appendChild(paginationContainer);
    
    // Add event listeners
    document.getElementById('prev-page').addEventListener('click', (e) => {
      e.preventDefault();
      if (this.currentPage > 1) {
        this.currentPage--;
        this.updateTablePage();
      }
    });
    
    document.getElementById('next-page').addEventListener('click', (e) => {
      e.preventDefault();
      const totalPages = Math.ceil(this.filteredFiles.length / this.itemsPerPage);
      if (this.currentPage < totalPages) {
        this.currentPage++;
        this.updateTablePage();
      }
    });
    
    document.getElementById('items-per-page').addEventListener('change', (e) => {
      this.itemsPerPage = parseInt(e.target.value);
      this.currentPage = 1;
      this.updateTablePage();
      this.setupPagination(); // Refresh pagination
    });
  }

  updateTablePage() {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    const end = start + this.itemsPerPage;
    const pageFiles = this.filteredFiles.slice(start, end);
    
    const tbody = document.getElementById('results-table-body');
    tbody.innerHTML = '';
    
    const fragment = document.createDocumentFragment();
    pageFiles.forEach(file => {
      fragment.appendChild(this.createFileRow(file, file._id));
    });
    tbody.appendChild(fragment);
    
    // Update pagination display
    const currentPageSpan = document.getElementById('current-page');
    const totalPages = Math.ceil(this.filteredFiles.length / this.itemsPerPage);
    if (currentPageSpan) {
      currentPageSpan.textContent = this.currentPage;
    }
    
    // Update prev/next button states
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    if (prevBtn) {
      prevBtn.classList.toggle('disabled', this.currentPage === 1);
    }
    if (nextBtn) {
      nextBtn.classList.toggle('disabled', this.currentPage === totalPages);
    }
  }

  populateTableView() { this.renderFastList(); }

  // New: ultra-fast windowed list
  renderFastList() {
    const viewport = document.getElementById('list-viewport');
    if (!viewport) return;
    viewport.innerHTML = '';

    const inner = document.createElement('div');
    inner.className = 'list-inner';
    viewport.appendChild(inner);

    const ROW_HEIGHT = 80;
    const BUFFER = 6;

    const files = this.filteredFiles || [];
    inner.style.height = (files.length * ROW_HEIGHT) + 'px';

    const getVisibleCount = () => Math.ceil((viewport.clientHeight || 600) / ROW_HEIGHT);
    let visible = getVisibleCount();
    let poolSize = Math.min(files.length, visible + BUFFER * 2 + 2);

    const pool = [];
    const buildRow = () => {
      const el = document.createElement('div');
      el.className = 'list-row';
      el.role = 'listitem';
      el.innerHTML = `
        <div class="col checkbox"><input class="form-check-input file-checkbox" type="checkbox"></div>
        <div class="col file file-col">
          <div class="file-info-compact">
            <div class="file-type-icon"></div>
            <div class="file-details">
              <div class="file-name-compact"></div>
            </div>
          </div>
        </div>
        <div class="col priority">
          <div class="priority-info">
            <span class="priority-badge-compact"></span>
            <span class="confidence-compact"></span>
          </div>
        </div>
        <div class="col movement movement-col">
          <div class="movement-flow">
            <div class="from"></div>
            <div class="arrow-compact">→</div>
            <div class="to"></div>
          </div>
        </div>
        <div class="col reasoning reasoning-col"></div>`;
      inner.appendChild(el);
      return el;
    };

    while (pool.length < poolSize) pool.push(buildRow());

    const updateRow = (rowEl, file, rowIndex) => {
      rowEl.style.transform = `translateY(${rowIndex * ROW_HEIGHT}px)`;
      if (rowEl._id === file._id) return;
      rowEl._id = file._id;
      // checkbox
      const cb = rowEl.querySelector('.file-checkbox');
      cb.dataset.index = file._id;
      cb.checked = this.selectedFiles.has(file._id);
      // icon and name
      const iconWrap = rowEl.querySelector('.file-type-icon');
      iconWrap.className = `file-type-icon ${this.getFileTypeClass(file.source)}`;
      iconWrap.innerHTML = `<i class="bi ${this.getFileIcon(file.source)}"></i>`;
      rowEl.querySelector('.file-name-compact').textContent = this.getFileName(file.source);
      // priority/confidence
      const badge = rowEl.querySelector('.priority-badge-compact');
      badge.className = `priority-badge-compact priority-${file.priority}`;
      badge.textContent = this.getPriorityText(file.priority).toUpperCase();
      rowEl.querySelector('.confidence-compact').textContent = `${Math.round(file.confidence * 100)}%`;
      // movement: show full source dir in grey and destination as relative from common root in blue
      const fromEl = rowEl.querySelector('.from');
      const toEl = rowEl.querySelector('.to');
      const sourceDir = this.getDirectoryPath(file.source);
      const destDir = this.getDirectoryPath(file.destination);
      const { relativeDest } = this.computeCommonAndRelative(sourceDir, destDir);
      fromEl.innerHTML = `<span class="path-label">FROM:</span> <span class="path-text path-from" title="${sourceDir}">${sourceDir}</span>`;
      toEl.innerHTML = `<span class="path-label">TO:</span> <span class="path-text path-to" title="${destDir}">${relativeDest}</span>`;
      // reasoning with emoji and better formatting
      const reasoningEl = rowEl.querySelector('.reasoning-col');
      const reasoningText = this.truncateReason(file.reason, 60);
      reasoningEl.innerHTML = `<span class="reasoning-content"><i class="bi bi-robot me-1"></i>${reasoningText}</span>`;
    };

    let start = 0, end = 0;
    const render = (force = false) => {
      const first = Math.max(0, Math.floor(viewport.scrollTop / ROW_HEIGHT) - BUFFER);
      const last = Math.min(files.length, first + visible + BUFFER * 2);
      if (!force && first === start && last === end) return;
      start = first; end = last;
      // ensure pool size
      const desired = Math.min(files.length, visible + BUFFER * 2 + 2);
      while (pool.length < desired) pool.push(buildRow());
      while (pool.length > desired) pool.pop().remove();
      // fill pool
      for (let i = 0; i < pool.length; i++) {
        const rowIndex = start + i;
        const el = pool[i];
        if (rowIndex < end) {
          const file = files[rowIndex];
          el.style.display = '';
          updateRow(el, file, rowIndex);
        } else {
          el.style.display = 'none';
          el._id = undefined;
        }
      }
    };

    // Event delegation for checkboxes
    viewport.onchange = (e) => {
      const t = e.target;
      if (t && t.classList.contains('file-checkbox')) {
        const idx = parseInt(t.dataset.index);
        this.toggleFileSelection(idx, t.checked);
      }
    };

    let ticking = false;
    viewport.addEventListener('scroll', () => {
      if (ticking) return; ticking = true;
      requestAnimationFrame(() => { ticking = false; render(false); });
    }, { passive: true });

    new ResizeObserver(() => { visible = getVisibleCount(); render(true); }).observe(viewport);
    render(true);
  }

  setupVirtualTable() {
    const container = document.querySelector('.results-table-container');
    if (!container) return;

    // Clone the existing table structure so we preserve the sticky header layout
    const baseTable = document.querySelector('.results-table');
    const clonedTable = baseTable ? baseTable.cloneNode(true) : null;
    if (!clonedTable) return;

    // Ensure tbody is empty; we'll render only the visible rows
    const clonedTbody = clonedTable.querySelector('#results-table-body');
    if (clonedTbody) clonedTbody.innerHTML = '';

    // Build the virtual scroller container
    const virtualContainer = document.createElement('div');
    virtualContainer.className = 'virtual-table-container';
    virtualContainer.style.height = '60vh';
    virtualContainer.style.overflowY = 'auto';
    virtualContainer.style.position = 'relative';

    // Spacers simulate the full height
    const topSpacer = document.createElement('div');
    const bottomSpacer = document.createElement('div');

    // Replace content
    container.innerHTML = '';
    virtualContainer.appendChild(topSpacer);
    virtualContainer.appendChild(clonedTable);
    virtualContainer.appendChild(bottomSpacer);
    container.appendChild(virtualContainer);

    // Re-bind header interactions lost during cloning (Select All)
    const selectAllEl = clonedTable.querySelector('#select-all-checkbox-new');
    if (selectAllEl) {
      selectAllEl.addEventListener('change', (e) => {
        if (e.target.checked) this.selectAllFiles(); else this.deselectAllFiles();
      });
    }

    // Virtualization settings with a fixed row pool (recycling)
    const ROW_HEIGHT = 80; // Keep in sync with CSS row height
    const BUFFER_ROWS = 6; // small buffer for smooth scrolling
    let visibleRows = Math.ceil((virtualContainer.clientHeight || 600) / ROW_HEIGHT);
    let startIndex = 0;
    let endIndex = Math.min(this.filteredFiles.length, visibleRows + BUFFER_ROWS);

    const tbody = clonedTable.querySelector('#results-table-body');
    const rowPool = [];

    const ensurePoolSize = (size) => {
      if (!tbody) return;
      while (rowPool.length < size) {
        const placeholder = document.createElement('tr');
        placeholder.style.display = 'none';
        tbody.appendChild(placeholder);
        rowPool.push(placeholder);
      }
      while (rowPool.length > size) {
        const old = rowPool.pop();
        old?.remove();
      }
    };

    const hydrateInitial = () => {
      const poolSize = Math.min(this.filteredFiles.length, visibleRows + BUFFER_ROWS * 2 + 2);
      ensurePoolSize(poolSize);
      for (let k = 0; k < rowPool.length; k++) {
        const i = startIndex + k;
        if (i < endIndex) {
          const row = this.createFileRow(this.filteredFiles[i], this.filteredFiles[i]._id);
          tbody.replaceChild(row, rowPool[k]);
          rowPool[k] = row;
        } else {
          rowPool[k].style.display = 'none';
          rowPool[k].dataset.fileIndex = '';
        }
      }
    };

    const renderWindow = (force = false) => {
      const scrollTop = virtualContainer.scrollTop;
      const newStart = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS);
      const newEnd = Math.min(this.filteredFiles.length, newStart + visibleRows + BUFFER_ROWS * 2);
      if (!force && newStart === startIndex && newEnd === endIndex) return;

      startIndex = newStart;
      endIndex = newEnd;
      topSpacer.style.height = `${startIndex * ROW_HEIGHT}px`;
      bottomSpacer.style.height = `${(this.filteredFiles.length - endIndex) * ROW_HEIGHT}px`;

      const desiredPool = Math.min(this.filteredFiles.length, visibleRows + BUFFER_ROWS * 2 + 2);
      ensurePoolSize(desiredPool);

      for (let k = 0; k < rowPool.length; k++) {
        const i = startIndex + k;
        const current = rowPool[k];
        if (i < endIndex) {
          const file = this.filteredFiles[i];
          // If already rendering this index, just ensure visible
          if (String(current.dataset.fileIndex) !== String(file._id)) {
            const newRow = this.createFileRow(file, file._id);
            if (current.parentNode === tbody) {
              tbody.replaceChild(newRow, current);
            } else {
              tbody.appendChild(newRow);
              current.remove();
            }
            rowPool[k] = newRow;
          }
          rowPool[k].style.display = '';
          rowPool[k].dataset.fileIndex = String(file._id);
        } else {
          current.style.display = 'none';
          current.dataset.fileIndex = '';
        }
      }
    };

    // rAF-based scroll handling for per-frame updates (no trailing debounce lag)
    let isTicking = false;
    virtualContainer.addEventListener('scroll', () => {
      if (isTicking) return;
      isTicking = true;
      requestAnimationFrame(() => {
        isTicking = false;
        renderWindow(false);
      });
    }, { passive: true });

    // Recalculate visible rows on container resize
    const resizeObserver = new ResizeObserver(() => {
      visibleRows = Math.ceil((virtualContainer.clientHeight || 600) / ROW_HEIGHT);
      renderWindow(true);
    });
    try { resizeObserver.observe(virtualContainer); } catch (_) {}

    // Initial render after mounting
    renderWindow(true);
  }

  createFileRow(file, index) {
    const row = document.createElement('tr');
    row.dataset.fileIndex = index;
    row.className = 'file-row';
    
    const confidenceClass = this.getConfidenceClass(file.confidence);
    const priorityClass = `priority-${file.priority}`;
    const fileIcon = this.getFileIcon(file.source);
    const fileName = this.getFileName(file.source);
    const sourcePath = this.getFormattedPath(file.source);
    const destPath = this.getFormattedPath(file.destination);
    const fileTypeClass = this.getFileTypeClass(file.source);
    
    row.innerHTML = `
      <td class="checkbox-cell">
        <input class="form-check-input file-checkbox" type="checkbox" data-index="${index}" checked>
      </td>
      <td class="file-info-cell">
        <div class="file-info-compact">
          <div class="file-type-icon ${fileTypeClass}">
            <i class="bi ${fileIcon}"></i>
          </div>
          <div class="file-details-compact">
            <div class="file-name-compact" title="${fileName}">${fileName}</div>
            ${file.is_duplicate ? '<span class="duplicate-badge-inline">DUPLICATE</span>' : ''}
          </div>
        </div>
      </td>
      <td class="priority-confidence-cell">
        <div class="priority-confidence-compact">
          <span class="priority-badge-compact ${priorityClass}">${this.getPriorityText(file.priority).toUpperCase()}</span>
          <span class="confidence-compact">${Math.round(file.confidence * 100)}%</span>
        </div>
      </td>
      <td class="movement-cell">
        <div class="movement-flow-compact">
          <div class="from-path-compact">
            <span class="path-label-small">FROM:</span>
            <span class="path-text-compact path-from" title="${this.getDirectoryPath(file.source)}">${this.getDirectoryPath(file.source)}</span>
          </div>
          <div class="arrow-compact">→</div>
          <div class="to-path-compact">
            <span class="path-label-small">TO:</span>
            <span class="path-text-compact path-to" title="${this.getDirectoryPath(file.destination)}">${this.computeCommonAndRelative(this.getDirectoryPath(file.source), this.getDirectoryPath(file.destination)).relativeDest}</span>
          </div>
        </div>
      </td>
      <td class="reasoning-cell">
        <div class="reasoning-text-inline" title="${file.reason}">
          ${this.truncateReason(file.reason, 80)}
        </div>
      </td>
    `;
    
    // Add event listeners
    const checkbox = row.querySelector('.file-checkbox');
    checkbox.addEventListener('change', () => {
      this.toggleFileSelection(index, checkbox.checked);
    });
    
    // Add row hover effects
    row.addEventListener('mouseenter', () => {
      row.classList.add('hovered');
    });
    
    row.addEventListener('mouseleave', () => {
      row.classList.remove('hovered');
    });
    
    return row;
  }

  getFileTypeClass(filePath) {
    const ext = filePath.split('.').pop().toLowerCase();
    
    // Images
    if (['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp'].includes(ext)) {
      return 'file-type-image';
    }
    // Documents
    if (['pdf', 'doc', 'docx', 'txt', 'md'].includes(ext)) {
      return 'file-type-document';
    }
    // Media
    if (['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'].includes(ext)) {
      return 'file-type-video';
    }
    if (['mp3', 'wav', 'flac', 'aac', 'm4a'].includes(ext)) {
      return 'file-type-audio';
    }
    // Archives
    if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) {
      return 'file-type-archive';
    }
    // Code
    if (['js', 'ts', 'html', 'css', 'py', 'java', 'cpp', 'c'].includes(ext)) {
      return 'file-type-code';
    }
    
    return 'file-type-other';
  }

  getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  }

  getFileIcon(filePath) {
    const ext = filePath.split('.').pop().toLowerCase();
    
    // Images
    if (['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp'].includes(ext)) {
      return 'bi-image';
    }
    // Documents
    if (['pdf'].includes(ext)) return 'bi-file-pdf';
    if (['doc', 'docx'].includes(ext)) return 'bi-file-word';
    if (['xls', 'xlsx'].includes(ext)) return 'bi-file-excel';
    if (['ppt', 'pptx'].includes(ext)) return 'bi-file-ppt';
    if (['txt', 'md'].includes(ext)) return 'bi-file-text';
    // Media
    if (['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'].includes(ext)) return 'bi-play-circle';
    if (['mp3', 'wav', 'flac', 'aac', 'm4a'].includes(ext)) return 'bi-music-note';
    // Archives
    if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) return 'bi-archive';
    // Code
    if (['js', 'ts', 'html', 'css', 'py', 'java', 'cpp', 'c'].includes(ext)) return 'bi-code-slash';
    // CAD/Design
    if (['dwg', 'dxf', 'ai', 'psd'].includes(ext)) return 'bi-palette';
    
    return 'bi-file-earmark';
  }

  getFileName(filePath) {
    return filePath.split('/').pop();
  }

  getFormattedPath(path) {
    const parts = path.split('/');
    if (parts.length <= 3) return path;
    
    // Show first part, middle ellipsis, and last 2 parts
    const first = parts[0];
    const lastTwo = parts.slice(-2);
    
    return `${first}/.../${lastTwo.join('/')}`;
  }

  getPriorityText(priority) {
    switch (priority) {
      case 5: return 'Critical';
      case 4: return 'High';
      case 3: return 'Medium';
      case 2: return 'Low';
      case 1: return 'Minimal';
      default: return 'Unknown';
    }
  }

  showEnhancedReasoningModal(file, index) {
    const modal = document.getElementById('reasoningModal');
    const fileName = this.getFileName(file.source);
    
    // Populate file details
    const fileDetails = document.getElementById('modal-file-details');
    fileDetails.innerHTML = `
      <strong>File:</strong> ${fileName}<br>
      <strong>Type:</strong> ${file.source.split('.').pop().toUpperCase()}<br>
      <strong>Priority:</strong> ${this.getPriorityText(file.priority)}<br>
      <strong>Confidence:</strong> ${Math.round(file.confidence * 100)}%
      ${file.is_duplicate ? '<br><strong>Status:</strong> <span class="duplicate-badge">DUPLICATE</span>' : ''}
    `;
    
    // Populate reasoning text
    document.getElementById('modal-reasoning-text').textContent = file.reason;
    
    // Populate paths
    document.getElementById('modal-source-path').textContent = this.shortenPath(file.source);
    document.getElementById('modal-destination-path').textContent = this.shortenPath(file.destination);
    
    // Set up approve button
    const approveBtn = document.getElementById('approve-this-move');
    approveBtn.onclick = () => {
      // Check the file in the main table
      const checkbox = document.querySelector(`input[data-index="${index}"]`);
      if (checkbox) {
        checkbox.checked = true;
        this.toggleFileSelection(index, true);
      }
      
      // Close modal
      bootstrap.Modal.getInstance(modal).hide();
      
      // Show notification
      this.showNotification(`File "${fileName}" approved for organization`, 'success');
    };
    
    // Show modal with proper positioning
    const bootstrapModal = new bootstrap.Modal(modal, {
      backdrop: 'static',
      keyboard: true
    });
    bootstrapModal.show();
  }

  shortenPath(path, maxLength = 30) {
    if (path.length <= maxLength) return path;
    
    const parts = path.split('/');
    if (parts.length <= 2) return path;
    
    // For very long paths, show first and last parts
    if (path.length > maxLength * 1.5) {
      return `${parts[0]}/.../${parts.slice(-1).join('/')}`;
    }
    
    // For moderately long paths, show last 2 parts
    return `.../${parts.slice(-2).join('/')}`;
  }

  getDirectoryPath(filePath) {
    const parts = filePath.split('/');
    const fileName = parts.pop(); // Remove filename
    const dirPath = parts.join('/');
    
    // Show meaningful directory path, not just filename
    if (dirPath.length > 40) {
      const pathParts = parts;
      if (pathParts.length > 3) {
        return `${pathParts[0]}/.../${pathParts.slice(-2).join('/')}`;
      }
    }
    
    return dirPath || '/';
  }

  // Compute common root and relative destination path for clearer display
  computeCommonAndRelative(sourceDir, destDir) {
    try {
      const srcParts = sourceDir.split('/').filter(Boolean);
      const dstParts = destDir.split('/').filter(Boolean);
      const len = Math.min(srcParts.length, dstParts.length);
      let i = 0;
      while (i < len && srcParts[i] === dstParts[i]) i++;
      const commonRoot = '/' + srcParts.slice(0, i).join('/');
      const relativeDest = '/' + dstParts.slice(i).join('/');
      return { commonRoot: commonRoot === '' ? '/' : commonRoot, relativeDest: relativeDest === '/' ? '/' : relativeDest };
    } catch (_) {
      return { commonRoot: '/', relativeDest: destDir };
    }
  }

  truncateReason(reason, maxLength = 80) {
    if (!reason || reason.length <= maxLength) return reason || 'AI analysis not available';
    return reason.substring(0, maxLength - 3) + '...';
  }

  updatePlanSummary() {
    const summary = this.calculateSummary();
    
    // Update individual stat elements efficiently (no innerHTML replacement)
    const totalFilesEl = document.getElementById('total-files-display');
    const categoriesEl = document.getElementById('categories-display');
    const duplicatesEl = document.getElementById('duplicates-display');
    const confidenceEl = document.getElementById('confidence-display');
    
    if (totalFilesEl) totalFilesEl.textContent = summary.totalFiles;
    if (categoriesEl) categoriesEl.textContent = summary.categories;
    if (duplicatesEl) duplicatesEl.textContent = summary.duplicates;
    if (confidenceEl) confidenceEl.textContent = `${Math.round(summary.avgConfidence * 100)}%`;
    
    // Also update header stat
    const headerStatEl = document.getElementById('total-files-stat');
    if (headerStatEl) headerStatEl.textContent = summary.totalFiles;
  }

  calculateSummary() {
    const categories = new Set();
    let duplicates = 0;
    let totalConfidence = 0;
    
    this.allFiles.forEach(file => {
      const parts = file.destination.split('/');
      if (parts.length > 0) categories.add(parts[0]);
      if (file.is_duplicate) duplicates++;
      totalConfidence += file.confidence;
    });
    
    return {
      totalFiles: this.allFiles.length,
      categories: categories.size,
      duplicates,
      avgConfidence: this.allFiles.length > 0 ? totalConfidence / this.allFiles.length : 0
    };
  }

  applyFilters() {
    this.recomputeFiltersAndRender();
  }

  recomputeFiltersAndRender() {
    const priorityFilter = document.querySelector('input[name="priority-filter"]:checked')?.value || 'all';
    const confidenceFilter = document.querySelector('input[name="confidence-filter"]:checked')?.value || 'all';
    const showDuplicates = document.getElementById('show-duplicates-new')?.checked ?? true;
    const searchTerm = document.getElementById('file-search')?.value.toLowerCase().trim() || '';

    // Start from all files and apply search + filters together
    this.filteredFiles = this.allFiles.filter(file => {
      // Search
      if (searchTerm) {
        const fileName = this.getFileName(file.source).toLowerCase();
        const sourcePath = file.source.toLowerCase();
        const destPath = file.destination.toLowerCase();
        const reason = (file.reason || '').toLowerCase();
        const matchesSearch = fileName.includes(searchTerm) || sourcePath.includes(searchTerm) || destPath.includes(searchTerm) || reason.includes(searchTerm);
        if (!matchesSearch) return false;
      }

      // Priority filter
      if (priorityFilter !== 'all') {
        if (priorityFilter === 'high' && !(file.priority >= 4)) return false;
        if (priorityFilter === 'medium' && !(file.priority === 3)) return false;
        if (priorityFilter === 'low' && !(file.priority <= 2)) return false;
      }

      // Confidence filter
      if (confidenceFilter !== 'all') {
        if (confidenceFilter === 'high' && !(file.confidence >= 0.8)) return false;
        if (confidenceFilter === 'medium' && !(file.confidence >= 0.5 && file.confidence < 0.8)) return false;
        if (confidenceFilter === 'low' && !(file.confidence < 0.5)) return false;
      }

      // Duplicates filter
      if (!showDuplicates && file.is_duplicate) return false;

      return true;
    });

    // Reset to first page and update display
    this.currentPage = 1;

    // Always use fast list renderer now
    this.renderFastList();

    this.updateSelectionSummary();
  }

  updatePaginationInfo() {
    const totalPages = Math.ceil(this.filteredFiles.length / this.itemsPerPage);
    const paginationInfo = document.querySelector('.pagination-container .page-item:nth-child(3) .page-link');
    if (paginationInfo) {
      paginationInfo.textContent = `of ${totalPages}`;
    }
  }

  toggleFileSelection(index, selected) {
    if (selected) {
      this.selectedFiles.add(index);
    } else {
      this.selectedFiles.delete(index);
    }
    
    this.updateSelectionSummary();
    this.updateApplyButton();
  }

  selectAllFiles() {
    // Select all currently filtered files
    this.filteredFiles.forEach(file => {
      this.selectedFiles.add(file._id);
    });
    // Reflect in UI checkboxes that are rendered
    this.updateAllCheckboxes();
    
    document.getElementById('select-all-checkbox-new').checked = true;
    this.updateSelectionSummary();
    this.updateApplyButton();
  }

  deselectAllFiles() {
    const checkboxes = document.querySelectorAll('.file-checkbox');
    checkboxes.forEach(checkbox => {
      checkbox.checked = false;
    });
    
    this.selectedFiles.clear();
    document.getElementById('select-all-checkbox-new').checked = false;
    this.updateSelectionSummary();
    this.updateApplyButton();
  }

  updateSelectionSummary() {
    const selectedCount = this.selectedFiles.size;
    const totalVisible = this.filteredFiles ? this.filteredFiles.length : 0;
    
    const summaryText = selectedCount > 0 
      ? `${selectedCount} of ${totalVisible} files selected`
      : 'No files selected';
    
    document.getElementById('selection-summary').textContent = summaryText;
  }

  updateApplyButton() {
    const applyBtn = document.getElementById('apply-organization-btn');
    applyBtn.disabled = this.selectedFiles.size === 0;
  }

  async applyOrganization() {
    if (this.selectedFiles.size === 0) return;
    
    // Prepare the plan for selected files
    const selectedPlan = Array.from(this.selectedFiles).map(index => ({
      source: this.allFiles[index].source,
      destination: this.allFiles[index].destination,
      reason: this.allFiles[index].reason
    }));
    
    try {
      const result = await window.electronAPI.applyPlan(selectedPlan);
      
      if (result.success) {
        this.showSuccessView(selectedPlan.length);
      } else {
        this.showError('Failed to apply organization', result.error);
      }
    } catch (error) {
      this.showError('Error applying organization', error.message);
    }
  }

  showSuccessView(filesOrganized) {
    // Update success statistics
    document.getElementById('files-moved-stat').textContent = filesOrganized;
    document.getElementById('folders-created-stat').textContent = this.calculateFoldersCreated();
    document.getElementById('duplicates-handled-stat').textContent = this.calculateDuplicatesHandled();
    document.getElementById('confidence-avg-stat').textContent = `${Math.round(this.calculateAverageConfidence() * 100)}%`;
    
    this.showView('success');
  }

  calculateFoldersCreated() {
    const folders = new Set();
    this.selectedFiles.forEach(index => {
      const file = this.allFiles[index];
      const folderPath = file.destination.substring(0, file.destination.lastIndexOf('/'));
      folders.add(folderPath);
    });
    return folders.size;
  }

  calculateDuplicatesHandled() {
    let count = 0;
    this.selectedFiles.forEach(index => {
      if (this.allFiles[index].is_duplicate) count++;
    });
    return count;
  }

  calculateAverageConfidence() {
    let total = 0;
    this.selectedFiles.forEach(index => {
      total += this.allFiles[index].confidence;
    });
    return this.selectedFiles.size > 0 ? total / this.selectedFiles.size : 0;
  }

  async viewOrganizedFolder() {
    try {
      await window.electronAPI.openFolder(this.selectedFolder);
    } catch (error) {
      this.showError('Failed to open folder', error.message);
    }
  }

  async undoOrganization() {
    try {
      const result = await window.electronAPI.undoLastMoves();
      if (result.success) {
        this.showNotification('Successfully undid recent file moves', 'success');
      } else {
        this.showError('Failed to undo moves', result.error);
      }
    } catch (error) {
      this.showError('Error undoing moves', error.message);
    }
  }

  reset() {
    this.currentView = 'welcome';
    this.selectedFolder = null;
    this.organizationPlan = [];
    this.selectedMethod = 'quick';
    this.customPrompt = '';
    this.selectedTemplate = '';
    this.selectedFiles.clear();
    this.allFiles = [];
    
    // Reset UI elements
    document.getElementById('continue-to-method-btn').disabled = true;
    document.getElementById('custom-prompt').value = '';
    document.querySelectorAll('.template-card').forEach(card => {
      card.classList.remove('selected');
    });
    
    this.showView('welcome');
  }

  showView(viewName) {
    const views = document.querySelectorAll('.view-container');
    
    // Hide all views
    views.forEach(view => {
      view.classList.add('d-none');
    });
    
    // Show target view with animation
    const targetView = document.getElementById(`${viewName}-view`);
    if (targetView) {
      targetView.classList.remove('d-none');
      targetView.classList.add('entering');
      
      // Trigger animation
      setTimeout(() => {
        targetView.classList.remove('entering');
        targetView.classList.add('active');
      }, 50);
      
      this.currentView = viewName;
    }
  }

  showError(title, message) {
    // Create a modern error notification
    const errorToast = document.createElement('div');
    errorToast.className = 'toast align-items-center text-white bg-danger border-0';
    errorToast.setAttribute('role', 'status');
    errorToast.setAttribute('aria-live', 'assertive');
    errorToast.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">
          <strong>${title}</strong><br>
          ${message}
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    `;
    
    document.body.appendChild(errorToast);
    const toast = new bootstrap.Toast(errorToast);
    toast.show();
    
    // Remove element after toast is hidden
    errorToast.addEventListener('hidden.bs.toast', () => {
      errorToast.remove();
    });
  }

  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `toast align-items-center text-white bg-${type} border-0`;
    notification.setAttribute('role', 'status');
    notification.setAttribute('aria-live', 'polite');
    notification.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    `;
    
    document.body.appendChild(notification);
    const toast = new bootstrap.Toast(notification);
    toast.show();
    
    notification.addEventListener('hidden.bs.toast', () => {
      notification.remove();
    });
  }

  showHelp() {
    const helpContent = `
      <h6>How Smart Organizer Works</h6>
      <p>Smart Organizer uses AI to understand your files and create intelligent organization structures.</p>
      
      <h6>Organization Methods</h6>
      <ul>
        <li><strong>Quick & Smart:</strong> AI automatically organizes your files</li>
        <li><strong>Custom Prompt:</strong> Describe exactly how you want files organized</li>
        <li><strong>Templates:</strong> Use proven organization patterns</li>
        <li><strong>Advanced:</strong> Fine-tune all settings</li>
      </ul>
      
      <h6>Tips for Best Results</h6>
      <ul>
        <li>Use descriptive prompts for custom organization</li>
        <li>Review the preview before applying changes</li>
        <li>Keep backups of important files</li>
        <li>Use undo feature if needed</li>
      </ul>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('helpModal'));
    document.querySelector('#helpModal .modal-body').innerHTML = helpContent;
    modal.show();
  }

  showSettings() {
    const settingsContent = `
      <div class="mb-3">
        <label class="form-label">Default Organization Method</label>
        <select class="form-select">
          <option value="quick">Quick & Smart</option>
          <option value="custom">Custom Prompt</option>
          <option value="template">Templates</option>
        </select>
      </div>
      
      <div class="mb-3">
        <label class="form-label">Default Confidence Threshold</label>
        <input type="range" class="form-range" min="0" max="100" value="50">
      </div>
      
      <div class="form-check">
        <input class="form-check-input" type="checkbox" checked>
        <label class="form-check-label">Enable duplicate detection</label>
      </div>
      
      <div class="form-check">
        <input class="form-check-input" type="checkbox" checked>
        <label class="form-check-label">Preserve existing folder structure</label>
      </div>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('settingsModal'));
    document.querySelector('#settingsModal .modal-body').innerHTML = settingsContent;
    modal.show();
  }

  // Theme management
  applyPersistedTheme() {
    try {
      const saved = localStorage.getItem('sfo-theme-v2') || 'auto';
      this.themeMode = saved; // 'auto' | 'light' | 'dark'
      this.applyThemeMode(saved);
      // listen for system changes when in auto mode
      this.mql = window.matchMedia('(prefers-color-scheme: dark)');
      this.mql.addEventListener?.('change', () => {
        if (this.themeMode === 'auto') this.applyThemeMode('auto');
      });
    } catch (_) {}
  }

  toggleTheme() {
    const next = this.themeMode === 'auto' ? 'light' : this.themeMode === 'light' ? 'dark' : 'auto';
    this.themeMode = next;
    this.applyThemeMode(next);
    try { localStorage.setItem('sfo-theme-v2', next); } catch (_) {}
  }

  applyThemeMode(mode) {
    const label = document.getElementById('theme-label');
    if (mode === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      if (label) label.textContent = 'Dark';
    } else if (mode === 'light') {
      document.documentElement.removeAttribute('data-theme');
      if (label) label.textContent = 'Light';
    } else {
      // auto
      const preferDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (preferDark) {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      if (label) label.textContent = 'Auto';
    }
    this.announce(`Theme: ${mode}`);
  }

  initTooltips() {
    try {
      document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
    } catch (_) {}
  }

  announce(message) {
    const region = document.getElementById('live-region');
    if (!region) return;
    region.textContent = '';
    setTimeout(() => { region.textContent = message; }, 50);
  }

  // Command palette
  openCommandPalette() {
    const modalEl = document.getElementById('commandPalette');
    const listEl = document.getElementById('command-list');
    const inputEl = document.getElementById('command-search');
    if (!modalEl || !listEl || !inputEl) return;

    const commands = this.getCommands();
    let activeIndex = 0;

    const render = (filter = '') => {
      listEl.innerHTML = '';
      const filtered = commands.filter(c => c.label.toLowerCase().includes(filter.toLowerCase()));
      filtered.forEach((cmd, idx) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'list-group-item list-group-item-action' + (idx === activeIndex ? ' active' : '');
        item.innerHTML = `<span><i class="bi ${cmd.icon} me-2"></i>${cmd.label}</span><span class="hint">${cmd.hint || ''}</span>`;
        item.addEventListener('click', () => {
          cmd.run();
          bsModal.hide();
        });
        listEl.appendChild(item);
      });
    };

    render('');
    const bsModal = new bootstrap.Modal(modalEl, { backdrop: 'static', keyboard: true });
    bsModal.show();
    setTimeout(() => inputEl.focus(), 120);

    const keyHandler = (e) => {
      const items = Array.from(listEl.querySelectorAll('.list-group-item'));
      if (e.key === 'ArrowDown') {
        activeIndex = Math.min(activeIndex + 1, items.length - 1);
        e.preventDefault();
      } else if (e.key === 'ArrowUp') {
        activeIndex = Math.max(activeIndex - 1, 0);
        e.preventDefault();
      } else if (e.key === 'Enter') {
        items[activeIndex]?.click();
        e.preventDefault();
      }
      items.forEach((it, i) => it.classList.toggle('active', i === activeIndex));
    };

    const inputHandler = (e) => {
      activeIndex = 0;
      render(e.target.value || '');
    };

    inputEl.addEventListener('input', inputHandler);
    modalEl.addEventListener('keydown', keyHandler);
    modalEl.addEventListener('hidden.bs.modal', () => {
      inputEl.removeEventListener('input', inputHandler);
      modalEl.removeEventListener('keydown', keyHandler);
    }, { once: true });
  }

  getCommands() {
    return [
      { label: 'Select Folder…', icon: 'bi-folder2-open', hint: 'Enter', run: () => this.selectFolder() },
      { label: 'Start Organization', icon: 'bi-play-circle', hint: 'Enter', run: () => this.startOrganization() },
      { label: 'Toggle Theme', icon: 'bi-moon-stars', hint: 'T', run: () => this.toggleTheme() },
      { label: 'Show Help', icon: 'bi-question-circle', run: () => this.showHelp() },
      { label: 'Open Settings', icon: 'bi-gear', run: () => this.showSettings() },
      { label: 'View: Table', icon: 'bi-table', run: () => this.switchViewMode('table') },
      { label: 'View: Cards', icon: 'bi-grid', run: () => this.switchViewMode('cards') },
      { label: 'View: Grouped', icon: 'bi-collection', run: () => this.switchViewMode('grouped') },
      { label: 'Approve High Confidence', icon: 'bi-check-circle', run: () => this.approveHighConfidence() },
      { label: 'Export Plan', icon: 'bi-download', run: () => this.exportPlan() },
    ];
  }
}

// Enhanced progress handling for Electron communication
window.electronAPI.onProgress?.((progressData) => {
  if (organizer && organizer.isProcessing) {
    const { percentage, message, stage } = progressData;
    organizer.updateProcessingStatus(message, percentage);
    
    // Update processing details based on stage
    if (stage === 'scan') {
      document.getElementById('files-scanned').textContent = progressData.current || 0;
    } else if (stage === 'classify') {
      document.getElementById('files-analyzed').textContent = progressData.current || 0;
    }
  }
});

// Enhanced Search and Filter Methods
SmartOrganizer.prototype.applySearch = function() {
  // Search now recomputed together with other filters
  this.recomputeFiltersAndRender();
};

SmartOrganizer.prototype.switchViewMode = function(mode) {
  // Hide all view modes
  document.querySelectorAll('.view-mode').forEach(view => {
    view.classList.add('d-none');
    view.classList.remove('active');
  });
  
  // Show selected view mode
  const selectedView = document.getElementById(`${mode}-view`);
  if (selectedView) {
    selectedView.classList.remove('d-none');
    selectedView.classList.add('active');
    
    // Populate the view with data
    this.populateViewMode(mode);
  }
};

SmartOrganizer.prototype.populateViewMode = function(mode) {
  switch (mode) {
    case 'table':
      this.renderFastList();
      break;
    case 'cards':
      this.populateCardView();
      break;
    case 'grouped':
      this.populateGroupedView();
      break;
  }
};

SmartOrganizer.prototype.populateCardView = function() {
  const container = document.getElementById('results-cards-container');
  container.innerHTML = '';
  
  this.allFiles.forEach((file, index) => {
    const card = this.createFileCard(file, index);
    container.appendChild(card);
  });
};

SmartOrganizer.prototype.populateGroupedView = function() {
  const container = document.getElementById('results-groups-container');
  container.innerHTML = '';
  
  // Group files by category
  const groups = {};
  this.allFiles.forEach((file, index) => {
    const category = this.getFileCategory(file);
    if (!groups[category]) {
      groups[category] = [];
    }
    groups[category].push({ file, index });
  });
  
  // Create groups
  Object.entries(groups).forEach(([category, files]) => {
    const groupElement = this.createCategoryGroup(category, files);
    container.appendChild(groupElement);
  });
};

SmartOrganizer.prototype.createFileCard = function(file, index) {
  const card = document.createElement('div');
  card.className = 'col-md-6 col-lg-4';
  card.innerHTML = `
    <div class="file-card" data-index="${index}">
      <div class="file-card-header">
        <div class="file-info">
          <input class="form-check-input file-checkbox" type="checkbox" data-index="${index}" checked>
          <div class="file-type-icon ${this.getFileTypeClass(file.source)}">
            <i class="bi ${this.getFileIcon(file.source)}"></i>
          </div>
          <div class="file-details-compact">
            <div class="file-name-compact" title="${this.getFileName(file.source)}">${this.getFileName(file.source)}</div>
            ${file.is_duplicate ? '<span class="duplicate-badge-inline">DUPLICATE</span>' : ''}
          </div>
        </div>
        <div class="priority-badge-compact ${this.getPriorityText(file.priority)}">
          ${this.getPriorityText(file.priority).toUpperCase()}
        </div>
      </div>
      <div class="file-card-body">
        <div class="priority-confidence-compact mb-2">
          <span class="confidence-compact">${Math.round(file.confidence * 100)}% Confidence</span>
        </div>
        <div class="movement-flow-compact">
          <div class="from-path-compact">
            <span class="path-label-small">FROM:</span>
            <span class="path-text-compact" title="${file.source}">${file.source}</span>
          </div>
          <div class="arrow-compact">→</div>
          <div class="to-path-compact">
            <span class="path-label-small">TO:</span>
            <span class="path-text-compact" title="${file.destination}">${file.destination}</span>
          </div>
        </div>
        <div class="reasoning-text-inline mt-3" title="${file.reason}">
          ${this.truncateReason(file.reason, 100)}
        </div>
      </div>
    </div>
  `;
  
  // Add event listener for checkbox
  const checkbox = card.querySelector('.file-checkbox');
  checkbox.addEventListener('change', () => {
    this.toggleFileSelection(index, checkbox.checked);
  });
  
  return card;
};

SmartOrganizer.prototype.createCategoryGroup = function(category, files) {
  const group = document.createElement('div');
  group.className = 'category-group mb-4';
  
  const filesHtml = files.map(({ file, index }) => this.createCategoryFileItem(file, index)).join('');
  
  group.innerHTML = `
    <div class="category-header">
      <div class="d-flex align-items-center">
        <i class="bi ${this.getCategoryIcon(category)} me-3 fs-4 text-primary"></i>
        <h4 class="category-title mb-0">${category}</h4>
      </div>
      <span class="badge bg-secondary-light text-secondary-dark rounded-pill">${files.length} files</span>
    </div>
    <div class="category-content">
      ${filesHtml}
    </div>
  `;
  
  // Add event listeners to new elements
  group.querySelectorAll('.file-checkbox').forEach(checkbox => {
    checkbox.addEventListener('change', (e) => {
      const index = parseInt(e.target.dataset.index);
      this.toggleFileSelection(index, e.target.checked);
    });
  });
  
  return group;
};

SmartOrganizer.prototype.createCategoryFileItem = function(file, index) {
  return `
    <div class="category-file-item" data-index="${index}">
      <input class="form-check-input file-checkbox" type="checkbox" data-index="${index}" checked>
      <div class="file-info ms-2">
        <div class="file-type-icon ${this.getFileTypeClass(file.source)} me-2">
          <i class="bi ${this.getFileIcon(file.source)}"></i>
        </div>
        <span class="file-name">${this.getFileName(file.source)}</span>
        ${file.is_duplicate ? '<span class="duplicate-badge ms-2">DUPLICATE</span>' : ''}
      </div>
      <div class="ms-auto d-flex align-items-center">
        <div class="confidence-badge confidence-${this.getConfidenceClass(file.confidence)} me-3">
          ${Math.round(file.confidence * 100)}% Conf.
        </div>
        <div class="reasoning-text-inline" title="${file.reason}">
          ${this.truncateReason(file.reason, 60)}
        </div>
      </div>
    </div>
  `;
};

SmartOrganizer.prototype.getFileCategory = function(file) {
  const ext = file.source.split('.').pop().toLowerCase();
  
  if (['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp'].includes(ext)) {
    return 'Images';
  }
  if (['pdf', 'doc', 'docx', 'txt', 'md'].includes(ext)) {
    return 'Documents';
  }
  if (['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'].includes(ext)) {
    return 'Videos';
  }
  if (['mp3', 'wav', 'flac', 'aac', 'm4a'].includes(ext)) {
    return 'Audio';
  }
  if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) {
    return 'Archives';
  }
  if (['js', 'ts', 'html', 'css', 'py', 'java', 'cpp', 'c'].includes(ext)) {
    return 'Code';
  }
  
  return 'Other';
};

SmartOrganizer.prototype.getCategoryIcon = function(category) {
  const icons = {
    'Images': 'bi-image',
    'Documents': 'bi-file-text',
    'Videos': 'bi-play-circle',
    'Audio': 'bi-music-note',
    'Archives': 'bi-archive',
    'Code': 'bi-code-slash',
    'Other': 'bi-file-earmark'
  };
  return icons[category] || 'bi-file-earmark';
};

// Quick Action Methods
SmartOrganizer.prototype.approveHighConfidence = function() {
  this.selectedFiles.clear();
  this.allFiles.forEach((file, index) => {
    if (file.confidence >= 0.8) {
      this.selectedFiles.add(index);
    }
  });
  this.updateAllCheckboxes();
  this.updateSelectionSummary();
  this.showNotification(`Selected ${this.selectedFiles.size} high-confidence files`, 'success');
};

SmartOrganizer.prototype.showFileTypeSelector = function() {
  const categories = {};
  this.allFiles.forEach(file => {
    const category = this.getFileCategory(file);
    categories[category] = (categories[category] || 0) + 1;
  });
  
  let modal = document.getElementById('file-type-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'file-type-modal';
    modal.className = 'modal fade';
    modal.innerHTML = `
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Select by File Type</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
          </div>
          <div class="modal-body">
            <div id="file-type-list">
              ${Object.entries(categories).map(([category, count]) => `
                <div class="form-check">
                  <input class="form-check-input" type="checkbox" id="type-${category}" value="${category}">
                  <label class="form-check-label" for="type-${category}">
                    <i class="bi ${this.getCategoryIcon(category)} me-2"></i>
                    ${category} (${count} files)
                  </label>
                </div>
              `).join('')}
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="button" class="btn btn-primary" onclick="organizer.selectFileTypes()">Select Files</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }
  
  const bootstrapModal = new bootstrap.Modal(modal);
  bootstrapModal.show();
};

SmartOrganizer.prototype.selectFileTypes = function() {
  const selectedTypes = Array.from(document.querySelectorAll('#file-type-list input:checked')).map(cb => cb.value);
  
  this.selectedFiles.clear();
  this.allFiles.forEach((file, index) => {
    const category = this.getFileCategory(file);
    if (selectedTypes.includes(category)) {
      this.selectedFiles.add(index);
    }
  });
  
  this.updateAllCheckboxes();
  this.updateSelectionSummary();
  
  // Close modal
  bootstrap.Modal.getInstance(document.getElementById('file-type-modal')).hide();
  this.showNotification(`Selected ${this.selectedFiles.size} files by type`, 'success');
};

SmartOrganizer.prototype.groupSimilarFiles = function() {
  // Switch to grouped view
  document.getElementById('view-grouped').checked = true;
  this.switchViewMode('grouped');
  this.showNotification('Files grouped by category', 'info');
};

SmartOrganizer.prototype.exportPlan = function() {
  const planData = {
    timestamp: new Date().toISOString(),
    folder: this.selectedFolder,
    totalFiles: this.allFiles.length,
    selectedFiles: this.selectedFiles.size,
    plan: this.allFiles.map((file, index) => ({
      ...file,
      selected: this.selectedFiles.has(index)
    }))
  };
  
  const blob = new Blob([JSON.stringify(planData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `organization-plan-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
  
  this.showNotification('Organization plan exported', 'success');
};

SmartOrganizer.prototype.updateAllCheckboxes = function() {
  document.querySelectorAll('.file-checkbox').forEach(checkbox => {
    const index = parseInt(checkbox.dataset.index);
    checkbox.checked = this.selectedFiles.has(index);
  });
};

// Initialize the Smart Organizer when page loads
let organizer;
document.addEventListener('DOMContentLoaded', () => {
  organizer = new SmartOrganizer();
});

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
  const target = e.target;
  const typingContext = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);

  // Ignore most shortcuts while typing, but still allow Esc
  const allowWhileTyping = e.key === 'Escape';
  if (typingContext && !allowWhileTyping) return;
  // Escape key to go back
  if (e.key === 'Escape' && organizer) {
    switch (organizer.currentView) {
      case 'directory':
        organizer.showView('welcome');
        break;
      case 'method':
        organizer.showView('directory');
        break;
      case 'results':
        organizer.showView('method');
        break;
    }
  }
  
  // Enter key to proceed
  if (e.key === 'Enter' && organizer) {
    const activeButton = document.querySelector('button:not([disabled]):focus, button.btn-primary:not([disabled])');
    if (activeButton) {
      activeButton.click();
    }
  }

  // Command palette shortcut: Cmd/Ctrl+K
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    organizer?.openCommandPalette();
    return;
  }

  // Theme toggle: "t"
  if (!e.metaKey && !e.ctrlKey && e.key.toLowerCase() === 't') {
    e.preventDefault();
    organizer?.toggleTheme();
    return;
  }

  // Help: "?" or F1
  if ((!e.metaKey && !e.ctrlKey && e.shiftKey && e.key === '?') || e.key === 'F1') {
    e.preventDefault();
    organizer?.showHelp();
    return;
  }

  // Open Settings: Cmd/Ctrl+,
  if ((e.metaKey || e.ctrlKey) && e.key === ',') {
    e.preventDefault();
    window.electronAPI.settings.open();
    return;
  }

  // View mode shortcuts when on results view
  if (organizer && organizer.currentView === 'results' && !e.metaKey && !e.ctrlKey) {
    if (e.key === '1') { organizer.switchViewMode('table'); return; }
    if (e.key === '2') { organizer.switchViewMode('cards'); return; }
    if (e.key === '3') { organizer.switchViewMode('grouped'); return; }
  }
}); 