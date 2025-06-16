const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFolder: () => ipcRenderer.invoke('dialog:selectFolder'),
  
  // Main organization methods
  organizePlan: (dirPath, options = {}) => ipcRenderer.invoke('organize:plan', dirPath, options),
  applyPlan: (itemsToApply) => ipcRenderer.invoke('organize:apply', itemsToApply),
  undoLastMoves: (dirPath, limit = 50) => ipcRenderer.invoke('organize:undo', dirPath, limit),
  getPlanSummary: (dirPath) => ipcRenderer.invoke('organize:summary', dirPath),
  
  // Legacy support for existing functionality
  runOrganizer: (args) => {
    // Parse arguments from command-line style format
    const dirPath = args.find(arg => args[args.indexOf(arg) - 1] === '--path');
    const customPrompt = args.includes('--prompt') ? args[args.indexOf('--prompt') + 1] : null;
    const template = args.includes('--template') ? args[args.indexOf('--template') + 1] : null;
    const includeDuplicates = args.includes('--include-duplicates');
    
    const options = {
      includeDuplicates,
      customPrompt,
      template
    };
    
    return ipcRenderer.invoke('organize:plan', dirPath, options);
  },
  
  generatePlan: (dirPath, options = {}) => ipcRenderer.invoke('organize:plan', dirPath, options),
  
  // Progress handling
  onProgress: (callback) => ipcRenderer.on('organize:progress', (_event, data) => callback(data)),
  onProgressUpdate: (callback) => ipcRenderer.on('organize:progress', (event, ...args) => callback(...args)),
  
  // Utility functions
  openFolder: (folderPath) => ipcRenderer.invoke('shell:openFolder', folderPath),
  
  // Settings / API key management
  settings: {
    getApiKey: () => ipcRenderer.invoke('settings:getApiKey'),
    setApiKey: (key) => ipcRenderer.invoke('settings:setApiKey', key),
    deleteApiKey: () => ipcRenderer.invoke('settings:deleteApiKey'),
    open: () => ipcRenderer.send('settings:open')
  }
}); 