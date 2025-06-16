const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const Store = require('electron-store');
const prompt = require('electron-prompt');

// Persistent store for configuration such as API keys
const store = new Store({
  name: 'smart-file-organizer',
  encryptionKey: undefined // Optionally set an encryption key
});

async function ensureApiKey() {
  let apiKey = store.get('geminiApiKey');
  if (!apiKey) {
    const result = await prompt({
      title: 'Gemini API Key Required',
      label: 'Enter your Gemini API Key:',
      inputAttrs: { type: 'password', placeholder: 'AIza...' },
      height: 150,
      width: 500,
      type: 'input',
      resizable: false,
    });

    if (result === null || result.trim() === '') {
      dialog.showErrorBox('API Key Required', 'A valid Gemini API Key is required for Smart File Organizer to function. The application will now quit.');
      app.quit();
      return false;
    }

    apiKey = result.trim();
    store.set('geminiApiKey', apiKey);
  }

  // Expose to backend process via environment variable
  process.env.GEMINI_API_KEY = apiKey;
  return true;
}

function createSettingsWindow() {
  const settingsWin = new BrowserWindow({
    width: 500,
    height: 300,
    parent: BrowserWindow.getFocusedWindow(),
    modal: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  settingsWin.loadFile(path.join(__dirname, 'renderer', 'settings.html'));
}

// IPC handlers for settings operations
ipcMain.handle('settings:getApiKey', () => {
  return store.get('geminiApiKey') || '';
});

ipcMain.handle('settings:setApiKey', (event, newKey) => {
  if (newKey && newKey.trim()) {
    store.set('geminiApiKey', newKey.trim());
    process.env.GEMINI_API_KEY = newKey.trim();
    return true;
  }
  return false;
});

ipcMain.handle('settings:deleteApiKey', () => {
  store.delete('geminiApiKey');
  delete process.env.GEMINI_API_KEY;
  return true;
});

ipcMain.on('settings:open', () => {
  createSettingsWindow();
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  if (process.env.NODE_ENV === 'development') {
    win.webContents.openDevTools();
  }
  return win; // Return the window object
}

// Switches must be appended before app is ready
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('force-gpu-mem-available-mb', '1024');   // Upped from 512MB

app.whenReady().then(async () => {
  // Ensure API key exists before launching main window
  const ok = await ensureApiKey();
  if (!ok) return; // Application quit if key not provided

  const mainWindow = createWindow();

  // Build basic application menu with Settings item
  const { Menu } = require('electron');
  const template = [
    ...(process.platform === 'darwin' ? [{
      label: app.name,
      submenu: [
        { label: 'Settings', accelerator: 'CmdOrCtrl+,', click: () => createSettingsWindow() },
        { role: 'about' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    }] : []),
    {
      label: 'File',
      submenu: [
        { label: 'Settings', accelerator: 'Ctrl+,', click: () => createSettingsWindow() },
        { type: 'separator' },
        { role: 'quit' }
      ]
    }
  ];
  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  // Store mainWindow to access it in runOrganizer
  global.mainWindow = mainWindow;
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// IPC: choose directory
ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

function runOrganizer(dirPath, operation = 'plan', options = {}) {
  return new Promise((resolve, reject) => {
    console.log('runOrganizer called:', { dirPath, operation, options });
    
    // Determine the path to the Python backend executable
    let backendExecutable;
    if (app.isPackaged) {
      // In a packaged app, the executable is in extraResources
      // Adjust path based on electron-builder's extraResources configuration and PyInstaller's --onedir output
      if (process.platform === 'darwin') {
        // macOS: The extraResources are copied to Contents/Resources/backend/smart-file-organizer-backend/
        backendExecutable = path.join(process.resourcesPath, 'backend', 'smart-file-organizer-backend', 'smart-file-organizer-backend');
      } else {
        // For other platforms, adjust as needed (e.g., Windows: resources/backend/smart-file-organizer-backend/smart-file-organizer-backend.exe)
        backendExecutable = path.join(process.resourcesPath, 'backend', 'smart-file-organizer-backend', 'smart-file-organizer-backend');
      }
    } else {
      // In development mode, run directly from the dist folder's executable within the --onedir output
      backendExecutable = path.join(__dirname, '..', 'dist', 'smart-file-organizer-backend', 'smart-file-organizer-backend');
    }

    const args = ['--path', dirPath];
    
    // Add operation-specific arguments
    switch (operation) {
      case 'apply':
        args.push('--apply');
        break;
      case 'undo':
        args.push('--undo');
        if (options.limit) {
          args.push('--limit', options.limit.toString());
        }
        break;
      case 'summary':
        args.push('--summary');
        break;
      case 'plan':
      default:
        // Default is plan generation
        if (options.includeDuplicates) {
          args.push('--include-duplicates');
        }
        
        // Add intelligent AI classifier by default for better reliability
        if (options.useIntelligent !== false) {  // Default to true unless explicitly disabled
          args.push('--intelligent');
        }
        
        // Add custom prompt support
        if (options.customPrompt) {
          args.push('--prompt', options.customPrompt);
        }
        
        // Add template support
        if (options.template) {
          args.push('--template', options.template);
        }
        
        break;
    }

    console.log('Python command args:', args);

    // Use virtual environment Python if available, fallback to system python
    // This logic is now replaced by directly calling the bundled backendExecutable
    // const scriptRoot = path.join(__dirname, '..'); // project root
    // const venvPython = path.join(scriptRoot, 'venv', 'bin', 'python3');
    
    // let pythonExecutable;
    // try {
    //   // Check if virtual environment Python exists
    //   if (fs.existsSync(venvPython)) {
    //     pythonExecutable = venvPython;
    //     console.log('Using venv Python:', venvPython);
    //   } else {
    //     // Fallback to system python
    //     pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    //     console.log('Using system Python:', pythonExecutable);
    //   }
    // } catch (error) {
    //   console.error('Error checking Python executable:', error);
    //   pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    // }

    // Set the current working directory for the Python process
    const pythonCwd = app.isPackaged ? path.dirname(backendExecutable) : path.join(__dirname, '..');

    const pythonProcess = spawn(backendExecutable, args, {
      cwd: pythonCwd,
      env: { ...process.env },
    });

    // For apply operations, send the items via stdin
    if (operation === 'apply' && options.itemsToApply) {
      pythonProcess.stdin.write(JSON.stringify(options.itemsToApply));
      pythonProcess.stdin.end();
    }

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      const stderrLine = data.toString().trim();
      try {
        const progressData = JSON.parse(stderrLine);
        if (progressData.type === 'progress') {
          // Send progress updates to the renderer process
          if (global.mainWindow) {
            global.mainWindow.webContents.send('organize:progress', progressData);
          }
        } else {
          // If it's not a progress message, log to console as before
          console.error(`Python stderr: ${stderrLine}`);
          stderr += stderrLine + '\n';
        }
      } catch (e) {
        // Not a JSON progress message, just append to stderr
        console.error(`Python stderr (non-JSON): ${stderrLine}`);
        stderr += stderrLine + '\n';
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code: ${code}`);
      if (stdout) console.log('Python stdout:', stdout);
      if (stderr) console.log('Python stderr:', stderr);
      
      if (code !== 0) {
        return reject(new Error(`Python script failed with code ${code}: ${stderr}`));
      }
      
      try {
        switch (operation) {
          case 'apply':
            // For apply operations, return success with any relevant info
            resolve({ 
              success: true, 
              message: stderr,
              // In a real implementation, you'd parse move history from the backend
              moveHistory: options.itemsToApply || []
            });
            break;
            
          case 'undo':
            // Parse undo plan or return success
            if (stdout.trim()) {
              const undoPlan = JSON.parse(stdout);
              resolve({ success: true, undoPlan, count: undoPlan.length });
            } else {
              resolve({ success: true, count: 0 });
            }
            break;
            
          case 'summary':
            // Parse summary data
            const summary = JSON.parse(stdout);
            resolve(summary);
            break;
            
          case 'plan':
          default:
            // Parse the JSON plan from stdout
            const plan = JSON.parse(stdout);
            console.log('Parsed plan:', plan);
            resolve(plan);
            break;
        }
      } catch (e) {
        console.error('Error parsing Python response:', e);
        reject(new Error(`Failed to parse response from Python script: ${e.message}`));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('Python process error:', error);
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

// IPC: run organizer (dry run with enhanced options)
ipcMain.handle('organize:plan', (event, dirPath, options = {}) => {
  console.log('organize:plan called with:', { dirPath, options });
  return runOrganizer(dirPath, 'plan', options);
});

// IPC: run organizer (apply changes)
ipcMain.handle('organize:apply', (event, itemsToApply) => {
  // Extract directory from first item for backward compatibility
  const dirPath = itemsToApply[0]?.source ? 
    path.dirname(itemsToApply[0].source) : 
    process.cwd();
  
  return runOrganizer(dirPath, 'apply', { itemsToApply });
});

// IPC: undo last moves
ipcMain.handle('organize:undo', (event, dirPath, limit = 50) => {
  return runOrganizer(dirPath, 'undo', { limit });
});

// IPC: get plan summary
ipcMain.handle('organize:summary', (event, dirPath) => {
  return runOrganizer(dirPath, 'summary');
});

// IPC: open folder in file manager
ipcMain.handle('shell:openFolder', async (event, folderPath) => {
  try {
    await shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// IPC: choose directory (updated method name for consistency)
ipcMain.handle('dialog:selectFolder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
    title: 'Select Folder to Organize'
  });
  
  return result;
}); 