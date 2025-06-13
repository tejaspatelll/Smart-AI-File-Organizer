const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

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

app.whenReady().then(() => {
  const mainWindow = createWindow();

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
    
    const args = ['-m', 'backend.cli', '--path', dirPath];
    
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
    const scriptRoot = path.join(__dirname, '..'); // project root
    const venvPython = path.join(scriptRoot, 'venv', 'bin', 'python3');
    
    let pythonExecutable;
    try {
      // Check if virtual environment Python exists
      if (fs.existsSync(venvPython)) {
        pythonExecutable = venvPython;
        console.log('Using venv Python:', venvPython);
      } else {
        // Fallback to system python
        pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
        console.log('Using system Python:', pythonExecutable);
      }
    } catch (error) {
      console.error('Error checking Python executable:', error);
      pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    }

    const pythonProcess = spawn(pythonExecutable, args, {
      cwd: scriptRoot,
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