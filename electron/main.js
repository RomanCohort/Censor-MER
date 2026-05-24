const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let streamlitProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    title: 'Censor - 微表情识别系统'
  });

  // 加载页面
  mainWindow.loadFile('index.html');

  // 打开开发者工具（可选）
  mainWindow.webContents.openDevTools();

  // 窗口关闭时退出
  mainWindow.on('closed', () => {
    mainWindow = null;
    if (streamlitProcess) {
      streamlitProcess.kill();
    }
  });
}

// 启动Streamlit
function startStreamlit() {
  const censorPath = path.join(__dirname, '..');

  console.log('Starting Streamlit server...');

  streamlitProcess = spawn('streamlit', ['run', 'frontend/app.py', '--server.port', '8501', '--server.headless', 'true'], {
    cwd: censorPath,
    shell: true,
    env: { ...process.env }
  });

  streamlitProcess.stdout.on('data', (data) => {
    console.log(`Streamlit: ${data}`);
  });

  streamlitProcess.stderr.on('data', (data) => {
    console.error(`Streamlit Error: ${data}`);
  });

  streamlitProcess.on('error', (err) => {
    console.error('Failed to start Streamlit:', err);
  });
}

app.whenReady().then(() => {
  createWindow();

  // 尝试启动Streamlit
  try {
    startStreamlit();
  } catch (e) {
    console.log('Streamlit not available, showing local page only');
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});