const { app, BrowserWindow, shell, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let streamlitProcess;
let isLoading = true;

// Streamlit 端口
const STREAMLIT_PORT = 7860;

// 检测运行模式：开发模式 vs 打包模式
const isPackaged = app.isPackaged;
const appPath = isPackaged ? path.dirname(app.getPath('exe')) : __dirname;

// Python 路径（优先使用嵌入版本）
function getPythonCommand() {
    const embeddedPythonPath = path.join(appPath, 'embedded', 'python', 'python.exe');
    const embeddedVenvPath = path.join(appPath, 'embedded', 'venv', 'Scripts', 'python.exe');

    if (isPackaged) {
        // 打包模式：使用嵌入 Python
        if (fs.existsSync(embeddedPythonPath)) {
            return embeddedPythonPath;
        }
        if (fs.existsSync(embeddedVenvPath)) {
            return embeddedVenvPath;
        }
        return 'python';  // 兜底
    } else {
        // 开发模式：使用系统 Python
        return 'python';
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 800,
        minHeight: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true
        },
        title: 'Censor - 微表情反馈收集',
        icon: path.join(__dirname, 'icon.ico'),
        show: false  // 先隐藏，加载完成后显示
    });

    // 创建菜单
    createMenu();

    // 加载启动等待页面
    mainWindow.loadFile('loading.html');

    // 窗口准备好后显示
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    // 窗口关闭时清理
    mainWindow.on('closed', () => {
        mainWindow = null;
        if (streamlitProcess) {
            console.log('Killing Streamlit process...');
            streamlitProcess.kill('SIGTERM');
        }
    });
}

// 创建应用菜单
function createMenu() {
    const menuTemplate = [
        {
            label: '文件',
            submenu: [
                { label: '刷新', accelerator: 'CmdOrCtrl+R', click: () => mainWindow.reload() },
                { type: 'separator' },
                { label: '退出', accelerator: 'CmdOrCtrl+Q', click: () => app.quit() }
            ]
        },
        {
            label: '视图',
            submenu: [
                { label: '放大', accelerator: 'CmdOrCtrl+Plus', click: () => mainWindow.webContents.zoomIn() },
                { label: '缩小', accelerator: 'CmdOrCtrl+-', click: () => mainWindow.webContents.zoomOut() },
                { label: '重置', accelerator: 'CmdOrCtrl+0', click: () => mainWindow.webContents.zoomReset() },
                { type: 'separator' },
                { label: '开发者工具', accelerator: 'F12', click: () => mainWindow.webContents.toggleDevTools() }
            ]
        },
        {
            label: '帮助',
            submenu: [
                { label: '关于', click: () => showAboutDialog() }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(menuTemplate);
    Menu.setApplicationMenu(menu);
}

function showAboutDialog() {
    const { dialog } = require('electron');
    dialog.showMessageBox(mainWindow, {
        title: '关于 Censor',
        message: 'Censor 微表情反馈收集系统',
        detail: '版本: 1.0.0\n用于收集微表情视频生成的人类评分数据',
        buttons: ['确定']
    });
}

// 启动 Streamlit 后端
function startStreamlit() {
    const pythonCmd = getPythonCommand();
    const censorPath = isPackaged ? path.join(appPath, 'resources', 'app') : path.join(__dirname, '..');

    // 设置 PYTHONPATH（嵌入包路径）
    const packagesPath = path.join(appPath, 'embedded', 'packages');
    const env = { ...process.env };

    if (fs.existsSync(packagesPath)) {
        env.PYTHONPATH = packagesPath;
    }

    console.log('Python command:', pythonCmd);
    console.log('Working directory:', censorPath);
    console.log('Packages path:', packagesPath);
    console.log('Is packaged:', isPackaged);
    console.log('Starting Streamlit server on port ' + STREAMLIT_PORT);

    // 启动 Streamlit
    streamlitProcess = spawn(
        pythonCmd,
        [
            '-m',
            'streamlit',
            'run',
            'interface/feedback_choice.py',
            '--server.port', String(STREAMLIT_PORT),
            '--server.headless', 'true',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ],
        {
            cwd: censorPath,
            shell: true,
            env: env
        }
    );

    streamlitProcess.stdout.on('data', (data) => {
        console.log(`[Streamlit] ${data.toString().trim()}`);

        // 检测是否启动完成
        if (data.toString().includes('You can now view your Streamlit app')) {
            console.log('Streamlit is ready!');
            loadStreamlitPage();
        }
    });

    streamlitProcess.stderr.on('data', (data) => {
        console.error(`[Streamlit Error] ${data.toString().trim()}`);
    });

    streamlitProcess.on('error', (err) => {
        console.error('Failed to start Streamlit:', err);
        showErrorPage(err.message);
    });

    streamlitProcess.on('exit', (code, signal) => {
        console.log(`Streamlit exited with code ${code}, signal ${signal}`);
    });

    // 备用方案：等待一段时间后尝试加载
    setTimeout(() => {
        if (isLoading) {
            console.log('Timeout reached, attempting to load Streamlit page...');
            loadStreamlitPage();
        }
    }, 10000);
}

// 加载 Streamlit 页面
function loadStreamlitPage() {
    if (mainWindow && !mainWindow.isDestroyed()) {
        isLoading = false;
        mainWindow.loadURL(`http://localhost:${STREAMLIT_PORT}`);
        console.log(`Loading http://localhost:${STREAMLIT_PORT}`);
    }
}

// 显示错误页面
function showErrorPage(errorMessage) {
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.loadFile('error.html');
    }
}

// 应用启动
app.whenReady().then(() => {
    createWindow();
  startStreamlit();
});

// 所有窗口关闭时退出
app.on('window-all-closed', () => {
  if (streamlitProcess) {
    streamlitProcess.kill('SIGTERM');
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// 应用退出前清理
app.on('before-quit', () => {
  if (streamlitProcess) {
    streamlitProcess.kill();
  }
});