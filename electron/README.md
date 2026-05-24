# Censor Electron 打包指南

## 方式一：直接使用 Electron（推荐开发时）

### 1. 安装 Electron
```bash
# 在 censor/electron 目录
cd D:/censor/electron
npm install
```

### 2. 运行
```bash
npm start
```

### 3. 打包为exe
```bash
npm run build
```

---

## 方式二：PyInstaller（无需Node.js）

如果无法安装Electron，可以用PyInstaller打包：

```bash
pip install pyinstaller
pyinstaller --onedir --name Censor --add-data "frontend/app.py;frontend" --add-data "docs;docs" --hidden-import=streamlit --hidden-import=torch --hidden-import=numpy --hidden-import=pandas --collect-all streamlit --collect-all torch --collect-all numpy frontend/app.py
```

---

## 文件结构

```
D:/censor/electron/
├── package.json          # Electron配置
├── main.js              # 主进程
├── preload.js           # 预加载脚本
├── index.html          # 启动页面
├── demo_local.html     # 本地演示页面（Streamlit不可用时）
└── README.md           # 本指南
```

---

## 快速开始（无Electron时）

直接用Streamlit：

```bash
cd D:/censor
streamlit run frontend/app.py
```

或者用demo脚本：

```bash
cd D:/censor
python demo.py
```