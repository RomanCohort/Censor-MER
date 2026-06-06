# Censor Electron 桌面应用（嵌入 Python 版）

将微表情反馈收集系统打包为完全独立的 Windows/Mac/Linux 桌面应用，**无需用户安装 Python**。

## 架构

```
Electron 前端 (桌面窗口)
    ↓
启动嵌入 Python (embedded/python/)
    ↓
运行 Streamlit 后端 (localhost:7860)
    ↓
加载 feedback_choice.py (简洁选择式界面)
```

## 界面预览

```
问题：哪个更像具有「微笑」微表情？

   [视频 A]        [视频 B]
     
   ⬅️ 选择 A       选择 B ➡️

底部选项：两个都不像 | 两个都很像 | 无法判断
```

---

## 一键打包（推荐）

### Windows

```bash
# 双击运行
build_full.bat

# 或命令行
cd D:/censor/electron
build_full.bat
```

这将自动：
1. 安装 Electron 依赖
2. 下载 Python embeddable (约 15MB)
3. 安装 Streamlit 及依赖
4. 打包为独立应用

### 输出文件

打包后生成在 `dist/` 目录：

| 文件 | 说明 | 大约大小 |
|------|------|---------|
| `Censor 微表情反馈 Setup.exe` | Windows 安装程序 | ~200MB |
| `Censor 微表情反馈.exe` | Windows 便携版 | ~200MB |

---

## 分步操作

如果一键打包失败，可以分步执行：

### 步骤 1: 准备嵌入 Python

```bash
# Windows
setup_embedded_python.bat

# Linux/Mac
chmod +x setup_embedded_python.sh
./setup_embedded_python.sh
```

这会下载：
- Python 3.11 embeddable (约 15MB)
- pip
- Streamlit + SQLAlchemy + pandas + openpyxl + plotly

### 步骤 2: 安装 Electron 依赖

```bash
npm install
```

### 步骤 3: 打包

```bash
# Windows
npm run build

# Mac
npm run build:mac

# Linux
npm run build:linux
```

---

## 开发模式

```bash
cd D:/censor/electron
npm install
npm start
```

开发模式使用系统 Python，打包模式使用嵌入 Python。

---

## 目录结构

```
D:/censor/electron/
├── main.js                  # Electron 主进程
├── preload.js               # 预加载脚本
├── loading.html             # 启动等待页面
├── error.html               # 错误页面
├── package.json             # Electron 配置
│
├── setup_embedded_python.bat # Python 环境准备脚本
├── build_full.bat           # 一键打包脚本
├── build.bat                # 快速打包脚本
│
├── embedded/                # 嵌入 Python 环境 (打包后)
│   ├── python/              # Python 3.11 embeddable
│   ├── packages/            # pip 安装的包
│   └── run_streamlit.bat    # 启动脚本
│
└── dist/                    # 打包输出
    ├── Censor 微表情反馈 Setup.exe
    └── Censor 微表情反馈.exe
```

---

## 注意事项

### 1. 图标文件

需自行准备应用图标放入 `electron/` 目录：
- `icon.ico` - Windows (256x256 像素)
- `icon.icns` - Mac
- `icon.png` - Linux

### 2. 打包大小

嵌入 Python + Streamlit + 依赖约 200MB，无法进一步压缩。

### 3. 端口冲突

默认使用 7860 端口，可在 `main.js` 修改 `STREAMLIT_PORT`。

### 4. 首次打包时间

首次运行 `build_full.bat` 需下载 Python 和依赖，约需 5-10 分钟。

---

## 常见问题

**Q: setup_embedded_python.bat 执行失败**

手动下载：
- Python: https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
- 解压到 `embedded/python/`
- 运行 `embedded\python\python.exe -m pip install streamlit --target=embedded\packages`

**Q: 打包后运行报错 "找不到 Python"**

检查：
1. `embedded/python/python.exe` 是否存在
2. `embedded/packages/streamlit` 是否存在

**Q: Streamlit 启动很慢**

首次启动约需 10 秒，后续会更快。

**Q: 如何对接真实视频生成**

修改 `interface/feedback_choice.py` 的 `generate_comparison_pair()` 函数，调用实际生成模型。

---

## 备用方案：直接 Streamlit

如果 Electron 打包失败，可直接使用 Streamlit：

```bash
cd D:/censor
pip install streamlit sqlalchemy pandas openpyxl
streamlit run interface/feedback_choice.py --server.port 7860
```