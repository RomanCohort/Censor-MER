// Preload script - exposes safe APIs to renderer
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Get app info
  getVersion: () => '1.0.0',

  // Open external links
  openExternal: (url) => require('electron').shell.openExternal(url),

  // Get platform info
  platform: process.platform
});