const DEFAULT_BASE_URL = 'http://127.0.0.1:8081';

App({
  globalData: {
    baseUrl: DEFAULT_BASE_URL,
    version: '0.1.0',
  },

  onLaunch() {
    const stored = wx.getStorageSync('BASE_URL');
    if (stored && typeof stored === 'string') {
      this.globalData.baseUrl = stored;
    }
  },

  setBaseUrl(url) {
    if (!url || typeof url !== 'string') return;
    this.globalData.baseUrl = url;
    wx.setStorageSync('BASE_URL', url);
  },
});
