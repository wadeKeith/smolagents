const BASE_URL_FALLBACK = 'http://127.0.0.1:8081';
const OPENID_STORAGE_KEY = 'WECHAT_OPENID';
const DEFAULT_TIMEOUT = 20000;

function getBaseUrl() {
  const app = getApp();
  if (app && app.globalData && app.globalData.baseUrl) {
    return app.globalData.baseUrl;
  }
  return BASE_URL_FALLBACK;
}

function wxLogin() {
  return new Promise((resolve, reject) => {
    wx.login({ success: resolve, fail: reject });
  });
}

function request({ url, method = 'GET', data = {}, header = {}, timeout = DEFAULT_TIMEOUT }) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${getBaseUrl()}${url}`,
      method,
      data,
      timeout,
      header: { 'content-type': 'application/json', ...header },
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data);
        } else {
          reject({ message: res.data?.detail || '请求失败', response: res });
        }
      },
      fail(err) {
        reject(err);
      },
    });
  });
}

async function ensureOpenId() {
  const cached = wx.getStorageSync(OPENID_STORAGE_KEY);
  if (cached) return cached;
  const loginResult = await wxLogin();
  const loginResponse = await request({ url: '/wechat/login', method: 'POST', data: { code: loginResult.code } });
  const openid = loginResponse.openid;
  if (openid) {
    wx.setStorageSync(OPENID_STORAGE_KEY, openid);
  }
  return openid;
}

function clearOpenId() {
  wx.removeStorageSync(OPENID_STORAGE_KEY);
}

function showErrorToast(message) {
  wx.showToast({ title: message || '请求出错', icon: 'none' });
}

module.exports = {
  request,
  ensureOpenId,
  clearOpenId,
  showErrorToast,
  getBaseUrl,
};
