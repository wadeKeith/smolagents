const { request, showErrorToast } = require('../../utils/api');

const STATUS_MAP = {
  pending: '待调度',
  processing: '生成中',
  completed: '已完成',
  error: '失败',
};

Page({
  data: {
    orderId: '',
    detail: null,
    statusLabel: '',
    updatedTime: '',
    isRefreshing: false,
  },

  onLoad(options) {
    const orderId = options.orderId || '';
    this.setData({ orderId });
    if (!orderId) {
      showErrorToast('缺少订单 ID');
      return;
    }
    this.fetchOrder();
  },

  onUnload() {
    this.clearTimer();
  },

  formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    const pad = (n) => (n < 10 ? `0${n}` : n);
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  },

  async fetchOrder(manual = false) {
    if (!this.data.orderId) return;
    if (manual) this.setData({ isRefreshing: true });
    try {
      const detail = await request({ url: `/wechat/orders/${this.data.orderId}` });
      const statusLabel = STATUS_MAP[detail.status] || detail.status;
      const updatedTime = this.formatTime(detail.updated_at);
      this.setData({ detail, statusLabel, updatedTime });
      if (detail.status === 'processing' || detail.status === 'pending') {
        this.scheduleNextPoll();
      } else {
        this.clearTimer();
      }
    } catch (err) {
      console.error('fetch order error', err);
      showErrorToast(err.message || '获取订单失败');
    } finally {
      if (manual) this.setData({ isRefreshing: false });
    }
  },

  scheduleNextPoll() {
    this.clearTimer();
    this.pollTimer = setTimeout(() => {
      this.fetchOrder();
    }, 4000);
  },

  clearTimer() {
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  },

  refreshOrder() {
    this.fetchOrder(true);
  },

  goHome() {
    wx.navigateBack({ delta: 1 });
  },
});
