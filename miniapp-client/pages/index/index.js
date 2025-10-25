const { request, ensureOpenId, showErrorToast, getBaseUrl } = require('../../utils/api');

const FALLBACK_CATALOG = {
  plans: [
    {
      id: 'standard',
      name: '单次尽调',
      price_cny: 25,
      tagline: '固定 ¥25，一次完成 24 个月尽调',
      features: [
        'RAG 检索 + 多智能体协同（Manager/Search/Critic）',
        '全链路：浏览器抓取 → RAG 精炼 → Playbook 写入',
        '报告正文 + Playbook 双输出，附引用与置信度',
      ],
    },
  ],
};

Page({
  data: {
    baseUrl: getBaseUrl(),
    plans: [],
    selectedPlan: 'standard',
    form: {
      company_name: '',
      company_site: '',
      jurisdiction_hint: 'CN',
      report_language: '中文',
      time_window_months: 24,
    },
    timeWindowOptions: [12, 24, 36],
    timeWindowIndex: 1,
    totalPrice: 0,
    orderSummary: {},
    isSubmitting: false,
    techHighlights: [
      'RAG 向量检索融合实时浏览器抓取，防止信息滞后',
      '多智能体（Manager / Search / Critic）分工协同，自动追踪缺口',
      'Playbook 自动归档，提供结构化引用与可审计链路',
    ],
  },

  onLoad() {
    this.fetchPricing();
  },

  async fetchPricing() {
    try {
      const catalog = await request({ url: '/wechat/pricing' });
      this.applyCatalog(catalog);
    } catch (err) {
      console.warn('pricing error', err);
      showErrorToast('获取定价失败，使用默认配置');
      this.applyCatalog(FALLBACK_CATALOG);
    }
  },

  applyCatalog(catalog) {
    const plans = catalog.plans || [];
    const selectedPlan = plans.length ? plans[0].id : this.data.selectedPlan;
    this.setData({ plans, selectedPlan }, () => {
      this.updateTotalPrice();
    });
  },

  handlePlanTap(event) {
    const planId = event.currentTarget.dataset.planId;
    this.setData({ selectedPlan: planId }, () => this.updateTotalPrice());
  },

  handleInput(event) {
    const field = event.currentTarget.dataset.field;
    const value = event.detail.value;
    const form = { ...this.data.form, [field]: value };
    this.setData({ form });
  },

  handleWindowChange(event) {
    const index = Number(event.detail.value);
    const months = this.data.timeWindowOptions[index];
    const form = { ...this.data.form, time_window_months: months };
    this.setData({ timeWindowIndex: index, form });
  },

  async handleSubmit() {
    if (!this.data.form.company_name) {
      showErrorToast('请填写企业名称');
      return;
    }
    this.setData({ isSubmitting: true });
    try {
      const openid = await ensureOpenId();
      const payload = {
        plan_id: this.data.selectedPlan,
        company: this.data.form,
        openid,
      };
      const summary = await request({ url: '/wechat/orders', method: 'POST', data: payload });
      wx.showToast({ title: '订单已创建', icon: 'success' });
      this.setData({ orderSummary: summary });
      wx.navigateTo({ url: `/pages/order/order?orderId=${summary.order_id}` });
    } catch (err) {
      console.error('submit error', err);
      showErrorToast(err.message || '创建订单失败');
    } finally {
      this.setData({ isSubmitting: false });
    }
  },

  updateTotalPrice() {
    const plan = this.data.plans.find((p) => p.id === this.data.selectedPlan);
    const planPrice = plan ? Number(plan.price_cny) : 0;
    this.setData({ totalPrice: planPrice });
  },

  goToOrder() {
    if (!this.data.orderSummary.order_id) return;
    wx.navigateTo({ url: `/pages/order/order?orderId=${this.data.orderSummary.order_id}` });
  },
});
