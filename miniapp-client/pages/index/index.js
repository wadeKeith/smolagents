const { request, ensureOpenId, showErrorToast, getBaseUrl } = require('../../utils/api');

const FALLBACK_CATALOG = {
  plans: [
    {
      id: 'standard',
      name: '标准版',
      price_cny: 99,
      tagline: '24 个月公开信息 + Playbook 更新',
      features: ['24 个月公开渠道扫描', '公司概况/合规风险', 'Playbook 实时快照'],
    },
    {
      id: 'deep',
      name: '深度版',
      price_cny: 199,
      tagline: '多轮主题 + RAG 导出',
      features: ['主题分析', 'Playbook 归档', '来源链路 + 可信度'],
    },
    {
      id: 'pro',
      name: '专业版',
      price_cny: 299,
      tagline: '定制检索关键词 + 回调',
      features: ['自定义关键词', '批量导出', 'API 回调'],
    },
  ],
  add_ons: [
    { id: 'playbook_export', name: 'Playbook 归档导出', price_cny: 20, description: '导出 PDF/JSON' },
    { id: 'custom_recency', name: '自定义时效 (12/36M)', price_cny: 30, description: '覆盖 12/36 月敏感窗口' },
    { id: 'human_review', name: '人工复核', price_cny: 100, description: '分析师二次校验' },
  ],
};

Page({
  data: {
    baseUrl: getBaseUrl(),
    plans: [],
    addOns: [],
    selectedPlan: 'standard',
    selectedAddOns: [],
    form: {
      company_name: '',
      company_site: '',
      jurisdiction_hint: 'CN',
      report_language: '中文',
      time_window_months: 24,
    },
    timeWindowOptions: [12, 24, 36],
    timeWindowIndex: 1,
    customWindowOptions: [12, 24, 36],
    customWindowIndex: 0,
    customWindowValue: 12,
    totalPrice: 0,
    orderSummary: {},
    isSubmitting: false,
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
    const addOns = catalog.add_ons || [];
    const selectedPlan = plans.length ? plans[0].id : this.data.selectedPlan;
    this.setData({ plans, addOns, selectedPlan }, () => {
      this.updateTotalPrice();
    });
  },

  handlePlanTap(event) {
    const planId = event.currentTarget.dataset.planId;
    this.setData({ selectedPlan: planId }, () => this.updateTotalPrice());
  },

  handleAddonChange(event) {
    const values = event.detail.value || [];
    let customWindowValue = this.data.customWindowValue;
    const hasCustom = values.indexOf('custom_recency') > -1;
    if (!hasCustom) {
      customWindowValue = null;
    } else if (!customWindowValue) {
      customWindowValue = this.data.customWindowOptions[this.data.customWindowIndex];
    }
    this.setData({ selectedAddOns: values, customWindowValue }, () => this.updateTotalPrice());
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

  handleCustomWindowChange(event) {
    const index = Number(event.detail.value);
    const value = this.data.customWindowOptions[index];
    this.setData({ customWindowIndex: index, customWindowValue: value });
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
        add_ons: this.data.selectedAddOns,
        company: this.data.form,
        openid,
      };
      if (this.data.selectedAddOns.indexOf('custom_recency') > -1 && this.data.customWindowValue) {
        payload.custom_time_window_months = this.data.customWindowValue;
      }
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
    let addonPrice = 0;
    this.data.selectedAddOns.forEach((addonId) => {
      const addon = (this.data.addOns || []).find((item) => item.id === addonId);
      if (addon) addonPrice += Number(addon.price_cny);
    });
    this.setData({ totalPrice: planPrice + addonPrice });
  },

  goToOrder() {
    if (!this.data.orderSummary.order_id) return;
    wx.navigateTo({ url: `/pages/order/order?orderId=${this.data.orderSummary.order_id}` });
  },
});
