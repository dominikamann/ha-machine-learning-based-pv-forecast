/*
 * PV forecast card for the Machine-Learning-Based PV Forecast integration.
 *
 * Shows today's actual production next to this morning's and the live forecast, and the hourly
 * forecast with its P10–P90 range for the next days. Rendering is delegated to apexcharts-card.
 *
 *   type: custom:ml-pv-forecast-card
 *   entity: sensor.pv_forecast_total_forecast_curve   # optional, found automatically
 *   days: 3                                            # 1–3, optional
 *   title: PV                                          # optional
 */

const TEXT = {
  en: {
    title: "PV forecast",
    actual: "Actual",
    forecast: "Forecast",
    morning: "This morning's forecast",
    range: "Range (P10–P90)",
    today: "Today",
    tomorrow: "Tomorrow",
    dayAfter: "Day after tomorrow",
    sofar: "so far",
    noApex:
      "This card needs apexcharts-card. Install it via HACS (Frontend → apexcharts-card) and reload the page.",
    noEntity: "No forecast found. Set up the Machine-Learning-Based PV Forecast integration first.",
  },
  de: {
    title: "PV-Prognose",
    actual: "Tatsächlich",
    forecast: "Prognose",
    morning: "Prognose von heute früh",
    range: "Bandbreite (P10–P90)",
    today: "Heute",
    tomorrow: "Morgen",
    dayAfter: "Übermorgen",
    sofar: "bisher",
    noApex:
      "Diese Karte braucht apexcharts-card. Installiere sie über HACS (Frontend → apexcharts-card) und lade die Seite neu.",
    noEntity:
      "Keine Prognose gefunden. Richte zuerst die Integration Machine-Learning-Based PV Forecast ein.",
  },
};

const COLORS = { forecast: "#29B6F6", band: "#81D4FA", actual: "#66BB6A", morning: "#B39DDB" };

function isForecastCurve(state) {
  const a = state && state.attributes;
  return !!a && Array.isArray(a.hourly) && Array.isArray(a.quarters) && a.daily !== undefined;
}

// Bodies of apexcharts-card data_generator functions (they receive `entity`).
const GEN = {
  hourly: (key) => `
    return (entity.attributes.hourly || []).map(r => [new Date(r.time).getTime(), r.${key} / 1000]);`,
  morning: `
    const hours = {};
    (entity.attributes.morning || []).forEach(r => {
      const t = new Date(r.time); t.setMinutes(0, 0, 0);
      (hours[t.getTime()] ??= []).push(r.p50);
    });
    return Object.entries(hours).map(([t, v]) => [Number(t), v.reduce((a, b) => a + b, 0) / v.length / 1000]);`,
  actual: `
    const hours = {};
    (entity.attributes.actual || []).forEach(r => {
      const t = new Date(r.time); t.setMinutes(0, 0, 0);
      (hours[t.getTime()] ??= []).push(r.w);
    });
    return Object.entries(hours).map(([t, v]) => [Number(t), v.reduce((a, b) => a + b, 0) / v.length / 1000]);`,
};

class MlPvForecastCard extends HTMLElement {
  setConfig(config) {
    const days = Number(config.days ?? 3);
    if (!(days >= 1 && days <= 3)) throw new Error("days must be 1, 2 or 3");
    this._config = { ...config, days };
    this._chart = undefined;
    this._chartEntity = undefined;
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._render();
  }

  getCardSize() {
    return 6;
  }

  static getStubConfig() {
    return {};
  }

  _t() {
    const lang = (this._hass?.locale?.language || this._hass?.language || "en").slice(0, 2);
    return TEXT[lang] || TEXT.en;
  }

  _entityId() {
    if (this._config.entity) return this._config.entity;
    const states = Object.values(this._hass.states).filter(isForecastCurve);
    const total = states.find((s) => s.entity_id.includes("_total_"));
    return (total || states[0])?.entity_id;
  }

  _render() {
    if (!this._config || !this._hass) return;
    if (!this._card) {
      this._card = document.createElement("ha-card");
      this._header = document.createElement("div");
      this._header.className = "header";
      this._body = document.createElement("div");
      const style = document.createElement("style");
      style.textContent = `
        .header { padding: 16px 16px 0; }
        .title { font-size: 1.2em; font-weight: 500; }
        .days { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 6px; }
        .day { display: flex; flex-direction: column; }
        .label { color: var(--secondary-text-color); font-size: 0.85em; }
        .value { font-size: 1.15em; font-weight: 500; }
        .sub { color: var(--secondary-text-color); font-size: 0.8em; }
        .message { padding: 16px; color: var(--secondary-text-color); }
      `;
      this._card.append(style, this._header, this._body);
      this.append(this._card);
    }
    const t = this._t();
    const entityId = this._entityId();
    const state = entityId && this._hass.states[entityId];
    if (!state) {
      this._header.innerHTML = "";
      this._body.innerHTML = `<div class="message">${t.noEntity}</div>`;
      return;
    }
    this._renderHeader(state, t);
    if (!customElements.get("apexcharts-card")) {
      this._body.innerHTML = `<div class="message">${t.noApex}</div>`;
      customElements.whenDefined("apexcharts-card").then(() => this._render());
      return;
    }
    if (!this._chart || this._chartEntity !== entityId) {
      this._chart = document.createElement("apexcharts-card");
      this._chart.setConfig(this._chartConfig(entityId, t));
      this._chartEntity = entityId;
      this._body.innerHTML = "";
      this._body.append(this._chart);
    }
    this._chart.hass = this._hass;
  }

  _renderHeader(state, t) {
    const daily = state.attributes.daily || {};
    const actualKwh =
      (state.attributes.actual || []).reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
    const fmt = (v) =>
      v === undefined || v === null
        ? "–"
        : `${Number(v).toLocaleString(this._hass.locale?.language, { maximumFractionDigits: 1 })} kWh`;
    const days = [
      [t.today, daily.today, `${t.sofar} ${fmt(actualKwh)}`],
      [t.tomorrow, daily.tomorrow],
      [t.dayAfter, daily.day_after_tomorrow],
    ].slice(0, Math.max(this._config.days, 1));
    this._header.innerHTML = `
      <div class="title">${this._config.title ?? t.title}</div>
      <div class="days">${days
        .map(
          ([label, value, sub]) => `
          <div class="day"><span class="label">${label}</span>
            <span class="value">${fmt(value)}</span>${sub ? `<span class="sub">${sub}</span>` : ""}
          </div>`
        )
        .join("")}</div>`;
  }

  _chartConfig(entity, t) {
    const line = (name, color, generator, extra = {}) => ({
      entity,
      name,
      color,
      type: "line",
      stroke_width: 2,
      extend_to: false,
      unit: "kW",
      float_precision: 2,
      show: { in_header: false, legend_value: false },
      data_generator: generator,
      ...extra,
    });
    const midnight = new Date();
    midnight.setHours(24, 0, 0, 0);
    const dayMarks = [t.tomorrow, t.dayAfter].slice(0, this._config.days - 1).map((label, i) => ({
      x: midnight.getTime() + i * 86400000,
      borderColor: "#9E9E9E",
      strokeDashArray: 0,
      label: {
        text: label,
        orientation: "horizontal",
        position: "top",
        textAnchor: "start",
        offsetX: 6,
        offsetY: -4,
        borderWidth: 0,
        style: { background: "transparent", color: "#9E9E9E", fontSize: "12px" },
      },
    }));
    return {
      type: "custom:apexcharts-card",
      graph_span: `${this._config.days * 24}h`,
      span: { start: "day" },
      now: { show: true, color: "#9E9E9E" },
      header: { show: false },
      yaxis: [{ min: 0, decimals: 1, apex_config: { title: { text: "kW" } } }],
      apex_config: {
        chart: { height: 280 },
        legend: { show: true, position: "bottom" },
        tooltip: { x: { format: "ddd HH:mm" } },
        annotations: { xaxis: dayMarks },
        stroke: { curve: "smooth" },
      },
      series: [
        line(t.actual, COLORS.actual, GEN.actual, { type: "column", stroke_width: 0, opacity: 0.85 }),
        line(t.forecast, COLORS.forecast, GEN.hourly("p50"), { type: "area", opacity: 0.25 }),
        line(t.range, COLORS.band, GEN.hourly("p10"), { stroke_width: 1, opacity: 0.8, stroke_dash: 4 }),
        line(t.range, COLORS.band, GEN.hourly("p90"), {
          stroke_width: 1,
          opacity: 0.8,
          stroke_dash: 4,
          show: { in_header: false, legend_value: false, in_legend: false },
        }),
        line(t.morning, COLORS.morning, GEN.morning, { stroke_width: 2, stroke_dash: 6 }),
      ],
    };
  }
}

customElements.define("ml-pv-forecast-card", MlPvForecastCard);
window.customCards = window.customCards || [];
window.customCards.push({
  type: "ml-pv-forecast-card",
  name: "PV Forecast",
  description: "Actual production and the machine-learning PV forecast for today and the next days.",
  preview: true,
  documentationURL: "https://github.com/dominikamann/ha-machine-learning-based-pv-forecast",
});
