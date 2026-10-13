/*
 * PV forecast card for the Machine-Learning-Based PV Forecast integration.
 *
 * Five views, switched with the tabs at the top:
 *   overview – today and the next days, one chart per day covering only the hours with sun
 *   today    – today only: production so far, expected total and the hit rate so far
 *   carousel – one day at a time (today, tomorrow, ...), switching automatically
 *   plants   – the next days per plant, stacked
 *   review   – the last 7 days: morning forecast against actual production and the hit rate
 * Rendering is delegated to apexcharts-card.
 *
 *   type: custom:ml-pv-forecast-card
 *   entity: sensor.pv_forecast_total_forecast_curve   # optional, found automatically
 *   days: 3                                            # overview: 1–3, optional
 *   views: [overview, today, carousel, plants, review]  # tabs shown, in this order
 *   view: overview              # start view, one of views (default: the first)
 *   interval: 10                                       # carousel: seconds per day
 *   layout: stacked                                    # overview: stacked | timeline
 *   tabs: true                     # false (or a single view) shows only the start view
 *   title: PV                                          # optional, no title by default
 */

const TEXT = {
  en: {
    overview: "Overview",
    carousel: "Days",
    plants: "Plants",
    total: "Total",
    barsHint: "lines: actual · areas: forecast",
    noPlants: "No plants found.",
    review: "Review",
    actual: "Actual",
    forecast: "Forecast",
    morning: "Morning forecast",
    range: "Range (P10–P90)",
    today: "Today",
    tomorrow: "Tomorrow",
    dayAfter: "Day after tomorrow",
    sofar: "so far",
    produced: "Produced so far",
    expected: "Expected today",
    until: "until",
    pessimistic: "too pessimistic",
    optimistic: "too optimistic",
    onTarget: "on target",
    hitRate: "Hit rate",
    inBand: "Hours in range",
    noMorning: "No morning forecast was recorded for this day.",
    noApex:
      "This card needs apexcharts-card. Install it via HACS (Frontend → apexcharts-card) and reload the page.",
    noEntity: "No forecast found. Set up the Machine-Learning-Based PV Forecast integration first.",
    noReview: "No past days recorded yet.",
  },
  de: {
    overview: "Übersicht",
    carousel: "Tage",
    plants: "Anlagen",
    total: "Gesamt",
    barsHint: "Linien: tatsächlich · Flächen: Prognose",
    noPlants: "Keine Anlagen gefunden.",
    review: "Rückblick",
    actual: "Tatsächlich",
    forecast: "Prognose",
    morning: "Morgenprognose",
    range: "Bandbreite (P10–P90)",
    today: "Heute",
    tomorrow: "Morgen",
    dayAfter: "Übermorgen",
    sofar: "bisher",
    produced: "Bisher erzeugt",
    expected: "Erwartet heute",
    until: "bis",
    pessimistic: "zu pessimistisch",
    optimistic: "zu optimistisch",
    onTarget: "passt",
    hitRate: "Trefferquote",
    inBand: "Stunden im Band",
    noMorning: "Für diesen Tag wurde keine Morgenprognose gespeichert.",
    noApex:
      "Diese Karte braucht apexcharts-card. Installiere sie über HACS (Frontend → apexcharts-card) und lade die Seite neu.",
    noEntity:
      "Keine Prognose gefunden. Richte zuerst die Integration Machine-Learning-Based PV Forecast ein.",
    noReview: "Noch keine vergangenen Tage aufgezeichnet.",
  },
};

const COLORS = { forecast: "#29B6F6", band: "#81D4FA", actual: "#66BB6A", morning: "#B39DDB" };
const VIEWS = ["overview", "today", "carousel", "plants", "review"];
const PLANT_COLORS = ["#29B6F6", "#FFCA28", "#AB47BC", "#26A69A", "#EF5350", "#8D6E63"];
// Lighter shades of the plant colours for the actual-production lines drawn over the areas.
const PLANT_LIGHT = ["#E1F5FE", "#FFF8E1", "#F3E5F5", "#E0F2F1", "#FFEBEE", "#EFEBE9"];
const HOUR = 3600000;
const DAY = 24 * HOUR;
// Hide the inner apexcharts card frame so the charts sit flush inside this card.
const FLUSH =
  "--ha-card-border-width:0;--ha-card-box-shadow:none;--ha-card-background:transparent;--ha-card-border-radius:0;";

function isForecastCurve(state) {
  const a = state && state.attributes;
  return !!a && Array.isArray(a.hourly) && Array.isArray(a.quarters) && a.daily !== undefined;
}

// Bodies of apexcharts-card data_generator functions (they receive `entity`).
const hourlyAverage = (source, field) => `
    const hours = {};
    (${source} || []).forEach(r => {
      const t = new Date(r.time); t.setMinutes(0, 0, 0);
      (hours[t.getTime()] ??= []).push(r.${field});
    });
    return Object.entries(hours).map(([t, v]) => [Number(t), v.reduce((a, b) => a + b, 0) / v.length / 1000]);`;
const GEN = {
  // The live forecast from the current hour on: hourly averages of the 15-minute values (they are
  // corrected by the current production) while they last, the hourly values after that. Past
  // hours are left out – the bars show them.
  live: (key) => `
    const hour = new Date(); hour.setMinutes(0, 0, 0);
    const sums = {};
    (entity.attributes.quarters || []).forEach(r => {
      const t = new Date(r.time); t.setMinutes(0, 0, 0);
      if (t >= hour) (sums[t.getTime()] ??= []).push(r.${key});
    });
    const quarters = Object.entries(sums)
      .filter(([, v]) => v.length === 4)
      .map(([t, v]) => [Number(t), v.reduce((a, b) => a + b, 0) / 4000]);
    const last = quarters.length ? quarters[quarters.length - 1][0] : hour.getTime() - 1;
    const hours = (entity.attributes.hourly || [])
      .map(r => [new Date(r.time).getTime(), r.${key} / 1000])
      .filter(([t]) => t > last);
    return [...quarters, ...hours];`,
  morning: hourlyAverage("entity.attributes.morning", "p50"),
  actual: hourlyAverage("entity.attributes.actual", "w"),
  // Sum of the hourly P50 of several plants (apexcharts-card passes `hass` to generators).
  stacked: (ids) => `
    const sums = {};
    for (const id of ${JSON.stringify(ids)}) {
      (hass.states[id]?.attributes.hourly || []).forEach(r => {
        const t = new Date(r.time).getTime();
        sums[t] = (sums[t] || 0) + r.p50 / 1000;
      });
    }
    return Object.entries(sums).map(([t, v]) => [Number(t), v]).sort((a, b) => a[0] - b[0]);`,
  // Hourly actual power summed over `ids`, on the hours any of `allIds` has data (aligned series).
  stackedActual: (ids, allIds) => `
    const add = (id, into) => (hass.states[id]?.attributes.actual || []).forEach(r => {
      const t = new Date(r.time); t.setMinutes(0, 0, 0);
      (into[t.getTime()] ??= []).push(r.w);
    });
    const all = {};
    for (const id of ${JSON.stringify(allIds)}) add(id, all);
    const sums = {};
    for (const id of ${JSON.stringify(ids)}) {
      const own = {};
      add(id, own);
      for (const [t, v] of Object.entries(own)) sums[t] = (sums[t] || 0) + v.reduce((a, b) => a + b, 0) / v.length / 1000;
    }
    return Object.keys(all).map(t => [Number(t), sums[t] || 0]).sort((a, b) => a[0] - b[0]);`,
  review: (index, key) => `
    const day = (entity.attributes.review || [])[${index}];
    if (!day) return [];
    return day.hourly.filter(h => h.${key} !== undefined)
      .map(h => [new Date(h.time).getTime(), h.${key} / 1000]);`,
};

// WMO weather codes as plain text symbols (U+FE0E asks for the monochrome text style, not emoji).
const TEXT_STYLE = "\uFE0E";
function weatherSymbol(code) {
  if (code === null || code === undefined) return "";
  if (code === 0) return "☀" + TEXT_STYLE;
  if (code <= 2) return "⛅" + TEXT_STYLE;
  if (code === 3) return "☁" + TEXT_STYLE;
  if (code === 45 || code === 48) return "≡";
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return "❄" + TEXT_STYLE;
  if (code >= 95) return "⚡" + TEXT_STYLE;
  return "☂" + TEXT_STYLE; // drizzle, rain, showers
}

// Hours of one day with sun: the most frequent weather symbol and the highest temperature.
function dayWeather(hourly, dayStart) {
  const rows = (hourly || []).filter((r) => {
    const t = new Date(r.time).getTime();
    return t >= dayStart && t < dayStart + DAY;
  });
  const sunny = rows.filter((r) => r.p50 > 0 && r.wc !== null && r.wc !== undefined);
  const counts = {};
  sunny.forEach((r) => (counts[weatherSymbol(r.wc)] = (counts[weatherSymbol(r.wc)] || 0) + 1));
  const symbol = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
  const temps = rows.map((r) => r.t).filter((t) => t !== null && t !== undefined);
  const tmax = temps.length ? `${Math.round(Math.max(...temps))}°` : "";
  return [symbol, tmax].filter(Boolean).join(" ");
}

// Hours between weather symbols so they keep ~26 px apart at the given chart width.
function weatherStep(widthPx, hours) {
  const pxPerHour = Math.max(120, widthPx - 70) / hours;
  return [1, 2, 3, 4, 6, 8, 12, 24].find((s) => s * pxPerHour >= 26) ?? 24;
}

// Small grey weather symbols along the top of a chart, every `step` hours within [from, to).
// Only hours with sun get one: WMO code 0 means a clear sky at night as well.
function weatherMarks(hourly, from, to, step, offsetY = -2) {
  return (hourly || [])
    .filter((r) => {
      const t = new Date(r.time).getTime();
      return (
        t >= from &&
        t < to &&
        r.p50 > 0 &&
        r.wc !== null &&
        r.wc !== undefined &&
        (t - from) % (step * HOUR) === 0
      );
    })
    .map((r) => ({
      x: new Date(r.time).getTime() + HOUR / 2,
      borderColor: "transparent",
      label: {
        text: weatherSymbol(r.wc),
        orientation: "horizontal",
        position: "top",
        textAnchor: "middle",
        offsetY,
        borderWidth: 0,
        style: { background: "transparent", color: "#9E9E9E", fontSize: "13px" },
      },
    }));
}

function localMidnight(offsetDays = 0) {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + offsetDays);
  return d.getTime();
}

// Hours of a day [start, end) that are worth showing: production or forecast, plus one hour.
function sunWindow(dayStart, points) {
  const day = points.filter(([t]) => t >= dayStart && t < dayStart + DAY);
  // Relative threshold: meters report tiny values at night that are not sun.
  const threshold = Math.max(0.005, 0.03 * Math.max(0, ...day.map(([, v]) => v)));
  const hours = day
    .filter(([, v]) => v > threshold)
    .map(([t]) => Math.floor((t - dayStart) / HOUR));
  if (!hours.length) return [6, 20];
  let start = Math.max(0, Math.min(...hours) - 1);
  let end = Math.min(24, Math.max(...hours) + 2);
  if (end - start < 6) {
    const pad = Math.ceil((6 - (end - start)) / 2);
    start = Math.max(0, start - pad);
    end = Math.min(24, end + pad);
  }
  return [start, end];
}

class MlPvForecastCard extends HTMLElement {
  setConfig(config) {
    const days = Number(config.days ?? 3);
    if (!(days >= 1 && days <= 3)) throw new Error("days must be 1, 2 or 3");
    const views = config.views ?? VIEWS;
    if (!Array.isArray(views) || !views.length || views.some((v) => !VIEWS.includes(v))) {
      throw new Error(`views must be a list of ${VIEWS.join(", ")}`);
    }
    const view = config.view ?? views[0];
    if (!views.includes(view)) throw new Error(`view must be one of ${views.join(", ")}`);
    const interval = Number(config.interval ?? 10);
    if (!(interval >= 2)) throw new Error("interval must be at least 2 seconds");
    const layout = config.layout ?? "stacked";
    if (!["stacked", "timeline"].includes(layout)) throw new Error("layout must be stacked or timeline");
    this._config = { tabs: true, ...config, days, views: [...new Set(views)], view, interval, layout };
    this._view = view;
    this._reviewIndex = 0;
    this._dayIndex = 0;
    this._restartCarousel();
    this._chartsKey = undefined;
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._render();
  }

  getCardSize() {
    return 7;
  }

  static getStubConfig() {
    return {};
  }

  _t() {
    const lang = (this._hass?.locale?.language || this._hass?.language || "en").slice(0, 2);
    return TEXT[lang] || TEXT.en;
  }

  _lang() {
    return this._hass?.locale?.language || this._hass?.language || "en";
  }

  _entityId() {
    if (this._config.entity) return this._config.entity;
    const states = Object.values(this._hass.states).filter(isForecastCurve);
    const total = states.find((s) => s.entity_id.includes("_total_"));
    return (total || states[0])?.entity_id;
  }

  _kwh(value) {
    if (value === undefined || value === null) return "–";
    return `${Number(value).toLocaleString(this._lang(), { maximumFractionDigits: 1 })} kWh`;
  }

  _time(value) {
    return new Date(value).toLocaleTimeString(this._lang(), { hour: "2-digit", minute: "2-digit" });
  }

  _setView(view) {
    this._view = view;
    this._dayIndex = 0;
    this._chartsKey = undefined;
    this._restartCarousel();
    this._render();
  }

  _showDay(index) {
    this._dayIndex = index;
    this._chartsKey = undefined;
    this._restartCarousel(); // a click restarts the timer so the chosen day stays a while
    this._render();
  }

  _restartCarousel() {
    clearInterval(this._timer);
    this._timer = undefined;
    if (this._view !== "carousel" || !this.isConnected) return;
    this._timer = setInterval(() => {
      this._dayIndex = (this._dayIndex + 1) % this._config.days;
      this._chartsKey = undefined;
      this._render();
    }, this._config.interval * 1000);
  }

  connectedCallback() {
    this._restartCarousel();
  }

  disconnectedCallback() {
    clearInterval(this._timer);
    this._timer = undefined;
  }

  _shiftReview(delta) {
    this._reviewIndex = Math.max(0, Math.min(6, this._reviewIndex + delta));
    this._chartsKey = undefined;
    this._render();
  }

  _render() {
    if (!this._config || !this._hass) return;
    if (!this._card) this._build();
    const t = this._t();
    const entityId = this._entityId();
    const state = entityId && this._hass.states[entityId];
    if (!state) {
      this._header.innerHTML = "";
      this._body.innerHTML = `<div class="message">${t.noEntity}</div>`;
      return;
    }
    this._header.innerHTML = this._headerHtml(state, t);
    if (!customElements.get("apexcharts-card")) {
      this._body.innerHTML = `<div class="message">${t.noApex}</div>`;
      customElements.whenDefined("apexcharts-card").then(() => this._render());
      return;
    }
    const message = this._message(state, t);
    if (message) {
      this._charts = [];
      this._chartsKey = undefined;
      this._body.innerHTML = `<div class="message">${message}</div>`;
      return;
    }
    // Rebuild the charts when the view, the day or the shown hours change (once an hour at most).
    const specs = this._specs(entityId, state, t);
    const key = `${entityId}|${this._view}|${this._reviewIndex}|${this._cardWidth() < 500}|${weatherStep(this._cardWidth(), 24)}|${specs.map((s) => s.key).join(",")}`;
    if (this._chartsKey !== key) {
      this._chartsKey = key;
      this._charts = [];
      const grid = document.createElement("div");
      grid.className = "charts";
      for (const spec of specs) {
        const column = document.createElement("div");
        column.className = "column";
        if (spec.head) column.innerHTML = spec.head;
        const chart = document.createElement("apexcharts-card");
        chart.style.cssText = FLUSH;
        chart.setConfig(spec.config);
        column.append(chart);
        grid.append(column);
        this._charts.push(chart);
      }
      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = this._legend(t);
      this._body.innerHTML = "";
      this._body.append(grid, legend);
    } else {
      // Update the per-day headings (values change during the day) without rebuilding charts.
      this._body.querySelectorAll(".column").forEach((column, i) => {
        const heading = column.querySelector(".col-head");
        if (heading && specs[i]?.head) heading.outerHTML = specs[i].head;
      });
    }
    for (const chart of this._charts) chart.hass = this._hass;
  }

  _build() {
    this._card = document.createElement("ha-card");
    this._header = document.createElement("div");
    this._header.className = "header";
    this._body = document.createElement("div");
    this._body.className = "body";
    const style = document.createElement("style");
    style.textContent = `
      .header { padding: 12px 16px 0; }
      .title { font-size: 1.15em; font-weight: 500; margin-bottom: 8px; }
      .tabs { display: flex; border: 1px solid var(--divider-color); border-radius: 8px; overflow: hidden; }
      .tab { flex: 1 1 0; min-width: 0; border: none; border-left: 1px solid var(--divider-color);
             background: none; color: var(--secondary-text-color); padding: 6px 2px; font-size: 0.8em;
             cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .tab:first-child { border-left: none; }
      .tab.active { background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
               gap: 4px 16px; margin-top: 12px; }
      .stat .label, .col-head .label { color: var(--secondary-text-color); font-size: 0.8em; }
      .stat .value, .col-head .value { font-size: 1.2em; font-weight: 500; line-height: 1.3; }
      .sub { color: var(--secondary-text-color); font-size: 0.8em; }
      .nav { display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 10px; }
      .nav span { min-width: 12em; text-align: center; font-weight: 500; }
      .nav button { background: none; border: 1px solid var(--divider-color); border-radius: 50%;
                    width: 28px; height: 28px; color: var(--primary-text-color); cursor: pointer; }
      .nav button:disabled { opacity: 0.3; cursor: default; }
      .body { padding: 4px 4px 8px; }
      .charts { display: flex; flex-direction: column; }
      .column { min-width: 0; overflow: hidden; border-top: 1px solid var(--divider-color); }
      .column:first-child { border-top: none; }
      .col-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; padding: 10px 12px 0; }
      .col-head .label { font-size: 0.9em; }
      .col-head .value { font-size: 1.05em; }
      .weather { color: var(--secondary-text-color); font-weight: normal; margin-left: 4px; }
      .legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 4px 16px;
                font-size: 0.8em; color: var(--secondary-text-color); padding: 0 12px; }
      .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
                             border-radius: 2px; margin-right: 6px; background: var(--c); }
      .legend span.plain::before { display: none; }
      .legend span.dash::before { height: 0; border-top: 2px dashed var(--c); background: none;
                                  vertical-align: middle; }
      .legend span.line::before { height: 0; border-top: 3px solid var(--c); background: none;
                                  vertical-align: middle; }
      .message { padding: 16px; color: var(--secondary-text-color); }
      table.plants { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
      table.plants th { color: var(--secondary-text-color); font-weight: normal; text-align: right; }
      table.plants th:first-child, table.plants td:first-child { text-align: left; }
      table.plants td { text-align: right; padding: 2px 0; }
      table.plants tr.total td { border-top: 1px solid var(--divider-color); padding-top: 4px; }
      table.plants td:first-child::before { content: ""; display: inline-block; width: 10px; height: 10px;
        border-radius: 2px; margin-right: 6px; background: var(--c); }
      .dots { display: flex; justify-content: center; gap: 8px; margin-top: 8px; }
      .dots button { width: 9px; height: 9px; border-radius: 50%; border: none; padding: 0;
                     background: var(--divider-color); cursor: pointer; }
      .dots button.active { background: var(--primary-color); }
    `;
    this._card.append(style, this._header, this._body);
    this.append(this._card);
    // Charts measure their width when first drawn; dashboards often settle on a narrower column
    // afterwards (phones, masonry). Redraw whenever the card's width changes.
    this._width = 0;
    new ResizeObserver((entries) => {
      const width = Math.round(entries[0].contentRect.width);
      if (!width || width === this._width) return;
      this._width = width;
      clearTimeout(this._resizeTimer);
      this._resizeTimer = setTimeout(() => {
        this._render(); // symbol spacing and day names depend on the width
        window.dispatchEvent(new Event("resize"));
      }, 100);
    }).observe(this._card);
    this._header.addEventListener("click", (ev) => {
      const target = ev.target.closest("[data-view],[data-shift],[data-day]");
      if (!target || target.disabled) return;
      if (target.dataset.view) this._setView(target.dataset.view);
      if (target.dataset.day) this._showDay(Number(target.dataset.day));
      if (target.dataset.shift) this._shiftReview(Number(target.dataset.shift));
    });
  }

  _message(state, t) {
    if (this._view === "review" && !(state.attributes.review || [])[this._reviewIndex]) {
      return t.noReview;
    }
    if (this._view === "plants" && !this._plantStates().length) return t.noPlants;
    return undefined;
  }

  _legend(t) {
    // style: "dash" or "line"; a filled square otherwise
    const item = (label, color, style) =>
      `<span class="${style === true ? "dash" : style || ""}" style="--c:${color}">${label}</span>`;
    const items = {
      overview: [
        item(t.actual, COLORS.actual),
        item(t.forecast, COLORS.forecast, "line"),
        item(t.range, COLORS.band, true),
        item(t.morning, COLORS.morning, true),
      ],
      forecast: [item(t.forecast, COLORS.forecast, "line"), item(t.range, COLORS.band, true)],
      review: [
        item(t.actual, COLORS.actual),
        item(t.morning, COLORS.morning, true),
        item(t.range, COLORS.band, true),
      ],
    };
    if (this._view === "carousel") {
      return (this._dayIndex === 0 ? items.overview : items.forecast).join("");
    }
    if (this._view === "today") return items.overview.join("");
    if (this._view === "plants") {
      const plants = this._plantStates()
        .map((p, i) => item(p.attributes.plant, PLANT_COLORS[i % PLANT_COLORS.length]))
        .join("");
      return `${plants}<span class="plain">${t.barsHint}</span>`;
    }
    return items[this._view].join("");
  }

  _headerHtml(state, t) {
    const title = this._config.title ? `<div class="title">${this._config.title}</div>` : "";
    const label = {
      overview: t.overview,
      today: t.today,
      carousel: t.carousel,
      plants: t.plants,
      review: t.review,
    };
    const { views } = this._config;
    const tabs =
      this._config.tabs && views.length > 1
        ? `<div class="tabs">${views
            .map(
              (v) =>
                `<button class="tab ${v === this._view ? "active" : ""}" data-view="${v}">${label[v]}</button>`
            )
            .join("")}</div>`
        : "";
    const top = title + tabs;
    if (this._view === "review") return top + this._reviewStats(state, t);
    if (this._view === "carousel") return top + this._carouselStats(state, t);
    if (this._view === "today") return top + this._todayStats(state, t);
    if (this._view === "plants") return top + this._plantsTable(t);
    return top;
  }

  _stat(label, value, sub) {
    return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div>${
      sub ? `<div class="sub">${sub}</div>` : ""
    }</div>`;
  }

  _colHead(label, value, sub) {
    return `<div class="col-head"><span class="label">${label}</span><span class="value">${value}</span>${
      sub ? `<span class="sub">· ${sub}</span>` : ""
    }</div>`;
  }

  _carouselStats(state, t) {
    const a = state.attributes;
    const daily = a.daily || {};
    const i = this._dayIndex;
    const weather = dayWeather(a.hourly, localMidnight(i));
    const label = `${[t.today, t.tomorrow, t.dayAfter][i]}${weather ? ` · ${weather}` : ""}`;
    const value = [daily.today, daily.tomorrow, daily.day_after_tomorrow][i];
    const actualKwh = (a.actual || []).reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
    const dots = Array.from(
      { length: this._config.days },
      (_, d) => `<button class="${d === i ? "active" : ""}" data-day="${d}" aria-label="${d}"></button>`
    ).join("");
    return `<div class="stats">${this._stat(
      label,
      this._kwh(value),
      i === 0 ? `${t.sofar} ${this._kwh(actualKwh)}` : undefined
    )}</div><div class="dots">${dots}</div>`;
  }

  _todayStats(state, t) {
    const a = state.attributes;
    const today = a.today || {};
    const actual = a.actual || [];
    // Expected: what was produced so far plus the live forecast for the rest of the day.
    const done = actual.length ? new Date(actual[actual.length - 1].time).getTime() + 15 * 60000 : 0;
    const tomorrow = localMidnight(1);
    const rest = (a.quarters || [])
      .filter((r) => {
        const t0 = new Date(r.time).getTime();
        return t0 >= Math.max(done, Date.now() - 15 * 60000) && t0 < tomorrow;
      })
      .reduce((sum, r) => sum + r.p50, 0);
    const produced = actual.reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
    // Fallback while the integration is older than the card (no `today` attribute yet).
    const morningKwh = a.morning?.length
      ? a.morning.reduce((sum, r) => sum + r.p50, 0) / 4 / 1000
      : undefined;
    const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v)} %`);
    const until = today.until ? `${t.until} ${this._time(today.until)}` : undefined;
    const direction = this._direction(today.deviation_pct, t);
    return `<div class="stats">${[
      this._stat(t.produced, this._kwh(produced)),
      this._stat(t.expected, this._kwh(produced + rest / 4 / 1000)),
      this._stat(t.morning, this._kwh(today.forecast_kwh ?? morningKwh)),
      this._stat(t.hitRate, pct(today.hit_rate), [direction, until].filter(Boolean).join(" · ")),
      this._stat(t.inBand, pct(today.in_band), until),
    ].join("")}</div>`;
  }

  // Which way the morning forecast was off: ▲ more came than forecast, ▼ less.
  _direction(deviation, t) {
    if (deviation === null || deviation === undefined) return undefined;
    if (Math.abs(deviation) <= 3) return t.onTarget;
    const amount = `${Math.round(Math.abs(deviation))} %`;
    return deviation < 0 ? `▲ ${amount} ${t.pessimistic}` : `▼ ${amount} ${t.optimistic}`;
  }

  _reviewStats(state, t) {
    const days = state.attributes.review || [];
    const day = days[this._reviewIndex];
    const date = day
      ? new Date(day.date).toLocaleDateString(this._lang(), {
          weekday: "long",
          day: "numeric",
          month: "long",
        })
      : "";
    const nav = `<div class="nav">
      <button data-shift="1" ${this._reviewIndex >= days.length - 1 ? "disabled" : ""}>‹</button>
      <span>${date}</span>
      <button data-shift="-1" ${this._reviewIndex === 0 ? "disabled" : ""}>›</button></div>`;
    if (!day) return nav;
    const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v)} %`);
    const note = day.forecast_kwh === null ? `<div class="sub">${t.noMorning}</div>` : "";
    return `${nav}<div class="stats">${[
      this._stat(t.morning, this._kwh(day.forecast_kwh)),
      this._stat(t.actual, this._kwh(day.actual_kwh)),
      this._stat(t.hitRate, pct(day.hit_rate), this._direction(day.deviation_pct, t)),
      this._stat(t.inBand, pct(day.in_band)),
    ].join("")}</div>${note}`;
  }

  _series(entity, name, color, generator, extra = {}) {
    return {
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
    };
  }

  _band(entity, t, low, high) {
    const style = { stroke_width: 1, opacity: 0.8, stroke_dash: 4 };
    return [
      this._series(entity, t.range, COLORS.band, low, style),
      this._series(entity, t.range, COLORS.band, high, style),
    ];
  }

  _chart({ entity, start, hours, max, yAxis, now, series, height = 230 }) {
    const offset = Math.round((start - localMidnight()) / HOUR);
    return {
      type: "custom:apexcharts-card",
      header: { show: false },
      graph_span: `${hours}h`,
      span: offset ? { start: "day", offset: `${offset > 0 ? "+" : ""}${offset}h` } : { start: "day" },
      now: { show: now, color: "#9E9E9E" },
      yaxis: [
        {
          min: 0,
          max,
          decimals: 1,
          apex_config: { show: yAxis, tickAmount: 4, title: { text: yAxis ? "kW" : "" } },
        },
      ],
      apex_config: {
        chart: { height, width: "100%", parentHeightOffset: 0, redrawOnParentResize: true },
        legend: { show: false },
        grid: { padding: { left: 8, right: 8 } }, // keeps the kW labels off the grid lines
        xaxis: { labels: { datetimeUTC: false, format: "HH:mm" }, tickAmount: 4 },
        tooltip: { x: { format: "ddd HH:mm" } },
        stroke: { curve: "smooth" },
      },
      series: series.map((s) => ({ entity, ...s })),
    };
  }

  _maxKw(values) {
    const max = Math.max(0.5, ...values) * 1.1;
    return Math.ceil(max * 2) / 2;
  }

  _specs(entity, state, t) {
    const a = state.attributes;
    if (this._view === "review") {
      const i = this._reviewIndex;
      const day = a.review[i];
      const dayStart = new Date(day.date).getTime();
      const points = day.hourly.flatMap((h) => [
        [new Date(h.time).getTime(), (h.w ?? 0) / 1000],
        [new Date(h.time).getTime(), (h.p90 ?? 0) / 1000],
      ]);
      const [from, to] = sunWindow(dayStart, points);
      const max = this._maxKw(points.map(([, v]) => v));
      return [
        {
          key: `${from}-${to}`,
          weight: 1,
          config: this._chart({
            entity,
            start: dayStart + from * HOUR,
            hours: to - from,
            max,
            yAxis: true,
            now: false,
            series: [
              this._series(entity, t.actual, COLORS.actual, GEN.review(i, "w"), {
                type: "column",
                stroke_width: 0,
                opacity: 0.85,
              }),
              this._series(entity, t.morning, COLORS.morning, GEN.review(i, "p50"), {
                stroke_dash: 6,
              }),
              ...this._band(entity, t, GEN.review(i, "p10"), GEN.review(i, "p90")),
            ],
          }),
        },
      ];
    }
    if (this._view === "plants") return [this._plantsSpec(t)];
    if (this._view === "overview" && this._config.layout === "timeline") {
      return [this._timelineSpec(entity, state, t)];
    }
    // Overview: one chart per day, each covering only the hours with sun, sharing one kW scale.
    // the corrected 15-minute values can lie above the hourly ones, so both set the scale
    const hourly = [...(a.hourly || []), ...(a.quarters || [])].map((r) => [
      new Date(r.time).getTime(),
      r.p90 / 1000,
    ]);
    const actual = (a.actual || []).map((r) => [new Date(r.time).getTime(), r.w / 1000]);
    // Today alone gets its own kW scale; the other views share one scale across the days.
    const scaled = this._view === "today" ? hourly.filter(([t]) => t < localMidnight(1)) : hourly;
    const max = this._maxKw([...scaled, ...actual].map(([, v]) => v));
    const daily = a.daily || {};
    const actualKwh = (a.actual || []).reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
    const withWeather = (label, i) => {
      const weather = dayWeather(a.hourly, localMidnight(i));
      return weather ? `${label} <span class="weather">${weather}</span>` : label;
    };
    const heads = [
      this._colHead(withWeather(t.today, 0), this._kwh(daily.today), `${t.sofar} ${this._kwh(actualKwh)}`),
      this._colHead(withWeather(t.tomorrow, 1), this._kwh(daily.tomorrow)),
      this._colHead(withWeather(t.dayAfter, 2), this._kwh(daily.day_after_tomorrow)),
    ];
    const specs = [];
    const single = this._view === "carousel" || this._view === "today";
    const indices =
      this._view === "carousel"
        ? [this._dayIndex]
        : this._view === "today"
          ? [0]
          : Array.from({ length: this._config.days }, (_, i) => i);
    for (const i of indices) {
      const dayStart = localMidnight(i);
      const [from, to] = sunWindow(dayStart, i === 0 ? [...hourly, ...actual] : hourly);
      const series = [
        this._series(entity, t.forecast, COLORS.forecast, GEN.live("p50"), {
          stroke_width: 2.5,
        }),
        ...this._band(entity, t, GEN.live("p10"), GEN.live("p90")),
      ];
      if (i === 0) {
        series.unshift(
          this._series(entity, t.actual, COLORS.actual, GEN.actual, {
            type: "column",
            stroke_width: 0,
            opacity: 0.85,
          })
        );
        series.push(this._series(entity, t.morning, COLORS.morning, GEN.morning, { stroke_dash: 6 }));
      }
      const config = this._chart({
        entity,
        start: dayStart + from * HOUR,
        hours: to - from,
        max,
        yAxis: true,
        height: single ? 230 : 160,
        now: i === 0,
        series,
      });
      config.apex_config.grid.padding.top = 22;
      config.apex_config.annotations = {
        xaxis: weatherMarks(
          a.hourly,
          dayStart + from * HOUR,
          dayStart + to * HOUR,
          weatherStep(this._cardWidth(), to - from),
          -22 // above the top grid line, in the padding strip below
        ),
      };
      specs.push({
        key: `${from}-${to}`,
        weight: to - from,
        head: single ? undefined : heads[i],
        config,
      });
    }
    return specs;
  }
}

MlPvForecastCard.prototype._timelineSpec = function (entity, state, t) {
  // All days in one continuous chart (nights included), with the day totals as heading.
  const a = state.attributes;
  const daily = a.daily || {};
  const actualKwh = (a.actual || []).reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
  const weather = (i) => {
    const w = dayWeather(a.hourly, localMidnight(i));
    return w ? ` <span class="weather">${w}</span>` : "";
  };
  const parts = [
    `${t.today}${weather(0)} <b>${this._kwh(daily.today)}</b> <span class="sub">(${t.sofar} ${this._kwh(actualKwh)})</span>`,
    `${this._dayName(1, t)}${weather(1)} <b>${this._kwh(daily.tomorrow)}</b>`,
    `${this._dayName(2, t)}${weather(2)} <b>${this._kwh(daily.day_after_tomorrow)}</b>`,
  ].slice(0, this._config.days);
  const marks = [1, 2].slice(0, this._config.days - 1).map((d, i) => ({
    x: localMidnight(i + 1),
    borderColor: "#9E9E9E",
    strokeDashArray: 0,
    label: {
      text: this._dayName(i + 1, t),
      orientation: "horizontal",
      position: "top",
      textAnchor: "start",
      offsetX: 6,
      offsetY: -4,
      borderWidth: 0,
      style: { background: "transparent", color: "#9E9E9E", fontSize: "12px" },
    },
  }));
  const config = this._chart({
    entity,
    start: localMidnight(),
    hours: this._config.days * 24,
    max: this._maxKw([...(a.hourly || []), ...(a.quarters || [])].map((r) => r.p90 / 1000)),
    yAxis: true,
    now: true,
    height: 260,
    series: [
      this._series(entity, t.actual, COLORS.actual, GEN.actual, {
        type: "column",
        stroke_width: 0,
        opacity: 0.85,
      }),
      this._series(entity, t.forecast, COLORS.forecast, GEN.live("p50"), {
        stroke_width: 2.5,
      }),
      ...this._band(entity, t, GEN.live("p10"), GEN.live("p90")),
      this._series(entity, t.morning, COLORS.morning, GEN.morning, { stroke_dash: 6 }),
    ],
  });
  const daylight = (a.hourly || []).filter((r) => r.p50 > 0);
  // one row below the day labels, so they never overlap
  const weatherAnnotations = weatherMarks(
    daylight,
    localMidnight(),
    localMidnight(this._config.days),
    weatherStep(this._cardWidth(), this._config.days * 24),
    16
  );
  config.apex_config.annotations = { xaxis: [...marks, ...weatherAnnotations] };
  config.apex_config.xaxis = {
    labels: { datetimeUTC: false, format: "HH:mm" },
    tickAmount: this._config.days * 4, // a label every 6 hours
  };
  return {
    key: "timeline",
    head: `<div class="col-head">${parts.join('<span class="sub">·</span>')}</div>`,
    config,
  };
};

MlPvForecastCard.prototype._cardWidth = function () {
  return this._width || this.clientWidth || 800;
};

// Narrow cards: weekday names ("Tue") instead of "Tomorrow" / "Day after tomorrow".
MlPvForecastCard.prototype._dayName = function (offset, t) {
  if (offset === 0) return t.today;
  if (this._cardWidth() >= 500) return [t.tomorrow, t.dayAfter][offset - 1];
  const day = new Date(localMidnight(offset));
  return day.toLocaleDateString(this._lang(), { weekday: "short" });
};

MlPvForecastCard.prototype._plantStates = function () {
  return Object.values(this._hass.states)
    .filter((s) => isForecastCurve(s) && s.attributes.plant)
    .sort((a, b) => a.attributes.plant.localeCompare(b.attributes.plant));
};

MlPvForecastCard.prototype._plantsTable = function (t) {
  const days = [0, 1, 2].slice(0, this._config.days).map((d) => this._dayName(d, t));
  const keys = ["today", "tomorrow", "day_after_tomorrow"];
  // Today: forecast and, in brackets, the actual production so far.
  const actualKwh = (s) => (s.attributes.actual || []).reduce((sum, r) => sum + r.w, 0) / 4 / 1000;
  const sofar = (s) =>
    ` <span class="sub">(${actualKwh(s).toLocaleString(this._lang(), { maximumFractionDigits: 1 })})</span>`;
  const rows = this._plantStates().map((p, i) => {
    const daily = p.attributes.daily || {};
    const cells = days
      .map((_, d) => `<td>${this._kwh(daily[keys[d]])}${d === 0 ? sofar(p) : ""}</td>`)
      .join("");
    return `<tr><td style="--c:${PLANT_COLORS[i % PLANT_COLORS.length]}">${p.attributes.plant}</td>${cells}</tr>`;
  });
  const total = Object.values(this._hass.states).find(
    (s) => isForecastCurve(s) && !s.attributes.plant
  );
  if (total && rows.length > 1) {
    const daily = total.attributes.daily || {};
    const cells = days
      .map((_, d) => `<td><b>${this._kwh(daily[keys[d]])}</b>${d === 0 ? sofar(total) : ""}</td>`)
      .join("");
    rows.push(`<tr class="total"><td style="--c:transparent"><b>${t.total}</b></td>${cells}</tr>`);
  }
  return `<table class="plants"><tr><th></th>${days.map((d) => `<th>${d}</th>`).join("")}</tr>${rows.join("")}</table>`;
};

MlPvForecastCard.prototype._plantsSpec = function (t) {
  // The next days per plant, stacked, in one continuous chart.
  const plants = this._plantStates();
  const ids = plants.map((p) => p.entity_id);
  const hourly = (s) => s.attributes.hourly || [];
  const totals = {};
  plants.forEach((p) =>
    hourly(p).forEach((r) => (totals[r.time] = (totals[r.time] || 0) + r.p50 / 1000))
  );
  const marks = [1, 2].slice(0, this._config.days - 1).map((d, i) => ({
    x: localMidnight(i + 1),
    borderColor: "#9E9E9E",
    strokeDashArray: 0,
    label: {
      text: this._dayName(i + 1, t),
      orientation: "horizontal",
      position: "top",
      textAnchor: "start",
      offsetX: 6,
      offsetY: -4,
      borderWidth: 0,
      style: { background: "transparent", color: "#9E9E9E", fontSize: "12px" },
    },
  }));
  const config = this._chart({
    entity: plants[0].entity_id,
    start: localMidnight(),
    hours: this._config.days * 24,
    max: this._maxKw(Object.values(totals)),
    yAxis: true,
    now: true,
    height: 260,
    // Stacked by hand: series i shows the sum of plants 0..i. Drawn from the largest sum down, so
    // each plant's band sits on top of the ones before it.
    // Forecast as areas, today's actual production as light step lines, both stacked by plant.
    // (Bars of several series would be drawn side by side, not on top of each other.)
    series: [
      ...plants
        .map((p, i) =>
          this._series(p.entity_id, p.attributes.plant, PLANT_COLORS[i % PLANT_COLORS.length], GEN.stacked(ids.slice(0, i + 1)), {
            type: "area",
            opacity: 0.85,
            stroke_width: 1,
          })
        )
        .reverse(),
      ...plants
        .map((p, i) =>
          this._series(p.entity_id, `${p.attributes.plant} (${t.actual})`, PLANT_LIGHT[i % PLANT_LIGHT.length], GEN.stackedActual(ids.slice(0, i + 1), ids), {
            type: "line",
            curve: "stepline",
            stroke_width: 2.5,
          })
        )
        .reverse(),
    ],
  });
  // The tooltip shows each plant's own power: its sum minus the sum below it, which is the next
  // series within the same group (forecast areas first, then actual bars, n series each).
  config.apex_config.tooltip = {
    x: { format: "ddd HH:mm" },
    y: {
      formatter: `EVAL:function(value, { seriesIndex, dataPointIndex, w }) { const n = ${plants.length}; const below = seriesIndex % n < n - 1 ? w.globals.series[seriesIndex + 1] : null; const own = value - (below ? below[dataPointIndex] || 0 : 0); return own.toFixed(2) + ' kW'; }`,
    },
  };
  const daylight = hourly(plants[0]).filter((r) => r.p50 > 0);
  config.apex_config.annotations = {
    xaxis: [...marks, ...weatherMarks(
    daylight,
    localMidnight(),
    localMidnight(this._config.days),
    weatherStep(this._cardWidth(), this._config.days * 24),
    16
  )],
  };
  config.apex_config.xaxis = {
    labels: { datetimeUTC: false, format: "HH:mm" },
    tickAmount: this._config.days * 4,
  };
  return { key: `plants-${plants.length}`, config };
};

// Home Assistant swaps in a scoped custom element registry while it boots. This module is loaded
// next to the frontend and can run before that; a card defined then is lost. Define it once
// Home Assistant itself is defined, which happens after the swap.
const defineCard = () => {
  if (!customElements.get("ml-pv-forecast-card")) {
    customElements.define("ml-pv-forecast-card", MlPvForecastCard);
  }
};
if (customElements.get("home-assistant")) defineCard();
else customElements.whenDefined("home-assistant").then(defineCard);
window.customCards = window.customCards || [];
window.customCards.push({
  type: "ml-pv-forecast-card",
  name: "PV Forecast",
  description:
    "Machine-learning PV forecast: the next days (sun hours only), today with the hit rate so far, per plant and a review of past days.",
  preview: true,
  documentationURL: "https://github.com/dominikamann/ha-machine-learning-based-pv-forecast",
});
