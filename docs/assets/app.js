/* 台股趨勢雷達 - 共用 JS（無外部依賴，離線可用） */
"use strict";

var App = (function () {
  var NO_DATA = "資料暫無";

  /* ---------- 資料讀取 ---------- */
  function fetchJSON(path) {
    return fetch(path, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) { return { ok: true, data: data }; })
      .catch(function (e) { return { ok: false, error: e }; });
  }

  /* ---------- 格式化（null 安全） ---------- */
  function isValidNumber(x) {
    return typeof x === "number" && isFinite(x);
  }

  // 機率（0~1 分數）→ 百分比，一律標示為估計值由呼叫端加上 est-tag
  function fmtPct(x, digits) {
    if (!isValidNumber(x)) return NO_DATA;
    var d = digits == null ? 1 : digits;
    var v = x > 1 ? x : x * 100; // 防禦：若已是百分比則直接用
    if (x <= 1) v = x * 100;
    return v.toFixed(d) + "%";
  }

  function fmtPrice(x) {
    if (!isValidNumber(x)) return NO_DATA;
    // 台股報價慣例：只顯示數字（元/股），不加 NT$ 等貨幣前綴
    return x.toLocaleString("zh-TW", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtSignedPct(x, digits) {
    if (!isValidNumber(x)) return NO_DATA;
    var d = digits == null ? 1 : digits;
    var s = x > 0 ? "+" : "";
    return s + (x * 100).toFixed(d) + "%";
  }

  function fmtDate(s) {
    if (!s) return NO_DATA;
    return String(s).slice(0, 10);
  }

  function fmtDateTime(s) {
    if (!s) return NO_DATA;
    var t = String(s).replace("T", " ").slice(0, 16);
    return t;
  }

  function text(v) {
    if (v == null || v === "") return NO_DATA;
    return String(v);
  }

  /* ---------- DOM ---------- */
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  function estTag() {
    var s = el("span", "est-tag", "估計值");
    return s;
  }

  function trendBadge(trend) {
    var map = { "偏多": "bull", "盤整": "flat", "偏空": "bear" };
    var b = el("span", "trend " + (map[trend] || "flat"), text(trend));
    return b;
  }

  function probRow(labelText, value) {
    var row = el("div", "prob-row");
    var lab = el("div", "label", labelText + " ");
    lab.appendChild(estTag());
    var val = el("div", "value num", fmtPct(value));
    var bar = el("div", "prob-bar");
    var fill = el("div");
    if (isValidNumber(value)) {
      var v = Math.max(0, Math.min(1, value > 1 ? value / 100 : value));
      fill.style.width = (v * 100).toFixed(1) + "%";
    } else {
      fill.style.width = "0%";
    }
    bar.appendChild(fill);
    row.appendChild(lab);
    row.appendChild(val);
    row.appendChild(bar);
    return row;
  }

  function kvRow(k, v) {
    var row = el("div", "kv");
    row.appendChild(el("span", "k", k));
    var vv = el("span", "v");
    if (typeof v === "string") vv.textContent = v;
    else vv.appendChild(v);
    row.appendChild(vv);
    return row;
  }

  function sigList(items, cls) {
    var ul = el("ul", "sig-list " + cls);
    if (!items || !items.length) {
      ul.appendChild(el("li", "muted", "目前無資料"));
      return ul;
    }
    items.forEach(function (s) { ul.appendChild(el("li", "", String(s))); });
    return ul;
  }

  /* ---------- 價格走勢圖（SVG 自繪） ---------- */
  // prices: [[date, close, volume], ...] 依日期遞增
  function drawPriceChart(container, prices, opts) {
    opts = opts || {};
    container.innerHTML = "";
    var tip = el("div", "chart-tip");
    container.appendChild(tip);

    var W = 720, H = 300, padL = 8, padR = 64, padT = 16, padB = 44;
    var volH = 56; // 成交量區高度
    var priceH = H - padT - padB - volH;

    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "近 " + prices.length + " 個交易日收盤價走勢圖");

    function mk(tag, attrs) {
      var n = document.createElementNS(svgNS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      return n;
    }

    if (!prices || prices.length < 2) {
      var t = mk("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#8b96b3", "font-size": "16" });
      t.textContent = NO_DATA;
      svg.appendChild(t);
      container.appendChild(svg);
      return;
    }

    var closes = prices.map(function (p) { return p[1]; }).filter(isValidNumber);
    var vols = prices.map(function (p) { return p[2]; }).filter(isValidNumber);
    var minP = Math.min.apply(null, closes), maxP = Math.max.apply(null, closes);
    var span = (maxP - minP) || 1;
    minP -= span * 0.08; maxP += span * 0.08;
    var maxV = Math.max.apply(null, vols.concat([1]));

    var n = prices.length;
    function x(i) { return padL + (i / (n - 1)) * (W - padL - padR); }
    function y(p) { return padT + (1 - (p - minP) / (maxP - minP)) * priceH; }
    function yv(v) { return padT + priceH + 8 + (1 - v / maxV) * (volH - 8); }

    // 成交量柱
    var bw = Math.max(1, (W - padL - padR) / n * 0.6);
    prices.forEach(function (p, i) {
      if (!isValidNumber(p[2])) return;
      var r = mk("rect", {
        x: (x(i) - bw / 2).toFixed(1), y: yv(p[2]).toFixed(1),
        width: bw.toFixed(1), height: Math.max(1, (padT + priceH + volH - yv(p[2]))).toFixed(1),
        fill: "#26314d", opacity: "0.8"
      });
      svg.appendChild(r);
    });

    // 價格線
    var d = prices.map(function (p, i) {
      return (i === 0 ? "M" : "L") + x(i).toFixed(1) + "," + y(p[1]).toFixed(1);
    }).join(" ");
    var up = closes[closes.length - 1] >= closes[0];
    svg.appendChild(mk("path", {
      d: d, fill: "none", stroke: up ? "#e5484d" : "#3dd68c",
      "stroke-width": "2.2", "stroke-linejoin": "round", "stroke-linecap": "round"
    }));

    // 最高/最低標示
    var iMax = closes.indexOf(Math.max.apply(null, closes));
    var iMin = closes.indexOf(Math.min.apply(null, closes));
    [[iMax, "高"], [iMin, "低"]].forEach(function (pair) {
      var i = pair[0];
      svg.appendChild(mk("circle", { cx: x(i).toFixed(1), cy: y(prices[i][1]).toFixed(1), r: 3.5, fill: "#e8edf7" }));
      var t = mk("text", {
        x: Math.min(x(i), W - padR - 40), y: (y(prices[i][1]) - 8).toFixed(1),
        fill: "#8b96b3", "font-size": "11"
      });
      t.textContent = pair[1] + " " + prices[i][1].toFixed(2);
      svg.appendChild(t);
    });

    // 最新價虛線
    var lastP = closes[closes.length - 1];
    svg.appendChild(mk("line", {
      x1: padL, x2: W - padR, y1: y(lastP).toFixed(1), y2: y(lastP).toFixed(1),
      stroke: "#4f7cff", "stroke-dasharray": "5,4", "stroke-width": "1"
    }));
    var lt = mk("text", { x: W - padR + 4, y: (y(lastP) + 4).toFixed(1), fill: "#4f7cff", "font-size": "12", "font-weight": "bold" });
    lt.textContent = lastP.toFixed(2);
    svg.appendChild(lt);

    // X 軸日期（頭/中/尾）
    [0, Math.floor(n / 2), n - 1].forEach(function (i) {
      var t = mk("text", { x: x(i).toFixed(1), y: H - 8, fill: "#8b96b3", "font-size": "11", "text-anchor": "middle" });
      t.textContent = String(prices[i][0]).slice(5, 10);
      svg.appendChild(t);
    });

    // 互動：觸控/滑鼠顯示十字線
    var cross = mk("line", { y1: padT, y2: padT + priceH + volH, stroke: "#8b96b3", "stroke-width": "1", opacity: "0" });
    var dot = mk("circle", { r: 4, fill: "#fff", stroke: "#4f7cff", "stroke-width": "2", opacity: "0" });
    svg.appendChild(cross); svg.appendChild(dot);

    function showTip(evt) {
      var rect = svg.getBoundingClientRect();
      var cx = (evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left;
      var scaleX = W / rect.width;
      var svgX = cx * scaleX;
      var i = Math.round((svgX - padL) / ((W - padL - padR) / (n - 1)));
      i = Math.max(0, Math.min(n - 1, i));
      var p = prices[i];
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", "1");
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(p[1])); dot.setAttribute("opacity", "1");
      tip.style.display = "block";
      tip.textContent = fmtDate(p[0]) + " 收 " + (isValidNumber(p[1]) ? p[1].toFixed(2) : NO_DATA);
      var wrapRect = container.getBoundingClientRect();
      var tipX = (evt.touches ? evt.touches[0].clientX : evt.clientX) - wrapRect.left + 12;
      tip.style.left = Math.min(tipX, wrapRect.width - 150) + "px";
      tip.style.top = "8px";
    }
    function hideTip() {
      cross.setAttribute("opacity", "0"); dot.setAttribute("opacity", "0");
      tip.style.display = "none";
    }
    svg.addEventListener("pointermove", showTip);
    svg.addEventListener("pointerleave", hideTip);
    svg.addEventListener("pointerdown", showTip);

    container.appendChild(svg);
  }

  /* ---------- 可排序表格 ---------- */
  // columns: [{key, label, sortable, align, render(row)->node|string, getValue(row)->number|string}]
  function buildSortableTable(columns, rows, defaultSort) {
    var wrap = el("div", "table-wrap");
    var table = el("table", "rank");
    var thead = el("thead");
    var tr = el("tr");
    var state = { key: defaultSort.key, dir: defaultSort.dir }; // dir: 1 升冪, -1 降冪

    function renderHead() {
      tr.innerHTML = "";
      columns.forEach(function (c) {
        var th = el("th", c.sortable ? "sortable" : "", c.label);
        th.setAttribute("scope", "col");
        if (c.sortable) {
          th.setAttribute("role", "button");
          th.setAttribute("tabindex", "0");
          if (state.key === c.key) {
            var ind = el("span", "sort-ind", state.dir === -1 ? "▼" : "▲");
            th.appendChild(ind);
            th.setAttribute("aria-sort", state.dir === -1 ? "descending" : "ascending");
          }
          var toggle = function () {
            if (state.key === c.key) state.dir = -state.dir;
            else { state.key = c.key; state.dir = c.defaultDir || -1; }
            renderHead(); renderBody();
          };
          th.addEventListener("click", toggle);
          th.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
          });
        }
        tr.appendChild(th);
      });
    }

    var tbody = el("tbody");
    function renderBody() {
      tbody.innerHTML = "";
      var col = null;
      columns.forEach(function (c) { if (c.key === state.key) col = c; });
      var sorted = rows.slice();
      if (col && col.getValue) {
        sorted.sort(function (a, b) {
          var va = col.getValue(a), vb = col.getValue(b);
          var na = (va == null || !isFinite(va)) ? -Infinity : va;
          var nb = (vb == null || !isFinite(vb)) ? -Infinity : vb;
          return (na - nb) * state.dir;
        });
      }
      if (!sorted.length) {
        var emptyTr = el("tr");
        var td = el("td", "muted center", NO_DATA);
        td.colSpan = columns.length;
        emptyTr.appendChild(td);
        tbody.appendChild(emptyTr);
        return;
      }
      sorted.forEach(function (row) {
        var rtr = el("tr");
        columns.forEach(function (c) {
          var td = el("td");
          var v = c.render(row);
          if (typeof v === "string") td.textContent = v;
          else td.appendChild(v);
          rtr.appendChild(td);
        });
        if (row.symbol) {
          rtr.addEventListener("click", function () {
            var url = "stock.html?symbol=" + encodeURIComponent(row.symbol);
            if (window.__archiveDate) {
              url += "&date=" + encodeURIComponent(window.__archiveDate);
            }
            window.location.href = url;
          });
        }
        tbody.appendChild(rtr);
      });
    }

    renderHead(); renderBody();
    thead.appendChild(tr);
    table.appendChild(thead); table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function probCell(value) {
    var span = el("span", "num", fmtPct(value));
    if (isValidNumber(value)) span.appendChild(estTag());
    return span;
  }

  function getQueryParam(name) {
    try {
      return new URLSearchParams(window.location.search).get(name);
    } catch (e) { return null; }
  }

  return {
    NO_DATA: NO_DATA,
    fetchJSON: fetchJSON,
    isValidNumber: isValidNumber,
    fmtPct: fmtPct,
    fmtPrice: fmtPrice,
    fmtSignedPct: fmtSignedPct,
    fmtDate: fmtDate,
    fmtDateTime: fmtDateTime,
    text: text,
    el: el,
    estTag: estTag,
    trendBadge: trendBadge,
    probRow: probRow,
    probCell: probCell,
    kvRow: kvRow,
    sigList: sigList,
    drawPriceChart: drawPriceChart,
    buildSortableTable: buildSortableTable,
    getQueryParam: getQueryParam
  };
})();
