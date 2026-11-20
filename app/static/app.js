/* Property Guide chat client — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const els = {
    chat: $("#chat"), messages: $("#messages"), empty: $("#empty"), suggestions: $("#suggestions"),
    stats: $("#stats"), form: $("#composer"), input: $("#input"), send: $("#send"), newChat: $("#newChat"),
    subtitle: $("#subtitle"), themeBtn: $("#themeBtn"),
  };
  const STORE_KEY = "pg.chat.v1";
  const THEME_KEY = "pg.theme";
  let history = [];          // [{role, content}] sent to the API
  let busy = false;
  let lastQuestion = "";

  // ---------------------------------------------------------------- storage (best effort only)
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
    del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
  };

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children) if (c != null) n.append(c);
    return n;
  }
  const isArabic = (s) => (s.match(/[؀-ۿ]/g) || []).length > s.replace(/\s/g, "").length * 0.3;
  const safeUrl = (u) => { try { const x = new URL(u, location.href); return /^https?:$/.test(x.protocol) ? x.href : "#"; } catch { return "#"; } };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const ICON_COPY = ["M9 9h11v11H9z", "M5 15H4V4h11v1"];
  const ICON_CHECK = ["M5 12l5 5L20 7"];
  function icon(paths) {
    const s = document.createElementNS(SVG_NS, "svg");
    s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("aria-hidden", "true");
    for (const d of paths) { const p = document.createElementNS(SVG_NS, "path"); p.setAttribute("d", d); s.append(p); }
    return s;
  }
  const scrollDown = () => { els.chat.scrollTop = els.chat.scrollHeight; };
  const nearBottom = () => els.chat.scrollHeight - els.chat.scrollTop - els.chat.clientHeight < 120;

  function renderMarkdown(target, text, sources) {
    const libs = window.marked && window.DOMPurify;
    if (!libs) { target.textContent = text; return; }
    const html = window.DOMPurify.sanitize(window.marked.parse(text, { breaks: true, gfm: true }), {
      FORBID_TAGS: ["style", "img", "iframe", "form", "input"], FORBID_ATTR: ["style"],
    });
    target.innerHTML = html;
    target.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; a.href = safeUrl(a.getAttribute("href")); });
    linkCitations(target, sources || []);
    target.dir = isArabic(text) ? "rtl" : "ltr";
  }

  // Turn "[3]" in text nodes into small citation links (DOM walk -> no HTML injection).
  function linkCitations(root, sources) {
    if (!sources.length) return;
    const byN = new Map(sources.map((s) => [String(s.n), s]));
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement.closest("a, code, pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) if (/\[\d+(?:\s*,\s*\d+)*\]/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const frag = document.createDocumentFragment();
      let last = 0;
      const text = node.nodeValue;
      text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, nums, idx) => {
        frag.append(text.slice(last, idx));
        for (const n of nums.split(/\s*,\s*/)) {
          const s = byN.get(n);
          frag.append(s ? el("a", { class: "cite", href: safeUrl(s.url), target: "_blank", rel: "noopener noreferrer", title: s.title, text: n }) : `[${n}]`);
        }
        last = idx + m.length;
        return m;
      });
      frag.append(text.slice(last));
      node.replaceWith(frag);
    }
  }

  // ---------------------------------------------------------------- message views
  function addUser(text) {
    els.empty.hidden = true;
    const li = el("li", { class: "msg user" }, el("div", { class: "bubble", text, dir: isArabic(text) ? "rtl" : "auto" }));
    els.messages.append(li);
    scrollDown();
  }

  function addBot() {
    els.empty.hidden = true;
    const view = {
      li: el("li", { class: "msg bot" }),
      status: el("div", { class: "status", "aria-live": "polite" }),
      bubble: el("div", { class: "bubble" }, el("span", { class: "typing", "aria-label": "Thinking" }, el("i"), el("i"), el("i"))),
      sources: el("section", { class: "sources", "aria-label": "Sources", hidden: true }),
      srcFoot: el("div", { class: "attrib" }),
      followups: el("div", { class: "followups", hidden: true }),
      toolbar: el("div", { class: "toolbar" }),
      disclaimer: el("div", { class: "disclaimer", hidden: true, text: "This is an AI-generated response. Please verify the information before taking action." }),
      text: "", srcList: [], cardItems: [], srcOpen: false, requestId: null, started: false,
    };
    view.li.append(view.status, view.bubble, view.disclaimer, view.toolbar, view.sources, view.followups);
    els.messages.append(view.li);
    scrollDown();
    return view;
  }

  function setStatus(view, text) {
    view.status.querySelectorAll(".step.active").forEach((s) => s.classList.remove("active"));
    view.status.append(el("div", { class: "step active", text }));
  }

  // Sources: one row of small chips under the answer, each source once (cards only add sources the
  // answer's list lacks). Only the first SRC_PREVIEW show until "View all".
  const SRC_PREVIEW = 3;

  function renderCards(view, items) { view.cardItems = items || []; renderSourceBox(view); }
  function renderSources(view, sources) { view.srcList = sources || []; renderSourceBox(view); }

  function renderSourceBox(view) {
    const byN = new Map(view.srcList.map((s) => [String(s.n), { ...s }]));
    for (const c of view.cardItems) {
      if (!byN.has(String(c.n))) byN.set(String(c.n), { n: c.n, site: c.site, url: c.url, title: c.title });
    }
    const all = [...byN.values()].sort((a, b) => a.n - b.n);
    const box = view.sources;
    box.hidden = all.length === 0;
    if (!all.length) return;
    const chips = all.map((s, i) => el("a", { class: i < SRC_PREVIEW ? "src" : "src extra", href: safeUrl(s.url), target: "_blank", rel: "noopener noreferrer", title: s.title },
      el("span", { class: "n", text: s.n }),
      el("span", { class: `badge ${s.site}`, text: s.site === "wasalt" ? "Wasalt" : "DarGlobal" }),
      el("span", { class: "tt", text: s.title || s.url })));
    const toggle = all.length > SRC_PREVIEW ? el("button", { type: "button", class: "src-toggle", "aria-expanded": String(view.srcOpen),
      text: view.srcOpen ? "Show less" : `View all ${all.length}`,
      onclick: () => { view.srcOpen = !view.srcOpen; renderSourceBox(view); } }) : null;
    box.classList.toggle("open", view.srcOpen);
    box.replaceChildren(el("div", { class: "label", text: "Sources" }),
      el("div", { class: "src-row" }, ...chips, toggle), view.srcFoot);
  }

  let rafPending = false;
  function paint(view) {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      const stick = nearBottom();
      renderMarkdown(view.bubble, view.text, view.srcList);
      if (stick) scrollDown();
    });
  }

  function finish(view, done) {
    view.status.querySelectorAll(".step.active").forEach((s) => s.classList.remove("active"));
    if (view.text) renderMarkdown(view.bubble, view.text, view.srcList);
    if (view.srcList.some((s) => s.site === "wasalt") && !/Source:\s*Wasalt\.sa/i.test(view.text)) {
      view.srcFoot.textContent = "Source: Wasalt.sa";
    }
    view.requestId = done && done.request_id;
    const copyBtn = el("button", { type: "button", title: "Copy answer", "aria-label": "Copy answer",
      onclick: async () => {
        try { await navigator.clipboard.writeText(view.text); copyBtn.replaceChildren(icon(ICON_CHECK)); copyBtn.title = "Copied"; }
        catch { copyBtn.title = "Copy failed"; }
        setTimeout(() => { copyBtn.replaceChildren(icon(ICON_COPY)); copyBtn.title = "Copy answer"; }, 1500);
      } }, icon(ICON_COPY));
    const vote = (rating, btn) => async () => {
      view.toolbar.querySelectorAll("[aria-pressed]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      try { await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message_id: view.requestId || "unknown", rating }) }); } catch { /* ignore */ }
    };
    const up = el("button", { type: "button", title: "Helpful", "aria-label": "Helpful", "aria-pressed": "false", text: "👍" });
    const down = el("button", { type: "button", title: "Not helpful", "aria-label": "Not helpful", "aria-pressed": "false", text: "👎" });
    up.addEventListener("click", vote("up", up));
    down.addEventListener("click", vote("down", down));
    const meta = [];
    if (done && done.cached) meta.push("cached");
    if (done && done.model) meta.push(done.model.replace(/:free$/, ""));
    if (done && done.latency_ms != null) meta.push(`${(done.latency_ms / 1000).toFixed(1)}s`);
    view.toolbar.append(copyBtn, up, down, el("span", { text: meta.join(" · ") }));
    if (done && done.model) view.disclaimer.hidden = false;  // not on canned refusals
  }

  function showError(view, message) {
    view.status.querySelectorAll(".step.active").forEach((s) => s.classList.remove("active"));
    view.bubble.className = "bubble error";
    view.bubble.replaceChildren(message, el("button", { type: "button", text: "Retry", onclick: () => { view.li.remove(); history = history.slice(0, -1); ask(lastQuestion, true); } }));
  }

  // ---------------------------------------------------------------- streaming
  async function ask(question, isRetry = false) {
    question = (question || "").trim();
    if (!question || busy) return;
    busy = true; lastQuestion = question;
    els.send.disabled = true;
    if (!isRetry) addUser(question);
    const view = addBot();
    const sentHistory = history.slice(-10).map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));
    history.push({ role: "user", content: question });
    let done = null;
    try {
      const resp = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message: question.slice(0, 1000), history: sentHistory }),
      });
      if (!resp.ok || !resp.body) {
        let msg = resp.status === 429 ? "You're sending messages quickly — please wait a minute and try again." : "The server couldn't answer that.";
        try { const j = await resp.json(); if (resp.status !== 422 && j.detail && typeof j.detail === "string") msg = j.detail; } catch { /* ignore */ }
        throw new Error(msg);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done: eof } = await reader.read();
        if (eof) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
          let event = "message", data = "";
          for (const line of raw.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;
          let payload; try { payload = JSON.parse(data); } catch { continue; }
          handle(view, event, payload);
          if (event === "done") done = payload;
          if (event === "error") throw new Error(payload.message || "Something went wrong.");
        }
      }
      if (!view.text) throw new Error("No answer was received. Please try again.");
      finish(view, done);
      history.push({ role: "assistant", content: view.text });
      persist();
    } catch (err) {
      history.pop();
      showError(view, err.message || "Network error — please check your connection.");
    } finally {
      busy = false;
      els.send.disabled = false;
      els.input.focus();
    }
  }

  function handle(view, event, d) {
    switch (event) {
      case "status": setStatus(view, d.text); break;
      case "meta": renderSources(view, d.sources); if (view.text) paint(view); break;
      case "cards": renderCards(view, d.items || []); break;
      case "token": view.text += d.text; paint(view); break;
      case "followups":
        view.followups.replaceChildren(...(d.items || []).slice(0, 3).map((q) => el("button", { type: "button", class: "chip", text: q, onclick: () => ask(q) })));
        view.followups.hidden = !(d.items || []).length;
        break;
      default: break;
    }
  }

  // ---------------------------------------------------------------- persistence
  function persist() { store.set(STORE_KEY, history.slice(-20)); }
  function restore() {
    const saved = store.get(STORE_KEY);
    if (!Array.isArray(saved) || !saved.length) return;
    history = saved.filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string");
    for (const m of history) {
      if (m.role === "user") addUser(m.content);
      else { const v = addBot(); v.text = m.content; renderMarkdown(v.bubble, m.content, []); }
    }
  }

  // ---------------------------------------------------------------- boot
  function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
  applyTheme(store.get(THEME_KEY));
  els.themeBtn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    applyTheme(next); store.set(THEME_KEY, next);
  });

  els.form.addEventListener("submit", (e) => { e.preventDefault(); const q = els.input.value; els.input.value = ""; autosize(); ask(q); });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.form.requestSubmit(); }
  });
  const autosize = () => {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
    els.input.style.overflowY = els.input.scrollHeight > 160 ? "auto" : "hidden";  // scrollbar only once it's full
  };
  els.input.addEventListener("input", autosize);
  els.newChat.addEventListener("click", () => {
    if (busy) return;
    history = []; store.del(STORE_KEY); els.messages.replaceChildren(); els.empty.hidden = false; els.input.focus();
  });

  fetch("/api/suggestions").then((r) => r.json()).then((d) => {
    els.suggestions.replaceChildren(...(d.items || []).map((q) => el("button", { type: "button", class: "chip", role: "listitem", text: q, onclick: () => ask(q) })));
  }).catch(() => {});
  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (h.scrape_date) els.subtitle.textContent = `Answers from public data collected on ${h.scrape_date}`;
    els.stats.textContent = `${(h.listings || 0).toLocaleString()} Wasalt listings · ${h.projects || 0} DarGlobal projects · ${(h.chunks || 0).toLocaleString()} indexed passages`;
    if (h.repo_url && safeUrl(h.repo_url) !== "#") { $("#repoLink").href = safeUrl(h.repo_url); $("#repoWrap").hidden = false; }
  }).catch(() => {});
  restore();
  els.input.focus();
})();
