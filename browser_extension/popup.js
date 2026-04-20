const analyzeButton = document.querySelector("#analyze-button");
const statusLine = document.querySelector("#status-line");
const tabUrl = document.querySelector("#tab-url");
const modePill = document.querySelector("#mode-pill");
const emailSender = document.querySelector("#email-sender");
const emailSubject = document.querySelector("#email-subject");
const emailLinks = document.querySelector("#email-links");
const emailLinkList = document.querySelector("#email-link-list");
const decisionPill = document.querySelector("#decision-pill");
const scoreLine = document.querySelector("#score-line");
const urlScore = document.querySelector("#url-score");
const structuralScore = document.querySelector("#structural-score");
const nlpScore = document.querySelector("#nlp-score");
const keywords = document.querySelector("#keywords");

function setStatus(message) {
  statusLine.textContent = message;
}

function renderKeywords(items) {
  keywords.innerHTML = "";
  if (!items || items.length === 0) {
    const tag = document.createElement("span");
    tag.className = "tag muted";
    tag.textContent = "No explanation keywords";
    keywords.appendChild(tag);
    return;
  }
  items.forEach((item) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item;
    keywords.appendChild(tag);
  });
}

function renderLinks(items) {
  emailLinkList.innerHTML = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No extracted email links.";
    emailLinkList.appendChild(item);
    return;
  }
  items.slice(0, 5).forEach((entry) => {
    const item = document.createElement("li");
    const risk = entry.url_analysis
      ? ` | url risk ${Math.round((entry.url_analysis.final_risk_score || 0) * 100)}%`
      : "";
    item.textContent = `${entry.url}${risk}`;
    emailLinkList.appendChild(item);
  });
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function captureTabContext(tabId) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      const hostname = window.location.hostname.toLowerCase();
      const gmailMode = hostname.includes("mail.google.com");
      const outlookMode = hostname.includes("outlook.office.com") || hostname.includes("outlook.live.com");

      const textFromNodes = (selectors) => {
        for (const selector of selectors) {
          const node = document.querySelector(selector);
          if (node && node.innerText && node.innerText.trim()) {
            return node.innerText.trim();
          }
        }
        return "";
      };

      const attrFromNodes = (selectors, attribute) => {
        for (const selector of selectors) {
          const node = document.querySelector(selector);
          const value = node?.getAttribute?.(attribute);
          if (value && value.trim()) {
            return value.trim();
          }
        }
        return "";
      };

      const genericText = `${document.title || ""}\n${document.body?.innerText || ""}`.slice(0, 12000);
      const html = (document.documentElement?.outerHTML || "").slice(0, 120000);

      let sender = "";
      let subject = "";
      let bodyText = "";
      let mode = "page";
      let bodyNode = null;
      let mailboxHint = "";

      if (gmailMode) {
        mode = "email";
        subject = textFromNodes(["h2.hP", "h2[data-thread-perm-id]"]);
        sender =
          attrFromNodes(["span[email]", "h3 span[email]"], "email") ||
          textFromNodes(["span[email]", "h3 span[email]"]);
        bodyNode = document.querySelector("div.a3s") || document.querySelector("div[role='listitem'] div.a3s");
        bodyText = bodyNode?.innerText?.trim() || "";
        mailboxHint = window.location.hash || "";
      } else if (outlookMode) {
        mode = "email";
        subject = textFromNodes(["div[role='heading']", "[aria-label='Message header'] div"]);
        sender = textFromNodes([
          "[aria-label='From'] span",
          "[title][data-testid='messageSender']",
          "span[title*='@']",
        ]);
        bodyNode = document.querySelector("div[role='document']") || document.querySelector("[data-app-section='MailReadCompose']");
        bodyText = bodyNode?.innerText?.trim() || "";
        mailboxHint = window.location.pathname || "";
      }

      const allLinks = Array.from((mode === "email" && bodyNode ? bodyNode.querySelectorAll("a[href]") : document.querySelectorAll("a[href]")))
        .map((node) => node.href)
        .filter((href) => /^https?:/i.test(href));
      const filteredLinks =
        mode === "email"
          ? allLinks.filter((href) => {
              const lower = href.toLowerCase();
              return !(
                lower.includes("mail.google.com") ||
                lower.includes("support.google.com/mail") ||
                lower.includes("outlook.office.com/mail") ||
                lower.includes("outlook.live.com/mail")
              );
            })
          : allLinks;
      const dedupedLinks = [...new Set(filteredLinks)];

      const emailText = `${subject}\n${bodyText}`.trim();
      return {
        mode,
        text: genericText,
        html,
        email: {
          sender,
          subject,
          body: bodyText,
          combinedText: emailText,
          links: dedupedLinks.slice(0, 20),
          mailboxHint,
        },
      };
    },
  });
  return result?.result || {
    mode: "page",
    text: "",
    html: "",
    email: { sender: "", subject: "", body: "", combinedText: "", links: [], mailboxHint: "" },
  };
}

async function analyzeUrl(url) {
  const response = await fetch("http://127.0.0.1:8080/api/analyze/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    return null;
  }
  return await response.json();
}

function applyAnalysisToUi(payload, { mode, emailMeta, linkAnalyses }) {
  const explanation = payload.explanation || {};
  const nlp = explanation.nlp || {};
  modePill.className = mode === "email" ? "pill good" : "pill neutral";
  modePill.textContent = mode === "email" ? "Email Mode" : "Generic Page";
  emailSender.textContent = emailMeta.sender || "-";
  emailSubject.textContent = emailMeta.subject || "-";
  emailLinks.textContent = String((emailMeta.links || []).length);
  renderLinks(linkAnalyses);

  decisionPill.className = payload.decision === "BLOCK" ? "pill bad" : "pill good";
  decisionPill.textContent = payload.decision;
  scoreLine.textContent = `Final risk score ${Math.round((payload.final_risk_score || 0) * 100)}%`;
  if (mode === "email") {
    urlScore.textContent = `${Math.round((explanation.max_link_risk || 0) * 100)}%`;
    structuralScore.textContent = "N/A";
  } else {
    urlScore.textContent = `${Math.round((explanation.url?.final_risk_score || 0) * 100)}%`;
    structuralScore.textContent = `${Math.round((explanation.structural?.final_risk_score || 0) * 100)}%`;
  }
  nlpScore.textContent = `${Math.round((nlp.phishing_probability || 0) * 100)}%`;
  renderKeywords(nlp.explanation_keywords || []);
}

async function analyzeCurrentTab() {
  setStatus("Inspecting active tab...");
  const tab = await getActiveTab();
  if (!tab || !tab.url || !/^https?:/i.test(tab.url)) {
    throw new Error("Open a normal http/https page first.");
  }
  tabUrl.textContent = tab.url;
  const context = await captureTabContext(tab.id);
  const emailMeta = context.email || { sender: "", subject: "", combinedText: "", links: [] };
  const analysisText =
    context.mode === "email"
      ? (emailMeta.combinedText || context.text || tab.title || tab.url)
      : (context.text || tab.title || tab.url);
  const endpoint = context.mode === "email" ? "/api/analyze/email" : "/api/analyze/full";
  const requestPayload =
    context.mode === "email"
      ? {
          sender: emailMeta.sender,
          subject: emailMeta.subject,
          body: emailMeta.body,
          links: emailMeta.links,
          mailbox_hint: emailMeta.mailboxHint,
        }
      : {
          url: tab.url,
          text: analysisText,
          html_content: context.html,
        };

  setStatus(context.mode === "email" ? "Email mode detected. Analyzing message..." : "Sending page to local phishing detector...");
  const response = await fetch(`http://127.0.0.1:8080${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Local detector request failed.");
  }

  const linkAnalyses = [];
  for (const url of (emailMeta.links || []).slice(0, 3)) {
    const urlAnalysis = await analyzeUrl(url);
    linkAnalyses.push({ url, url_analysis: urlAnalysis });
  }
  applyAnalysisToUi(payload, { mode: context.mode, emailMeta, linkAnalyses });
  setStatus(`Analysis complete. Decision: ${payload.decision}.`);
}

analyzeButton.addEventListener("click", () => {
  analyzeCurrentTab().catch((error) => {
    setStatus(error.message);
    decisionPill.className = "pill neutral";
    decisionPill.textContent = "Error";
  });
});
