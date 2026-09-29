const conversation = document.getElementById("conversation");
const emptyChatInvitation = document.getElementById("empty-chat-invitation");
const jumpToLatest = document.getElementById("jump-to-latest");
const recoveryPanel = document.getElementById("recovery-panel");
const chatAnnouncements = document.getElementById("chat-announcements");
const form = document.getElementById("chat-form");
const prompt = document.getElementById("prompt");
const sendButton = document.getElementById("send-button");
const socketBanner = document.getElementById("socket-banner");
const previewPlaceholder = document.getElementById("preview-placeholder");
const previewHeader = document.getElementById("preview-header");
const previewTitle = document.getElementById("preview-title");
const previewSubtitle = document.getElementById("preview-subtitle");
const previewOpenLink = document.getElementById("preview-open-link");
const previewMetadata = document.getElementById("preview-metadata");
const previewFrameWrap = document.getElementById("preview-frame-wrap");
const previewState = document.getElementById("preview-state");
const previewFallback = document.getElementById("preview-fallback");
const previewFallbackLink = document.getElementById("preview-fallback-link");
let previewFrame = document.getElementById("preview-frame");
const historyList = document.getElementById("history-list");
const historyStatus = document.getElementById("history-status");
const historyFeedback = document.getElementById("history-feedback");
const archiveStatus = document.getElementById("archive-status");
const historyRefreshButton = document.getElementById("history-refresh");
const chatLayout = document.getElementById("chat-layout");
const historyColumn = document.getElementById("history-panel");
const previewColumn = document.getElementById("preview-panel");
const historyToggle = document.getElementById("history-toggle");
const previewClose = document.getElementById("preview-close");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const connectionStatus = document.getElementById("connection-status");
const helpDialog = document.getElementById("help-dialog");
const helpButton = document.getElementById("help-button");
const newConversationButton = document.getElementById("new-conversation");
const renameConversationButton = document.getElementById("rename-conversation");
const deleteConversationButton = document.getElementById("delete-conversation");
const renameDialog = document.getElementById("rename-dialog");
const renameForm = document.getElementById("rename-form");
const renameInput = document.getElementById("rename-input");
const renameError = document.getElementById("rename-error");
const renameSaveButton = document.getElementById("rename-save");
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const basePath = window.location.pathname.replace(/\/chat\/?$/, "");
const wsPath = `${basePath}/ws/chat`.replace(/\/{2,}/g, "/");
let socket = null;
let socketGeneration = 0;
let activeTurn = null;
let recovery = null;
let recoveryNeedsRefresh = false;
let readerNearBottom = true;
let markdownFallbackShown = false;
let composing = false;
const turns = new Map();
let selectedConversationId = null;
let conversations = [];
let historyReady = false;
let lifecycleLoading = false;
let lifecycleMutation = false;
let chatBusy = false;
let historyRequest = 0;
let selectionRequest = 0;
let historyRetry = null;
let drawerOpener = null;
let openDrawerName = null;
let selectedSource = null;
let previewGeneration = 0;

updateLifecycleControls();

function drawerIsModal(name) {
  return name === "history"
    ? window.matchMedia("(max-width: 1199px)").matches
    : window.matchMedia("(max-width: 1439px)").matches;
}

function setDrawerInert(name) {
  const modal = Boolean(name && drawerIsModal(name));
  document.querySelector(".paperless-topbar").inert = modal;
  document.querySelector(".chat-column").inert = modal;
  historyColumn.inert =
    (drawerIsModal("history") && name !== "history") ||
    (modal && name !== "history");
  previewColumn.inert =
    (drawerIsModal("inspector") && name !== "inspector") ||
    (modal && name !== "inspector") ||
    (window.matchMedia("(min-width: 1440px)").matches &&
      chatLayout.classList.contains("inspector-closed"));
  historyColumn.setAttribute("aria-modal", String(modal && name === "history"));
  previewColumn.setAttribute(
    "aria-modal",
    String(modal && name === "inspector"),
  );
}
setDrawerInert(null);

function openDrawer(name, opener = document.activeElement) {
  if (
    (name === "history" && window.matchMedia("(min-width:1200px)").matches) ||
    (name === "inspector" && window.matchMedia("(min-width:1440px)").matches)
  ) {
    if (name === "inspector") {
      drawerOpener = opener;
      chatLayout.classList.remove("inspector-closed");
      setDrawerInert(null);
    }
    return;
  }
  openDrawerName = name;
  drawerOpener = opener;
  historyToggle.setAttribute("aria-expanded", String(name === "history"));
  chatLayout.dataset.historyOpen = String(name === "history");
  chatLayout.dataset.inspectorOpen = String(name === "inspector");
  drawerBackdrop.classList.add("active");
  setDrawerInert(name);
  const pane = name === "history" ? historyColumn : previewColumn;
  pane.querySelector("button, a, [tabindex='0']")?.focus();
}

function closeDrawer({ restoreFocus = true } = {}) {
  const previous = openDrawerName;
  openDrawerName = null;
  historyToggle.setAttribute("aria-expanded", "false");
  delete chatLayout.dataset.historyOpen;
  delete chatLayout.dataset.inspectorOpen;
  drawerBackdrop.classList.remove("active");
  if (!previous && window.matchMedia("(min-width:1440px)").matches) {
    chatLayout.classList.add("inspector-closed");
  }
  setDrawerInert(null);
  if (
    restoreFocus &&
    drawerOpener?.isConnected &&
    !drawerOpener.closest("[inert]")
  )
    drawerOpener.focus();
  drawerOpener = null;
}

function syncDrawerMode() {
  if (openDrawerName && !drawerIsModal(openDrawerName)) {
    const name = openDrawerName;
    closeDrawer({ restoreFocus: false });
    if (name === "inspector") {
      chatLayout.classList.remove("inspector-closed");
      setDrawerInert(null);
    }
    document.getElementById("page-heading").focus({ preventScroll: true });
  } else if (openDrawerName) {
    setDrawerInert(openDrawerName);
  } else if (selectedSource && drawerIsModal("inspector")) {
    openDrawer("inspector", selectedSource.opener);
  } else {
    setDrawerInert(null);
  }
  if (document.activeElement.closest("[inert]"))
    document.getElementById("page-heading").focus({ preventScroll: true });
}

historyToggle.addEventListener("click", (event) =>
  openDrawer("history", event.currentTarget),
);
previewClose.addEventListener("click", () =>
  resetPreview({ restoreFocus: true }),
);
drawerBackdrop.addEventListener("click", () => {
  if (openDrawerName === "inspector") resetPreview({ restoreFocus: true });
  else closeDrawer();
});
window.addEventListener("resize", syncDrawerMode);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && openDrawerName) {
    event.preventDefault();
    if (openDrawerName === "inspector") resetPreview({ restoreFocus: true });
    else closeDrawer();
  }
  if (event.key !== "Tab" || !openDrawerName || !drawerIsModal(openDrawerName))
    return;
  const pane = openDrawerName === "history" ? historyColumn : previewColumn;
  const focusable = [
    ...pane.querySelectorAll(
      "button:not(:disabled),a[href],textarea:not(:disabled),[tabindex]:not([tabindex='-1'])",
    ),
  ].filter((element) => !element.hidden && element.getClientRects().length);
  const first = focusable[0];
  const last = focusable.at(-1);
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last?.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first?.focus();
  }
});
helpButton.addEventListener("click", () => helpDialog.showModal());
helpDialog.addEventListener("close", () => helpButton.focus());

function apiPath(path) {
  return `${basePath}${path}`.replace(/\/{2,}/g, "/");
}

async function api(path, options = {}) {
  const response = await fetch(apiPath(path), {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || "Request failed.");
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

function lifecycleBusy() {
  return lifecycleLoading || chatBusy;
}

function updateLifecycleControls() {
  const mutationDisabled = lifecycleBusy() || !historyReady;
  newConversationButton.disabled = mutationDisabled;
  renameConversationButton.hidden = !selectedConversationId;
  deleteConversationButton.hidden = !selectedConversationId;
  renameConversationButton.disabled =
    mutationDisabled || !selectedConversationId;
  deleteConversationButton.disabled =
    mutationDisabled || !selectedConversationId;
  historyList.querySelectorAll("button").forEach((button) => {
    button.disabled =
      chatBusy ||
      lifecycleMutation ||
      button.dataset.conversationId === selectedConversationId;
  });
  historyRefreshButton.disabled = lifecycleLoading;
  sendButton.disabled =
    !prompt.value.trim() ||
    !historyReady ||
    lifecycleLoading ||
    chatBusy ||
    recoveryNeedsRefresh ||
    !socket ||
    socket.readyState !== WebSocket.OPEN;
}

// Step 03 calls this hook when a turn starts and completes.
function setChatBusy(busy) {
  chatBusy = Boolean(busy);
  historyFeedback.textContent = chatBusy
    ? "A response is in progress. Conversation controls are temporarily unavailable."
    : "";
  updateLifecycleControls();
}

function setLifecycleLoading(loading) {
  lifecycleLoading = loading;
  updateLifecycleControls();
}

function setLifecycleMutation(loading) {
  lifecycleMutation = loading;
  setLifecycleLoading(loading);
}

function setHistoryError(message, retry) {
  historyStatus.replaceChildren();
  historyFeedback.textContent = "";
  const text = document.createElement("span");
  text.textContent = message;
  historyStatus.appendChild(text);
  historyRetry = retry;
  if (retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-sm btn-outline-danger ms-2";
    button.textContent = "Retry";
    button.addEventListener("click", () => historyRetry?.());
    historyStatus.appendChild(button);
  }
}

function clearHistoryError() {
  historyStatus.replaceChildren();
  historyFeedback.textContent = "";
  historyRetry = null;
}

function resetPreview({ restoreFocus = false } = {}) {
  const opener = selectedSource?.opener;
  previewGeneration += 1;
  previewFrame.removeAttribute("src");
  previewFrame.replaceWith(previewFrame.cloneNode(false));
  previewFrame = document.getElementById("preview-frame");
  previewFrameWrap.classList.remove("active");
  previewHeader.classList.remove("active");
  previewMetadata.hidden = true;
  previewMetadata.querySelector("dl").replaceChildren();
  previewFallback.hidden = true;
  previewState.textContent = "";
  previewPlaceholder.hidden = false;
  previewOpenLink.removeAttribute("href");
  previewFallbackLink.removeAttribute("href");
  selectedSource = null;
  document
    .querySelectorAll(".source-card.selected")
    .forEach((card) => card.classList.remove("selected"));
  document
    .querySelectorAll(
      ".source-select[aria-pressed='true'], .source-preview[aria-pressed='true']",
    )
    .forEach((button) => button.setAttribute("aria-pressed", "false"));
  chatLayout.classList.add("inspector-closed");
  if (openDrawerName === "inspector") closeDrawer({ restoreFocus: false });
  else setDrawerInert(null);
  if (restoreFocus && opener?.isConnected) opener.focus();
}

function clearConversationView() {
  resetPreview();
  conversation.replaceChildren();
  emptyChatInvitation.hidden = false;
  conversation.appendChild(emptyChatInvitation);
  turns.clear();
  readerNearBottom = true;
  jumpToLatest.hidden = true;
}

function dateGroup(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Older";
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const day = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const ordinal = (value) =>
    Date.UTC(value.getFullYear(), value.getMonth(), value.getDate()) / 86400000;
  const diff = ordinal(today) - ordinal(day);
  return diff === 0
    ? "Today"
    : diff >= 1 && diff <= 7
      ? "Previous 7 days"
      : "Older";
}

function formatHistoryTime(timestamp, group) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return group === "Today"
    ? new Intl.DateTimeFormat(navigator.languages, {
        hour: "numeric",
        minute: "2-digit",
      }).format(date)
    : new Intl.DateTimeFormat(navigator.languages, {
        month: "short",
        day: "numeric",
        year:
          date.getFullYear() === new Date().getFullYear()
            ? undefined
            : "numeric",
      }).format(date);
}

function renderConversationList() {
  historyList.replaceChildren();
  if (!conversations.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = historyReady
      ? "No conversations yet."
      : "Loading conversations…";
    historyList.appendChild(empty);
  }
  const groups = new Map([
    ["Today", []],
    ["Previous 7 days", []],
    ["Older", []],
  ]);
  conversations.forEach((item) =>
    groups.get(dateGroup(item.updated_at)).push(item),
  );
  groups.forEach((items, label) => {
    if (!items.length) return;
    const group = document.createElement("section");
    group.className = "history-group";
    const heading = document.createElement("h3");
    heading.textContent = label;
    group.appendChild(heading);
    items.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `history-item ${item.id === selectedConversationId ? "active" : ""}`;
      button.dataset.conversationId = item.id;
      button.title = item.title;
      button.setAttribute(
        "aria-label",
        `${item.title}, updated ${new Date(item.updated_at).toLocaleString(navigator.languages)}`,
      );
      if (item.id === selectedConversationId)
        button.setAttribute("aria-current", "page");
      const title = document.createElement("span");
      title.className = "history-item-title";
      title.textContent = item.title;
      const time = document.createElement("time");
      time.dateTime = item.updated_at;
      time.textContent = formatHistoryTime(item.updated_at, label);
      time.title = new Date(item.updated_at).toLocaleString(
        navigator.languages,
      );
      button.append(title, time);
      button.addEventListener("click", () => selectConversation(item.id));
      group.appendChild(button);
    });
    historyList.appendChild(group);
  });
  updateLifecycleControls();
}

async function refreshConversations() {
  const request = ++historyRequest;
  const items = (await api("/conversations")).items;
  if (request !== historyRequest) return conversations;
  conversations = items;
  renderConversationList();
  return conversations;
}

async function createConversation({ preserveDraft = false } = {}) {
  if (lifecycleBusy()) return;
  if (
    selectedConversationId &&
    !conversation.querySelector(".bubble.user, .turn") &&
    !prompt.value.trim()
  ) {
    prompt.focus();
    return;
  }
  if (
    !preserveDraft &&
    prompt.value.trim() &&
    !window.confirm("Discard the unsent draft and start a new conversation?")
  )
    return;
  const draftRevision = promptRevision;
  setLifecycleMutation(true);
  clearHistoryError();
  try {
    const item = await api("/conversations", { method: "POST", body: "{}" });
    selectedConversationId = item.id;
    historyReady = true;
    clearConversationView();
    if (!preserveDraft && promptRevision === draftRevision) prompt.value = "";
    renderRecovery();
    if (openDrawerName === "history") closeDrawer({ restoreFocus: false });
    conversations = [
      item,
      ...conversations.filter((conversation) => conversation.id !== item.id),
    ];
    renderConversationList();
    await refreshConversations().catch((error) =>
      setHistoryError(
        `Conversation created, but history could not refresh: ${error.message}`,
        () =>
          refreshConversations()
            .then(clearHistoryError)
            .catch((retryError) =>
              setHistoryError(
                `Unable to refresh conversations: ${retryError.message}`,
                () => retryHistory(),
              ),
            ),
      ),
    );
    prompt.focus();
  } catch (error) {
    setHistoryError(`Unable to create a conversation: ${error.message}`, () =>
      refreshConversations()
        .then(clearHistoryError)
        .catch((retryError) =>
          setHistoryError(
            `Unable to refresh conversations: ${retryError.message}`,
            () => retryHistory(),
          ),
        ),
    );
  } finally {
    setLifecycleMutation(false);
  }
}

async function selectConversation(
  conversationId,
  { discardDraft = false, force = false, preserveDraft = false } = {},
) {
  if (
    !conversationId ||
    chatBusy ||
    lifecycleMutation ||
    (conversationId === selectedConversationId && !force)
  )
    return;
  if (
    !discardDraft &&
    prompt.value.trim() &&
    !window.confirm("Discard the unsent draft and switch conversations?")
  )
    return;
  const request = ++selectionRequest;
  const draftRevision = promptRevision;
  setLifecycleLoading(true);
  clearHistoryError();
  historyFeedback.textContent = "Loading conversation…";
  try {
    const item = await api(`/conversations/${conversationId}`);
    if (request !== selectionRequest) return false;
    if (!preserveDraft && promptRevision !== draftRevision) {
      historyFeedback.textContent =
        "Draft changed while loading. Conversation switch canceled; the draft was kept.";
      return false;
    }
    selectedConversationId = item.id;
    historyReady = true;
    clearConversationView();
    if (openDrawerName === "history") closeDrawer({ restoreFocus: false });
    item.messages.forEach((message) => {
      if (message.role === "user") {
        addUserBubble(message.content, message.created_at);
        return;
      }
      createTurn(message.id, message.created_at);
      setAssistantMessage(message.id, message.content);
      (message.tool_activity || []).forEach((tool, position) =>
        updateTool(message.id, tool, false, position),
      );
      updateUsage(message.id, {
        available: Boolean(message.usage),
        model: message.model,
        ...(message.usage || {}),
      });
      renderSources(message.id, message.sources || []);
    });
    addContextResetNotice();
    if (!preserveDraft) prompt.value = "";
    renderConversationList();
    historyFeedback.textContent = "";
    renderRecovery();
    await refreshConversations().catch((error) =>
      setHistoryError(
        `Conversation loaded, but history could not refresh: ${error.message}`,
        () => retryHistory(),
      ),
    );
    return true;
  } catch (error) {
    if (request !== selectionRequest) return false;
    if (error.status === 404) {
      if (selectedConversationId === conversationId) {
        selectedConversationId = null;
        historyReady = false;
        clearConversationView();
      }
      historyFeedback.textContent = "";
      setHistoryError(
        "This conversation is unavailable. Refresh history and choose another conversation.",
        () => retryHistory(),
      );
      await refreshConversations().catch(() => {});
    } else {
      historyFeedback.textContent = "";
      setHistoryError(
        `Unable to load this conversation: ${error.message}`,
        () => selectConversation(conversationId),
      );
    }
    return false;
  } finally {
    if (request === selectionRequest) setLifecycleLoading(false);
  }
}

async function retryHistory() {
  try {
    await refreshConversations();
    if (selectedConversationId)
      await selectConversation(selectedConversationId, {
        discardDraft: true,
        force: true,
        preserveDraft: true,
      });
    else if (conversations.length)
      await selectConversation(conversations[0].id);
    else {
      historyReady = true;
      updateLifecycleControls();
      await createConversation();
    }
  } catch (error) {
    setHistoryError(`Unable to refresh conversations: ${error.message}`, () =>
      retryHistory(),
    );
  }
}

async function loadArchiveStatus() {
  archiveStatus.textContent = "Loading status…";
  try {
    const response = await fetch(apiPath("/health"));
    const body = await response.json();
    if (!body || !["ok", "degraded"].includes(body.status))
      throw new Error("Invalid health response");
    archiveStatus.textContent =
      body.status === "ok" ? "Processing available" : "Processing degraded";
    if (body.pending && typeof body.pending === "object") {
      const pending = Object.values(body.pending).reduce(
        (total, count) => total + (Number(count) || 0),
        0,
      );
      archiveStatus.textContent += ` · ${pending} pending tasks`;
    }
  } catch {
    archiveStatus.textContent = "Status unavailable";
  }
}

function scrollConversation(force = false) {
  if (!force && !readerNearBottom) {
    jumpToLatest.hidden = false;
    return;
  }
  conversation.scrollTop = conversation.scrollHeight;
  jumpToLatest.hidden = true;
  readerNearBottom = true;
}

conversation.addEventListener("scroll", () => {
  readerNearBottom =
    conversation.scrollHeight -
      conversation.scrollTop -
      conversation.clientHeight <=
    64;
  jumpToLatest.hidden = readerNearBottom;
});
jumpToLatest.addEventListener("click", () => scrollConversation(true));

function autosizePrompt() {
  prompt.style.height = "auto";
  const maxHeight = Math.min(160, Math.max(88, window.innerHeight * 0.2));
  prompt.style.height = `${Math.min(Math.max(prompt.scrollHeight, sendButton.offsetHeight), maxHeight)}px`;
}

function setSocketBanner(kind, text, reconnect = false) {
  socketBanner.className = `socket-banner alert alert-${kind} mb-0 active`;
  socketBanner.replaceChildren(document.createTextNode(text));
  if (reconnect) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-sm btn-outline-secondary ms-2";
    button.textContent = "Reconnect";
    button.addEventListener("click", reconnectSocket);
    socketBanner.appendChild(button);
  }
  connectionStatus.textContent =
    kind === "danger"
      ? "Connection error"
      : kind === "warning"
        ? "Disconnected"
        : "Connected";
}

function clearSocketBanner() {
  socketBanner.className = "socket-banner alert alert-warning mb-0";
  socketBanner.replaceChildren();
  connectionStatus.textContent = "Connected";
}

function renderMarkdown(text) {
  const source = String(text ?? "").trim() || "(no response)";
  if (!window.marked?.parse || !window.DOMPurify?.sanitize) return null;
  try {
    return window.DOMPurify.sanitize(
      window.marked.parse(source, { gfm: true, breaks: false }),
    );
  } catch {
    return null;
  }
}

function refreshEmptyState() {
  emptyChatInvitation.hidden =
    conversation.querySelector(".bubble.user, .turn") !== null;
}

function addUserBubble(content, createdAt = null) {
  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = content;
  conversation.appendChild(bubble);
  const time = document.createElement("time");
  time.className = "message-time";
  const date = createdAt ? new Date(createdAt) : new Date();
  if (!Number.isNaN(date.getTime())) {
    time.dateTime = date.toISOString();
    time.textContent = new Intl.DateTimeFormat(navigator.languages, {
      hour: "numeric",
      minute: "2-digit",
    }).format(date);
    time.title = createdAt
      ? date.toLocaleString(navigator.languages)
      : "Received in this browser";
    conversation.appendChild(time);
  }
  refreshEmptyState();
  scrollConversation(true);
  return bubble;
}

let promptRevision = 0;
prompt.addEventListener("input", () => {
  promptRevision += 1;
  autosizePrompt();
});
window.addEventListener("load", autosizePrompt);
window.addEventListener("resize", autosizePrompt);
autosizePrompt();

function createTurn(turnId, createdAt = null) {
  if (turns.has(turnId)) return turns.get(turnId);
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.dataset.turnId = turnId;
  turn.setAttribute("aria-label", "Assistant response");

  const time = document.createElement("time");
  time.className = "turn-time";
  if (createdAt) {
    const date = new Date(createdAt);
    if (!Number.isNaN(date.getTime())) {
      time.dateTime = date.toISOString();
      time.textContent = new Intl.DateTimeFormat(navigator.languages, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
    }
  }

  const status = document.createElement("div");
  status.className = "turn-status";
  status.textContent = "Waiting for the assistant…";
  const answer = document.createElement("div");
  answer.className = "bubble assistant is-pending";
  const timeline = document.createElement("div");
  timeline.className = "timeline";
  const usage = document.createElement("div");
  usage.className = "usage";
  const toolsPanel = document.createElement("section");
  toolsPanel.className = "tools-panel";
  const toolsSummary = document.createElement("h3");
  toolsSummary.textContent = "Activity";
  const toolList = document.createElement("div");
  toolList.className = "tool-list";
  toolsPanel.append(toolsSummary, toolList);
  const sourcesPanel = document.createElement("details");
  sourcesPanel.className = "sources-panel";
  const sourcesSummary = document.createElement("summary");
  sourcesSummary.textContent = "Sources";
  const sourceList = document.createElement("div");
  sourceList.className = "source-list";
  sourcesPanel.append(sourcesSummary, sourceList);
  toolsPanel.hidden = true;
  sourcesPanel.hidden = true;
  turn.append(time, status, timeline, toolsPanel, answer, sourcesPanel, usage);
  conversation.appendChild(turn);

  const state = {
    root: turn,
    answer,
    status,
    timeline,
    usage,
    toolsPanel,
    toolList,
    toolsSummary,
    sourcesPanel,
    sourceList,
    sourcesSummary,
    toolEntries: new Map(),
  };
  turns.set(turnId, state);
  refreshEmptyState();
  scrollConversation();
  return state;
}

function addTimelineItem(turnId, text) {
  const turn = turns.get(turnId);
  if (!turn) return;
  turn.status.textContent = text;
  chatAnnouncements.textContent = text;
  scrollConversation();
}

function renderUsageChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "usage-chip";
  chip.textContent = `${label}: ${value}`;
  return chip;
}

function updateUsage(turnId, payload) {
  const turn = turns.get(turnId);
  if (!turn) return;
  turn.usage.replaceChildren();
  if (payload.model)
    turn.usage.appendChild(renderUsageChip("Model", payload.model));
  if (payload.available !== true) {
    turn.usage.appendChild(renderUsageChip("Tokens", "unavailable"));
    return;
  }
  for (const [label, key] of [
    ["Prompt", "prompt_tokens"],
    ["Completion", "completion_tokens"],
    ["Total", "total_tokens"],
  ]) {
    turn.usage.appendChild(
      renderUsageChip(
        label,
        Number.isFinite(payload[key]) ? payload[key] : "unavailable",
      ),
    );
  }
}

function formatJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function toolLabel(name) {
  return (
    {
      get_available_metadata: "Check archive metadata",
      search_documents: "Search documents",
      read_document: "Read document",
      read_full_document: "Read document",
    }[name] ||
    name ||
    "Unknown tool"
  );
}

function getOrCreateToolCard(turnId, payload, position = 0) {
  const turn = turns.get(turnId);
  if (!turn) return null;
  turn.toolsPanel.hidden = false;
  const key = String(payload.tool_call_id || `position-${position}`);
  if (turn.toolEntries.has(key)) return turn.toolEntries.get(key);
  const card = document.createElement("div");
  card.className = "tool-card";
  const header = document.createElement("div");
  header.className = "tool-card-header";
  const label = document.createElement("strong");
  label.textContent = toolLabel(payload.name);
  const state = document.createElement("span");
  state.className = "tool-card-status";
  header.append(label, state);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.title = "Show activity details";
  summary.appendChild(header);
  const body = document.createElement("div");
  body.className = "tool-body";
  const meta = document.createElement("div");
  meta.className = "tool-meta";
  const preview = document.createElement("pre");
  preview.className = "tool-preview";
  const args = document.createElement("pre");
  args.className = "tool-arguments";
  args.textContent = formatJson(payload.arguments ?? {});
  body.append(meta, preview, args);
  details.append(summary, body);
  card.appendChild(details);
  turn.toolList.appendChild(card);
  const tool = { card, details, meta, preview, args, label, state };
  turn.toolEntries.set(key, tool);
  turn.toolsSummary.textContent = `Activity (${turn.toolEntries.size})`;
  return tool;
}

function updateTool(turnId, payload, started, position = 0) {
  const tool = getOrCreateToolCard(turnId, payload, position);
  if (!tool) return;
  tool.label.textContent = toolLabel(payload.name);
  tool.args.textContent = formatJson(payload.arguments ?? {});
  tool.card.dataset.state = started
    ? "running"
    : payload.interrupted
      ? "interrupted"
      : "complete";
  if (started) {
    tool.state.textContent = "Running…";
    tool.meta.replaceChildren();
    tool.preview.textContent = "";
  } else {
    tool.state.textContent = payload.interrupted
      ? "Interrupted; outcome unknown"
      : payload.summary || "Complete";
    const pieces = [];
    if (payload.duration_ms != null)
      pieces.push(`Duration ${payload.duration_ms} ms`);
    tool.meta.replaceChildren(
      ...pieces.map((text) => {
        const span = document.createElement("span");
        span.textContent = text;
        return span;
      }),
    );
    tool.preview.textContent = payload.preview || "";
  }
  scrollConversation();
}

function interruptRunningTools(turnId) {
  const turn = turns.get(turnId);
  if (!turn) return;
  turn.toolEntries.forEach((tool) => {
    if (tool.state.textContent === "Running…") {
      tool.state.textContent = "Interrupted; outcome unknown";
      tool.card.dataset.interrupted = "true";
    }
  });
}

function getTurn(turnId) {
  return turns.get(turnId) || null;
}

function safeSourceText(value, fallback = "Not available") {
  return (typeof value === "string" || typeof value === "number") &&
    String(value).trim()
    ? String(value)
    : fallback;
}

function sourceDocumentId(source) {
  const value = source?.id;
  if (
    typeof value !== "number" &&
    !(typeof value === "string" && /^[1-9]\d*$/.test(value))
  )
    return null;
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : null;
}

function sourcePath(id, kind) {
  return kind === "detail"
    ? `/documents/${id}/detail`
    : kind === "thumb"
      ? `/api/documents/${id}/thumb/`
      : `/api/documents/${id}/preview/`;
}

function formatDocumentDate(value) {
  if (typeof value !== "string" || !value) return "Not available";
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  let date;
  if (dateOnly) {
    const [year, month, day] = dateOnly.slice(1).map(Number);
    date = new Date(0);
    date.setFullYear(year, month - 1, day);
    date.setHours(0, 0, 0, 0);
    if (
      date.getFullYear() !== year ||
      date.getMonth() !== month - 1 ||
      date.getDate() !== day
    )
      return "Not available";
  } else date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat(navigator.languages, {
    dateStyle: "medium",
  }).format(date);
}

function renderSourceBadges(source) {
  const badges = [];
  if (source.available === false) badges.push(["Unavailable", ""]);
  if (source.matched) badges.push(["Matched", "match"]);
  if (source.inspected) badges.push(["Read in full", "read"]);
  return badges;
}

function updateSourceSelection() {
  document.querySelectorAll(".source-card").forEach((card) => {
    const selected =
      Boolean(selectedSource) && card.dataset.sourceKey === selectedSource.key;
    card.classList.toggle("selected", selected);
    card
      .querySelectorAll(".source-select, .source-preview")
      .forEach((button) => {
        button.setAttribute("aria-pressed", String(selected));
      });
  });
}

function appendMetadataRow(list, label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  if (value instanceof Node) description.appendChild(value);
  else description.textContent = value;
  list.append(term, description);
}

function openPreview(source, turnId, opener = document.activeElement) {
  const id = sourceDocumentId(source);
  const isAvailable = source.available !== false && id !== null;
  const title = safeSourceText(
    source.title,
    id ? `Document ${id}` : "Document unavailable",
  );
  const key = JSON.stringify([
    selectedConversationId,
    turnId,
    id ?? safeSourceText(source.id, "unknown"),
  ]);
  selectedSource = { key, opener };
  updateSourceSelection();
  previewPlaceholder.hidden = true;
  previewHeader.classList.add("active");
  previewMetadata.hidden = false;
  previewTitle.textContent = title;
  previewSubtitle.textContent = id
    ? `Document ${id}`
    : "Document ID unavailable";
  previewOpenLink.hidden = id === null;
  previewFallbackLink.hidden = id === null;
  if (id !== null) {
    previewOpenLink.href = sourcePath(id, "detail");
    previewFallbackLink.href = sourcePath(id, "detail");
  } else {
    previewOpenLink.removeAttribute("href");
    previewFallbackLink.removeAttribute("href");
  }

  const metadata = previewMetadata.querySelector("dl");
  metadata.replaceChildren();
  const tags = Array.isArray(source.tag_names)
    ? source.tag_names.filter((tag) => typeof tag === "string" && tag.trim())
    : [];
  const tagList = document.createElement("div");
  tagList.className = "preview-tags";
  if (tags.length) {
    tags.forEach((tag) => {
      const chip = document.createElement("span");
      chip.className = "source-badge";
      chip.textContent = tag;
      tagList.appendChild(chip);
    });
  } else tagList.textContent = "No tags";
  appendMetadataRow(
    metadata,
    "Correspondent",
    safeSourceText(source.correspondent_name),
  );
  appendMetadataRow(
    metadata,
    "Document date",
    formatDocumentDate(source.created),
  );
  appendMetadataRow(
    metadata,
    "Document type",
    safeSourceText(source.document_type_name),
  );
  appendMetadataRow(metadata, "Tags", tagList);
  appendMetadataRow(metadata, "Added", formatDocumentDate(source.added));
  appendMetadataRow(
    metadata,
    "File name",
    safeSourceText(source.original_filename),
  );
  appendMetadataRow(
    metadata,
    "Document ID",
    id === null ? "Not available" : String(id),
  );
  appendMetadataRow(
    metadata,
    "Archive serial number",
    safeSourceText(source.archive_serial_number),
  );
  appendMetadataRow(
    metadata,
    "Storage path",
    safeSourceText(source.storage_path_name),
  );

  previewGeneration += 1;
  const generation = previewGeneration;
  previewFrame.removeAttribute("src");
  const nextFrame = previewFrame.cloneNode(false);
  previewFrame.replaceWith(nextFrame);
  previewFrame = nextFrame;
  previewFrame.title = `Preview of ${title}`;
  previewFrameWrap.classList.add("active");
  previewFallback.hidden = true;
  if (!isAvailable) {
    previewState.textContent =
      source.available === false
        ? "Document unavailable. Saved metadata is shown; preview and thumbnail are disabled."
        : "Preview unavailable because the document ID is invalid.";
    previewFallback.hidden = source.available === false || id === null;
  } else {
    previewState.textContent = "Loading preview…";
    previewFallback.hidden = false;
    nextFrame.addEventListener("load", () => {
      if (generation === previewGeneration && previewFrame === nextFrame)
        previewState.textContent = "";
    });
    nextFrame.addEventListener("error", () => {
      if (generation === previewGeneration && previewFrame === nextFrame)
        previewState.textContent = "Preview could not be loaded.";
    });
    nextFrame.src = sourcePath(id, "preview");
  }
  openDrawer("inspector", opener);
}

function renderSources(turnId, items) {
  const turn = getTurn(turnId);
  if (!turn) return;
  turn.sourcesPanel.hidden = items.length === 0;
  turn.sourcesPanel.open = true;
  const uniqueItems = [
    ...new Map(
      items.map((source) => [
        String(sourceDocumentId(source) ?? source.id),
        source,
      ]),
    ).values(),
  ];
  turn.sourcesSummary.textContent = `Sources (${uniqueItems.length})`;
  turn.sourceList.replaceChildren();
  uniqueItems.forEach((source) => {
    const id = sourceDocumentId(source);
    const titleText = safeSourceText(
      source.title,
      id ? `Document ${id}` : "Document unavailable",
    );
    const key = JSON.stringify([
      selectedConversationId,
      turnId,
      id ?? safeSourceText(source.id, "unknown"),
    ]);
    const card = document.createElement("article");
    card.className = "source-card";
    card.dataset.sourceKey = key;

    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "source-select source-thumb";
    selectButton.setAttribute("aria-label", `Inspect ${titleText}`);
    selectButton.setAttribute("aria-pressed", "false");
    const thumb = document.createElement("img");
    thumb.className = "source-thumb";
    thumb.loading = "lazy";
    thumb.alt = "";
    if (source.available !== false && id !== null) {
      thumb.src = sourcePath(id, "thumb");
      thumb.onerror = () => {
        const placeholder = document.createElement("span");
        placeholder.className = "source-thumb-placeholder";
        placeholder.setAttribute("aria-hidden", "true");
        placeholder.textContent = "▤";
        thumb.replaceWith(placeholder);
      };
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "source-thumb-placeholder";
      placeholder.setAttribute("aria-hidden", "true");
      placeholder.textContent = "▤";
      selectButton.appendChild(placeholder);
    }
    if (source.available !== false && id !== null)
      selectButton.appendChild(thumb);
    selectButton.addEventListener("click", () =>
      openPreview(source, turnId, selectButton),
    );

    const content = document.createElement("div");
    content.className = "source-content";

    const title = document.createElement("h3");
    title.className = "source-title";
    const titleButton = document.createElement("button");
    titleButton.type = "button";
    titleButton.className = "source-select";
    titleButton.textContent = titleText;
    titleButton.setAttribute("aria-pressed", "false");
    titleButton.addEventListener("click", () =>
      openPreview(source, turnId, titleButton),
    );
    title.appendChild(titleButton);

    const badges = document.createElement("div");
    badges.className = "source-badges";
    renderSourceBadges(source).forEach(([label, klass]) => {
      const badge = document.createElement("span");
      badge.className = `source-badge ${klass}`.trim();
      badge.textContent = label;
      badges.appendChild(badge);
    });

    const meta = document.createElement("div");
    meta.className = "source-meta";
    const sourceTags = Array.isArray(source.tag_names)
      ? source.tag_names.filter((tag) => typeof tag === "string" && tag.trim())
      : [];
    [
      formatDocumentDate(source.created),
      safeSourceText(source.correspondent_name, ""),
      ...sourceTags.slice(0, 2),
      sourceTags.length > 2 ? `+${sourceTags.length - 2} more` : "",
    ]
      .filter((item) => item && item !== "Not available")
      .forEach((item) => {
        const span = document.createElement("span");
        span.textContent = item;
        meta.appendChild(span);
      });

    const actions = document.createElement("div");
    actions.className = "source-actions";
    const previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "source-preview btn btn-sm btn-outline-secondary";
    previewButton.textContent =
      source.available === false || id === null
        ? "Preview unavailable"
        : "Preview document";
    previewButton.disabled = source.available === false || id === null;
    previewButton.setAttribute("aria-pressed", "false");
    previewButton.addEventListener("click", () =>
      openPreview(source, turnId, previewButton),
    );

    let openLink;
    if (id !== null) {
      openLink = document.createElement("a");
      openLink.className = "btn btn-sm btn-link px-0";
      openLink.href = sourcePath(id, "detail");
      openLink.target = "_blank";
      openLink.rel = "noopener noreferrer";
      openLink.textContent = "Open in Paperless";
    }

    const inspectButton = document.createElement("button");
    inspectButton.type = "button";
    inspectButton.className = "btn btn-sm btn-link px-0";
    inspectButton.textContent = "Inspect details";
    inspectButton.addEventListener("click", () =>
      openPreview(source, turnId, inspectButton),
    );

    actions.appendChild(previewButton);
    if (source.available === false || id === null)
      actions.appendChild(inspectButton);
    if (openLink) actions.appendChild(openLink);

    content.appendChild(title);
    content.appendChild(badges);
    content.appendChild(meta);
    card.appendChild(selectButton);
    card.appendChild(content);
    card.appendChild(actions);
    turn.sourceList.appendChild(card);
  });
  updateSourceSelection();
}

function setAssistantMessage(turnId, content) {
  const turn = turns.get(turnId);
  if (!turn) return;
  turn.answer.classList.remove("is-pending");
  const html = renderMarkdown(content);
  if (html === null) {
    turn.answer.textContent = String(content ?? "(no response)");
    if (!markdownFallbackShown) {
      const notice = document.createElement("p");
      notice.className = "markdown-fallback";
      notice.textContent =
        "Markdown formatting is unavailable; showing the answer as plain text.";
      turn.answer.appendChild(notice);
      markdownFallbackShown = true;
    }
  } else {
    turn.answer.innerHTML = `<div class="markdown-body">${html}</div>`;
  }
  turn.status.textContent = "";
  scrollConversation();
}

function setTurnError(turnId, content) {
  const turn = turns.get(turnId);
  if (!turn) return;
  turn.answer.classList.remove("is-pending");
  turn.answer.textContent = content;
  turn.answer.setAttribute("role", "alert");
  turn.answer.classList.add(
    "border",
    "border-danger-subtle",
    "bg-danger-subtle",
  );
  turn.status.textContent = "Response failed. The prompt was not saved.";
  scrollConversation();
}

function renderRecovery() {
  recoveryPanel.replaceChildren();
  recoveryPanel.classList.toggle("active", Boolean(recovery));
  if (!recovery) return;
  const message = document.createElement("div");
  message.textContent = recovery.failed
    ? "This prompt failed and was not saved. You can restore it to the composer and try again."
    : "The previous response may still be saving. Refresh history before resending; the request may already have completed.";
  const submitted = document.createElement("div");
  submitted.textContent = `Submitted prompt: ${recovery.content}`;
  const actions = document.createElement("div");
  actions.className = "d-flex flex-wrap gap-2";
  const refresh = document.createElement("button");
  refresh.type = "button";
  refresh.className = "btn btn-sm btn-outline-secondary";
  refresh.textContent = "Refresh history";
  refresh.hidden = recovery.failed;
  refresh.disabled = !selectedConversationId || lifecycleLoading;
  refresh.addEventListener("click", async () => {
    if (!selectedConversationId) return;
    const loaded = await selectConversation(selectedConversationId, {
      discardDraft: true,
      force: true,
      preserveDraft: true,
    });
    if (loaded) {
      recoveryNeedsRefresh = false;
      renderRecovery();
      updateLifecycleControls();
    }
  });
  const reuse = document.createElement("button");
  reuse.type = "button";
  reuse.className = "btn btn-sm btn-outline-danger";
  reuse.textContent = "Use this prompt again";
  reuse.disabled =
    recoveryNeedsRefresh ||
    (!recovery.failed && selectedConversationId !== recovery.conversationId);
  reuse.addEventListener("click", () => {
    if (!recovery.failed && selectedConversationId !== recovery.conversationId)
      return;
    const warning = recovery.failed
      ? "Restore this failed prompt to the composer?"
      : "The previous response may still be saving. Resending may duplicate a completed request. Continue?";
    if (!window.confirm(warning)) return;
    if (
      prompt.value &&
      !window.confirm("Replace the current draft with the submitted prompt?")
    )
      return;
    prompt.value = recovery.content;
    promptRevision += 1;
    autosizePrompt();
    prompt.focus();
  });
  actions.append(refresh, reuse);
  recoveryPanel.append(message, submitted, actions);
}

function addContextResetNotice() {
  const notice = document.createElement("p");
  notice.className = "context-reset-notice";
  notice.textContent =
    "Saved messages are shown here. Earlier messages are not included in a new session’s AI context.";
  conversation.appendChild(notice);
  refreshEmptyState();
}

async function recoverHistory() {
  if (!selectedConversationId) return false;
  recoveryNeedsRefresh = true;
  updateLifecycleControls();
  const loaded = await selectConversation(selectedConversationId, {
    discardDraft: true,
    force: true,
    preserveDraft: true,
  });
  if (loaded) {
    recoveryNeedsRefresh = false;
    renderRecovery();
    updateLifecycleControls();
    return true;
  }
  updateLifecycleControls();
  return false;
}

function handleSocketMessage(ws, event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    setSocketBanner(
      "danger",
      "Received an unreadable chat event. Reconnect to continue.",
      true,
    );
    return;
  }
  if (
    !payload ||
    typeof payload !== "object" ||
    typeof payload.type !== "string"
  )
    return;
  const turnId = payload.turn_id;
  if (
    payload.type === "error" &&
    (turnId === "invalid" || turnId === "unavailable")
  ) {
    setSocketBanner("danger", payload.content || "Chat is unavailable.", true);
    if (activeTurn && !activeTurn.turnId) {
      recovery = {
        content: activeTurn.content,
        conversationId: activeTurn.conversationId,
        failed: true,
      };
      recoveryNeedsRefresh = false;
      renderRecovery();
      activeTurn = null;
      setChatBusy(false);
    }
    if (turnId === "invalid") {
      historyReady = false;
      retryHistory();
    }
    updateLifecycleControls();
    return;
  }
  if (payload.type === "turn_started") {
    if (
      !activeTurn ||
      activeTurn.turnId ||
      activeTurn.conversationId !== selectedConversationId ||
      payload.conversation_id !== selectedConversationId ||
      !turnId
    )
      return;
    activeTurn.turnId = turnId;
    const turn = createTurn(turnId);
    turn.status.textContent = "Starting response…";
    return;
  }
  if (
    !activeTurn ||
    !turnId ||
    activeTurn.turnId !== turnId ||
    activeTurn.conversationId !== selectedConversationId ||
    ws !== socket
  )
    return;
  if (payload.type === "status") {
    if (payload.phase === "model")
      addTimelineItem(turnId, payload.content || "Thinking…");
  } else if (payload.type === "tool_call_started") {
    updateTool(turnId, payload, true, activeTurn.toolPosition++);
  } else if (payload.type === "tool_call_completed") {
    updateTool(
      turnId,
      payload,
      false,
      Math.max(0, activeTurn.toolPosition - 1),
    );
  } else if (payload.type === "usage" && payload.scope === "total") {
    updateUsage(turnId, payload);
  } else if (payload.type === "assistant_message") {
    setAssistantMessage(turnId, payload.content);
  } else if (payload.type === "sources") {
    renderSources(turnId, Array.isArray(payload.items) ? payload.items : []);
  } else if (payload.type === "error") {
    setTurnError(turnId, payload.content || "Chat failed.");
  } else if (payload.type === "turn_completed") {
    if (payload.success !== true) {
      if (turns.get(turnId)?.answer.textContent === "")
        setTurnError(turnId, "The response failed before it could be saved.");
      interruptRunningTools(turnId);
      if (turns.has(turnId))
        turns.get(turnId).status.textContent =
          "Response failed. The prompt was not saved.";
      recovery = {
        content: activeTurn.content,
        conversationId: selectedConversationId,
        failed: true,
      };
      recoveryNeedsRefresh = false;
      renderRecovery();
    } else if (turns.has(turnId)) {
      turns.get(turnId).status.textContent = "";
      chatAnnouncements.textContent = "Answer ready.";
      refreshConversations().catch((error) =>
        setSocketBanner("warning", error.message, true),
      );
    }
    activeTurn = null;
    setChatBusy(false);
    updateLifecycleControls();
  }
}

function preserveUncertainTurn() {
  if (!activeTurn) return;
  recovery = {
    content: activeTurn.content,
    conversationId: activeTurn.conversationId,
  };
  recoveryNeedsRefresh = true;
  if (activeTurn.turnId) {
    interruptRunningTools(activeTurn.turnId);
    const turn = turns.get(activeTurn.turnId);
    if (turn)
      turn.status.textContent =
        "Connection lost. The previous response may still be saving.";
  }
  activeTurn = null;
  setChatBusy(false);
  renderRecovery();
}

function createSocket() {
  const generation = ++socketGeneration;
  const ws = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);
  socket = ws;
  ws.addEventListener("open", async () => {
    if (socket !== ws || generation !== socketGeneration) return;
    clearSocketBanner();
    connectionStatus.textContent = "Connected";
    if (generation > 1) {
      setSocketBanner(
        "warning",
        "Connection restored. Reloading saved messages; earlier messages are not included in a new session’s AI context.",
      );
      await recoverHistory();
      if (recovery && !recovery.failed)
        setSocketBanner(
          "warning",
          "Connection restored. The previous response may still be saving. Refresh history before resending.",
          true,
        );
      else clearSocketBanner();
    }
    updateLifecycleControls();
  });
  ws.addEventListener("close", () => {
    if (socket !== ws || generation !== socketGeneration) return;
    preserveUncertainTurn();
    setSocketBanner(
      "warning",
      "Connection closed. Reconnect to continue.",
      true,
    );
    updateLifecycleControls();
  });
  ws.addEventListener("error", () => {
    if (socket === ws && generation === socketGeneration)
      setSocketBanner(
        "danger",
        "The chat connection encountered an error. Reconnect to continue.",
        true,
      );
  });
  ws.addEventListener("message", (event) => {
    if (socket === ws && generation === socketGeneration)
      handleSocketMessage(ws, event);
  });
}

function reconnectSocket() {
  const previousSocket = socket;
  socket = null;
  preserveUncertainTurn();
  if (previousSocket && previousSocket.readyState !== WebSocket.CLOSED)
    previousSocket.close();
  createSocket();
  connectionStatus.textContent = "Connecting…";
  setSocketBanner("warning", "Connecting to chat…");
  updateLifecycleControls();
}

createSocket();

newConversationButton.addEventListener("click", () => createConversation());
renameConversationButton.addEventListener("click", () => {
  const item = conversations.find(
    (conversation) => conversation.id === selectedConversationId,
  );
  if (!item || lifecycleBusy()) return;
  renameInput.value = item.title;
  renameError.textContent = "";
  renameDialog.showModal();
  renameInput.focus();
  renameInput.select();
});
document
  .getElementById("rename-cancel")
  .addEventListener("click", () => renameDialog.close());
renameDialog.addEventListener("close", () => renameConversationButton.focus());
renameForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const title = renameInput.value.trim();
  if (!title || title.length > 120 || !selectedConversationId) {
    renameError.textContent = "Enter a name between 1 and 120 characters.";
    return;
  }
  const targetId = selectedConversationId;
  setLifecycleMutation(true);
  renameSaveButton.disabled = true;
  renameError.textContent = "";
  try {
    const updated = await api(`/conversations/${targetId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    if (selectedConversationId === targetId) {
      conversations = conversations.map((item) =>
        item.id === targetId ? updated : item,
      );
      renderConversationList();
    }
    renameDialog.close();
    await refreshConversations().catch((error) =>
      setHistoryError(
        `Conversation renamed, but history could not refresh: ${error.message}`,
        () => retryHistory(),
      ),
    );
  } catch (error) {
    if (error.status === 404) {
      selectedConversationId = null;
      historyReady = false;
      clearConversationView();
      renameDialog.close();
      setHistoryError(
        "This conversation is unavailable. Refresh history and choose another conversation.",
        () => retryHistory(),
      );
      await refreshConversations().catch(() => {});
    } else {
      renameError.textContent =
        error.message || "Unable to rename this conversation.";
    }
  } finally {
    renameSaveButton.disabled = false;
    setLifecycleMutation(false);
  }
});
deleteConversationButton.addEventListener("click", async () => {
  const target = conversations.find(
    (item) => item.id === selectedConversationId,
  );
  if (!target || lifecycleBusy()) return;
  if (
    prompt.value.trim() &&
    !window.confirm("Discard the unsent draft and delete this conversation?")
  )
    return;
  if (
    !window.confirm(
      `Delete “${target.title}”? This deletes only chat history, never Paperless documents.`,
    )
  )
    return;
  const targetId = selectedConversationId;
  const draftRevision = promptRevision;
  setLifecycleMutation(true);
  clearHistoryError();
  try {
    await api(`/conversations/${targetId}`, { method: "DELETE" });
    selectedConversationId = null;
    historyReady = false;
    clearConversationView();
    if (promptRevision === draftRevision) prompt.value = "";
    conversations = conversations.filter((item) => item.id !== targetId);
    renderConversationList();
    setLifecycleMutation(false);
    let historyRefreshed = true;
    await refreshConversations().catch((error) => {
      historyRefreshed = false;
      setHistoryError(
        `Conversation deleted, but history could not refresh: ${error.message}`,
        () => retryHistory(),
      );
    });
    if (!historyRefreshed) return;
    if (conversations.length)
      await selectConversation(conversations[0].id, {
        discardDraft: true,
        preserveDraft: true,
      });
    else await createConversation({ preserveDraft: true });
  } catch (error) {
    if (error.status === 404) {
      selectedConversationId = null;
      historyReady = false;
      clearConversationView();
      setHistoryError(
        "This conversation is unavailable. Refresh history and choose another conversation.",
        () => retryHistory(),
      );
      await refreshConversations().catch(() => {});
    } else {
      setHistoryError(
        `Unable to delete this conversation: ${error.message}`,
        () => retryHistory(),
      );
    }
  } finally {
    setLifecycleMutation(false);
  }
});
historyRefreshButton.addEventListener("click", async () => {
  const reloadSelected = !lifecycleBusy() && !recoveryNeedsRefresh;
  clearHistoryError();
  await Promise.all([
    refreshConversations().catch((error) =>
      setHistoryError(`Unable to refresh history: ${error.message}`, () =>
        retryHistory(),
      ),
    ),
    loadArchiveStatus(),
  ]);
  if (
    reloadSelected &&
    !lifecycleBusy() &&
    !recoveryNeedsRefresh &&
    selectedConversationId
  ) {
    await selectConversation(selectedConversationId, {
      discardDraft: true,
      force: true,
      preserveDraft: true,
    });
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (composing) return;
  const content = prompt.value.trim();
  if (!content) return;
  if (
    !historyReady ||
    lifecycleLoading ||
    chatBusy ||
    recoveryNeedsRefresh ||
    !selectedConversationId
  ) {
    historyFeedback.textContent =
      "Conversation history is still loading or the previous response needs a history refresh.";
    return;
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    setSocketBanner(
      "warning",
      "The connection is not open. Reconnect to continue.",
      true,
    );
    return;
  }
  const conversationId = selectedConversationId;
  try {
    socket.send(JSON.stringify({ content, conversation_id: conversationId }));
  } catch {
    setSocketBanner(
      "danger",
      "The message could not be sent. Your draft was kept; reconnect and try again.",
      true,
    );
    return;
  }
  activeTurn = { content, conversationId, turnId: null, toolPosition: 0 };
  recovery = null;
  recoveryNeedsRefresh = false;
  renderRecovery();
  setChatBusy(true);
  addUserBubble(content);
  prompt.value = "";
  autosizePrompt();
  prompt.focus();
  updateLifecycleControls();
});

prompt.addEventListener("compositionstart", () => {
  composing = true;
});
prompt.addEventListener("compositionend", () => {
  composing = false;
});
prompt.addEventListener("keydown", (event) => {
  if (
    event.key !== "Enter" ||
    event.shiftKey ||
    composing ||
    event.isComposing ||
    event.keyCode === 229
  )
    return;
  if (event.altKey && !event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  form.requestSubmit();
});

prompt.addEventListener("input", updateLifecycleControls);

loadArchiveStatus();
refreshConversations()
  .then(() => {
    historyReady = true;
    renderConversationList();
    updateLifecycleControls();
    return conversations.length
      ? selectConversation(conversations[0].id, {
          discardDraft: true,
          preserveDraft: true,
        })
      : createConversation();
  })
  .catch((error) =>
    setHistoryError(`Unable to load chat history: ${error.message}`, () =>
      retryHistory(),
    ),
  );
