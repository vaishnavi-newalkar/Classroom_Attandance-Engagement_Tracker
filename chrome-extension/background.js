const API_URL = "http://localhost:8000/api/ext";

// Listen for network requests forwarded from the restricted Content Script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "postData") {
    fetch(`${API_URL}${request.endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request.payload)
    })
    .then(res => res.json())
    .then(data => sendResponse(data))
    .catch(err => {
      console.error("BG Fetch Error:", err);
      sendResponse({ success: false, error: err.toString() });
    });
    return true; // Keep message channel open for async response
  }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  chrome.storage.local.get(['enrollmentId', 'activeSession'], async (data) => {
    if (data.enrollmentId && data.activeSession) {
      if (data.activeSession.tabId && data.activeSession.tabId !== activeInfo.tabId) {
        // User switched tabs during class! Add a distraction tick.
        try {
          await fetch(`${API_URL}/log-engagement`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              enrollment_id: data.enrollmentId,
              session_id: data.activeSession.sessionId,
              action: "switched_tab"
            })
          });
        } catch (e) { console.error("Ext error", e); }
      }
    }
  });
});
