document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['enrollmentId', 'sessionId'], (res) => {
    if (res.enrollmentId && res.sessionId) {
      showConnected(res.enrollmentId, res.sessionId);
    } else {
      showLogin();
    }
  });

  document.getElementById('connectBtn').addEventListener('click', () => {
    const eid = document.getElementById('enrollId').value.trim();
    const sid = parseInt(document.getElementById('sessionId').value.trim());
    if (eid && sid) {
      chrome.storage.local.set({ enrollmentId: eid, sessionId: sid }, () => {
        showConnected(eid, sid);
      });
    }
  });

  document.getElementById('disconnectBtn').addEventListener('click', () => {
    chrome.storage.local.remove(['enrollmentId', 'sessionId'], () => {
      showLogin();
    });
  });
});

function showConnected(eid, sid) {
  document.getElementById('login-section').style.display = 'none';
  document.getElementById('connected-section').style.display = 'block';
  document.getElementById('displayId').textContent = eid;
  document.getElementById('displaySession').textContent = sid;
}

function showLogin() {
  document.getElementById('login-section').style.display = 'block';
  document.getElementById('connected-section').style.display = 'none';
}
