let enrollmentId = null;
let quizActive = false;
let sessionTarget = 68;

// ─── INIT ────────────────────────────────────────────────────────────────────
function initMeeting() {
  chrome.storage.local.get(['enrollmentId', 'sessionId'], (res) => {
    enrollmentId = res.enrollmentId || "CS202401";
    sessionTarget = res.sessionId ? parseInt(res.sessionId) : 68;

    // Mark attendance
    chrome.runtime.sendMessage({
      action: "postData", endpoint: "/attendance",
      payload: { session_id: sessionTarget, enrollment_id: enrollmentId }
    });

    chrome.storage.local.set({ activeSession: { sessionId: sessionTarget } });

    // Start fixed 30-second interval for Math Quizzes
    startMathQuizInterval();
  });
}

// ─── AUTOMATIC MATH QUIZ EVERY 30 SECONDS ────────────────────────────────────
function startMathQuizInterval() {
  setInterval(() => {
    if (quizActive) return;
    
    // Generate simple random math question (e.g. 5 + 3, 12 - 4)
    const ops = ['+', '-'];
    const op = ops[Math.floor(Math.random() * ops.length)];
    const num1 = Math.floor(Math.random() * 15) + 1;
    const num2 = Math.floor(Math.random() * 15) + 1;
    
    let answer = 0;
    let questionText = "";
    
    if (op === '+') {
      answer = num1 + num2;
      questionText = `What is ${num1} + ${num2}?`;
    } else {
      // Ensure no negative answers for simplicity
      const big = Math.max(num1, num2);
      const small = Math.min(num1, num2);
      answer = big - small;
      questionText = `What is ${big} - ${small}?`;
    }
    
    // Generate options
    let options = [answer, answer + 1, answer - 1, answer + 2];
    // Shuffle options
    options = options.sort(() => Math.random() - 0.5);
    const correctIdx = options.indexOf(answer);
    
    const quizData = {
      question: questionText,
      options: options.map(String),
      correct_index: correctIdx
    };
    
    injectQuizUI(quizData);
    
  }, 30000); // Trigger every 30 seconds
}


// ─── SAFE DOM INJECTION ───────────────────────────────────────────────────────
function injectQuizUI(quizData) {
  quizActive = true;
  
  const existing = document.getElementById('classroom-ai-quiz-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'classroom-ai-quiz-overlay';
  overlay.style.cssText = `
    position:fixed;top:0;left:0;width:100vw;height:100vh;
    background:rgba(0,0,0,0.5);backdrop-filter:blur(8px);
    z-index:9999999;display:flex;justify-content:center;align-items:center;
    font-family:'Segoe UI',sans-serif;
  `;

  const modal = document.createElement('div');
  modal.style.cssText = `
    background:#111827;border:1px solid #1f2937;border-radius:12px;
    width:350px;padding:28px;box-shadow:0 20px 40px rgba(0,0,0,0.6);color:#fff;
  `;

  const header = document.createElement('div');
  header.textContent = '🧠 Attentiveness Check';
  header.style.cssText = 'color:#4f8ef7;font-weight:700;font-size:13px;margin-bottom:14px;text-transform:uppercase;letter-spacing:1px;';

  const question = document.createElement('div');
  question.textContent = quizData.question;
  question.style.cssText = 'font-size:22px;line-height:1.5;margin-bottom:20px;font-weight:700;text-align:center;color:#43d98c;';

  const optionsDiv = document.createElement('div');
  optionsDiv.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:10px;';

  quizData.options.forEach((opt, idx) => {
    const optBtn = document.createElement('button');
    optBtn.textContent = opt;
    optBtn.dataset.idx = idx;
    optBtn.style.cssText = `
      background:#1f2937;color:#fff;border:1px solid #374151;padding:14px;
      border-radius:8px;text-align:center;font-size:18px;font-weight:bold;cursor:pointer;
      transition:background 0.15s;
    `;
    optBtn.onmouseover = () => { optBtn.style.background = '#374151'; optBtn.style.borderColor = '#4f8ef7'; };
    optBtn.onmouseout = () => { optBtn.style.background = '#1f2937'; optBtn.style.borderColor = '#374151'; };
    optionsDiv.appendChild(optBtn);
  });

  const timerEl = document.createElement('div');
  timerEl.id = 'cai-timer';
  timerEl.textContent = '⏱ 10s remaining';
  timerEl.style.cssText = 'margin-top:18px;font-size:12px;color:#f25c5c;text-align:center;font-weight:600;';

  modal.appendChild(header);
  modal.appendChild(question);
  modal.appendChild(optionsDiv);
  modal.appendChild(timerEl);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  // 10 Second Timer
  let timeLeft = 10;
  const timer = setInterval(() => {
    timeLeft--;
    timerEl.textContent = `⏱ ${timeLeft}s remaining`;
    if (timeLeft <= 0) { clearInterval(timer); submitAnswer(false); }
  }, 1000);

  optionsDiv.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', (e) => {
      clearInterval(timer);
      submitAnswer(parseInt(e.target.dataset.idx) === quizData.correct_index);
    });
  });

  function submitAnswer(isCorrect) {
    overlay.remove();
    quizActive = false;

    const toast = document.createElement('div');
    toast.style.cssText = `
      position:fixed;top:20px;right:20px;z-index:9999999;
      background:${isCorrect ? '#43d98c' : '#f25c5c'};color:#fff;
      padding:14px 22px;border-radius:10px;font-weight:bold;
      font-family:sans-serif;box-shadow:0 4px 16px rgba(0,0,0,0.3);font-size:15px;
    `;
    toast.textContent = isCorrect ? '✅ Correct! +5 Engagement points' : '❌ Wrong / Timeout — Penalty Applied';
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);

    // Send result to backend
    chrome.runtime.sendMessage({
      action: "postData", endpoint: "/log-engagement",
      payload: {
        enrollment_id: enrollmentId,
        session_id: sessionTarget,
        action: isCorrect ? "answered_correct" : "answered_wrong"
      }
    });
  }
}

// Start
setTimeout(initMeeting, 4000);
