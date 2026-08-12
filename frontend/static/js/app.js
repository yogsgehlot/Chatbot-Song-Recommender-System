/* ============================================================
   MusicMind — AI Music Assistant
   Frontend Application Logic
   ============================================================ */

'use strict';

// ----------------------------------------------------------
// 1. Constants & Configuration
// ----------------------------------------------------------
const API_BASE_URL = 'http://localhost:8000';

const MOOD_MESSAGES = {
  calm:      'Give me calm and soothing songs to relax',
  happy:     'Recommend me some upbeat happy songs to lift my spirits',
  sad:       "I'm feeling sad, recommend some emotional and melancholic songs",
  energetic: 'Give me high-energy, pump-up songs to get me going',
  romantic:  'Suggest romantic and love songs for a special evening',
  workout:   'I need intense workout music for the gym session',
  study:     'Give me calm study music to help me focus and concentrate',
  party:     'Recommend party songs for a night out with friends',
};

const COVER_GRADIENTS = [
  'cover-grad-1', 'cover-grad-2', 'cover-grad-3',
  'cover-grad-4', 'cover-grad-5', 'cover-grad-6',
  'cover-grad-7', 'cover-grad-8', 'cover-grad-9', 'cover-grad-10',
];

// ----------------------------------------------------------
// 2. Session Management
// ----------------------------------------------------------
let sessionId = sessionStorage.getItem('musicmind_session_id');
if (!sessionId) {
  sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 9);
  sessionStorage.setItem('musicmind_session_id', sessionId);
}

// ----------------------------------------------------------
// 3. DOM References
// ----------------------------------------------------------
const chatMessages      = document.getElementById('chatMessages');
const chatInput         = document.getElementById('chatInput');
const sendButton        = document.getElementById('sendButton');
const typingIndicator   = document.getElementById('typingIndicator');
const emptyState        = document.getElementById('emptyState');
const songCardsGrid     = document.getElementById('songCardsGrid');
const recommendationsCount = document.getElementById('recommendationsCount');
const clearRecsBtn      = document.getElementById('clearRecsBtn');
const statusDot         = document.getElementById('statusDot');
const statusLabel       = document.getElementById('statusLabel');
const providerBadge     = document.getElementById('providerBadge');
const providerName      = document.getElementById('providerName');

// ----------------------------------------------------------
// 4. State
// ----------------------------------------------------------
let isSending = false;
let currentRecommendations = [];

// ----------------------------------------------------------
// 5. Utility Helpers
// ----------------------------------------------------------

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str ?? '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Pick a cover gradient class based on an index.
 * @param {number} index
 * @returns {string}
 */
function pickGradient(index) {
  return COVER_GRADIENTS[index % COVER_GRADIENTS.length];
}

/**
 * Format energy value (0–1 or 0–100) to a percentage.
 * @param {number|string} raw
 * @returns {{ pct: number, label: string }}
 */
function parseEnergy(raw) {
  let val = parseFloat(raw);
  if (isNaN(val)) return { pct: 50, label: '50%' };
  if (val <= 1) val = Math.round(val * 100);
  val = Math.min(100, Math.max(0, Math.round(val)));
  return { pct: val, label: val + '%' };
}

/**
 * Auto-resize textarea to fit content.
 */
function autoResizeTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

/**
 * Scroll chat container to the latest message smoothly.
 */
function scrollToLatest() {
  requestAnimationFrame(() => {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
  });
}

// ----------------------------------------------------------
// 6. Status Indicator
// ----------------------------------------------------------

/**
 * Update the connection status indicator in the top bar.
 * @param {'connected'|'error'|'connecting'} state
 * @param {string} [label]
 */
function updateStatusIndicator(state, label) {
  statusDot.className = 'status-dot';
  if (state === 'connected') {
    statusDot.classList.add('connected');
    statusLabel.textContent = label || 'Connected';
  } else if (state === 'error') {
    statusDot.classList.add('error');
    statusLabel.textContent = label || 'Offline';
  } else {
    statusLabel.textContent = label || 'Connecting…';
  }
}

/**
 * Update the LLM provider badge text.
 * @param {string} name
 */
function updateProviderBadge(name) {
  if (name && providerName) {
    providerName.textContent = name;
  }
}

// ----------------------------------------------------------
// 7. Health Check
// ----------------------------------------------------------

/**
 * Check if the FastAPI backend is reachable.
 * Updates the status indicator accordingly.
 */
async function checkApiHealth() {
  updateStatusIndicator('connecting');
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      updateStatusIndicator('connected', 'Connected');
      if (data.llm_provider) updateProviderBadge(data.llm_provider);
      else if (data.provider)  updateProviderBadge(data.provider);
    } else {
      updateStatusIndicator('error', 'Backend error');
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      updateStatusIndicator('error', 'Timeout');
    } else {
      updateStatusIndicator('error', 'Offline');
    }
  }
}

// ----------------------------------------------------------
// 8. Typing Indicator
// ----------------------------------------------------------

/** Show the animated typing dots indicator. */
function showTypingIndicator() {
  typingIndicator.style.display = 'flex';
  typingIndicator.setAttribute('aria-hidden', 'false');
  scrollToLatest();
}

/** Hide the typing dots indicator. */
function hideTypingIndicator() {
  typingIndicator.style.display = 'none';
  typingIndicator.setAttribute('aria-hidden', 'true');
}

// ----------------------------------------------------------
// 9. Message Bubbles
// ----------------------------------------------------------

/**
 * Append a chat message bubble to the conversation.
 * @param {'user'|'bot'|'error'} role
 * @param {string} text  - Plain text or minimal HTML for bot messages.
 */
function appendMessage(role, text) {
  const row = document.createElement('div');
  row.className = `message-row ${role === 'user' ? 'user' : 'bot'}`;
  row.style.animationFillMode = 'both';

  const bubbleClass = role === 'user'
    ? 'user-bubble'
    : role === 'error'
      ? 'bot-bubble error-bubble'
      : 'bot-bubble';

  // Avatar only for bot messages
  const avatarHtml = (role !== 'user') ? `
    <div class="message-avatar" aria-hidden="true">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M9 18V5l12-2v13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="6" cy="18" r="3" stroke="currentColor" stroke-width="2"/>
        <circle cx="18" cy="16" r="3" stroke="currentColor" stroke-width="2"/>
      </svg>
    </div>
  ` : '';

  // For user messages, escape HTML. For bot, allow safe HTML.
  const contentHtml = role === 'user'
    ? `<p>${escapeHtml(text)}</p>`
    : text;  // Bot messages come from our own render logic

  row.innerHTML = `
    ${avatarHtml}
    <div class="message-bubble ${bubbleClass}">${contentHtml}</div>
  `;

  chatMessages.appendChild(row);
  scrollToLatest();
  return row;
}

// ----------------------------------------------------------
// 10. Recommendation Skeleton
// ----------------------------------------------------------

/**
 * Show shimmer skeleton placeholders while loading.
 * @param {number} [count=3]
 */
function showRecommendationSkeleton(count = 3) {
  emptyState.style.display = 'none';
  songCardsGrid.style.display = 'grid';
  songCardsGrid.innerHTML = '';

  for (let i = 0; i < count; i++) {
    const card = document.createElement('div');
    card.className = 'skeleton-card';
    card.innerHTML = `
      <div class="skeleton-header">
        <div class="skeleton-circle"></div>
        <div class="skeleton-lines">
          <div class="skeleton-block skeleton-line-wide"></div>
          <div class="skeleton-block skeleton-line-mid"></div>
        </div>
      </div>
      <div class="skeleton-block skeleton-line-short" style="margin-top:0.25rem;"></div>
      <div class="skeleton-block skeleton-bar"></div>
      <div style="display:flex;flex-direction:column;gap:0.4rem;margin-top:0.5rem;">
        <div class="skeleton-block skeleton-line-wide"></div>
        <div class="skeleton-block skeleton-line-mid"></div>
        <div class="skeleton-block skeleton-line-short"></div>
      </div>
    `;
    songCardsGrid.appendChild(card);
  }

  recommendationsCount.style.display = 'none';
  clearRecsBtn.style.display = 'none';
}

// ----------------------------------------------------------
// 11. Clear Recommendations
// ----------------------------------------------------------

/** Clear all recommendation cards and show empty state. */
function clearRecommendations() {
  currentRecommendations = [];
  songCardsGrid.innerHTML = '';
  songCardsGrid.style.display = 'none';
  emptyState.style.display = 'flex';
  recommendationsCount.style.display = 'none';
  clearRecsBtn.style.display = 'none';
  updateTasteDashboard([]);
}

// ----------------------------------------------------------
// 12. Song Card Renderer
// ----------------------------------------------------------

/**
 * Build the full HTML string for a single song card.
 * @param {Object} rec   - Recommendation object from the API.
 * @param {number} index - Zero-based index for gradient & animation delay.
 * @returns {string}     - HTML string for the card.
 */
function renderSongCard(rec, index) {
  if (!rec) return '';
  const song = rec.song || rec;
  const gradClass  = pickGradient(index);
  const energy     = parseEnergy(song.energy ?? song.energy_level ?? 0.5);
  const title      = escapeHtml(song.song_title || song.title || 'Unknown Title');
  const artist     = escapeHtml(song.artist || song.artist_name || 'Unknown Artist');
  const genre      = escapeHtml(song.genre || 'Unknown Genre');
  const language   = escapeHtml(song.language || 'Unknown');
  const year       = song.year || song.release_year || '';
  const songId     = escapeHtml(String(song.song_id || song.id || index));
  const reasons    = Array.isArray(rec.reasons) ? rec.reasons : (rec.reason ? [rec.reason] : (Array.isArray(song.reasons) ? song.reasons : ['Great match for your request']));
  const delay      = (index * 0.06).toFixed(2);

  const reasonsHtml = reasons.slice(0, 3).map(r => `
    <div class="reason-item">${escapeHtml(String(r))}</div>
  `).join('');

  const yearBadge = year ? `<span class="badge badge-year">${escapeHtml(String(year))}</span>` : '';

  return `
    <article
      class="song-card"
      style="animation-delay: ${delay}s;"
      data-song-id="${songId}"
      data-title="${title}"
      aria-label="${title} by ${artist}"
    >
      <div class="song-card-header">
        <div class="song-cover">
          <div class="song-cover-gradient ${gradClass}"></div>
          <div class="song-cover-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M9 18V5l12-2v13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="6" cy="18" r="3" stroke="white" stroke-width="2.2"/>
              <circle cx="18" cy="16" r="3" stroke="white" stroke-width="2.2"/>
            </svg>
          </div>
        </div>
        <div class="song-info">
          <div class="song-title-row">
            <div class="song-title" title="${title}">${title}</div>
            <div class="play-links">
              <a
                href="https://open.spotify.com/search/${encodeURIComponent(artist + ' ' + title)}"
                target="_blank"
                rel="noopener noreferrer"
                class="play-link spotify"
                title="Listen on Spotify"
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm4.586 14.424c-.18.295-.565.387-.86.207-2.377-1.454-5.37-1.783-8.893-.982-.336.076-.67-.135-.746-.47-.077-.337.135-.67.472-.747 3.852-.88 7.15-.505 9.822 1.13.295.18.387.563.205.862zm1.225-2.72c-.226.367-.707.487-1.074.26-2.72-1.672-6.87-2.157-10.078-1.182-.413.125-.85-.107-.975-.522-.125-.413.107-.85.522-.975 3.67-1.114 8.24-.57 11.346 1.343.366.226.486.707.26 1.075zm.105-2.81c-3.26-1.937-8.65-2.12-11.77-1.173-.5.152-1.025-.13-1.176-.63-.153-.5.13-1.027.63-1.178 3.626-1.1 9.563-.89 13.317 1.34.45.267.6.846.333 1.296-.267.45-.847.6-1.297.33z"/>
                </svg>
              </a>
              <a
                href="https://www.youtube.com/results?search_query=${encodeURIComponent(artist + ' ' + title)}"
                target="_blank"
                rel="noopener noreferrer"
                class="play-link youtube"
                title="Listen on YouTube"
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M23.498 6.163a3.003 3.003 0 00-2.11-2.11C19.518 3.545 12 3.545 12 3.545s-7.517 0-9.388.508a3.003 3.003 0 00-2.11 2.11C0 8.033 0 12 0 12s0 3.967.502 5.837a3.003 3.003 0 002.11 2.11c1.871.508 9.388.508 9.388.508s7.517 0 9.388-.508a3.002 3.002 0 002.11-2.11C24 15.967 24 12 24 12s0-3.967-.502-5.837zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                </svg>
              </a>
            </div>
          </div>
          <div class="song-artist" title="${artist}">${artist}</div>
        </div>
      </div>

      <div class="song-badges">
        <span class="badge badge-genre">${genre}</span>
        <span class="badge badge-language">${language}</span>
        ${yearBadge}
      </div>

      <div class="song-energy-row">
        <span class="energy-label">Energy</span>
        <div class="energy-track">
          <div
            class="energy-fill"
            data-energy-pct="${energy.pct}"
            style="width: 0%;"
          ></div>
        </div>
        <span class="energy-value">${energy.label}</span>
      </div>

      <div class="song-card-divider"></div>

      <div class="song-reasons">
        <div class="reasons-label">Why this song</div>
        ${reasonsHtml}
      </div>

      <div class="song-card-actions">
        <button
          class="action-btn like-btn"
          data-song-id="${songId}"
          data-feedback="like"
          aria-label="Like ${title}"
          title="Like"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
            <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Like
        </button>
        <button
          class="action-btn dislike-btn"
          data-song-id="${songId}"
          data-feedback="dislike"
          aria-label="Dislike ${title}"
          title="Dislike"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
            <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zm7-13h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Dislike
        </button>
        <button
          class="action-btn similar-btn"
          data-song-id="${songId}"
          data-title="${title}"
          aria-label="Find songs similar to ${title}"
          title="Find similar"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          Similar
        </button>
      </div>
    </article>
  `;
}

/**
 * Animate energy bars to their target widths after cards are in the DOM.
 */
function animateEnergyBars() {
  requestAnimationFrame(() => {
    const fills = songCardsGrid.querySelectorAll('.energy-fill[data-energy-pct]');
    fills.forEach((fill, i) => {
      const pct = parseInt(fill.dataset.energyPct, 10) || 0;
      setTimeout(() => {
        fill.style.width = pct + '%';
      }, 150 + i * 60);
    });
  });
}

// ----------------------------------------------------------
// 13. Render Song Cards
// ----------------------------------------------------------

/**
 * Render all song cards from a recommendations array.
 * @param {Array} recommendations
 */
function renderSongCards(recommendations) {
  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    clearRecommendations();
    return;
  }

  currentRecommendations = recommendations;
  emptyState.style.display = 'none';
  songCardsGrid.style.display = 'grid';
  songCardsGrid.innerHTML = '';

  const html = recommendations.map((rec, i) => renderSongCard(rec, i)).join('');
  songCardsGrid.innerHTML = html;

  // Update count badge
  recommendationsCount.textContent = recommendations.length;
  recommendationsCount.style.display = 'inline-flex';
  clearRecsBtn.style.display = 'inline-flex';

  // Animate energy bars
  animateEnergyBars();

  // Attach event listeners to action buttons
  attachCardActionListeners();

  // Update Taste Analysis Dashboard
  updateTasteDashboard(recommendations);
}

// ----------------------------------------------------------
// 14. Card Action Event Listeners
// ----------------------------------------------------------

/**
 * Attach like/dislike/similar button listeners to freshly rendered cards.
 */
function attachCardActionListeners() {
  // Like / Dislike buttons
  songCardsGrid.querySelectorAll('.like-btn, .dislike-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const songId   = btn.dataset.songId;
      const type     = btn.dataset.feedback;
      submitFeedback(sessionId, songId, type, btn);
    });
  });

  // Similar buttons
  songCardsGrid.querySelectorAll('.similar-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const songTitle = btn.dataset.title;
      findSimilar(songTitle);
    });
  });
}

// ----------------------------------------------------------
// 15. Submit Feedback
// ----------------------------------------------------------

/**
 * POST feedback (like/dislike) to the API and update button state.
 * @param {string} sid     - Session ID
 * @param {string} songId  - Song identifier
 * @param {string} type    - 'like' or 'dislike'
 * @param {HTMLElement} clickedBtn - The button that was clicked
 */
async function submitFeedback(sid, songId, type, clickedBtn) {
  // Optimistic UI: mark button active immediately
  const card = clickedBtn.closest('.song-card');
  if (card) {
    card.querySelectorAll('.like-btn, .dislike-btn').forEach(b => b.classList.remove('active'));
  }
  clickedBtn.classList.add('active');
  clickedBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sid,
        song_id:    songId,
        feedback_type: type,
      }),
    });

    if (!res.ok) {
      // Revert on failure
      clickedBtn.classList.remove('active');
      clickedBtn.disabled = false;
      console.warn('[MusicMind] Feedback submission failed:', res.status);
    }
  } catch (err) {
    clickedBtn.classList.remove('active');
    clickedBtn.disabled = false;
    console.error('[MusicMind] Feedback error:', err);
  }
}

// ----------------------------------------------------------
// 16. Find Similar Songs
// ----------------------------------------------------------

/**
 * Send a "find similar" query to the chat.
 * @param {string} songTitle
 */
function findSimilar(songTitle) {
  const message = `Find songs similar to "${songTitle}"`;
  chatInput.value = message;
  autoResizeTextarea();
  sendMessage(message);
}

// ----------------------------------------------------------
// 17. Send Chat Message
// ----------------------------------------------------------

/**
 * Send a user message to the chat API and handle the response.
 * @param {string} text
 */
async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || isSending) return;

  isSending = true;
  sendButton.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Append user bubble
  appendMessage('user', trimmed);

  // Read filter values
  const countryFilter = document.getElementById('filterCountry').value;
  const langFilter = document.getElementById('filterLanguage').value;
  const genreFilter = document.getElementById('filterGenre').value;
  
  let payloadMessage = trimmed;
  let filterTags = [];
  
  if (countryFilter && countryFilter !== 'all') {
    filterTags.push(`region: ${countryFilter}`);
  }
  if (langFilter && langFilter !== 'all') {
    filterTags.push(`language: ${langFilter}`);
  }
  if (genreFilter && genreFilter !== 'all') {
    filterTags.push(`genre: ${genreFilter}`);
  }
  
  if (filterTags.length > 0) {
    payloadMessage += ` [Filter: ${filterTags.join(', ')}]`;
  }

  // Show typing + skeleton
  showTypingIndicator();
  showRecommendationSkeleton(3);

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);

    const res = await fetch(`${API_BASE_URL}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        message:    payloadMessage,
        session_id: sessionId,
      }),
    });

    clearTimeout(timeout);
    hideTypingIndicator();

    if (!res.ok) {
      let errMsg = `Server error (${res.status})`;
      try {
        const errData = await res.json();
        errMsg = errData.detail || errData.message || errMsg;
      } catch {}
      appendMessage('error', `<p>⚠️ ${escapeHtml(errMsg)}</p>`);
      clearRecommendations();
      return;
    }

    const data = await res.json();

    // Display bot response
    const botText = data.message || data.response || data.reply || 'Here are some songs I think you\'ll love!';
    appendMessage('bot', `<p>${escapeHtml(botText)}</p>`);

    // Render recommendations
    const recs = data.recommendations || data.songs || data.results || [];
    renderSongCards(recs);

    // Update provider info if returned
    if (data.llm_provider || data.provider) {
      updateProviderBadge(data.llm_provider || data.provider);
    }

    // Mark status as connected
    updateStatusIndicator('connected', 'Connected');

  } catch (err) {
    hideTypingIndicator();
    clearRecommendations();

    if (err.name === 'AbortError') {
      appendMessage('error', '<p>⏳ Request timed out. The server took too long to respond. Please try again.</p>');
    } else if (err.message === 'Failed to fetch') {
      appendMessage('error', '<p>🔌 Cannot connect to MusicMind backend. Make sure the server is running at <code>' + API_BASE_URL + '</code></p>');
      updateStatusIndicator('error', 'Offline');
    } else {
      appendMessage('error', `<p>⚠️ Something went wrong: ${escapeHtml(err.message)}</p>`);
    }
  } finally {
    isSending = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
}

// ----------------------------------------------------------
// 18. Event Listeners
// ----------------------------------------------------------

// Send button click
sendButton.addEventListener('click', () => {
  sendMessage(chatInput.value);
});

// Enter key sends (Shift+Enter = newline)
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(chatInput.value);
  }
});

// Auto-resize textarea
chatInput.addEventListener('input', autoResizeTextarea);

// Mood chip clicks
document.querySelectorAll('.mood-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const message = chip.dataset.message;
    if (!message) return;

    // Briefly highlight the chip
    chip.style.transition = 'transform 0.1s ease';
    chip.style.transform = 'scale(0.92)';
    setTimeout(() => {
      chip.style.transform = '';
    }, 150);

    // Fill input and auto-send
    chatInput.value = message;
    autoResizeTextarea();
    sendMessage(message);
  });
});

// Clear recommendations button
clearRecsBtn.addEventListener('click', () => {
  clearRecommendations();
  appendMessage('bot', '<p>Recommendations cleared! Ask me anything to discover new music 🎵</p>');
});

// Paste handler: trim pasted text
chatInput.addEventListener('paste', () => {
  requestAnimationFrame(autoResizeTextarea);
});

// ----------------------------------------------------------
// 19. Initialisation
// ----------------------------------------------------------

/** Initialise the application on DOM load. */
function init() {
  // Focus input on load
  chatInput.focus();

  // Initialize filters
  initFilters();

  // Health check
  checkApiHealth();

  // Repeat health check every 30 seconds
  setInterval(checkApiHealth, 30000);

  // Ensure empty state is shown
  emptyState.style.display = 'flex';
  songCardsGrid.style.display = 'none';

  // Log session info
  console.log('[MusicMind] Session ID:', sessionId);
  console.log('[MusicMind] API Base URL:', API_BASE_URL);
}

// ----------------------------------------------------------
// 20. Taste Profile Dashboard Updates
// ----------------------------------------------------------

/**
 * Calculate acoustic averages of recommendations and update the dashboard UI.
 * @param {Array} recs
 */
function updateTasteDashboard(recs) {
  const dashboard = document.getElementById('tasteProfileDashboard');
  if (!recs || recs.length === 0) {
    dashboard.style.display = 'none';
    return;
  }
  
  let totalEnergy = 0;
  let totalValence = 0;
  let totalDanceability = 0;
  let totalAcousticness = 0;
  let count = 0;
  
  recs.forEach(item => {
    const song = item.song || item;
    if (song) {
      totalEnergy += (song.energy || 0);
      totalValence += (song.valence || 0);
      totalDanceability += (song.danceability || 0);
      totalAcousticness += (song.acousticness || 0);
      count++;
    }
  });
  
  if (count > 0) {
    const avgEnergy = Math.round((totalEnergy / count) * 100);
    const avgValence = Math.round((totalValence / count) * 100);
    const avgDanceability = Math.round((totalDanceability / count) * 100);
    const avgAcousticness = Math.round((totalAcousticness / count) * 100);
    
    document.getElementById('metricEnergyVal').textContent = `${avgEnergy}%`;
    document.getElementById('metricEnergyBar').style.width = `${avgEnergy}%`;
    
    document.getElementById('metricValenceVal').textContent = `${avgValence}%`;
    document.getElementById('metricValenceBar').style.width = `${avgValence}%`;
    
    document.getElementById('metricDanceabilityVal').textContent = `${avgDanceability}%`;
    document.getElementById('metricDanceabilityBar').style.width = `${avgDanceability}%`;
    
    document.getElementById('metricAcousticnessVal').textContent = `${avgAcousticness}%`;
    document.getElementById('metricAcousticnessBar').style.width = `${avgAcousticness}%`;
    
    dashboard.style.display = 'block';
  } else {
    dashboard.style.display = 'none';
  }
}

// ----------------------------------------------------------
// 21. Filters Bar Functionality
// ----------------------------------------------------------

/** Initialize dropdown filter listeners and reset actions. */
function initFilters() {
  const filterCountry = document.getElementById('filterCountry');
  const filterLanguage = document.getElementById('filterLanguage');
  const filterGenre = document.getElementById('filterGenre');
  const filterResetBtn = document.getElementById('filterResetBtn');

  const handleFilterChange = () => {
    let genreText = filterGenre.value !== 'all' ? filterGenre.value : '';
    let langText = filterLanguage.value !== 'all' ? filterLanguage.value : '';
    let countryText = filterCountry.value !== 'all' ? filterCountry.value : '';
    
    // Check constraints to align country & language dynamically
    if (countryText === 'india') {
      if (langText === 'english' || langText === 'spanish') {
        filterLanguage.value = 'all';
        langText = '';
      }
    } else if (countryText === 'international') {
      if (langText === 'hindi' || langText === 'punjabi') {
        filterLanguage.value = 'all';
        langText = '';
      }
    }
    
    // Auto-formulate prompt message
    let queryPrompt = "Recommend me some songs";
    if (genreText) {
      queryPrompt = `Recommend some ${genreText} songs`;
    }
    if (langText) {
      queryPrompt += ` in ${langText}`;
    } else if (countryText === 'india') {
      queryPrompt += ` from India`;
    }
    
    chatInput.value = queryPrompt;
    autoResizeTextarea();
    sendMessage(queryPrompt);
  };

  filterCountry.addEventListener('change', handleFilterChange);
  filterLanguage.addEventListener('change', handleFilterChange);
  filterGenre.addEventListener('change', handleFilterChange);

  filterResetBtn.addEventListener('click', () => {
    filterCountry.value = 'all';
    filterLanguage.value = 'all';
    filterGenre.value = 'all';
    
    const resetPrompt = "Suggest some popular hits";
    chatInput.value = resetPrompt;
    autoResizeTextarea();
    sendMessage(resetPrompt);
  });
}

// Run on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
