const replayUrl = "./data/replay.json?v=1.0.1";

const state = {
  data: null,
  episodes: [],
  selectedEpisode: 0,
  selectedStep: 0,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function averageStepReward(episode) {
  if (!episode.steps?.length) return 0;
  return episode.steps.reduce((sum, step) => sum + Number(step.reward ?? 0), 0) / episode.steps.length;
}

function runModel(run) {
  return run.model ?? run.config?.agent_config?.model ?? "model";
}

function normalizeEpisodes(data) {
  return (data.episodes ?? []).map((episode, index) => {
    const replaySteps = data.replay?.[episode.id];
    const steps = Array.isArray(episode.steps) ? episode.steps : replaySteps ?? [];
    return {
      ...episode,
      label: episode.label ?? `Episode ${index + 1}`,
      steps,
    };
  });
}

function tileClass(value) {
  if (!value) return "empty";
  return `tile-${Math.min(Number(value), 2048)}`;
}

function currentEpisode() {
  return state.episodes[state.selectedEpisode];
}

function currentStep() {
  return currentEpisode()?.steps[state.selectedStep];
}

function renderMetrics() {
  const run = state.data.run ?? {};
  const steps = state.episodes.flatMap((episode) => episode.steps);
  const correct = steps.filter((step) => step.info?.correct === true).length;
  const penalties = steps.filter((step) => Number(step.reward) < 0).length;

  $("averageReward").textContent = Number(run.scores?.average_reward ?? 0).toFixed(2);
  $("runMeta").textContent = `${runModel(run)} · ${state.episodes.length} episodes · vow ${state.data.binding_vow_version}`;
  $("runIdPill").textContent = run.id ?? "--";
  $("correctCount").textContent = `${correct}/${steps.length}`;
  $("penaltyCount").textContent = `${penalties}/${steps.length}`;

  $("metrics").innerHTML = [
    ["Status", run.status ?? "unknown"],
    ["Episodes", state.episodes.length],
    ["Reward Range", "-2.0 to +2.0"],
    ["Policy", "MCTS rollouts"],
  ]
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderTabs() {
  $("episodeTabs").innerHTML = state.episodes
    .map(
      (episode, index) => `
        <button class="tab ${index === state.selectedEpisode ? "active" : ""}" data-index="${index}">
          Seed ${escapeHtml(episode.seed)} · ${averageStepReward(episode).toFixed(2)}
        </button>
      `,
    )
    .join("");

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedEpisode = Number(button.dataset.index);
      state.selectedStep = 0;
      renderReplay();
    });
  });
}

function renderBoard(board) {
  $("board").innerHTML = board
    .flat()
    .map((value) => `<div class="tile ${tileClass(value)}">${value ? escapeHtml(value) : ""}</div>`)
    .join("");
}

function renderPolicy(policy, chosenMove) {
  $("policyTable").innerHTML = `
    <div class="policy-row policy-head">
      <span>Rank</span><span>Move</span><span>Value</span><span>Score</span><span>Empty</span>
    </div>
    ${policy
      .map(
        (row) => `
          <div class="policy-row ${row.move === chosenMove ? "chosen" : ""}">
            <span>#${escapeHtml(row.rank)}</span>
            <strong>${escapeHtml(row.move)}</strong>
            <span>${Number(row.mean_value).toFixed(1)}</span>
            <span>${escapeHtml(row.immediate_score)}</span>
            <span>${escapeHtml(row.empty_after)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

function renderChat(step) {
  const info = step.info ?? {};
  const resultClass = Number(step.reward) >= 0 ? "reward-message" : "penalty-message";
  $("chatWindow").innerHTML = `
    <div class="chat-message env-message">
      <div class="chat-meta"><span>Environment</span><small>Step ${escapeHtml(step.step)}</small></div>
      <div class="chat-bubble">Valid moves: ${escapeHtml(step.observation.valid_moves.join(", "))}. Use the MCTS table to choose one direction.</div>
    </div>
    <div class="chat-message thinking-message">
      <div class="chat-meta"><span>Model Reasoning</span><small>Policy comparison</small></div>
      <div class="chat-bubble">${escapeHtml(step.reasoning)}</div>
    </div>
    <div class="chat-message model-message">
      <div class="chat-meta"><span>Action</span><small>${escapeHtml(info.valid ? "submitted move" : "invalid action")}</small></div>
      <div class="chat-bubble">${escapeHtml(step.action)}</div>
    </div>
    <div class="chat-message ${resultClass}">
      <div class="chat-meta"><span>Reward</span><small>${escapeHtml(info.reason)}</small></div>
      <div class="chat-bubble">
        ${Number(step.reward).toFixed(2)} · rank ${escapeHtml(info.rank ?? "--")} ·
        correct streak ${escapeHtml(info.correct_streak ?? 0)} · wrong streak ${escapeHtml(info.wrong_streak ?? 0)}
      </div>
    </div>
  `;
}

function renderReplay() {
  const episode = currentEpisode();
  const step = currentStep();
  if (!episode || !step) return;

  const obs = step.observation;
  const info = step.info ?? {};
  $("episodeMeta").textContent = `${episode.label ?? "Episode"} · step ${step.step}/${episode.steps.length}`;
  $("episodeTitle").textContent = `Seed ${episode.seed}: choose the next 2048 move`;
  $("rewardPill").textContent = `Step reward ${Number(step.reward).toFixed(2)}`;
  $("chosenMove").textContent = step.action;
  $("bestMove").textContent = info.best_move ?? obs.mcts_policy?.[0]?.move ?? "--";
  $("boardStats").innerHTML = `
    <span>Score <strong>${escapeHtml(obs.score)}</strong></span>
    <span>Max tile <strong>${escapeHtml(obs.max_tile)}</strong></span>
    <span>Empty <strong>${escapeHtml(obs.empty_cells)}</strong></span>
  `;
  $("nextStepButton").textContent = state.selectedStep + 1 >= episode.steps.length ? "Restart Episode" : "Next Step";

  renderBoard(obs.board);
  renderPolicy(obs.mcts_policy ?? [], step.action);
  renderChat(step);
}

function nextStep() {
  const episode = currentEpisode();
  if (!episode) return;
  state.selectedStep = state.selectedStep + 1 >= episode.steps.length ? 0 : state.selectedStep + 1;
  renderReplay();
}

function render() {
  renderMetrics();
  renderTabs();
  renderReplay();
}

async function init() {
  const response = await fetch(replayUrl);
  if (!response.ok) throw new Error(`Could not load ${replayUrl}`);
  state.data = await response.json();
  state.episodes = normalizeEpisodes(state.data);
  if (!state.episodes.length) throw new Error("Replay contains no episodes.");
  render();
  $("nextStepButton").addEventListener("click", nextStep);
}

init().catch((error) => {
  $("episodeTitle").textContent = "Could not load replay data";
  $("chatWindow").innerHTML = `<div class="chat-message env-message">${escapeHtml(error.message)}</div>`;
});
