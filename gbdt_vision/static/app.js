const byId = (id) => document.getElementById(id);
const labels = {occupied: 'Occupied', empty: 'Empty', stale: 'Stale', unknown: 'Unknown'};
let mode = new URLSearchParams(location.search).get('mode') === 'demo' ? 'demo' : 'live';
let selectedRoom = null;
let snapshot = null;
let controller = null;
let receivedAt = 0;
let disconnected = false;
let advisorMode = 'room';
let advisorKey = null;
let advisorRequest = null;

function advisorRoom() { return snapshot?.rooms.find((room) => room.id === selectedRoom); }

function advisorInputs() {
  if (advisorMode === 'room') {
    const room = advisorRoom();
    const condition = room ? lightingFor(room).status : 'unknown';
    if (disconnected || room?.occupancy_count == null || !['low', 'normal', 'excessive'].includes(condition)) return null;
    return {people_count: room.occupancy_count, lighting_condition: condition};
  }
  const people = byId('advisor-people');
  const condition = byId('advisor-lighting').value;
  if (!people.value || !people.checkValidity() || !condition) return null;
  return {people_count: Number(people.value), lighting_condition: condition};
}

function currentAdvisorKey() {
  const inputs = advisorInputs();
  if (!inputs) return null;
  const room = advisorRoom();
  return JSON.stringify([advisorMode, advisorMode === 'room' ? mode : 'manual',
    advisorMode === 'room' ? [selectedRoom, room?.source_type] : null, inputs]);
}

function clearAdvisor(message = 'No recommendation generated.') {
  advisorRequest?.abort(); advisorRequest = null; advisorKey = null;
  byId('advisor-result').hidden = true;
  byId('advisor-output').setAttribute('aria-busy', 'false');
  text('advisor-status', message);
}

function renderAdvisor() {
  const room = advisorRoom();
  const roomMode = advisorMode === 'room';
  if (roomMode) {
    byId('advisor-people').value = room?.occupancy_count ?? '';
    const condition = room ? lightingFor(room).status : 'unknown';
    byId('advisor-lighting').value = ['low', 'normal', 'excessive'].includes(condition) ? condition : '';
  }
  byId('advisor-people').readOnly = roomMode;
  byId('advisor-lighting').disabled = roomMode;
  text('advisor-context', roomMode ? `${room?.name || 'No room selected'} / ${mode === 'demo' ? 'Demo room data' : room?.source_type === 'video' ? 'Recorded video' : 'Live room data'}` : 'Manual what-if scenario');
  document.querySelectorAll('[data-advisor-mode]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.advisorMode === advisorMode)));
  const key = currentAdvisorKey();
  if (advisorKey && advisorKey !== key) clearAdvisor('Inputs changed or expired. Previous recommendation cleared.');
  byId('advisor-generate').disabled = !key || !!advisorRequest || disconnected;
  byId('advisor-generate').textContent = advisorRequest ? 'Generating...' : 'Generate recommendations';
}

async function generateAdvice(event) {
  event.preventDefault();
  if (!byId('advisor-form').reportValidity() || advisorRequest) return;
  const inputs = advisorInputs();
  const key = currentAdvisorKey();
  if (!inputs || !key) return;
  clearAdvisor('Generating recommendations...');
  advisorKey = key;
  const request = new AbortController(); advisorRequest = request;
  const timeout = setTimeout(() => request.abort(), 20000);
  byId('advisor-output').setAttribute('aria-busy', 'true');
  renderAdvisor();
  try {
    const payload = advisorMode === 'room' ? {mode, room_id: selectedRoom} : {mode: 'manual', ...inputs};
    const response = await fetch('/api/recommendations', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload), signal: request.signal});
    const result = await response.json();
    if (advisorRequest !== request || currentAdvisorKey() !== key) return;
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Recommendation request failed.');
    const current = advisorInputs();
    if (result.inputs.people_count !== current.people_count || result.inputs.lighting_condition !== current.lighting_condition) {
      clearAdvisor('Room inputs changed. Recommendation discarded.'); return;
    }
    const source = result.source === 'genai' ? `GenAI / ${result.provider}${result.cached ? ' / cached' : ''}` :
      `Rule-based fallback / ${result.fallback_reason === 'not_configured' ? 'AI not configured' : 'AI unavailable'}`;
    text('advisor-source', source.toUpperCase());
    text('advisor-summary', result.summary);
    const basis = result.basis.mode === 'manual' ? 'Manual what-if scenario' : `${result.basis.mode === 'demo' ? 'Demo' : result.basis.source_type === 'video' ? 'Recorded video' : 'Live'} / ${result.basis.room_name}`;
    text('advisor-basis', `${basis} / ${result.inputs.people_count} people / ${result.inputs.lighting_condition} lighting`);
    const list = byId('advisor-actions'); list.replaceChildren();
    result.actions.forEach((action) => { const item = document.createElement('li'); item.textContent = action; list.append(item); });
    text('advisor-caution', result.caution);
    byId('advisor-result').hidden = false;
    text('advisor-status', 'Recommendation ready.');
  } catch (error) {
    if (advisorRequest === request) {
      advisorKey = null;
      text('advisor-status', error.name === 'AbortError' ? 'Generation timed out. No recommendation was applied.' : error.message);
    }
  } finally {
    clearTimeout(timeout);
    if (advisorRequest === request) {
      advisorRequest = null;
      byId('advisor-output').setAttribute('aria-busy', 'false');
      renderAdvisor();
    }
  }
}

async function loadAdvisorStatus() {
  try {
    const response = await fetch('/api/recommendations/status', {cache: 'no-store'});
    if (!response.ok) throw new Error('Status unavailable');
    const status = await response.json();
    text('advisor-provider', status.configured ? `GENAI CONFIGURED / ${status.provider.toUpperCase()}` : 'RULE-BASED FALLBACK / AI NOT CONFIGURED');
  } catch { text('advisor-provider', 'PROVIDER STATUS UNAVAILABLE'); }
}

function lightingFor(room) {
  return room.lighting_analysis || {status: 'unknown', label: 'Unknown lighting', brightness: null, evidence: 'none',
    reading_age_seconds: null, low_threshold: 0.30, high_threshold: 0.75, message: 'No fresh frame brightness is available.'};
}

function invalidateLighting(room) {
  const analysis = room.lighting_analysis;
  if (!analysis) return;
  analysis.status = 'unknown'; analysis.label = 'Unknown lighting';
  analysis.brightness = null; analysis.evidence = 'none';
  analysis.message = 'No fresh frame brightness is available.';
}

function renderLighting() {
  for (const status of ['low', 'excessive', 'normal']) {
    text(`lighting-${status}`, snapshot.rooms.filter((room) => lightingFor(room).status === status).length);
  }
  const list = byId('lighting-alerts');
  const signature = JSON.stringify(snapshot.rooms.map((room) => {
    const analysis = lightingFor(room);
    return [room.id, room.name, analysis.status, analysis.brightness, analysis.scope, analysis.evidence, analysis.message];
  }));
  if (list.dataset.signature === signature) return;
  list.dataset.signature = signature;
  const focusedRoom = document.activeElement?.closest('.lighting-room')?.dataset.roomId;
  list.replaceChildren();
  snapshot.rooms.forEach((room) => {
    const analysis = lightingFor(room);
    const row = document.createElement('div'); row.className = `lighting-alert ${analysis.status}`;
    const body = document.createElement('div'); body.className = 'lighting-alert-body';
    const name = document.createElement('button'); name.type = 'button'; name.className = 'lighting-room'; name.dataset.roomId = room.id;
    name.textContent = room.name; name.addEventListener('click', () => selectRoom(room.id));
    const label = document.createElement('span'); label.className = 'lighting-alert-label';
    label.textContent = analysis.label;
    label.classList.add(analysis.status);
    const evidence = document.createElement('p');
    const scope = analysis.scope === 'configured_region' ? 'configured region' : analysis.scope === 'full_frame' ? 'full frame' : 'camera observation';
    evidence.textContent = analysis.brightness === null ? 'Brightness unavailable' :
      `${(analysis.brightness * 100).toFixed(1)}% brightness / ${scope}${analysis.evidence === 'simulated' ? ' / simulated' : ''}`;
    const message = document.createElement('p'); message.textContent = analysis.message;
    body.append(name, label, evidence, message);
    row.append(body);
    list.append(row);
  });
  if (focusedRoom) [...list.querySelectorAll('button')].find((button) => button.dataset.roomId === focusedRoom)?.focus({preventScroll: true});
}

function text(id, value) { byId(id).textContent = value; }
function ageLabel(age) {
  if (age === null) return 'No observation';
  if (age < 60) return `${Math.floor(age)}s ago`;
  if (age < 3600) return `${Math.floor(age / 60)}m ago`;
  if (age < 86400) return `${Math.floor(age / 3600)}h ago`;
  return `${Math.floor(age / 86400)}d ago`;
}
function node(tag, attrs = {}, content = null) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  if (content !== null) element.textContent = content;
  return element;
}
function badge(status) {
  const element = document.createElement('span');
  element.className = `status-badge ${status}`;
  element.textContent = labels[status];
  return element;
}
function selectRoom(id) { selectedRoom = id; renderRooms(); renderAdvisor(); }

function furniture(group, room, index) {
  if (index === 0) {
    for (let desk = 0; desk < 6; desk++) {
      const left = room.x + 42 + (desk % 3) * ((room.width - 100) / 3);
      const top = room.y + 180 + Math.floor(desk / 3) * 105;
      group.append(node('rect', {x: left, y: top, width: 90, height: 50, rx: 3, class: 'furniture'}));
      group.append(node('rect', {x: left + 30, y: top + 56, width: 30, height: 15, rx: 4, class: 'furniture'}));
    }
  } else {
    group.append(node('rect', {x: room.x + room.width - 125, y: room.y + 60, width: 80, height: 100, rx: index === 1 ? 25 : 3, class: 'furniture'}));
    for (let chair = 0; chair < 4; chair++) {
      group.append(node('rect', {x: room.x + room.width - 145 + (chair % 2) * 110, y: room.y + 75 + Math.floor(chair / 2) * 55, width: 15, height: 22, rx: 3, class: 'furniture'}));
    }
  }
}

function renderRooms() {
  if (!snapshot) return;
  const focused = document.activeElement?.closest('.plan-room')?.dataset.roomId;
  const focusedTable = document.activeElement?.closest('.room-link')?.dataset.roomId;
  const plan = byId('floor-plan');
  plan.replaceChildren(node('title', {}, `${snapshot.floor.level} room occupancy`));
  plan.setAttribute('viewBox', `0 0 ${snapshot.floor.width} ${snapshot.floor.height}`);
  plan.append(node('rect', {x: 30, y: 40, width: snapshot.floor.width - 60, height: snapshot.floor.height - 85, fill: '#f5f7f2', stroke: '#b9c4b8', 'stroke-width': 3}));
  plan.append(node('text', {x: 65, y: snapshot.floor.height - 60, class: 'circulation'}, 'CIRCULATION / ENTRANCE'));
  const table = byId('room-table');
  table.replaceChildren();
  snapshot.rooms.forEach((room, index) => {
    const group = node('g', {class: `plan-room ${room.status}${room.id === selectedRoom ? ' selected' : ''}`, tabindex: 0, role: 'button', 'aria-label': `${room.name}, ${labels[room.status]}, ${room.occupancy_count === null ? 'occupancy unknown' : room.occupancy_count + ' people'}`, 'aria-pressed': room.id === selectedRoom});
    group.dataset.roomId = room.id;
    group.append(node('rect', {x: room.x, y: room.y, width: room.width, height: room.height, class: 'room-shape'}));
    furniture(group, room, index);
    group.append(node('text', {x: room.x + 22, y: room.y + 32, class: 'room-index'}, room.label));
    group.append(node('text', {x: room.x + 22, y: room.y + 66, class: 'room-title', 'font-size': Math.min(24, (room.width - 40) / (room.name.length * 0.58))}, room.name));
    group.append(node('text', {x: room.x + 22, y: room.y + 117, class: 'room-count'}, room.occupancy_count === null ? '--' : String(room.occupancy_count)));
    group.append(node('text', {x: room.x + 22, y: room.y + 145, class: 'room-state'}, room.occupancy_count === null ? labels[room.status] : 'visible people'));
    const lighting = lightingFor(room);
    group.append(node('text', {x: room.x + 22, y: room.y + room.height - 18, class: `room-light-status ${lighting.status}`},
      lighting.label.toUpperCase()));
    group.addEventListener('click', () => selectRoom(room.id));
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectRoom(room.id); } });
    plan.append(group);
    const row = document.createElement('tr');
    row.classList.toggle('selected-row', room.id === selectedRoom);
    const nameCell = document.createElement('td');
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'room-link'; button.setAttribute('aria-pressed', String(room.id === selectedRoom));
    button.dataset.roomId = room.id;
    const number = document.createElement('span'); number.className = 'table-room-index'; number.textContent = room.label;
    button.append(number, document.createTextNode(room.name));
    button.addEventListener('click', () => selectRoom(room.id));
    nameCell.append(button); row.append(nameCell);
    const statusCell = document.createElement('td'); statusCell.append(badge(room.status)); row.append(statusCell);
    const lightingCell = document.createElement('td');
    const lightingLabel = document.createElement('span'); lightingLabel.className = `lighting-table-label ${lighting.status}`; lightingLabel.textContent = lighting.label;
    lightingCell.append(lightingLabel); row.append(lightingCell);
    [room.occupancy_count ?? '--', room.capacity, room.camera_id, ageLabel(room.age_seconds)].forEach((value) => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
    table.append(row);
  });
  if (focused) [...plan.querySelectorAll('.plan-room')].find((element) => element.dataset.roomId === focused)?.focus({preventScroll: true});
  if (focusedTable) [...table.querySelectorAll('.room-link')].find((element) => element.dataset.roomId === focusedTable)?.focus({preventScroll: true});
  const room = snapshot.rooms.find((entry) => entry.id === selectedRoom) || snapshot.rooms[0];
  text('room-number', room.label); text('room-name', room.name); text('room-use', room.use);
  text('room-status', labels[room.status]); byId('room-status').className = `status-badge ${room.status}`;
  text('room-count', room.occupancy_count ?? '--'); text('room-capacity', `/ ${room.capacity} capacity`);
  const meter = byId('capacity-meter'); meter.max = room.capacity; meter.value = room.occupancy_count ?? 0;
  meter.hidden = room.occupancy_count === null;
  text('room-age', ageLabel(room.age_seconds)); text('room-camera', room.camera_id); text('room-vision-id', room.vision_room_id);
  text('room-source', {camera: 'Live camera', video: 'Recorded video', demo: 'Simulated'}[room.source_type] || '--');
  text('room-last-count', room.last_known_count ?? '--');
  text('room-brightness', room.brightness === null ? '--' : `${(room.brightness * 100).toFixed(1)}%`);
  const analysis = lightingFor(room);
  text('light-scenario', analysis.label);
  byId('light-scenario').dataset.status = analysis.status;
  text('light-evidence', {simulated: 'Simulated brightness', camera_brightness: 'Camera brightness', none: 'None'}[analysis.evidence] || 'None');
  text('light-age', ageLabel(analysis.reading_age_seconds));
  text('light-scope', {full_frame: 'Full frame', configured_region: 'Configured region'}[analysis.scope] || '--');
  text('light-range', `${Math.round(analysis.low_threshold * 100)}%-${Math.round(analysis.high_threshold * 100)}%`);
  const brightnessMeter = byId('brightness-meter');
  brightnessMeter.hidden = analysis.brightness === null;
  brightnessMeter.low = analysis.low_threshold; brightnessMeter.high = analysis.high_threshold;
  brightnessMeter.optimum = (analysis.low_threshold + analysis.high_threshold) / 2;
  brightnessMeter.value = analysis.brightness ?? 0;
  text('light-message', analysis.message);
}

function render(offline = false) {
  if (!snapshot) return;
  if (!selectedRoom || !snapshot.rooms.some((room) => room.id === selectedRoom)) selectedRoom = snapshot.rooms[0].id;
  const summary = snapshot.summary;
  text('building-name', `${snapshot.floor.building} / ${snapshot.floor.level}`.toUpperCase());
  text('floor-name', snapshot.floor.name);
  text('people', summary.reporting_rooms ? summary.visible_people : '--');
  text('occupied', summary.reporting_rooms ? `${summary.occupied_rooms} / ${summary.total_rooms}` : '--');
  text('coverage', `${summary.reporting_rooms} / ${summary.total_rooms}`);
  text('freshness', `Freshness limit: ${snapshot.stale_seconds} seconds`);
  text('last-sync', new Date(snapshot.generated_at).toLocaleTimeString([], {hour12: false}));
  text('room-total', `${summary.total_rooms} SPACES`);
  text('source-label', mode === 'demo' ? 'Simulated observations' : 'Live observation file');
  document.querySelectorAll('[data-mode]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  text('connection-text', offline ? 'Disconnected' : mode === 'demo' ? 'Demo active' : summary.reporting_rooms ? 'Receiving data' : 'No fresh data');
  byId('connection-dot').className = `dot ${offline ? 'stale' : mode === 'demo' || summary.reporting_rooms ? 'occupied' : 'unknown'}`;
  const notice = byId('notice');
  notice.hidden = false;
  if (offline) notice.textContent = 'Connection lost. Current occupancy is unknown.';
  else if (mode === 'demo') notice.textContent = 'DEMO DATA · Occupancy and brightness are simulated. No live observations are changed.';
  else if (snapshot.ingestion_error) notice.textContent = snapshot.ingestion_error;
  else if (!summary.reporting_rooms) notice.textContent = 'No fresh camera observations. Room occupancy is unknown.';
  else notice.hidden = true;
  const invalid = snapshot.invalid_lines + snapshot.rejected_records;
  text('data-quality', invalid ? `${invalid} invalid records skipped` : mode === 'demo' ? 'SIMULATED DATA / SAMPLE FLOOR PLAN' : `${summary.reporting_rooms} of ${summary.total_rooms} rooms reporting / SAMPLE FLOOR PLAN`);
  renderRooms();
  renderLighting();
  renderAdvisor();
}

async function refresh(force = false) {
  if (controller && !force) return;
  controller?.abort();
  const request = new AbortController(); controller = request;
  const timeout = setTimeout(() => request.abort(), 8000);
  try {
    const response = await fetch(`/api/twin?mode=${mode}`, {signal: request.signal, cache: 'no-store'});
    if (!response.ok) throw new Error('Twin unavailable');
    const result = await response.json();
    if (controller !== request) return;
    snapshot = result;
    receivedAt = performance.now();
    snapshot.rooms.forEach((room) => { room.receivedAge = room.age_seconds; });
    disconnected = false;
    render();
  } catch (error) {
    if (controller !== request) return;
    disconnected = true;
    if (snapshot) {
      snapshot.rooms.forEach((room) => { room.status = room.observed_at ? 'stale' : 'unknown'; room.occupancy_count = null; room.brightness = null; invalidateLighting(room); });
      snapshot.summary.reporting_rooms = 0; snapshot.summary.occupied_rooms = 0; snapshot.summary.visible_people = 0;
      render(true);
    } else {
      text('connection-text', 'Disconnected');
      byId('notice').hidden = false; text('notice', 'The floor twin is unavailable. Retrying connection.');
    }
  } finally { clearTimeout(timeout); if (controller === request) controller = null; }
}

document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  if (mode === button.dataset.mode) return;
  mode = button.dataset.mode;
  const url = new URL(location.href); url.searchParams.set('mode', mode); history.replaceState(null, '', url);
  snapshot = null;
  clearAdvisor(); renderAdvisor();
  byId('lighting-alerts').replaceChildren();
  delete byId('lighting-alerts').dataset.signature;
  for (const status of ['low', 'excessive', 'normal']) text(`lighting-${status}`, '--');
  text('light-scenario', 'Unknown lighting'); byId('light-scenario').dataset.status = 'unknown'; text('light-evidence', 'None'); text('light-age', '--'); text('light-scope', '--');
  text('light-message', 'Waiting for current frame brightness'); byId('brightness-meter').hidden = true;
  text('people', '--'); text('occupied', '--'); text('coverage', '--');
  byId('floor-plan').replaceChildren(); byId('room-table').replaceChildren();
  text('room-count', '--'); text('room-brightness', '--'); text('room-last-count', '--');
  text('room-status', 'Loading'); byId('capacity-meter').hidden = true;
  text('connection-text', 'Connecting'); byId('connection-dot').className = 'dot unknown';
  byId('notice').hidden = false; text('notice', mode === 'demo' ? 'Loading simulated observations.' : 'Connecting to live observations.');
  document.querySelectorAll('[data-mode]').forEach((entry) => entry.setAttribute('aria-pressed', String(entry.dataset.mode === mode)));
  refresh(true);
}));
byId('refresh').addEventListener('click', () => refresh(true));
byId('advisor-form').addEventListener('submit', generateAdvice);
document.querySelectorAll('[data-advisor-mode]').forEach((button) => button.addEventListener('click', () => {
  if (advisorMode === button.dataset.advisorMode) return;
  advisorMode = button.dataset.advisorMode;
  clearAdvisor();
  if (advisorMode === 'manual') {
    if (!byId('advisor-people').value) byId('advisor-people').value = '0';
    if (!byId('advisor-lighting').value) byId('advisor-lighting').value = 'normal';
  }
  renderAdvisor();
}));
for (const id of ['advisor-people', 'advisor-lighting']) byId(id).addEventListener('input', () => { if (advisorMode === 'manual') { clearAdvisor('Inputs changed. Previous recommendation cleared.'); renderAdvisor(); } });
loadAdvisorStatus();
refresh();
setInterval(() => refresh(), 3000);
setInterval(() => {
  if (!snapshot) return;
  const elapsed = (performance.now() - receivedAt) / 1000;
  snapshot.rooms.forEach((room) => {
    if (room.receivedAge === null) return;
    room.age_seconds = room.receivedAge + elapsed;
    if (room.lighting_analysis) room.lighting_analysis.reading_age_seconds = room.age_seconds;
    if (room.age_seconds > snapshot.stale_seconds) {
      room.status = 'stale'; room.occupancy_count = null; room.brightness = null;
      invalidateLighting(room);
    }
  });
  const fresh = snapshot.rooms.filter((room) => room.status === 'occupied' || room.status === 'empty');
  snapshot.summary.reporting_rooms = fresh.length;
  snapshot.summary.visible_people = fresh.reduce((sum, room) => sum + room.occupancy_count, 0);
  snapshot.summary.occupied_rooms = fresh.filter((room) => room.status === 'occupied').length;
  render(disconnected);
}, 1000);