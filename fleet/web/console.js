const $ = id => document.getElementById(id);
const states = new Map();
let ws;
let reconnect;
function message(text, kind = '') {
  const box = document.createElement('div');
  box.className = `message ${kind}`;
  box.textContent = text;
  $('messages').append(box);
  while ($('messages').children.length > 150) $('messages').firstChild.remove();
  $('messages').scrollTop = $('messages').scrollHeight;
  return box;
}
function send(event) {
  if (ws?.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(event));
  return true;
}
function clearConfirmations() { $('confirmations').replaceChildren(); }
function stop() {
  if (send({type: 'stop_all'})) { clearConfirmations(); message('STOP requested.'); }
}
async function health() {
  try {
    const data = await (await fetch('/health')).json();
    $('mode').textContent = data.agent === 'grok'
      ? `Grok · ${data.model} · ${data.agent_ready ? 'configured (API access checked on first mission)' : 'key missing'}`
      : `Offline stub · ${data.message}`;
  } catch { $('mode').textContent = 'Unable to check agent configuration.'; }
}
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    $('connection').textContent = 'Connected';
    $('send').disabled = $('stop').disabled = false;
    health();
  };
  ws.onclose = () => {
    $('connection').textContent = 'Disconnected · reconnecting…';
    $('send').disabled = $('stop').disabled = true;
    clearConfirmations();
    clearTimeout(reconnect);
    reconnect = setTimeout(connect, 2000);
  };
  ws.onmessage = ({data}) => {
    const event = JSON.parse(data);
    if (event.type === 'device_state') {
      const d = event.device;
      if (!states.has(d.id)) {
        const card = document.createElement('article'); card.className = 'device';
        const title = document.createElement('h3'); title.textContent = `${d.id} · simulator`;
        const img = document.createElement('img'); img.src = `/video/${encodeURIComponent(d.id)}`; img.alt = `${d.id} camera`;
        const info = document.createElement('p');
        card.append(title, img, info); $('devices').append(card); states.set(d.id, info);
      }
      states.get(d.id).textContent = `${d.connected ? 'Connected' : 'Offline'} · ${d.flying ? 'Flying' : 'Landed'} · Battery ${d.battery_pct}% · Height ${d.height_cm} cm · Yaw ${d.yaw_deg}°`;
    } else if (event.type === 'mission_status') {
      $('send').disabled = event.status === 'started';
      if (event.status !== 'started') { clearConfirmations(); message(`Mission ${event.status}.`); }
    } else if (event.type === 'chat') {
      message(`${event.role === 'user' ? 'You' : 'Agent'}: ${event.text}`, event.role);
      if (/^(STOP completed|Mission failed)/.test(event.text)) clearConfirmations();
    } else if (event.type === 'tool_call') {
      message(`${event.name} ${JSON.stringify(event.args)} → ${JSON.stringify(event.result)}`, 'tool');
      if (event.name === 'takeoff') clearConfirmations();
    } else if (event.type === 'photo') {
      const box = message(event.caption); const img = document.createElement('img');
      // Only local photos are accepted; model text never becomes HTML or a URL.
      if (/^\/photos\/[a-f0-9]+\.jpg$/.test(event.url)) { img.src = event.url; img.alt = event.caption; box.append(img); }
    } else if (event.type === 'confirm_request') {
      const box = document.createElement('div'); box.className = 'confirmation';
      const reason = document.createElement('p'); reason.textContent = `${event.device_id}: ${event.reason}`; box.append(reason);
      for (const approved of [true, false]) {
        const button = document.createElement('button'); button.textContent = approved ? 'Approve' : 'Reject';
        button.onclick = () => { if (send({type: 'confirm_response', request_id: event.request_id, approved})) box.remove(); };
        box.append(button);
      }
      $('confirmations').append(box);
    }
  };
}
$('mission').onsubmit = event => {
  event.preventDefault();
  const text = $('prompt').value.trim();
  if (text && send({type: 'user_message', text})) $('prompt').value = '';
};
$('stop').onclick = stop;
document.addEventListener('keydown', event => { if (event.key === 'Escape') stop(); });
connect();
