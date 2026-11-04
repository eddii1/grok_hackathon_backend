"""SEEK app: camera, live map (2D as Grok sees it, 3D point cloud), manual control and the Grok agent.
http://localhost:8001

    uv run fleet-viewer --webcam 0          # your laptop camera: you carry it and do what the app says
    uv run fleet-viewer --sim room_360.jpg  # simulated drone inside a 360 photo
    uv run fleet-viewer --tello             # the real drone (laptop on its Wi-Fi)
    ... --goal "a red mug"                  # start straight in agent mode with this goal

The map is built live from the camera the whole time. Two modes, switchable at any moment:
  - Manual: you send the commands (buttons).
  - Agent: type what to find and press Start; Grok decides every move until you confirm a find or press Stop.
With the webcam every command becomes an instruction on screen ("turn right 90 degrees"); turns finish by
themselves when the camera has turned, moves when you press Done.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from fleet import tracing
from fleet.drone import Drone, SimBackend, TelloBackend, WebcamBackend
from fleet.grid import jpg
from fleet.harness import Seek, decision_config
from fleet.mapper import LiveMap, depth

app = FastAPI(telemetry={"tracing": False, "metrics": False, "logs": False})  # keep viewer polls out of Phoenix
ctx: dict = {"mode": "manual", "seek": None, "approve_each": True}


# ---------- agent control ----------


def start_agent(goal: str) -> None:
    stop_agent()
    out = ctx["out"] / f"mission-{time.strftime('%H%M%S')}"
    seek = Seek(ctx["drone"], ctx["map"], goal, out, decision_config())
    seek.approve_each = ctx["approve_each"]
    ctx.update(mode="agent", seek=seek)
    print(f"agent: searching for {goal!r}   files: {out}")
    threading.Thread(target=seek.run, daemon=True).start()


def stop_agent() -> None:
    if ctx["seek"] is not None:
        ctx["seek"].stop()


@app.post("/mode/{mode}")
def set_mode(mode: str) -> dict:
    if mode not in ("manual", "agent"):
        raise HTTPException(400, "mode is manual or agent")
    if mode == "manual":
        stop_agent()
    ctx["mode"] = mode
    return {"mode": mode}


@app.post("/agent/start")
def agent_start(goal: str = Body(..., embed=True)) -> dict:
    if not goal.strip():
        raise HTTPException(400, "type what to find")
    start_agent(goal.strip())
    return {"ok": True}


@app.post("/agent/stop")
def agent_stop() -> dict:
    stop_agent()
    return {"ok": True}


@app.post("/confirm/{answer}")
def confirm(answer: str) -> dict:
    if ctx["seek"] is not None:
        ctx["seek"].answer(answer == "yes")
    return {"ok": True}


@app.post("/approve")
def approve(ok: bool = Body(...), note: str = Body("")) -> dict:
    """The operator approved or rejected the agent's next action."""
    if ctx["seek"] is not None:
        ctx["seek"].decide(ok, note)
    return {"ok": True}


@app.post("/auto/{on}")
def auto(on: str) -> dict:
    """auto on: the agent acts without asking. auto off: every action waits for approval."""
    ctx["approve_each"] = on != "on"
    if ctx["seek"] is not None:
        ctx["seek"].approve_each = ctx["approve_each"]
    return {"approve_each": ctx["approve_each"]}


@app.post("/map/reset")
def map_reset() -> dict:
    """A new place: forget the map (not while the agent is searching)."""
    seek = ctx["seek"]
    if seek is not None and not seek.status.startswith("ended"):
        raise HTTPException(409, "stop the agent first")
    ctx["map"].reset()
    ctx["seek"] = None
    return {"ok": True}


@app.post("/done")
def done() -> dict:
    """The person carrying the camera finished the instruction."""
    if hasattr(ctx["drone"].b, "done"):
        ctx["drone"].b.done()
    return {"ok": True}


@app.get("/found/{name}")
def found_photo(name: str) -> FileResponse:
    seek = ctx["seek"]
    path = seek.out / Path(name).name if seek else None
    if not path or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ---------- state for the page ----------


@app.get("/state")
def state() -> dict:
    seek, d = ctx["seek"], ctx["drone"]
    running = seek is not None and not seek.status.startswith("ended")
    return {
        "kind": ctx["kind"],
        "mode": ctx["mode"],
        "running": running,
        "goal": seek.goal if seek else "",
        "status": seek.status if seek else "",
        "confirm": seek.confirm if seek else None,
        "plan": {k: v for k, v in seek.plan.items() if k != "path"} if seek and seek.plan else None,
        "waiting": bool(seek and seek.plan and seek.status.startswith("waiting for your approval")),
        "approve_each": ctx["approve_each"],
        "turns": seek.turns[-30:] if seek else [],
        "instruction": getattr(d.b, "instruction", None),
        "progress": d.b.progress() if getattr(d.b, "progress", None) else None,
        "pose": d.pose(),
        "log": d.log[-15:],
    }


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.get("/video")
def video() -> StreamingResponse:
    def gen():
        while True:
            f = ctx["drone"].frame()
            if f is not None:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg(cv2.resize(f, (640, 360))) + b"\r\n"
            time.sleep(1 / 15)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/map.png")
def map_png() -> Response:
    seek = ctx["seek"]
    visits, plan = (seek.mem.visits, seek.plan) if seek else (None, None)
    return Response(ctx["map"].bird(visited=visits, plan=plan), media_type="image/jpeg")


@app.get("/points")
def points() -> JSONResponse:
    m = ctx["map"]
    return JSONResponse({"points": m.points(max_points=25000), "pose": m.pose, "state": m.state()})


@app.post("/cmd/{name}")
def command(name: str, arg: int = 0) -> dict:
    d = ctx["drone"]
    if name == "emergency":
        stop_agent()
    elif ctx["mode"] == "agent" and ctx["seek"] is not None and not ctx["seek"].status.startswith("ended"):
        return {"reply": "refused: the agent is flying - switch to Manual first"}
    calls = {
        "takeoff": d.takeoff,
        "land": d.land,
        "spin": d.spin,
        "forward": lambda: d.forward(arg),
        "back": lambda: d.back(arg),
        "left": lambda: d.left(arg),
        "right": lambda: d.right(arg),
        "turn_left": lambda: d.turn_left(arg),
        "turn_right": lambda: d.turn_right(arg),
        "emergency": d.emergency,
    }
    if name not in calls:
        return {"reply": f"unknown command {name}"}
    if name == "emergency":
        if hasattr(d.b, "cancel"):
            d.b.cancel()  # a person holding the camera: drop the current instruction
        return {"reply": d.emergency()}
    # Commands can take long (a webcam move waits for Done), so they run in the background and this request
    # returns at once: a request held open per button would use up the browser's 6 connections per site and
    # freeze the camera stream and the page. One command at a time.
    busy = ctx.get("busy")
    if busy is not None and busy.is_alive():
        return {"reply": f"busy: still doing '{ctx['busy_name']}' - finish it first"}
    ctx["busy"] = threading.Thread(target=calls[name], daemon=True)
    ctx["busy_name"] = name
    ctx["busy"].start()
    return {"reply": "started"}


PAGE = """<!doctype html><meta charset=utf-8><title>SEEK</title>
<style>
:root{--bg:#0b0f14;--panel:#111821;--line:#1e2a36;--text:#dbe4ee;--dim:#7f8fa0;--blue:#1f6feb;--red:#d1242f;
--green:#2da44e;--amber:#ffd479}
*{box-sizing:border-box}
body{margin:0;font:13px system-ui;background:var(--bg);color:var(--text);display:grid;grid-template-columns:1fr 1fr;
grid-template-rows:auto 42vh 1fr;height:100vh;gap:6px;padding:6px;overflow:hidden}
.p{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;position:relative;min-height:0}
.p h3{margin:0;padding:6px 10px;font-size:12px;color:var(--dim);font-weight:600}
button{background:var(--blue);color:#fff;border:0;border-radius:5px;padding:6px 10px;cursor:pointer;font:inherit}
button:disabled{opacity:.35;cursor:default}.stop{background:var(--red)}.go{background:var(--green)}
#bar{grid-column:span 2;display:flex;align-items:center;gap:8px;padding:8px 10px}
#bar b{font-size:15px;letter-spacing:1px;margin-right:6px}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:transparent;border-radius:0;color:var(--dim)}.seg button.on{background:var(--blue);color:#fff}
#goalin{flex:1;min-width:120px;background:#0b121a;color:var(--text);border:1px solid var(--line);border-radius:5px;
padding:7px 9px;font:inherit}#status{color:var(--amber);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:32%}
#side{display:flex;flex-direction:column}#camwrap{position:relative;flex:1 1 0;min-height:0;display:flex;
justify-content:center;background:#000}#cam{max-width:100%;max-height:100%;object-fit:contain;display:block}
#ins{position:absolute;inset:auto 8px 8px 8px;background:rgba(255,212,121,.95);color:#111;border-radius:6px;
padding:10px 12px;font-size:16px;font-weight:600;display:none;align-items:center;gap:10px}
#ins span{flex:1}#ins button{background:#111;font-size:14px}
#st{padding:4px 10px;font-size:11px;color:#9fb0c0}
#btns{display:flex;flex-wrap:wrap;gap:4px;padding:4px 6px;align-items:center}#flash{color:var(--amber);font-size:12px;margin-left:6px}
#log{margin:0;padding:4px 10px;font-size:11px;line-height:14px;color:#9fb0c0;flex:0 0 46px;overflow:auto;
white-space:pre}
#map{width:100%;height:calc(100% - 28px);object-fit:contain;display:block}
#three{position:absolute;inset:28px 0 0 0}#msg{position:absolute;top:40px;left:12px;color:#ff9b9b;font-size:12px}
#agent{display:flex;flex-direction:column}#turns{padding:0 10px;overflow:auto;flex:1}
.t{border-top:1px solid var(--line);padding:6px 0}.t b{color:#7cc4ff}.t .r{color:#9fb0c0}.t .s{color:var(--dim)}
#plan{display:none;margin:0 10px 8px;padding:10px;border:2px solid #3cdc3c;border-radius:6px;background:#0f2014}
#plan .pl{font-size:14px;margin-bottom:4px}#plan b{color:#7ef07e}#plan .s{color:var(--dim);margin-bottom:8px}
.prow{display:flex;gap:6px}.prow input{flex:1;background:#0b121a;color:var(--text);border:1px solid var(--line);
border-radius:5px;padding:6px 8px;font:inherit}
#autol{color:var(--dim);white-space:nowrap;display:flex;align-items:center;gap:4px;cursor:pointer}
.ghost{background:transparent;border:1px solid var(--line);color:var(--dim)}
#confirm{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);align-items:center;justify-content:center;z-index:9}
#confirm .box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;width:min(640px,92vw)}
#confirm img{width:100%;border-radius:4px;margin:8px 0}#confirm .row{display:flex;gap:8px;justify-content:flex-end}
</style>
<div class=p id=bar><b>SEEK</b>
 <div class=seg><button id=m-manual onclick="setMode('manual')">Manual</button><button id=m-agent onclick="setMode('agent')">Agent</button></div>
 <input id=goalin placeholder="What should it find? e.g. a red mug" onkeydown="if(event.key==='Enter')start()">
 <label id=autol title="Off: you approve every action before it runs"><input type=checkbox id=auto onchange="post('/auto/'+(this.checked?'on':'off'))"> automatic</label>
 <button class=go id=startb onclick="start()">Start</button><button class=stop id=stopb onclick="stopAgent()">Stop agent</button>
 <button class=ghost id=resetb onclick="post('/map/reset')" title="Forget the map: start over in a new place">New map</button>
 <span id=status></span></div>
<div class=p id=side><h3>CAMERA</h3>
 <div id=camwrap><img id=cam src="/video"><div id=ins><span id=instext></span><button onclick="post('/done')">Done</button></div></div>
 <div id=st></div>
 <div id=btns>
  <button class=man data-k=drone onclick="cmd('takeoff')">Take off</button><button class=man onclick="cmd('spin')">Spin 360</button>
  <button class=man onclick="cmd('forward',50)">Fwd 50</button><button class=man onclick="cmd('back',50)">Back 50</button>
  <button class=man onclick="cmd('left',50)">Left 50</button><button class=man onclick="cmd('right',50)">Right 50</button>
  <button class=man onclick="cmd('turn_left',45)">Turn L 45</button><button class=man onclick="cmd('turn_right',45)">Turn R 45</button>
  <button class=man data-k=drone onclick="cmd('land')">Land</button><button class=stop onclick="cmd('emergency')">STOP</button><span id=flash></span>
 </div><pre id=log></pre></div>
<div class=p><h3>MAP AS GROK SEES IT (light free, red obstacle, dark unknown, blue visited, orange path)</h3><img id=map src="/map.png"></div>
<div class=p><h3>3D POINT CLOUD, BUILT LIVE (drag to rotate, scroll to zoom)</h3><div id=three></div><div id=msg>loading 3D view...</div></div>
<div class=p id=agent><h3>GROK DECISIONS</h3>
 <div id=plan><div class=pl>Grok wants to: <b id=ptext></b></div><div class=s id=pwhy></div>
  <div class=prow><input id=pnote placeholder="Why not? (optional, Grok reads it)" onkeydown="if(event.key==='Enter')decide(false)">
  <button class=stop onclick="decide(false)">Reject</button><button class=go onclick="decide(true)">Approve</button></div></div>
 <div id=turns></div></div>
<div id=confirm><div class=box><b>Grok thinks it found it. Is this right?</b><div id=cev></div><img id=cimg>
 <div class=row><button class=stop onclick="answer('no')">No, keep searching</button><button class=go onclick="answer('yes')">Yes, that's it</button></div></div></div>
<script>addEventListener('error',e=>{const m=document.getElementById('msg');if(m)m.textContent='3D view error: '+e.message},true)</script>
<script type=importmap>{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const el=document.getElementById('three');const r=new THREE.WebGLRenderer({antialias:true});el.appendChild(r.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0f14);
const cam=new THREE.PerspectiveCamera(60,1,0.05,100);cam.position.set(0,6,6);cam.up.set(0,1,0);
const ctl=new OrbitControls(cam,r.domElement);scene.add(new THREE.GridHelper(8.5,17,0x334455,0x1c2732));
const drone=new THREE.Mesh(new THREE.ConeGeometry(0.15,0.4,12),new THREE.MeshBasicMaterial({color:0x3399ff}));
drone.rotation.x=-Math.PI/2;scene.add(drone);let cloud=null;const mat=new THREE.PointsMaterial({size:0.06,vertexColors:true});
function size(){const w=el.clientWidth,h=el.clientHeight;r.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix()}
addEventListener('resize',size);size();
async function load(){const j=await (await fetch('/points',{signal:AbortSignal.timeout(5000)})).json();const n=j.points.length;
 const pos=new Float32Array(n*3),col=new Float32Array(n*3);
 j.points.forEach((p,i)=>{pos.set([p[0],p[2],-p[1]],i*3);col.set([p[3],p[4],p[5]],i*3)});
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));
 g.setAttribute('color',new THREE.BufferAttribute(col,3));
 if(cloud){scene.remove(cloud);cloud.geometry.dispose()}
 cloud=new THREE.Points(g,mat);scene.add(cloud);
 drone.position.set(j.pose.x,j.pose.z,-j.pose.y);drone.rotation.z=-j.pose.yaw*Math.PI/180;
 document.getElementById('st').textContent=`cell ${j.state.drone_cell}  heading ${Math.round(j.pose.yaw)%360}deg  `+
  `frames mapped ${j.state.frames_used}  free ${j.state.cells.free}  obstacle ${j.state.cells.obstacle}`}
async function loop3d(){try{await load();document.getElementById('msg').textContent=''}catch(e){}setTimeout(loop3d,1000)}
loop3d();
const mapimg=document.getElementById('map');let mapT=0;  // next map only after the last one arrived (or 5 s)
function nextMap(){clearTimeout(mapT);mapT=setTimeout(()=>{mapimg.src='/map.png?'+Date.now();mapT=setTimeout(nextMap,5000)},700)}
mapimg.onload=mapimg.onerror=nextMap;nextMap();
(function loop(){requestAnimationFrame(loop);ctl.update();r.render(scene,cam)})();
</script>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function post(url,body){const r=await fetch(url,{method:'POST',signal:AbortSignal.timeout(5000),headers:{'Content-Type':'application/json'},
 body:body?JSON.stringify(body):undefined});refresh();return r.json()}
let flashT=0;
async function cmd(name,arg){let j;try{j=await post('/cmd/'+name+(arg?'?arg='+arg:''))}catch(e){j={reply:'no answer from the app'}}
 if(j.reply!=='started'){$('flash').textContent=j.reply;clearTimeout(flashT);flashT=setTimeout(()=>$('flash').textContent='',4000)}}
function setMode(m){post('/mode/'+m)}
function start(){const g=$('goalin').value.trim();if(!g){$('goalin').focus();return}post('/agent/start',{goal:g})}
function stopAgent(){post('/agent/stop')}
function answer(a){post('/confirm/'+a)}
function decide(ok){const n=$('pnote').value;$('pnote').value='';post('/approve',{ok:ok,note:n})}
let shownConfirm=null;
async function refresh(){let j;try{j=await (await fetch('/state',{signal:AbortSignal.timeout(5000)})).json()}catch(e){return}
 $('m-manual').classList.toggle('on',j.mode==='manual');$('m-agent').classList.toggle('on',j.mode==='agent');
 const agent=j.mode==='agent';$('goalin').style.display=$('startb').style.display=$('autol').style.display=agent?'':'none';
 $('auto').checked=!j.approve_each;$('resetb').disabled=j.running;
 $('plan').style.display=j.waiting?'block':'none';
 if(j.plan){$('ptext').textContent=j.plan.text;$('pwhy').textContent='Why: '+j.plan.thought}
 $('stopb').style.display=agent&&j.running?'':'none';$('startb').disabled=j.running;
 if(j.running&&document.activeElement!==$('goalin'))$('goalin').value=j.goal;
 $('status').textContent=j.goal?`${j.running?'searching':'last mission'}: ${j.goal} - ${j.status}`:'';
 document.querySelectorAll('.man').forEach(b=>{b.disabled=agent&&j.running;
  b.style.display=b.dataset.k==='drone'&&j.kind==='webcam'?'none':''});
 $('ins').style.display=j.instruction?'flex':'none';
 if(j.instruction)$('instext').textContent=j.instruction+(j.progress?`  -  ${j.progress}`:'');
 $('log').textContent=j.log.slice().reverse().map(e=>`${e.cmd.padEnd(14)} ${String(e.reply).padEnd(8)} -> x ${e.after.x} y ${e.after.y} yaw ${e.after.yaw}`).join('\\n');
 $('turns').innerHTML=j.turns.length?j.turns.slice().reverse().map(t=>{const a=t.reply;
  const what=a.action+(a.action==='goto'||a.action==='found'?' '+a.cell:a.action==='look'?' '+a.heading+'deg':'');
  return `<div class=t><b>#${t.turn} ${esc(t.cell)} -> ${esc(what)}</b> <span class=s>${t.seconds}s</span><br>`+
   `${esc(a.thought)}<br><span class=s>saw: ${esc(a.seen)}</span><br><span class=r>result: ${esc(t.result)}</span></div>`}).join('')
  :`<div class=s style="padding:8px 0">${j.running?'Looking around first: follow the instruction on the camera.':agent?'Type what to find and press Start.':'Manual mode: use the buttons. Switch to Agent to let Grok search.'}</div>`;
 const c=j.confirm;$('confirm').style.display=c?'flex':'none';
 if(c&&shownConfirm!==c.photo){shownConfirm=c.photo;$('cimg').src='/found/'+c.photo;
  $('cev').textContent=`In ${c.cell}: ${c.evidence}`}}
(async function poll(){await refresh();setTimeout(poll,700)})();
// camera watchdog: if the picture has not changed for 4 s, reconnect the stream
(function(){const img=$('cam'),c=document.createElement('canvas');c.width=32;c.height=18;const x=c.getContext('2d');
 let last=0,same=0;setInterval(()=>{let h=0;try{x.drawImage(img,0,0,32,18);const d=x.getImageData(0,0,32,18).data;
  for(let i=0;i<d.length;i+=8)h=(h*31+d[i])|0}catch(e){}
  same=h===last?same+1:0;last=h;if(same>=4){same=0;img.src='/video?'+Date.now()}},1000)})();
</script>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", type=int, help="camera index (0 = built-in)")
    src.add_argument("--sim", help="360 photo for the simulated drone")
    src.add_argument("--tello", action="store_true")
    p.add_argument("--goal", help="start in agent mode, searching for this")
    p.add_argument("--fov", type=float, help="diagonal field of view in degrees (default: 78 webcam, 82.6 drone)")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()

    out = Path("runs") / f"seek-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    tracing.setup(out)
    print("loading depth model...")
    depth(np.zeros((360, 640, 3), np.uint8))  # load once at startup, not during the mission

    if a.webcam is not None:
        backend, fov, kind = WebcamBackend(a.webcam), a.fov or WebcamBackend.FOV_DIAG, "webcam"
    else:
        backend, fov, kind = (SimBackend(a.sim) if a.sim else TelloBackend()), a.fov or 82.6, "drone"
    drone = Drone(backend, log_dir=out)
    m = LiveMap(fov_diag=fov)
    drone.attach(m)
    if kind == "webcam":
        drone.takeoff()  # a hand-held camera is always "flying"
    ctx.update(kind=kind, drone=drone, map=m, out=out)
    if a.goal:
        start_agent(a.goal)
    print(f"SEEK: http://localhost:{a.port}   files: {out}   traces: http://localhost:6006 (project seek)")
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        stop_agent()
        m.save_ply(str(out / "points.ply"))
        tracing.flush()


if __name__ == "__main__":
    main()
