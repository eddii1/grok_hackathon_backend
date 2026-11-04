"""Live feed + chat with a vision LLM. One file, no build step.

    uv run fleet-live --source 0                 # laptop webcam
    uv run fleet-live --source clip.mp4          # recorded drone video (loops)
    uv run fleet-live --tello                    # live RoboMaster TT / Tello stream
    open http://localhost:8000

Every chat message sends the newest frame (or an attached image) to the model.
"Watch" re-asks a standing question every N seconds, one request at a time.
"""

import argparse
import base64
import socket
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from fleet.llm import ask, config

SYSTEM = (
    "You are the vision system of a search drone. You see the newest camera frame. "
    "Answer briefly and only from what is visible; say when you are unsure."
)

latest: dict = {"jpeg": None}
history: list[tuple[str, str]] = []
app = FastAPI()
cfg: dict = {}


def capture(source: str) -> None:
    """Keep only the newest frame; old frames are dropped, never queued."""
    is_file = not source.isdigit() and "://" not in source
    while True:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            latest["jpeg"] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            if delay:
                time.sleep(delay)
        cap.release()
        time.sleep(0.5)  # file ended (loop it) or stream dropped (reconnect)


def tello_stream() -> str:
    """Put a Tello in SDK mode, start video, keep it awake. Returns the stream URL."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 9000))
    s.settimeout(5)
    for cmd in (b"command", b"streamon"):
        s.sendto(cmd, ("192.168.10.1", 8889))
        print(cmd.decode(), "->", s.recvfrom(1024)[0].decode())

    def keepalive() -> None:
        while True:
            time.sleep(5)
            s.sendto(b"battery?", ("192.168.10.1", 8889))

    threading.Thread(target=keepalive, daemon=True).start()
    return "udp://@0.0.0.0:11111"


class Ask(BaseModel):
    text: str
    image: str | None = None  # optional attached image as a data URL


@app.post("/ask")
def post_ask(q: Ask) -> dict:
    jpeg = base64.b64decode(q.image.split(",", 1)[1]) if q.image else latest["jpeg"]
    if jpeg is None:
        return {"answer": "No frame yet.", "seconds": 0, "frame": None}
    past = "\n".join(f"User: {u}\nYou: {a}" for u, a in history[-6:])
    prompt = f"{SYSTEM}\n\n{past}\nUser: {q.text}".strip()
    try:
        answer, secs = ask(cfg, prompt, jpeg)
    except Exception as e:  # show API errors in the chat instead of a 500
        return {"answer": f"Error: {e}", "seconds": 0, "frame": None}
    history.append((q.text, answer))
    return {"answer": answer, "seconds": round(secs, 1), "frame": base64.b64encode(jpeg).decode()}


@app.get("/video")
def video() -> StreamingResponse:
    def gen():
        while True:
            if latest["jpeg"]:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + latest["jpeg"] + b"\r\n"
            time.sleep(1 / 15)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE.replace("{{MODEL}}", f"{cfg['provider']} / {cfg['model']}"))


PAGE = """<!doctype html><meta charset=utf-8><title>SEEK live</title>
<style>
body{margin:0;font:14px system-ui;background:#0b0f14;color:#dbe4ee;display:grid;grid-template-columns:1fr 420px;height:100vh}
#feed{background:#000;display:flex;align-items:center;justify-content:center}#feed img{max-width:100%;max-height:100vh}
#side{display:flex;flex-direction:column;border-left:1px solid #1e2a36}
#log{flex:1;overflow:auto;padding:12px}.m{margin:8px 0;padding:8px 10px;border-radius:8px;white-space:pre-wrap}
.u{background:#16324d}.a{background:#151c24}.a img{width:120px;display:block;margin-top:6px;border-radius:4px}
.t{color:#7f8fa0;font-size:12px}form,#watch{display:flex;gap:6px;padding:8px;border-top:1px solid #1e2a36}
input[type=text]{flex:1;background:#0f151c;color:inherit;border:1px solid #2a3a4a;border-radius:6px;padding:8px}
button{background:#1f6feb;color:#fff;border:0;border-radius:6px;padding:8px 12px;cursor:pointer}
</style>
<div id=feed><img src="/video"></div>
<div id=side>
 <div class=t style="padding:8px 12px">Model: {{MODEL}}</div>
 <div id=log></div>
 <div id=watch><input type=text id=wq placeholder="Watch: e.g. Is there a person in a red jacket?">
  <input id=ws type=number value=2 min=1 style="width:48px"><button id=wb type=button>Watch</button></div>
 <form id=f><input type=file id=img accept="image/*" style="width:90px">
  <input type=text id=q placeholder="Ask about the live frame..." autofocus><button>Send</button></form>
</div>
<script>
const log=document.getElementById('log');let busy=false,timer=null;
function add(cls,text,frame,secs){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;
 if(secs!==undefined){const t=document.createElement('div');t.className='t';t.textContent=secs+'s';d.appendChild(t)}
 if(frame){const i=document.createElement('img');i.src='data:image/jpeg;base64,'+frame;d.appendChild(i)}
 log.appendChild(d);log.scrollTop=log.scrollHeight}
async function send(text,image){if(busy)return;busy=true;add('u',text);
 try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,image})});
  const j=await r.json();add('a',j.answer,j.frame,j.seconds)}catch(e){add('a','Error: '+e)}busy=false}
const file=f=>new Promise(r=>{const x=new FileReader();x.onload=()=>r(x.result);x.readAsDataURL(f)});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q'),im=document.getElementById('img');
 if(!q.value)return;const image=im.files[0]?await file(im.files[0]):null;send(q.value,image);q.value='';im.value=''};
document.getElementById('wb').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='Watch';return}
 const q=document.getElementById('wq').value;if(!q)return;e.target.textContent='Stop';
 timer=setInterval(()=>send(q),document.getElementById('ws').value*1000)};
</script>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index, video file or stream URL")
    p.add_argument("--tello", action="store_true", help="connect to a Tello/TT on its own Wi-Fi")
    p.add_argument("--provider", help="xai, fireworks, ... (default: LLM_PROVIDER)")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    cfg.update(config(a.provider))
    source = tello_stream() if a.tello else a.source
    threading.Thread(target=capture, args=(source,), daemon=True).start()
    print(f"{cfg['provider']} / {cfg['model']} -> http://localhost:{a.port}")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
