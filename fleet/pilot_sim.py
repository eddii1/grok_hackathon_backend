"""In-memory SDK/telemetry plant for the exact dashboard pilot adapter. No sockets.

It can deliberately ignore land while takeoff is busy, reproducing the incident
where the old 3-second landing burst ended before an 8-second takeoff finished.
"""
import math
import time

from .pilot_adapter import PilotAdapter, PilotTransport
from .scan_simulator import ScanSimAdapter
from .tello_transport import matches_reply


class PilotSimTransport(PilotTransport):
    def __init__(self, *, takeoff_seconds=1.0):
        import threading
        super().__init__()
        self.room=ScanSimAdapter('pilot-sim')
        self.takeoff_seconds=takeoff_seconds
        self._plant_lock=threading.RLock()
        self._replies=[]
        self._rc=(0,0,0,0)
        self._rise_at=None
        self._rise_done=0
        self._descending=False
        self._yaw_target=None

    async def start(self,on_state,on_frame):
        self.on_state,self.on_frame=on_state,on_frame
        self._thread(self._plant)
        await self.request('command',3)
        await self.request('streamon',3)

    def send(self,command):
        self.validate_command(command)
        if self._quit.is_set(): raise OSError('Simulation link closed')
        now=time.monotonic()
        self.wire_events.append({'monotonic':now,'direction':'send','command':command})
        del self.wire_events[:-2000]
        with self._plant_lock:
            reply='ok'
            delay=.01
            if command=='takeoff':
                self._descending=False;self._rc=(0,0,0,0);self._yaw_target=None
                self._rise_at=now;self._rise_done=now+self.takeoff_seconds
                delay=self.takeoff_seconds
            elif command=='land':
                self._rc=(0,0,0,0);self._yaw_target=None
                if self._rise_at is not None and now<self._rise_done:
                    reply='error busy'
                else: self._descending=True
            elif command=='stop': self._rc=(0,0,0,0);self._yaw_target=None
            elif command=='emergency':
                self._rise_at=None;self._descending=False;self._rc=(0,0,0,0)
                self.room.height=0;self._yaw_target=None
            elif command=='cw 30':
                self._yaw_target=(self.room.yaw+30)%360;delay=.9
            elif command.startswith('rc '): self._rc=tuple(map(int,command.split()[1:]));return
            self._replies.append((now+delay,reply))

    def _reply(self,response):
        with self._pending_lock: pending=self._pending
        matched=bool(pending and matches_reply(pending[2],response))
        self.wire_events.append({'monotonic':time.monotonic(),'direction':'receive','response':response,'matched':matched})
        if matched:
            loop,future,_=pending
            def deliver():
                if not future.done(): future.set_result(response)
            try: loop.call_soon_threadsafe(deliver)
            except RuntimeError: pass

    def _plant(self):
        previous=time.monotonic();last_frame=0
        while not self._quit.wait(.04):
            now=time.monotonic();dt=min(now-previous,.1);previous=now
            with self._plant_lock:
                old_h=self.room.height
                if self._rise_at is not None:
                    self.room.height=min(80,80*(now-self._rise_at)/self.takeoff_seconds)
                    if now>=self._rise_done: self._rise_at=None
                if self._descending: self.room.height=max(0,self.room.height-100*dt)
                if self.room.height>=20 and not self._descending:
                    lr,fb,ud,yaw=self._rc
                    a=math.radians(self.room.yaw)
                    x=self.room.position[0]+(fb*math.sin(a)+lr*math.cos(a))*dt
                    y=self.room.position[1]+(fb*math.cos(a)-lr*math.sin(a))*dt
                    if self.room.clear(x/100,y/100): self.room.position=[x,y]
                    self.room.height=max(20,min(200,self.room.height+ud*dt))
                    self.room.yaw=(self.room.yaw+yaw*dt)%360
                    if self._yaw_target is not None:
                        delta=(self._yaw_target-self.room.yaw+540)%360-180
                        self.room.yaw=(self.room.yaw+max(-40*dt,min(40*dt,delta)))%360
                        if abs(delta)<1:self._yaw_target=None
                state={'bat':90,'h':round(self.room.height),'yaw':round(self.room.yaw),
                       'vgz':round((self.room.height-old_h)/max(dt,.001))}
                due=[r for r in self._replies if r[0]<=now]
                self._replies=[r for r in self._replies if r[0]>now]
            # Never call into the adapter while holding the plant lock.
            self.on_state(state)
            for _,response in due:self._reply(response)
            if now-last_frame>.2:
                frame=self.room.observation()[0]
                self.on_frame(frame);last_frame=now

    def close(self):
        self._quit.set();self.room.close()


class PilotSimAdapter(PilotAdapter):
    simulated=True
    def __init__(self,device_id,*,transport=None,**kwargs):
        super().__init__(device_id,transport=transport or PilotSimTransport(),**kwargs)

    def state(self):
        return {**super().state(),'pose':self.transport.room.pose(),'pose_source':'simulator'}
