"""A labelled synthetic room. Camera and depth share the same ray intersections.

The agent never receives the hidden room geometry. Depth is simulated, not a
measurement or a validation of the optional monocular model on real footage.
"""
import math
import time

import cv2
import numpy as np

from .simulator import SimAdapter


class ScanSimAdapter(SimAdapter):
    simulated = True
    fov_diag = 82.6
    boxes = [((-2, -.9, 0), (-1.2, .1, 1.6)), ((.9, .8, 0), (1.7, 1.6, 1.2))]

    def pose(self):
        return {'x': self.position[0]/100, 'y': self.position[1]/100,
                'z': self.height/100, 'yaw': self.yaw}

    def state(self):
        return {**super().state(), 'simulated': True, 'scan_capable': True,
                'pose': self.pose(), 'pose_source': 'simulator', 'depth_source': 'synthetic-raycast'}

    async def takeoff(self):
        await super().takeoff()
        self.height = 100

    def clear(self, x, y, margin=.3):
        return abs(x) < 3-margin and abs(y) < 3-margin and not any(
            low[0]-margin <= x <= high[0]+margin and low[1]-margin <= y <= high[1]+margin
            for low, high in self.boxes)

    async def move(self, forward_cm, left_cm, up_cm):
        self._require_flight()
        if up_cm or abs(forward_cm)+abs(left_cm) > 50:
            raise ValueError('Scan simulator accepts horizontal steps up to 50 cm')
        a = math.radians(self.yaw)
        dx = forward_cm*math.sin(a)-left_cm*math.cos(a)
        dy = forward_cm*math.cos(a)+left_cm*math.sin(a)
        x, y = self.position
        if not all(self.clear((x+dx*t)/100, (y+dy*t)/100) for t in np.linspace(0, 1, 21)):
            raise ValueError('Simulator collision envelope blocked movement')
        self.position = [x+dx, y+dy]
        self._log('move', forward_cm=forward_cm, left_cm=left_cm, up_cm=up_cm)

    def observation(self):
        h, w = 108, 192
        focal = np.hypot(w/2, h/2)/np.tan(np.radians(self.fov_diag/2))
        v, u = np.mgrid[:h, :w]
        right, down = (u-w/2)/focal, (v-h/2)/focal
        a = math.radians(self.yaw)
        rays = np.stack([right*math.cos(a)+math.sin(a),
                         -right*math.sin(a)+math.cos(a), -down], axis=-1)
        pose = self.pose()
        origin = np.array([pose['x'], pose['y'], max(.15, pose['z'])])
        distance = np.full((h, w), 20.)
        colours = np.zeros((h, w, 3), np.uint8)
        for axis, value, colour in [(0,-3,(80,130,170)),(0,3,(150,125,85)),
                                    (1,-3,(110,150,125)),(1,3,(155,130,105)),
                                    (2,0,(90,95,100)),(2,2.5,(180,180,175))]:
            component = rays[..., axis]
            t = np.divide(value-origin[axis], component, out=np.full((h,w), np.inf), where=abs(component)>1e-8)
            hit = (t > 0) & (t < distance)
            distance[hit], colours[hit] = t[hit], colour
        for low, high in self.boxes:
            safe_rays = np.where(abs(rays)<1e-8, 1e-8, rays)
            t1, t2 = (np.array(low)-origin)/safe_rays, (np.array(high)-origin)/safe_rays
            enter, leave = np.minimum(t1,t2).max(-1), np.maximum(t1,t2).min(-1)
            hit = (enter > 0) & (leave >= enter) & (enter < distance)
            distance[hit], colours[hit] = enter[hit], (85,155,195)
        points = origin + rays*distance[...,None]
        shade = (.75+.25*((np.floor(points[...,0]*4)+np.floor(points[...,1]*4)+np.floor(points[...,2]*4))%2))
        frame = (colours*shade[...,None]).astype(np.uint8)
        cv2.putText(frame, 'SIMULATED ROOM', (5,13), cv2.FONT_HERSHEY_SIMPLEX,.32,(235,245,240),1)
        return frame, distance.astype(np.float32), pose

    def latest_frame(self):
        if not self.connected: return None
        return self.observation()[0], time.time()
