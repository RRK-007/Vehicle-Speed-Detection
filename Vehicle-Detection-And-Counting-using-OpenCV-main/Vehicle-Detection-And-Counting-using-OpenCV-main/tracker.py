import math
import numpy as np

class EuclideanDistTracker:
    def __init__(self, max_frames_missing=5):
        self.center_points = {}
        self.id_count = 0
        self.max_frames_missing = max_frames_missing

    def update(self, obj_rect):
        obj_bbx_ids = []

        # Update existing object positions
        for obj_id, (prev_center, missing_frames) in list(self.center_points.items()):
            if missing_frames >= self.max_frames_missing:
                # Remove object if it's been missing for too many frames
                del self.center_points[obj_id]
                continue
            
            # Try to find a matching detection for this object
            match_found = False
            for rect in obj_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist < 25:
                    # Update object position and reset missing frames counter
                    self.center_points[obj_id] = ((cx, cy), 0)
                    obj_bbx_ids.append([x, y, w, h, obj_id])
                    match_found = True
                    break
            
            # If no match found, increment missing frames counter
            if not match_found:
                self.center_points[obj_id] = (prev_center, missing_frames + 1)

        # Assign IDs to new objects
        for rect in obj_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            matched = False
            for obj_id, (prev_center, _) in self.center_points.items():
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist < 25:
                    matched = True
                    break
            if not matched:
                self.center_points[self.id_count] = ((cx, cy), 0)
                obj_bbx_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        return obj_bbx_ids
