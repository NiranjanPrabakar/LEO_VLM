#!/usr/bin/env python3
"""
real_time_video.py  —  Live Bed-Making Robot Pipeline  [v3: Perception-Guided Visual Prompting]
================================================================================================
Changes from v2 (marked with # v3):
  1. burn_accessibility_onto_image() — burns HEAD/FOOT/LEFT/RIGHT labels onto image edges
  2. _vlm_auto_loop now sends _latest_rgb_yolo (YOLO annotated) instead of _latest_rgb (raw)
  3. Annotated image = YOLO boxes + class labels + accessibility zone labels + depth values
     LLaVA-7B now sees exactly what the perception pipeline knows, not raw pixels.
  Everything else UNCHANGED.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
import queue
import json
import sys
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from ultralytics import YOLO

from accessibility_analyzer import AccessibilityAnalyzer
from symbolic_state_encoder import SymbolicStateEncoder
from enhanced_state_machine import TaskState, TaskStep, ENHANCED_STEP_REGISTRY, Phase
from depth_mapper import (
    InteractivePanoramicMapper, colorize_depth,
    draw_yolo_boxes, aggregate_objects,
    filter_and_clean_detections,
)
from vlm_planner import StepPlanner, CPU_MODELS, PromptBuilder
from sam_integration import SAMRefiner, draw_sam_masks


# ─────────────────────────────────────────────────────────
# Terminal pretty-print helpers
# ─────────────────────────────────────────────────────────

W = 68

def box_top(title=""):
    if title:
        return f"┌─ {title} {'─'*max(0,W-4-len(title))}┐"
    return "┌" + "─"*W + "┐"

def box_bot():
    return "└" + "─"*W + "┘"

def box_line(text=""):
    text = str(text)
    if len(text) > W-2: text = text[:W-5]+"…"
    return f"│ {text:<{W-2}} │"

def print_step_header(task_state):
    defn  = task_state.get_current_definition()
    done  = len(task_state.completed_steps)
    total = len(ENHANCED_STEP_REGISTRY)
    best  = task_state.get_best_side_for_current_step()
    print("\n" + box_top(f"STEP {done+1}/{total}  [{defn.phase.value.upper()}]"))
    print(box_line(f"Step     : {task_state.current_step.value}"))
    print(box_line(f"Action   : {defn.action_type}"))
    print(box_line(f"Approach : {best or '?'}  (preferred: {defn.preferred_sides})"))
    print(box_line(f"Object   : {defn.object_location or 'n/a'}"))
    print(box_line())
    goal = defn.description; words = goal.split(); line = "Goal     : "
    for w in words:
        if len(line)+len(w)+1 > W-2: print(box_line(line)); line = "           "+w+" "
        else: line += w+" "
    if line.strip(): print(box_line(line))
    print(box_line())
    print(box_line(f"VLM hint : {defn.vlm_context_hint[:W-14]}"))
    print(box_bot())

def print_prompt(prompt, model_id):
    print(f"\n{'─'*20} PROMPT → {model_id} {'─'*20}")
    for line in prompt.strip().split('\n'): print(f"  {line}")
    print("─"*(W+2))

def print_vlm_response(raw, elapsed):
    print(f"\n{'─'*20} RAW RESPONSE ({elapsed:.1f}s) {'─'*20}")
    for line in raw.strip().split('\n')[:20]: print(f"  {line}")
    if len(raw.strip().split('\n')) > 20: print("  …(truncated)")
    print("─"*(W+2))

def print_parsed(output):
    print(f"\n  PARSED OUTPUT:")
    for k,v in output.items(): print(f"    {k}: {v}")

def print_validation(validation):
    ok=validation.get('valid',False); score=validation.get('score',0)
    viols=validation.get('violations',[])
    print(f"\n  VALIDATION: {'✓ OK' if ok else '✗ FAIL'}  score={score}/100")
    if viols:
        for v in viols: print(f"    ⚠  {v}")
    else:
        print("    No violations — safe to execute")

def print_accessibility(accessibility, stats):
    print("\n" + box_top("ACCESSIBILITY MAP"))
    icons={'free':'✓ FREE','partially_blocked':'⚠ PARTIAL','blocked':'✗ BLOCKED','unknown':'? UNKNOWN'}
    for side in ['head','foot','left','right']:
        s=accessibility.get(side,'unknown'); st=stats.get(side,{})
        d=st.get('median_depth_m',0.0); c=st.get('confidence',0.0); nv=st.get('num_views',0)
        print(box_line(f"  {side.upper():>5s}:  {icons[s]:20s}  ({d:.2f}m  conf={c:.2f}  views={nv})"))
    print(box_bot())


# ─────────────────────────────────────────────────────────
# v3: Perception-Guided Visual Annotation
# Burns YOLO + accessibility knowledge onto image for LLaVA
# ─────────────────────────────────────────────────────────

SIDE_COLOR_BGR = {
    'free':             (0, 220,   0),
    'partially_blocked':(0, 180, 255),
    'blocked':          (0,   0, 220),
    'unknown':          (160, 160, 160),
}

def burn_accessibility_onto_image(img, accessibility, stats=None):
    """
    v3: Burns HEAD/FOOT/LEFT/RIGHT accessibility labels onto image edges.

    LLaVA-7B receives this annotated image instead of raw RGB, so it
    directly reads what the depth-fusion pipeline already computed:
      - Which sides are FREE / BLOCKED / PARTIAL
      - Depth distances per side
      - YOLO bounding boxes + class labels (already on rgb_yolo input)

    This is 'perception-guided visual prompting' — no VLM fine-tuning needed.
    The perception pipeline's knowledge is injected visually at inference time.
    """
    out   = img.copy()
    h, w  = out.shape[:2]
    stats = stats or {}

    # (label_xy, depth_xy) for each side — positioned at image edges
    positions = {
        'head':  ((w//2 - 70,  22),    (w//2 - 35,  40)),
        'foot':  ((w//2 - 70,  h-30),  (w//2 - 35,  h-12)),
        'left':  ((4,           h//2 - 8),  (4,      h//2 + 14)),
        'right': ((w - 140,    h//2 - 8),  (w-110,   h//2 + 14)),
    }

    for side, (lpos, dpos) in positions.items():
        status = accessibility.get(side, 'unknown')
        color  = SIDE_COLOR_BGR[status]
        label  = f"{side.upper()}: {status.replace('_', ' ').upper()}"
        depth  = stats.get(side, {}).get('median_depth_m', 0.0)

        # Dark pill background so label is readable over any scene colour
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        px, py = lpos
        cv2.rectangle(out, (px-3, py-15), (px+tw+4, py+5), (0, 0, 0), -1)
        cv2.putText(out, label, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

        # Depth value on the line below
        if depth > 0:
            cv2.putText(out, f"{depth:.2f}m", dpos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────

SIDE_COLOR = {
    'free':(0,220,0),'partially_blocked':(0,180,255),
    'blocked':(0,0,220),'unknown':(160,160,160),
}
SIDE_LABEL = {'free':'FREE','partially_blocked':'PARTIAL','blocked':'BLOCKED','unknown':'?'}

def draw_accessibility_bar(width, accessibility, stats, best_side=None):
    H=88; panel=np.full((H,width,3),30,dtype=np.uint8); cw=width//4
    for i,side in enumerate(['head','foot','left','right']):
        status=accessibility.get(side,'unknown'); color=SIDE_COLOR[status]
        label=SIDE_LABEL[status]; s=stats.get(side,{})
        d_m=s.get('median_depth_m',s.get('median_depth',0)/1000)
        conf=s.get('confidence',0.0); is_best=(side==best_side); x=i*cw
        bg=panel.copy()
        cv2.rectangle(bg,(x+2,2),(x+cw-2,H-2),color,-1)
        cv2.addWeighted(bg,0.22,panel,0.78,0,panel)
        cv2.rectangle(panel,(x+2,2),(x+cw-2,H-2),color,3 if is_best else 1)
        if is_best: cv2.putText(panel,"BEST",(x+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,255,255),1,cv2.LINE_AA)
        cx=x+cw//2
        cv2.putText(panel,side.upper(),(cx-22,26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(panel,label,(cx-len(label)*5,52),cv2.FONT_HERSHEY_SIMPLEX,0.50,color,2,cv2.LINE_AA)
        cv2.putText(panel,f"{d_m:.2f}m",(cx-20,70),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1,cv2.LINE_AA)
        if conf>0: cv2.putText(panel,f"c={conf:.2f}",(cx-18,83),cv2.FONT_HERSHEY_SIMPLEX,0.30,(140,140,140),1,cv2.LINE_AA)
    return panel

def draw_step_panel(width,task_state,vlm_output,inference_time,validation,fps,paused,scan_running,auto_running):
    H=118; panel=np.full((H,width,3),18,dtype=np.uint8)
    defn=task_state.get_current_definition(); total=len(ENHANCED_STEP_REGISTRY)
    done=len(task_state.completed_steps); best=task_state.get_best_side_for_current_step()
    bar_w=max(1,int(width*done/total)); cv2.rectangle(panel,(0,0),(bar_w,3),(0,200,100),-1)
    cv2.putText(panel,f"STEP {done+1}/{total}  [{defn.phase.value.upper()}]  {task_state.current_step.value}",
        (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,220,255),1,cv2.LINE_AA)
    goal=defn.description[:90]+("…" if len(defn.description)>90 else "")
    cv2.putText(panel,goal,(8,42),cv2.FONT_HERSHEY_SIMPLEX,0.37,(185,185,185),1,cv2.LINE_AA)
    cv2.putText(panel,f"Approach: {best or '?'}  |  Action: {defn.action_type}",
        (8,62),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,180),1,cv2.LINE_AA)
    if vlm_output:
        v_ok=validation.get('valid',False); v_col=(0,200,0) if v_ok else (0,60,220)
        cv2.putText(panel,f"VLM: {'OK' if v_ok else 'FAIL'}  score={validation.get('score',0)}  {inference_time:.1f}s",
            (8,82),cv2.FONT_HERSHEY_SIMPLEX,0.40,v_col,1,cv2.LINE_AA)
    parts=[]
    if paused: parts.append("PAUSED")
    if scan_running: parts.append("RESCANNING…")
    if auto_running and not paused: parts.append("AUTO-VLM RUNNING")
    parts+=[f"FPS:{fps:.0f}",f"{done}/{total} steps","[SPACE=pause  p=rescan  r=reset  q=quit]"]
    cv2.putText(panel,"  |  ".join(parts),(8,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1,cv2.LINE_AA)
    return panel

def build_dashboard(rgb_yolo,depth_color,task_state,accessibility,stats,vlm_output,
                    inference_time,validation,fps,paused,scan_running,auto_running,
                    sam_panel=None):
    h,w=rgb_yolo.shape[:2]
    best=task_state.get_best_side_for_current_step()
    if sam_panel is not None:
        pw = w * 3
        labels=np.full((22,pw,3),38,dtype=np.uint8)
        cv2.putText(labels,"YOLO boxes",(w//2-40,16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
        cv2.putText(labels,"SAM masks",(w+w//2-35,16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,180),1)
        cv2.putText(labels,"Depth (red=near)",(w*2+w//2-60,16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
        row_cams=np.hstack([rgb_yolo, sam_panel, depth_color])
    else:
        pw = w * 2
        labels=np.full((22,pw,3),38,dtype=np.uint8)
        cv2.putText(labels,"RGB + YOLO + SAM",(w//2-65,16),cv2.FONT_HERSHEY_SIMPLEX,0.46,(200,200,200),1)
        cv2.putText(labels,"Depth  (red=near  blue=far)",(w+w//2-100,16),cv2.FONT_HERSHEY_SIMPLEX,0.46,(200,200,200),1)
        row_cams=np.hstack([rgb_yolo,depth_color])
    row_acc=draw_accessibility_bar(pw,accessibility,stats,best)
    row_step=draw_step_panel(pw,task_state,vlm_output,inference_time,validation,fps,paused,scan_running,auto_running)
    return np.vstack([labels,row_cams,row_acc,row_step])


# ─────────────────────────────────────────────────────────
# Main streaming class
# ─────────────────────────────────────────────────────────

class RealTimeVideoStream:

    def __init__(self):
        print("\n"+"="*65)
        print("  BED-MAKING ROBOT  —  Real-Time Video Stream  [v3: Visual Prompting]")
        print("="*65)

        print("\n[1/4] Loading YOLO…")
        model_path = Path("models/bed_detector_v1/weights/best.pt")
        if not model_path.exists():
            print(f"  ✗ Model not found: {model_path}"); sys.exit(1)
        self.yolo = YOLO(str(model_path))
        print("      ✓ YOLO loaded")

        print("\n[2/4] Loading SAM (vit_b, cpu)…")
        try:
            self.sam     = SAMRefiner(model_type="vit_b", device="cpu")
            self._sam_ok = True
            print("      ✓ SAM ready")
        except Exception as e:
            print(f"      ⚠ SAM unavailable ({e}) — YOLO-only mode")
            self.sam     = None
            self._sam_ok = False

        print("\n[3/4] Initialising analyzers…")
        self.acc_analyzer = AccessibilityAnalyzer()
        self.encoder      = SymbolicStateEncoder()
        self.planner      = StepPlanner()
        self.model_cfg    = CPU_MODELS[1]   # LLaVA-7B (benchmark winner)
        print("      ✓ Ready")

        print("\n[4/4] Camera will start in panoramic scan…")
        self.pipeline = None; self.align = None

        self.task_state     = TaskState()
        self.accessibility  = {}
        self.stats          = {}
        self.vlm_output     = {}
        self.inference_time = 0.0
        self.validation     = {'valid':False,'score':0,'violations':[]}
        self._lock          = threading.Lock()

        self._latest_rgb_yolo    = np.zeros((480,640,3),dtype=np.uint8)
        self._latest_depth_color = np.zeros((480,640,3),dtype=np.uint8)
        self._latest_detections  = {'bed':None,'bedding':[],'pillows':[],'obstacles':[],'walls':[]}
        self._latest_rgb         = None
        self._latest_depth       = None
        self._latest_sam_panel   = np.zeros((480,640,3), dtype=np.uint8)
        self._sam_input_rgb      = None
        self._sam_input_det      = None

        self._raw_queue    = queue.Queue(maxsize=4)
        self._frame_count  = 0
        self._paused       = False
        self._stop         = False
        self._scan_running = False
        self._auto_running = False
        self._fps_times    = []
        self._temp_dir     = Path("temp_demo"); self._temp_dir.mkdir(exist_ok=True)
        self._last_key_time: dict = {}
        self._key_debounce = 0.5

    def _key_allowed(self, key_char):
        now=time.time(); last=self._last_key_time.get(key_char,0.0)
        if now-last > self._key_debounce:
            self._last_key_time[key_char]=now; return True
        return False

    def _apply_sam(self, rgb, detections):
        if not self._sam_ok or self.sam is None: return detections
        try: return self.sam.refine_batch(rgb, detections)
        except Exception as e: print(f"[SAM] {e}"); return detections

    # ── Phase 1 ───────────────────────────────────────────

    def run_initial_scan(self):
        print("\n"+"▓"*42+"\n▓  PHASE 1: PANORAMIC ENVIRONMENT SCAN\n"+"▓"*42)
        mapper = InteractivePanoramicMapper(
            model_path="models/bed_detector_v1/weights/best.pt",
            sam=self.sam,
        )
        accessibility, stats, objects, frames = mapper.run()
        self.pipeline = mapper.pipeline; self.align = mapper.align
        with self._lock:
            self.accessibility = accessibility; self.stats = stats
            self.task_state.set_accessibility(accessibility)
            names = ((['Bed'] if objects.get('bed') else []) +
                     [o['class'] for o in objects.get('bedding',[])] +
                     [o['class'] for o in objects.get('pillows',[])])
            self.task_state.detected_objects = names
        print_accessibility(accessibility, stats)
        print(f"\n  Objects: {names}\n  Best approach: {self.task_state.best_approach_side}")
        return objects

    # ── Rescan ────────────────────────────────────────────

    def _run_rescan(self):
        self._scan_running = True; self._auto_running = False
        print("\n[RESCAN] Pausing for panoramic rescan…")
        cv2.destroyWindow("Bed-Making Robot [Live]")
        mapper = InteractivePanoramicMapper.__new__(InteractivePanoramicMapper)
        mapper.pipeline=self.pipeline; mapper.align=self.align; mapper.yolo=self.yolo
        mapper.sam=self.sam; mapper._sam_ok=self._sam_ok
        mapper.fusion=__import__('depth_mapper').MultiViewFusion()
        mapper.frames=[]; mapper._last_flash=-10.0
        mapper._live_detections={'bed':None,'bedding':[],'pillows':[],'obstacles':[],'walls':[]}
        mapper._frame_counter=0
        cv2.namedWindow("Panoramic Capture",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Panoramic Capture",1280,600)
        try:
            accessibility,stats,objects,frames = mapper.run()
            with self._lock:
                self.accessibility=accessibility; self.stats=stats
                self.task_state.set_accessibility(accessibility)
            print_accessibility(accessibility,stats)
        except Exception as e: print(f"[RESCAN] Error: {e}")
        cv2.namedWindow("Bed-Making Robot [Live]",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bed-Making Robot [Live]",1920,600)
        self._scan_running=False; self._auto_running=True
        print("[RESCAN] Done — resuming VLM loop")

    # ── VLM loop  (v3: sends perception-annotated image) ──────────────

    def _vlm_auto_loop(self):
        print("\n"+"▓"*42+"\n▓  PHASE 2: AUTO VLM TASK SEQUENCING\n"+"▓"*42)
        print("  VLM will plan each step automatically.")
        print("  v3: LLaVA receives YOLO+accessibility annotated image")
        print("  Press SPACE to pause/resume, p to rescan, r to reset.\n")
        while not self._stop:
            if self._paused or self._scan_running: time.sleep(0.2); continue
            with self._lock: state_copy=deepcopy(self.task_state)
            if state_copy.is_complete():
                print("\n"+"═"*70+"\n  ✓  TASK COMPLETE — All 35 steps finished!\n"+"═"*70)
                self._auto_running=False; break
            print_step_header(state_copy)

            # v3: grab YOLO-annotated frame + current accessibility
            with self._lock:
                rgb_yolo = self._latest_rgb_yolo.copy()  # YOLO boxes already drawn
                acc      = dict(self.accessibility)
                sts      = dict(self.stats)

            if rgb_yolo is None: time.sleep(0.5); continue

            # v3: burn accessibility zones onto image edges before sending to LLaVA
            vlm_frame = burn_accessibility_onto_image(rgb_yolo, acc, sts)

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = self._temp_dir / f"vlm_{ts}.png"
            cv2.imwrite(str(img_path), vlm_frame)  # LLaVA sees annotated image

            prompt=PromptBuilder.build(state_copy,"",model_id=self.model_cfg['id'])
            print_prompt(prompt,self.model_cfg['id'])
            print("  ⏳ Waiting for VLM response…")
            t0=time.time()
            result=self.planner.plan_step(self.model_cfg,state_copy,img_path,"",verbose=False)
            elapsed=time.time()-t0
            print_vlm_response(result['raw_response'],elapsed)
            print_parsed(result['output']); print_validation(result['validation'])
            with self._lock:
                self.vlm_output=result['output']; self.inference_time=result['inference_time']
                self.validation=result['validation']; self.task_state=result['updated_task_state']
                if not result['step_advanced']:
                    self.task_state.mark_complete(state_copy.current_step,note="auto-forced")
                    print("  ↳ Auto-advancing (step not confirmed complete by VLM)")
            print(f"\n  → Next step: {self.task_state.current_step.value}")
            time.sleep(1.5)

    # ── Capture thread ────────────────────────────────────

    def _capture_loop(self):
        while not self._stop:
            if self._paused or self._scan_running: time.sleep(0.03); continue
            try:
                frames=self.pipeline.wait_for_frames(timeout_ms=200)
                aligned=self.align.process(frames)
                color=aligned.get_color_frame(); depth_f=aligned.get_depth_frame()
                if not color or not depth_f: continue
                rgb=np.asanyarray(color.get_data()); depth=np.asanyarray(depth_f.get_data())
                if self._raw_queue.full():
                    try: self._raw_queue.get_nowait()
                    except: pass
                self._raw_queue.put((rgb,depth))
            except Exception: pass

    # ── Process thread ────────────────────────────────────

    def _process_loop(self):
        every=3
        while not self._stop:
            try: rgb,depth=self._raw_queue.get(timeout=0.5)
            except queue.Empty: continue
            self._frame_count+=1
            depth_color=colorize_depth(depth)
            with self._lock:
                self._latest_depth=depth; self._latest_rgb=rgb
                self._latest_depth_color=depth_color
            if self._frame_count % every != 0:
                with self._lock:
                    self._latest_rgb_yolo=draw_yolo_boxes(rgb,self._latest_detections)
                continue
            results=self.yolo(rgb,verbose=False)[0]
            h,w_frame=rgb.shape[:2]
            detections=filter_and_clean_detections(results.boxes,w_frame,h)
            rgb_yolo=draw_yolo_boxes(rgb,detections)
            with self._lock:
                self._latest_detections=detections
                self._latest_rgb_yolo=rgb_yolo
                self._sam_input_rgb = rgb.copy()
                self._sam_input_det = {k:(list(v) if isinstance(v,list) else v)
                                       for k,v in detections.items()}

    def _sam_loop(self):
        """Background thread: runs SAM ~1fps on CPU, updates _latest_sam_panel."""
        while not self._stop:
            with self._lock:
                rgb = getattr(self, '_sam_input_rgb', None)
                det = getattr(self, '_sam_input_det', None)
            if rgb is None or det is None:
                time.sleep(0.05); continue
            try:
                refined = self._apply_sam(rgb, det)
                panel   = draw_sam_masks(rgb, refined, alpha=0.55)
                for grp in ('bed','bedding','pillows','obstacles','walls'):
                    val   = refined.get(grp)
                    items = [val] if isinstance(val,dict) and val else (val or [])
                    for obj in items:
                        if obj.get('sam_mask') is None: continue
                        x1,y1 = int(obj['bbox'][0]), int(obj['bbox'][1])
                        cv2.putText(panel, obj.get('class','?'),
                            (x1+2,y1+14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,(255,255,255),1,cv2.LINE_AA)
                with self._lock:
                    self._latest_sam_panel = panel
                    self._sam_input_rgb    = None
            except Exception:
                pass
            time.sleep(0.02)

    # ── Display loop ──────────────────────────────────────

    def run_live_stream(self):
        cv2.namedWindow("Bed-Making Robot [Live]",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bed-Making Robot [Live]",1920,600)
        t_cap =threading.Thread(target=self._capture_loop, daemon=True)
        t_proc=threading.Thread(target=self._process_loop, daemon=True)
        t_sam =threading.Thread(target=self._sam_loop,     daemon=True)
        t_vlm =threading.Thread(target=self._vlm_auto_loop,daemon=True)
        t_cap.start(); t_proc.start(); t_sam.start()
        self._auto_running=True; t_vlm.start()

        while True:
            now=time.time()
            self._fps_times=[t for t in self._fps_times if now-t<1.0]
            self._fps_times.append(now); fps=float(len(self._fps_times))
            with self._lock:
                rgb_yolo   =self._latest_rgb_yolo.copy()
                depth_color=self._latest_depth_color.copy()
                sam_panel  =self._latest_sam_panel.copy()
                acc=dict(self.accessibility); sts=dict(self.stats)
                state_snap=deepcopy(self.task_state); vlm_out=dict(self.vlm_output)
                inf=self.inference_time; val=dict(self.validation)
            dash=build_dashboard(rgb_yolo,depth_color,state_snap,acc,sts,vlm_out,inf,val,
                                 fps,self._paused,self._scan_running,self._auto_running,
                                 sam_panel=sam_panel)
            cv2.imshow("Bed-Making Robot [Live]",dash)
            key=cv2.waitKey(1)&0xFF

            if key==ord('q'): print("\nQuit."); break
            elif key==ord(' ') and self._key_allowed(' '):
                self._paused=not self._paused
                print(f"[{'PAUSED — press SPACE to resume' if self._paused else 'RESUMED'}]")
            elif key==ord('p') and self._key_allowed('p'):
                if not self._scan_running:
                    threading.Thread(target=self._run_rescan,daemon=True).start()
                else: print("[RESCAN] Already running")
            elif key==ord('r') and self._key_allowed('r'):
                with self._lock:
                    self.task_state=TaskState(); self.task_state.set_accessibility(self.accessibility)
                    self.vlm_output={}; self.inference_time=0.0
                    self.validation={'valid':False,'score':0,'violations':[]}
                print("[RESET] Task state reset to step 1")

        self._stop=True
        t_cap.join(timeout=2.0); t_proc.join(timeout=2.0); t_sam.join(timeout=2.0)
        try: self.pipeline.stop()
        except: pass
        cv2.destroyAllWindows(); print("✓ Stopped")

    def run(self):
        self.run_initial_scan()
        self.run_live_stream()


def main():
    import cv2 as _cv2
    _cv2.namedWindow("Panoramic Capture", _cv2.WINDOW_NORMAL)
    _cv2.resizeWindow("Panoramic Capture", 1280, 600)
    _cv2.waitKey(1)

    stream=None
    try:
        stream=RealTimeVideoStream(); stream.run()
    except KeyboardInterrupt: print("\n\nInterrupted")
    except Exception as e:
        import traceback; print(f"\n✗ Error: {e}"); traceback.print_exc()
    finally:
        if stream:
            try:
                stream._stop=True
                if stream.pipeline: stream.pipeline.stop()
                cv2.destroyAllWindows()
            except: pass

if __name__=="__main__":
    main()
