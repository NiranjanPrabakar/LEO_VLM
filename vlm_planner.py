"""
Bed-Making VLM Planner v3  (CPU-Optimised, State-Machine-Aware)
================================================================
Changes from v2:
  - Fix 1: Added 'tuck' action to build_json() and build_prose()
            Steps 22-29 no longer send {"action":"unknown"} to LLaVA
  - Fix 2: Sensor depth data injected into assess prompt
            LLaVA confirms sensor truth instead of hallucinating
  - Fix 3: PlanValidator now accepts 'tuck' action type
  - Fix 4: prose_to_json handles 'tuck' for Moondream fallback

Models (CPU-safe):
  moondream  ~1.6 GB  parallel=True
  llava:7b   ~4.0 GB  parallel=False (sequential)
"""

import json
import base64
import requests
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple
from copy import deepcopy

from task_state_machine import (
    TaskState, TaskStep, StepDefinition,
    STEP_REGISTRY, Phase
)


# ─────────────────────────────────────────
# Model configs  (CPU-only)
# ─────────────────────────────────────────
CPU_MODELS = [
    {
        "id":          "moondream",
        "name":        "Moondream",
        "size_gb":     1.6,
        "parallel":    True,
        "max_workers": 3,
        "timeout":     120,
        "temperature": 0.1,
        "num_predict": 150,
    },
    {
        "id":          "llava:7b",
        "name":        "LLaVA-7B",
        "size_gb":     4.0,
        "parallel":    False,
        "max_workers": 1,
        "timeout":     180,
        "temperature": 0.1,
        "num_predict": 300,
    },
]


# ─────────────────────────────────────────
# Prompt Builder  (v3: tuck + sensor injection)
# ─────────────────────────────────────────
class PromptBuilder:
    """
    v3 changes:
      - 'tuck' action now has a real prompt (was falling through to unknown)
      - assess prompt includes sensor ground truth so VLM confirms not guesses
    """

    # ── JSON prompts: for LLaVA-7B ───────────────────────────────────────────
    @classmethod
    def build_json(cls, task_state: TaskState) -> str:
        defn      = task_state.get_current_definition()
        best_side = task_state.best_approach_side or "foot"
        acc       = task_state.accessibility
        acc_str   = ", ".join(f"{k}={v}" for k, v in acc.items()) if acc else "unknown"
        target    = defn.needs_object or "bedding"
        goal      = defn.description.split(".")[0]
        corner    = defn.object_location or "corner"

        if defn.action_type == "assess":
            # v3 Fix 2: inject real sensor data so VLM confirms rather than guesses
            sensor_hint = ""
            if acc:
                sensor_hint = (
                    f"\nSensor ground truth: {acc_str}. "
                    "Confirm or correct based on what you see in the image."
                )
            return (
                "Look at this bed image. For each side (head=top, foot=bottom, left, right) "
                "say if it is free, partially_blocked, or blocked. "
                "List objects you see: pillow, duvet, bed_cover, mattress_protector."
                + sensor_hint + "\n\n"
                "Reply with ONLY this JSON:\n"
                '{"action":"assess","head":"free","foot":"free","left":"free","right":"free",'
                '"objects":["pillow","duvet"],"step_complete":true}'
            )

        if defn.action_type == "navigate":
            return (
                f"You are a bed-making robot. Sensor data: {acc_str}. "
                f"Best accessible side: {best_side}.\n\n"
                "Reply with ONLY this JSON:\n"
                f'{{"action":"navigate","target_side":"{best_side}","reason":"most accessible","step_complete":false}}'
            )

        if defn.action_type == "grasp":
            return (
                f"You are a bed-making robot. Find the {target} in the image. "
                "Describe the best grasp point (corner, edge, centre).\n\n"
                "Reply with ONLY this JSON:\n"
                f'{{"action":"grasp","target_object":"{target}","grasp_point":"top-left corner","step_complete":false}}'
            )

        if defn.action_type == "manipulate":
            return (
                f"You are a bed-making robot. Task: {goal}. "
                "Describe the arm motion needed.\n\n"
                "Reply with ONLY this JSON:\n"
                '{"action":"manipulate","motion":"pull cover off toward you","step_complete":false}'
            )

        if defn.action_type == "place":
            return (
                f"You are a bed-making robot. Task: {goal}. "
                f"Approach from the {best_side} side.\n\n"
                "Reply with ONLY this JSON:\n"
                f'{{"action":"place","target_location":"head of bed centred","approach_side":"{best_side}","step_complete":false}}'
            )

        # v3 Fix 1: tuck action now has a real prompt
        if defn.action_type == "tuck":
            return (
                f"You are a bed-making robot. Task: {goal}. "
                f"Tuck the {corner} of the bedding under the mattress from the {best_side} side. "
                "Describe the arm motion needed.\n\n"
                "Reply with ONLY this JSON:\n"
                f'{{"action":"tuck","target_corner":"{corner}","approach_side":"{best_side}","motion":"lift mattress edge and tuck bedding underneath","step_complete":false}}'
            )

        return '{"action":"unknown","step_complete":false}'

    # ── Plain-English prompts: for Moondream ─────────────────────────────────
    @classmethod
    def build_prose(cls, task_state: TaskState) -> str:
        defn      = task_state.get_current_definition()
        best_side = task_state.best_approach_side or "foot"
        acc       = task_state.accessibility
        acc_str   = ", ".join(f"{k}={v}" for k, v in acc.items()) if acc else "unknown"
        target    = defn.needs_object or "bedding"
        goal      = defn.description.split(".")[0]
        corner    = defn.object_location or "corner"

        if defn.action_type == "assess":
            sensor_hint = f"\nSensor data: {acc_str}. Confirm or correct." if acc else ""
            return (
                "Look at this bed image. Answer these questions in order:\n"
                "1. Is the HEAD (top) side of the bed blocked by a wall or furniture? yes/no/partial\n"
                "2. Is the FOOT (bottom) side blocked? yes/no/partial\n"
                "3. Is the LEFT side blocked? yes/no/partial\n"
                "4. Is the RIGHT side blocked? yes/no/partial\n"
                "5. What objects do you see on the bed? List them."
                + sensor_hint
            )

        if defn.action_type == "navigate":
            return f"Which side of the bed is easiest to walk to — head, foot, left, or right? Answer in one word."

        if defn.action_type == "grasp":
            return f"Where is the {target} on the bed? Describe which corner or edge to grab it from."

        if defn.action_type == "manipulate":
            return f"Describe in one sentence the arm movement needed to: {goal}"

        if defn.action_type == "place":
            return f"Where exactly on the bed should the item be placed? Describe in one sentence."

        # v3 Fix 1: tuck action for Moondream
        if defn.action_type == "tuck":
            return f"You are tucking the {corner} of the bedding under the mattress from the {best_side} side. Describe the arm motion in one sentence."

        return "Describe what you see on this bed."

    # ── Dispatcher ────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, task_state: TaskState, symbolic_state: str,
              model_id: str = "llava:7b") -> str:
        if "moondream" in model_id:
            return cls.build_prose(task_state)
        return cls.build_json(task_state)


# ─────────────────────────────────────────
# VLM Client
# ─────────────────────────────────────────
class VLMClient:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        self.url = ollama_url

    def encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(
        self,
        model_cfg: dict,
        image_path: Path,
        prompt: str,
        verbose: bool = False,
    ) -> Tuple[dict, str]:
        try:
            image_b64 = self.encode_image(image_path)
            resp = requests.post(
                self.url,
                json={
                    "model":   model_cfg["id"],
                    "prompt":  prompt,
                    "images":  [image_b64],
                    "stream":  False,
                    "options": {
                        "temperature": model_cfg["temperature"],
                        "num_predict": model_cfg["num_predict"],
                    },
                },
                timeout=model_cfg["timeout"],
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "")
                if verbose:
                    print(f"\n{'─'*55}")
                    print(f"[{model_cfg['id']}] RAW RESPONSE:")
                    print(raw)
                    print(f"{'─'*55}")
                if "moondream" in model_cfg["id"]:
                    parsed = self._parse_json(raw)
                    if "error" in parsed:
                        parsed = self.prose_to_json(
                            raw,
                            action_type=model_cfg.get("_action_type", "assess"),
                            best_side=model_cfg.get("_best_side", "foot"),
                        )
                else:
                    parsed = self._parse_json(raw)
                return parsed, raw
            else:
                err = {"error": f"HTTP {resp.status_code}"}
                return err, ""
        except requests.exceptions.Timeout:
            return {"error": f"timeout after {model_cfg['timeout']}s"}, ""
        except Exception as e:
            return {"error": str(e)}, ""

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"error": "json_parse_failed", "raw": text[:500]}

    def prose_to_json(self, text: str, action_type: str,
                      best_side: str = "foot") -> dict:
        text_l = text.lower()
        result = {"action": action_type}

        if action_type == "assess":
            def side_status(fragment):
                if "partial" in fragment:  return "partially_blocked"
                if "yes" in fragment or "block" in fragment: return "blocked"
                return "free"
            lines = text_l.split("\n")
            sides = {}
            for i, side in enumerate(["head","foot","left","right"], 1):
                for line in lines:
                    if str(i)+"." in line or side in line:
                        sides[side] = side_status(line)
                        break
                if side not in sides:
                    sides[side] = "free"
            objects = []
            for obj in ["pillow","duvet","blanket","sheet","cover","mattress","protector"]:
                if obj in text_l:
                    objects.append(obj)
            if not objects:
                objects = ["unknown"]
            result.update(sides)
            result["objects"] = objects
            result["step_complete"] = True

        elif action_type == "navigate":
            for side in ["foot","left","right","head"]:
                if side in text_l:
                    result["target_side"] = side
                    break
            if "target_side" not in result:
                result["target_side"] = best_side
            result["reason"] = text.strip()[:100]
            result["step_complete"] = False

        elif action_type == "grasp":
            result["target_object"] = "bedding"
            result["grasp_point"] = text.strip()[:150]
            result["step_complete"] = False

        elif action_type == "manipulate":
            result["motion"] = text.strip()[:150]
            result["step_complete"] = False

        elif action_type == "place":
            result["target_location"] = text.strip()[:150]
            result["approach_side"] = best_side
            result["step_complete"] = False

        # v3 Fix 4: tuck fallback for Moondream
        elif action_type == "tuck":
            result["target_corner"] = "corner"
            result["approach_side"] = best_side
            result["motion"] = text.strip()[:150] or "lift mattress edge and tuck bedding underneath"
            result["step_complete"] = False

        else:
            result["notes"] = text.strip()[:150]
            result["step_complete"] = False

        return result


# ─────────────────────────────────────────
# Plan Validator  (v3: tuck action added)
# ─────────────────────────────────────────
class PlanValidator:
    REQUIRED_FIELDS = {
        "assess":     ["action"],
        "navigate":   ["action", "target_side"],
        "grasp":      ["action", "target_object"],
        "manipulate": ["action"],
        "place":      ["action", "target_location", "approach_side"],
        "tuck":       ["action", "target_corner", "approach_side"],   # v3 Fix 3
    }

    def validate(self, output: dict, task_state: TaskState, defn: StepDefinition) -> dict:
        if "error" in output:
            return {"valid": False, "score": 0, "violations": [f"VLM error: {output['error']}"]}

        violations = []
        action_type = defn.action_type

        for field in self.REQUIRED_FIELDS.get(action_type, ["action"]):
            if field not in output:
                violations.append(f"Missing field: '{field}'")

        # Safety: no navigating to blocked sides
        if action_type in ("navigate", "place", "tuck"):
            side_key = "target_side" if action_type == "navigate" else "approach_side"
            chosen   = str(output.get(side_key, "")).lower()
            if not chosen or chosen not in task_state.accessibility:
                chosen = task_state.best_approach_side or ""
                output[side_key] = chosen
            status = task_state.accessibility.get(chosen, "unknown")
            if status == "blocked":
                violations.append(f"SAFETY: '{chosen}' is blocked")

        score = max(0, 100 - len(violations) * 25)
        return {"valid": len(violations) == 0, "score": score, "violations": violations}


# ─────────────────────────────────────────
# Step Planner
# ─────────────────────────────────────────
class StepPlanner:
    def __init__(self):
        self.vlm       = VLMClient()
        self.validator = PlanValidator()

    def plan_step(
        self,
        model_cfg: dict,
        task_state: TaskState,
        image_path: Path,
        symbolic_state: str,
        verbose: bool = False,
    ) -> dict:
        defn   = task_state.get_current_definition()
        prompt = PromptBuilder.build(task_state, symbolic_state, model_id=model_cfg["id"])

        if verbose:
            print(f"\n[PROMPT SENT TO {model_cfg['id']}]")
            print(prompt)

        model_cfg_copy = dict(model_cfg)
        model_cfg_copy["_action_type"] = defn.action_type
        model_cfg_copy["_best_side"]   = task_state.best_approach_side or "foot"

        t0 = time.time()
        output, raw = self.vlm.call(model_cfg_copy, image_path, prompt, verbose=verbose)
        elapsed = round(time.time() - t0, 2)

        validation    = self.validator.validate(output, task_state, defn)
        updated_state = deepcopy(task_state)

        step_complete = output.get("step_complete")
        if step_complete is None:
            step_complete = (defn.action_type == "assess")

        if validation["valid"] and step_complete:
            note = str(output.get("notes", output.get("reason", "")))
            updated_state.mark_complete(defn.step, note=note)
            if defn.action_type == "assess":
                acc = {
                    k: output.get(k, "unknown")
                    for k in ["head", "foot", "left", "right"]
                }
                updated_state.set_accessibility(acc)
                updated_state.detected_objects = output.get("objects", [])

        return {
            "step":               defn.step.value,
            "phase":              defn.phase.value,
            "model":              model_cfg["id"],
            "output":             output,
            "raw_response":       raw[:300],
            "validation":         validation,
            "inference_time":     elapsed,
            "step_advanced":      updated_state.current_step != task_state.current_step,
            "updated_task_state": updated_state,
        }


# ─────────────────────────────────────────
# Full Task Runner
# ─────────────────────────────────────────
class FullTaskRunner:
    def __init__(self, model_cfg: dict):
        self.model   = model_cfg
        self.planner = StepPlanner()

    def run(self, frame_id, image_path, symbolic_state_path, accessibility) -> dict:
        print(f"\n{'='*60}")
        print(f"Full Task  |  {self.model['name']}  |  {frame_id}")
        print(f"{'='*60}")

        with open(symbolic_state_path) as f:
            symbolic_state = f.read()

        state = TaskState()
        state.set_accessibility(accessibility)

        history   = []
        max_steps = len(STEP_REGISTRY) + 5

        for i in range(max_steps):
            if state.is_complete():
                print("\n✅ Task complete!")
                break

            defn = state.get_current_definition()
            print(f"\n→ {i+1:02d} [{defn.phase.value}] {state.current_step.value}")

            result = self.planner.plan_step(self.model, state, image_path, symbolic_state)
            state  = result["updated_task_state"]

            v = result["validation"]
            print(f"   {'✓' if v['valid'] else '✗'} score={v['score']} time={result['inference_time']}s")
            for viol in v["violations"]:
                print(f"   ⚠ {viol}")

            r = {k: v for k, v in result.items() if k != "updated_task_state"}
            history.append(r)

            if not result["step_advanced"]:
                print("   ↳ forcing advance (eval mode)")
                state.mark_complete(defn.step, note="forced-advance")

        return {
            "frame_id":    frame_id,
            "model":       self.model["id"],
            "total_steps": len(history),
            "history":     history,
            "final_state": {
                "current_step":    state.current_step.value,
                "completed_steps": [s.value for s in state.completed_steps],
                "accessibility":   state.accessibility,
                "best_approach":   state.best_approach_side,
                "task_complete":   state.is_complete(),
                "notes":           state.notes,
            },
        }


# ─────────────────────────────────────────
# Model Comparison Runner
# ─────────────────────────────────────────
class ModelComparisonRunner:
    def __init__(self):
        self.results_dir = Path("results/vlm_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.planner = StepPlanner()

    def _load_dataset(self):
        with open("results/accessibility_dataset.json") as f:
            return json.load(f)

    def _smart_sample(self, dataset, n):
        import random
        patterns = {}
        for frame in dataset:
            if not frame.get("detections", {}).get("bed"):
                continue
            key = tuple(sorted(frame.get("accessibility", {}).items()))
            patterns.setdefault(key, []).append(frame)

        sampled = []
        per_p = max(1, n // max(len(patterns), 1))
        for frames in patterns.values():
            sampled.extend(random.sample(frames, min(per_p, len(frames))))
        if len(sampled) < n:
            rest = [f for f in dataset if f not in sampled and f.get("detections", {}).get("bed")]
            sampled.extend(random.sample(rest, min(n - len(sampled), len(rest))))
        return sampled[:n]

    def _process_frame(self, model_cfg, frame_data, verbose=False):
        frame_id   = frame_data["frame_id"]
        image_path = Path(f"data/keyframes/{frame_id}_rgb.png")
        sym_path   = Path(f"results/symbolic_states/symbolic_state_{frame_id}.txt")
        if not image_path.exists() or not sym_path.exists():
            return None

        with open(sym_path) as f:
            symbolic_state = f.read()

        state = TaskState()
        state.set_accessibility(frame_data.get("accessibility", {}))
        state.mark_complete(TaskStep.ANALYSE_ENVIRONMENT, note="skipped for comparison")

        result = self.planner.plan_step(model_cfg, state, image_path, symbolic_state,
                                        verbose=verbose)
        result["frame_id"]      = frame_id
        result["accessibility"] = frame_data.get("accessibility", {})
        result.pop("updated_task_state", None)
        return result

    def run(self, num_frames=30):
        dataset = self._load_dataset()
        frames  = self._smart_sample(dataset, num_frames)

        print(f"\n{'='*65}")
        print(f"CPU MODEL COMPARISON  |  {len(frames)} frames  |  2 models")
        print(f"Models: Moondream (~1.6 GB) vs LLaVA-7B (~4 GB)")
        print(f"{'='*65}\n")

        all_results = {}

        for model_cfg in CPU_MODELS:
            print(f"\n🔹 {model_cfg['name']}  ({model_cfg['size_gb']} GB)")
            results = []

            if model_cfg["parallel"]:
                with ThreadPoolExecutor(max_workers=model_cfg["max_workers"]) as ex:
                    futs = {ex.submit(self._process_frame, model_cfg, f): f for f in frames}
                    for fut in tqdm(as_completed(futs), total=len(futs),
                                    desc=f"  {model_cfg['name']}"):
                        r = fut.result()
                        if r:
                            results.append(r)
            else:
                for frame in tqdm(frames, desc=f"  {model_cfg['name']}"):
                    r = self._process_frame(model_cfg, frame)
                    if r:
                        results.append(r)

            for r in results:
                out = self.results_dir / f"{model_cfg['id'].replace(':','_')}_{r['frame_id']}.json"
                with open(out, "w") as f:
                    json.dump(r, f, indent=2)

            all_results[model_cfg["id"]] = results

        summary = self._build_summary(all_results, frames)
        with open(self.results_dir / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        print(f"\n📁 Results: {self.results_dir}")
        return all_results, summary

    def _build_summary(self, all_results, frames):
        summary = {"timestamp": datetime.now().isoformat(),
                   "total_frames": len(frames), "cpu_only": True, "models": {}}
        for mid, results in all_results.items():
            scores = [r["validation"]["score"] for r in results]
            times  = [r["inference_time"] for r in results]
            valids = sum(1 for r in results if r["validation"]["valid"])
            errors = sum(1 for r in results if "error" in r.get("output", {}))
            safety = sum(1 for r in results
                         if any("SAFETY" in v for v in r["validation"].get("violations", [])))
            if scores:
                avg = sum(scores) / len(scores)
                std = (sum((x - avg) ** 2 for x in scores) / len(scores)) ** 0.5
                summary["models"][mid] = {
                    "avg_score":         round(avg, 2),
                    "std_score":         round(std, 2),
                    "validity_rate_pct": round(valids / len(scores) * 100, 1),
                    "error_rate_pct":    round(errors / len(scores) * 100, 1),
                    "safety_violations": safety,
                    "avg_time_sec":      round(sum(times) / len(times), 2),
                    "total_tests":       len(scores),
                }
        return summary

    def _print_summary(self, summary):
        print(f"\n{'='*65}")
        print("COMPARISON RESULTS  (CPU-only, simplified prompts)")
        print(f"{'='*65}")
        print(f"Frames tested: {summary['total_frames']}\n")
        print(f"{'Model':<15} {'Score':<16} {'Valid%':<10} {'Time':<10} {'Errors%':<10} {'Safety'}")
        print("-" * 65)
        for mid, m in summary["models"].items():
            print(f"{mid:<15} {m['avg_score']:>5.1f}±{m['std_score']:<6.1f} "
                  f"{m['validity_rate_pct']:>6.1f}%  {m['avg_time_sec']:>6.2f}s  "
                  f"{m['error_rate_pct']:>5.1f}%    {m['safety_violations']} violations")
        print(f"\n{'='*65}\n")


# ─────────────────────────────────────────
# Diagnose Mode
# ─────────────────────────────────────────
def run_diagnose(frame_id: str, model_id: str):
    model_lookup = {m["id"]: m for m in CPU_MODELS}
    model_cfg    = model_lookup[model_id]
    planner      = StepPlanner()

    with open("results/accessibility_dataset.json") as f:
        dataset = json.load(f)
    frame_data = next((f for f in dataset if f["frame_id"] == frame_id), None)
    if not frame_data:
        frame_data = next((f for f in dataset if f.get("detections", {}).get("bed")), None)
        if not frame_data:
            print("No suitable frames found in dataset.")
            return
        frame_id = frame_data["frame_id"]
        print(f"(Using first available frame: {frame_id})")

    sym_path = Path(f"results/symbolic_states/symbolic_state_{frame_id}.txt")
    if not sym_path.exists():
        print(f"Symbolic state file not found: {sym_path}")
        return

    with open(sym_path) as f:
        symbolic_state = f.read()

    state = TaskState()
    state.set_accessibility(frame_data.get("accessibility", {}))

    print(f"\n{'='*60}")
    print(f"DIAGNOSE  |  model={model_id}  |  frame={frame_id}")
    print(f"Accessibility: {frame_data.get('accessibility', {})}")
    print(f"Best approach: {state.best_approach_side}")
    print(f"{'='*60}")

    result = planner.plan_step(
        model_cfg, state,
        Path(f"data/keyframes/{frame_id}_rgb.png"),
        symbolic_state,
        verbose=True,
    )

    print(f"\n[PARSED OUTPUT]")
    print(json.dumps(result["output"], indent=2))
    print(f"\n[VALIDATION]")
    print(json.dumps(result["validation"], indent=2))
    print(f"\nInference time: {result['inference_time']}s")
    print(f"Step advanced:  {result['step_advanced']}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bed-Making VLM Planner v3")
    parser.add_argument("--mode", choices=["diagnose", "single", "full", "compare"],
                        default="compare")
    parser.add_argument("--frame",  default=None)
    parser.add_argument("--model",  default="moondream", choices=["moondream", "llava:7b"])
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    model_lookup = {m["id"]: m for m in CPU_MODELS}

    if args.mode == "diagnose":
        frame_id = args.frame or "02_Feb_2026_3_vid01_frame000"
        run_diagnose(frame_id, args.model)

    elif args.mode == "compare":
        ModelComparisonRunner().run(num_frames=args.frames)

    elif args.mode == "full":
        frame_id  = args.frame or "02_Feb_2026_3_vid01_frame000"
        model_cfg = model_lookup[args.model]

        with open("results/accessibility_dataset.json") as f:
            dataset = json.load(f)
        frame_data = next((f for f in dataset if f["frame_id"] == frame_id), None)
        if not frame_data:
            print(f"Frame {frame_id} not found.")
            exit(1)

        result = FullTaskRunner(model_cfg).run(
            frame_id=frame_id,
            image_path=Path(f"data/keyframes/{frame_id}_rgb.png"),
            symbolic_state_path=Path(f"results/symbolic_states/symbolic_state_{frame_id}.txt"),
            accessibility=frame_data.get("accessibility", {}),
        )
        out = Path(f"results/full_task_{frame_id}_{args.model.replace(':','_')}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Saved: {out}")

    elif args.mode == "single":
        frame_id  = args.frame or "02_Feb_2026_3_vid01_frame000"
        model_cfg = model_lookup[args.model]

        with open("results/accessibility_dataset.json") as f:
            dataset = json.load(f)
        frame_data = next((f for f in dataset if f["frame_id"] == frame_id), None)
        sym_path   = Path(f"results/symbolic_states/symbolic_state_{frame_id}.txt")

        state = TaskState()
        state.set_accessibility(frame_data.get("accessibility", {}))
        with open(sym_path) as f:
            symbolic_state = f.read()

        result = StepPlanner().plan_step(
            model_cfg, state,
            Path(f"data/keyframes/{frame_id}_rgb.png"),
            symbolic_state,
        )
        print(json.dumps(result["output"], indent=2))
        print(f"\nValidation: {result['validation']}")
        print(f"Time: {result['inference_time']}s")
