"""
Task State Machine for Bed-Making Robot
========================================
Defines the full 14-step bed-making workflow as a state machine.
Each state knows:
  - what phase it belongs to (stripping / transition / making)
  - what preconditions must be met
  - what accessibility info it needs
  - what the next state is
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ─────────────────────────────────────────
# Phases
# ─────────────────────────────────────────
class Phase(str, Enum):
    INIT       = "init"        # environment analysis only
    STRIPPING  = "stripping"   # removing all bedding
    TRANSITION = "transition"  # all stripped, ready to make
    MAKING     = "making"      # putting clean bedding on


# ─────────────────────────────────────────
# Task Steps  (matches your 14-step list)
# ─────────────────────────────────────────
class TaskStep(str, Enum):
    # ── INIT ──────────────────────────────
    ANALYSE_ENVIRONMENT      = "analyse_environment"       # 1

    # ── STRIPPING ─────────────────────────
    PICK_PILLOW              = "pick_pillow"               # 6  (uses accessibility)
    REMOVE_PILLOW_COVER      = "remove_pillow_cover"       # 7
    PLACE_PILLOW_SIDE_TABLE  = "place_pillow_side_table"   # 8
    PICK_DUVET               = "pick_duvet"                # 9  (uses accessibility)
    REMOVE_DUVET_COVER       = "remove_duvet_cover"        # 10
    REMOVE_BED_COVER         = "remove_bed_cover"          # 11
    REMOVE_MATTRESS_PROTECTOR= "remove_mattress_protector" # 12

    # ── TRANSITION ────────────────────────
    STRIPPING_COMPLETE       = "stripping_complete"        # 13

    # ── MAKING ────────────────────────────
    PUT_PILLOW_IN_COVER      = "put_pillow_in_cover"       # 14
    PLACE_PILLOW             = "place_pillow"              # 15  (uses accessibility)
    PLACE_DUVET_COVER        = "place_duvet_cover"         # 16
    SPREAD_DUVET             = "spread_duvet"              # 17
    PLACE_BED_COVER          = "place_bed_cover"           # 18
    PLACE_MATTRESS_PROTECTOR = "place_mattress_protector"  # 19
    TASK_COMPLETE            = "task_complete"             # 20


# ─────────────────────────────────────────
# Step metadata
# ─────────────────────────────────────────
@dataclass
class StepDefinition:
    step: TaskStep
    phase: Phase
    description: str
    needs_accessibility: bool          # does this step need approach-side info?
    needs_object: Optional[str]        # what object must be detected?
    preconditions: List[TaskStep]      # which steps must be done first
    next_step: Optional[TaskStep]      # deterministic successor (None = end)
    action_type: str                   # navigate | grasp | manipulate | place | assess


# ─────────────────────────────────────────
# Full step registry
# ─────────────────────────────────────────
STEP_REGISTRY: Dict[TaskStep, StepDefinition] = {

    TaskStep.ANALYSE_ENVIRONMENT: StepDefinition(
        step=TaskStep.ANALYSE_ENVIRONMENT,
        phase=Phase.INIT,
        description="Analyse bed environment: detect objects, assess side accessibility (head/foot/left/right), identify what needs to be removed.",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[],
        next_step=TaskStep.PICK_PILLOW,
        action_type="assess"
    ),

    TaskStep.PICK_PILLOW: StepDefinition(
        step=TaskStep.PICK_PILLOW,
        phase=Phase.STRIPPING,
        description="Navigate to the most accessible side of the bed and grasp the pillow. Use accessibility data to choose approach side.",
        needs_accessibility=True,
        needs_object="pillow",
        preconditions=[TaskStep.ANALYSE_ENVIRONMENT],
        next_step=TaskStep.REMOVE_PILLOW_COVER,
        action_type="navigate"
    ),

    TaskStep.REMOVE_PILLOW_COVER: StepDefinition(
        step=TaskStep.REMOVE_PILLOW_COVER,
        phase=Phase.STRIPPING,
        description="Grasp pillow cover and remove it from the pillow. Hold pillow firmly, pull cover off.",
        needs_accessibility=False,
        needs_object="pillow",
        preconditions=[TaskStep.PICK_PILLOW],
        next_step=TaskStep.PLACE_PILLOW_SIDE_TABLE,
        action_type="manipulate"
    ),

    TaskStep.PLACE_PILLOW_SIDE_TABLE: StepDefinition(
        step=TaskStep.PLACE_PILLOW_SIDE_TABLE,
        phase=Phase.STRIPPING,
        description="Place the bare pillow and removed pillow cover on the side table. Keep them separate for later use.",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.REMOVE_PILLOW_COVER],
        next_step=TaskStep.PICK_DUVET,
        action_type="place"
    ),

    TaskStep.PICK_DUVET: StepDefinition(
        step=TaskStep.PICK_DUVET,
        phase=Phase.STRIPPING,
        description="Navigate to best accessible side and grasp the duvet/duvet cover. Use accessibility data to choose approach side.",
        needs_accessibility=True,
        needs_object="duvet",
        preconditions=[TaskStep.PLACE_PILLOW_SIDE_TABLE],
        next_step=TaskStep.REMOVE_DUVET_COVER,
        action_type="navigate"
    ),

    TaskStep.REMOVE_DUVET_COVER: StepDefinition(
        step=TaskStep.REMOVE_DUVET_COVER,
        phase=Phase.STRIPPING,
        description="Remove the duvet cover from the duvet insert. Pull cover off completely and set aside.",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.PICK_DUVET],
        next_step=TaskStep.REMOVE_BED_COVER,
        action_type="manipulate"
    ),

    TaskStep.REMOVE_BED_COVER: StepDefinition(
        step=TaskStep.REMOVE_BED_COVER,
        phase=Phase.STRIPPING,
        description="Grasp and remove the bed cover/top sheet. Pull back and fold or remove completely from the bed.",
        needs_accessibility=True,
        needs_object="bed_cover",
        preconditions=[TaskStep.REMOVE_DUVET_COVER],
        next_step=TaskStep.REMOVE_MATTRESS_PROTECTOR,
        action_type="manipulate"
    ),

    TaskStep.REMOVE_MATTRESS_PROTECTOR: StepDefinition(
        step=TaskStep.REMOVE_MATTRESS_PROTECTOR,
        phase=Phase.STRIPPING,
        description="Remove the mattress protector from all corners of the mattress. Work around accessible sides.",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.REMOVE_BED_COVER],
        next_step=TaskStep.STRIPPING_COMPLETE,
        action_type="manipulate"
    ),

    TaskStep.STRIPPING_COMPLETE: StepDefinition(
        step=TaskStep.STRIPPING_COMPLETE,
        phase=Phase.TRANSITION,
        description="All bedding has been removed. Bed is now bare. Transition to making phase.",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.REMOVE_MATTRESS_PROTECTOR],
        next_step=TaskStep.PUT_PILLOW_IN_COVER,
        action_type="assess"
    ),

    TaskStep.PUT_PILLOW_IN_COVER: StepDefinition(
        step=TaskStep.PUT_PILLOW_IN_COVER,
        phase=Phase.MAKING,
        description="Take the clean pillow cover and insert the pillow into it. Ensure all corners are filled and cover is straight.",
        needs_accessibility=False,
        needs_object="pillow",
        preconditions=[TaskStep.STRIPPING_COMPLETE],
        next_step=TaskStep.PLACE_PILLOW,
        action_type="manipulate"
    ),

    TaskStep.PLACE_PILLOW: StepDefinition(
        step=TaskStep.PLACE_PILLOW,
        phase=Phase.MAKING,
        description="Place the dressed pillow at the head of the bed. Use accessibility data to approach from the best available side.",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.PUT_PILLOW_IN_COVER],
        next_step=TaskStep.PLACE_DUVET_COVER,
        action_type="place"
    ),

    TaskStep.PLACE_DUVET_COVER: StepDefinition(
        step=TaskStep.PLACE_DUVET_COVER,
        phase=Phase.MAKING,
        description="Insert the duvet into the clean duvet cover. Ensure all corners are aligned.",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.PLACE_PILLOW],
        next_step=TaskStep.SPREAD_DUVET,
        action_type="manipulate"
    ),

    TaskStep.SPREAD_DUVET: StepDefinition(
        step=TaskStep.SPREAD_DUVET,
        phase=Phase.MAKING,
        description="Spread the dressed duvet evenly over the bed. Approach from accessible sides to smooth and align.",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.PLACE_DUVET_COVER],
        next_step=TaskStep.PLACE_BED_COVER,
        action_type="manipulate"
    ),

    TaskStep.PLACE_BED_COVER: StepDefinition(
        step=TaskStep.PLACE_BED_COVER,
        phase=Phase.MAKING,
        description="Place and smooth the bed cover/top sheet over the duvet. Tuck in accessible sides.",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.SPREAD_DUVET],
        next_step=TaskStep.PLACE_MATTRESS_PROTECTOR,
        action_type="place"
    ),

    TaskStep.PLACE_MATTRESS_PROTECTOR: StepDefinition(
        step=TaskStep.PLACE_MATTRESS_PROTECTOR,
        phase=Phase.MAKING,
        description="Fit the mattress protector under the mattress at all corners. Work around accessible sides only.",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.PLACE_BED_COVER],
        next_step=TaskStep.TASK_COMPLETE,
        action_type="manipulate"
    ),

    TaskStep.TASK_COMPLETE: StepDefinition(
        step=TaskStep.TASK_COMPLETE,
        phase=Phase.MAKING,
        description="Bed making task is complete. All bedding placed and smoothed.",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PLACE_MATTRESS_PROTECTOR],
        next_step=None,
        action_type="assess"
    ),
}


# ─────────────────────────────────────────
# Runtime task state  (tracks progress)
# ─────────────────────────────────────────
@dataclass
class TaskState:
    current_step: TaskStep = TaskStep.ANALYSE_ENVIRONMENT
    completed_steps: List[TaskStep] = field(default_factory=list)
    accessibility: Dict[str, str] = field(default_factory=dict)  # side → free/partially_blocked/blocked
    best_approach_side: Optional[str] = None
    detected_objects: List[str] = field(default_factory=list)
    phase: Phase = Phase.INIT
    notes: List[str] = field(default_factory=list)  # log of decisions

    def mark_complete(self, step: TaskStep, note: str = ""):
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        if note:
            self.notes.append(f"[{step.value}] {note}")
        # Advance to next step
        defn = STEP_REGISTRY[step]
        if defn.next_step:
            self.current_step = defn.next_step
            self.phase = STEP_REGISTRY[defn.next_step].phase

    def set_accessibility(self, accessibility: Dict[str, str]):
        self.accessibility = accessibility
        # Pick best approach side: free > partially_blocked
        for side in ["foot", "left", "right", "head"]:
            if accessibility.get(side) == "free":
                self.best_approach_side = side
                break
        if not self.best_approach_side:
            for side in ["foot", "left", "right", "head"]:
                if accessibility.get(side) == "partially_blocked":
                    self.best_approach_side = side
                    break

    def is_complete(self) -> bool:
        return self.current_step == TaskStep.TASK_COMPLETE

    def get_current_definition(self) -> StepDefinition:
        return STEP_REGISTRY[self.current_step]

    def to_context_string(self) -> str:
        """Serialise task state into a prompt-ready context string."""
        defn = self.get_current_definition()
        completed_names = [s.value for s in self.completed_steps]
        acc_str = ", ".join(f"{k}={v}" for k, v in self.accessibility.items()) or "unknown"
        obj_str = ", ".join(self.detected_objects) or "none detected"

        return f"""=== BED-MAKING TASK STATE ===
Current Phase   : {self.phase.value}
Current Step    : {self.current_step.value}  ({len(self.completed_steps)+1} of {len(STEP_REGISTRY)})
Step Goal       : {defn.description}
Action Type     : {defn.action_type}

Completed Steps : {completed_names if completed_names else 'none yet'}

Accessibility   : {acc_str}
Best Approach   : {self.best_approach_side or 'not yet determined'}
Detected Objects: {obj_str}

Needs Side Info : {'YES — use accessibility to choose approach side' if defn.needs_accessibility else 'NO'}
Target Object   : {defn.needs_object or 'none required'}
==========================="""
