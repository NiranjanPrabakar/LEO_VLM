"""
enhanced_state_machine.py  —  35-Step Bed-Making State Machine
==============================================================
Replaces task_state_machine.py with:
  • 35 granular steps (vs 14 before)
  • Side-aware reach logic: pillow is at HEAD → grab from LEFT or RIGHT, not FOOT
  • get_best_side_for_current_step() scores sides by accessibility + reachability
  • Mattress protector: 4-corner fitting steps
  • Tuck-in steps for fitted sheet and bed cover
  • VLM context hints per step (richer prompts)

Backwards compatible: TaskState API unchanged.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ─────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────

class Phase(str, Enum):
    INIT        = "init"
    STRIPPING   = "stripping"
    TRANSITION  = "transition"
    MAKING      = "making"


# ─────────────────────────────────────────────────────────
# Task Steps — 35 total
# ─────────────────────────────────────────────────────────

class TaskStep(str, Enum):
    # INIT ─────────────────────────────────────────────────
    ANALYSE_ENVIRONMENT           = "analyse_environment"            # 1

    # STRIPPING ────────────────────────────────────────────
    NAVIGATE_TO_PILLOW            = "navigate_to_pillow"             # 2
    GRASP_PILLOW                  = "grasp_pillow"                   # 3
    REMOVE_PILLOW_COVER           = "remove_pillow_cover"            # 4
    PLACE_PILLOW_ON_TABLE         = "place_pillow_on_table"          # 5
    PLACE_PILLOW_COVER_ON_TABLE   = "place_pillow_cover_on_table"    # 6

    NAVIGATE_TO_DUVET             = "navigate_to_duvet"              # 7
    GRASP_DUVET                   = "grasp_duvet"                    # 8
    PULL_DUVET_OFF_BED            = "pull_duvet_off_bed"             # 9
    REMOVE_DUVET_COVER            = "remove_duvet_cover"             # 10
    PLACE_DUVET_ON_TABLE          = "place_duvet_on_table"           # 11
    PLACE_DUVET_COVER_ON_TABLE    = "place_duvet_cover_on_table"     # 12

    NAVIGATE_TO_BED_COVER         = "navigate_to_bed_cover"         # 13
    GRASP_BED_COVER               = "grasp_bed_cover"               # 14
    PULL_BED_COVER_OFF            = "pull_bed_cover_off"            # 15
    PLACE_BED_COVER_ON_TABLE      = "place_bed_cover_on_table"      # 16

    NAVIGATE_TO_MATTRESS_PROTECTOR = "navigate_to_mattress_protector"  # 17
    UNCLIP_MATTRESS_PROT_CORNERS  = "unclip_mattress_prot_corners"  # 18
    REMOVE_MATTRESS_PROTECTOR     = "remove_mattress_protector"     # 19
    PLACE_MATTRESS_PROT_ON_TABLE  = "place_mattress_prot_on_table"  # 20

    STRIPPING_COMPLETE            = "stripping_complete"            # 21

    # MAKING ───────────────────────────────────────────────
    FIT_MATTRESS_PROT_HEAD_LEFT   = "fit_mattress_prot_head_left"   # 22
    FIT_MATTRESS_PROT_HEAD_RIGHT  = "fit_mattress_prot_head_right"  # 23
    FIT_MATTRESS_PROT_FOOT_LEFT   = "fit_mattress_prot_foot_left"   # 24
    FIT_MATTRESS_PROT_FOOT_RIGHT  = "fit_mattress_prot_foot_right"  # 25

    SPREAD_BED_COVER              = "spread_bed_cover"              # 26
    TUCK_BED_COVER_LEFT           = "tuck_bed_cover_left"           # 27
    TUCK_BED_COVER_RIGHT          = "tuck_bed_cover_right"          # 28
    TUCK_BED_COVER_FOOT           = "tuck_bed_cover_foot"           # 29

    PUT_DUVET_IN_COVER            = "put_duvet_in_cover"            # 30
    SPREAD_DUVET                  = "spread_duvet"                  # 31
    ALIGN_DUVET_EDGES             = "align_duvet_edges"             # 32

    PUT_PILLOW_IN_COVER           = "put_pillow_in_cover"           # 33
    PLACE_PILLOW_ON_BED           = "place_pillow_on_bed"           # 34

    TASK_COMPLETE                 = "task_complete"                 # 35


# ─────────────────────────────────────────────────────────
# Step definition
# ─────────────────────────────────────────────────────────

@dataclass
class EnhancedStepDefinition:
    step:               TaskStep
    phase:              Phase
    description:        str
    action_type:        str           # assess | navigate | grasp | manipulate | place | tuck
    needs_accessibility: bool
    needs_object:       Optional[str]
    preconditions:      List[TaskStep]
    next_step:          Optional[TaskStep]
    # Spatial reasoning
    preferred_sides:    List[str]     # ['left','right'] = not foot/head for this action
    object_location:    Optional[str] # where is the target object? 'head', 'centre', etc.
    vlm_context_hint:   str           # prepended to VLM prompt for this step


# ─────────────────────────────────────────────────────────
# Registry — all 35 steps
# ─────────────────────────────────────────────────────────

ENHANCED_STEP_REGISTRY: Dict[TaskStep, EnhancedStepDefinition] = {

    # ── 1 ─────────────────────────────────────────────────
    TaskStep.ANALYSE_ENVIRONMENT: EnhancedStepDefinition(
        step=TaskStep.ANALYSE_ENVIRONMENT,
        phase=Phase.INIT,
        description="Perform panoramic depth sweep. Identify all objects on the bed "
                    "(pillow, duvet, bed_cover, mattress_protector). Determine which "
                    "sides are free/partially_blocked/blocked.",
        action_type="assess",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[],
        next_step=TaskStep.NAVIGATE_TO_PILLOW,
        preferred_sides=["foot", "left", "right"],
        object_location=None,
        vlm_context_hint="This is the initial environment scan. List every object you "
                         "see on the bed and classify each side's accessibility.",
    ),

    # ── 2 ─────────────────────────────────────────────────
    TaskStep.NAVIGATE_TO_PILLOW: EnhancedStepDefinition(
        step=TaskStep.NAVIGATE_TO_PILLOW,
        phase=Phase.STRIPPING,
        description="Pillow is at the HEAD of the bed. Navigate to LEFT or RIGHT side "
                    "(NOT foot — too far to reach the head of the bed from there).",
        action_type="navigate",
        needs_accessibility=True,
        needs_object="pillow",
        preconditions=[TaskStep.ANALYSE_ENVIRONMENT],
        next_step=TaskStep.GRASP_PILLOW,
        preferred_sides=["left", "right"],   # head/foot not suitable for pillow reach
        object_location="head",
        vlm_context_hint="The pillow is near the headboard. To reach it you must "
                         "stand at the LEFT or RIGHT side of the bed (the foot end "
                         "is too far away to reach the pillow).",
    ),

    # ── 3 ─────────────────────────────────────────────────
    TaskStep.GRASP_PILLOW: EnhancedStepDefinition(
        step=TaskStep.GRASP_PILLOW,
        phase=Phase.STRIPPING,
        description="Grasp the pillow from the side you are standing at. "
                    "Grip the near edge / corner of the pillow closest to you.",
        action_type="grasp",
        needs_accessibility=False,
        needs_object="pillow",
        preconditions=[TaskStep.NAVIGATE_TO_PILLOW],
        next_step=TaskStep.REMOVE_PILLOW_COVER,
        preferred_sides=["left", "right"],
        object_location="head",
        vlm_context_hint="You are beside the bed. Locate the pillow near the headboard. "
                         "Describe which corner/edge of the pillow is closest to you.",
    ),

    # ── 4 ─────────────────────────────────────────────────
    TaskStep.REMOVE_PILLOW_COVER: EnhancedStepDefinition(
        step=TaskStep.REMOVE_PILLOW_COVER,
        phase=Phase.STRIPPING,
        description="Hold pillow firmly with one gripper. Use the other gripper to "
                    "grip the open end of the pillow cover and pull it off.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="pillow",
        preconditions=[TaskStep.GRASP_PILLOW],
        next_step=TaskStep.PLACE_PILLOW_ON_TABLE,
        preferred_sides=["left", "right"],
        object_location="in_hand",
        vlm_context_hint="You are holding the pillow. The cover must be pulled off. "
                         "Describe which end to grip and the pull direction.",
    ),

    # ── 5 ─────────────────────────────────────────────────
    TaskStep.PLACE_PILLOW_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_PILLOW_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Place the bare pillow flat on the side table to your left or right.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.REMOVE_PILLOW_COVER],
        next_step=TaskStep.PLACE_PILLOW_COVER_ON_TABLE,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Find the side table. Place the bare pillow flat on it.",
    ),

    # ── 6 ─────────────────────────────────────────────────
    TaskStep.PLACE_PILLOW_COVER_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_PILLOW_COVER_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Fold the removed pillow cover neatly and place it beside the pillow "
                    "on the side table.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PLACE_PILLOW_ON_TABLE],
        next_step=TaskStep.NAVIGATE_TO_DUVET,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Place the folded pillow cover beside the pillow on the table.",
    ),

    # ── 7 ─────────────────────────────────────────────────
    TaskStep.NAVIGATE_TO_DUVET: EnhancedStepDefinition(
        step=TaskStep.NAVIGATE_TO_DUVET,
        phase=Phase.STRIPPING,
        description="Duvet covers the whole bed. Navigate to the most accessible side "
                    "to grasp the duvet. Foot end is usually most accessible.",
        action_type="navigate",
        needs_accessibility=True,
        needs_object="duvet",
        preconditions=[TaskStep.PLACE_PILLOW_COVER_ON_TABLE],
        next_step=TaskStep.GRASP_DUVET,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="The duvet covers the whole bed. Use accessibility data to "
                         "pick the best side to approach from.",
    ),

    # ── 8 ─────────────────────────────────────────────────
    TaskStep.GRASP_DUVET: EnhancedStepDefinition(
        step=TaskStep.GRASP_DUVET,
        phase=Phase.STRIPPING,
        description="Grip the near edge of the duvet (closest to you). "
                    "Grip along the full edge width for even pull.",
        action_type="grasp",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.NAVIGATE_TO_DUVET],
        next_step=TaskStep.PULL_DUVET_OFF_BED,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="Locate the nearest edge of the duvet. Describe the grasp point.",
    ),

    # ── 9 ─────────────────────────────────────────────────
    TaskStep.PULL_DUVET_OFF_BED: EnhancedStepDefinition(
        step=TaskStep.PULL_DUVET_OFF_BED,
        phase=Phase.STRIPPING,
        description="Pull the duvet off the bed toward you. Let it gather on the floor "
                    "or drape over your arm.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.GRASP_DUVET],
        next_step=TaskStep.REMOVE_DUVET_COVER,
        preferred_sides=["foot", "left", "right"],
        object_location="in_hand",
        vlm_context_hint="Pull the duvet fully off the bed away from you. "
                         "Describe the pulling motion.",
    ),

    # ── 10 ────────────────────────────────────────────────
    TaskStep.REMOVE_DUVET_COVER: EnhancedStepDefinition(
        step=TaskStep.REMOVE_DUVET_COVER,
        phase=Phase.STRIPPING,
        description="Hold duvet insert with both grippers. Shake or peel the duvet "
                    "cover off the insert. The insert is typically lighter and stiffer.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.PULL_DUVET_OFF_BED],
        next_step=TaskStep.PLACE_DUVET_ON_TABLE,
        preferred_sides=["left", "right", "foot"],
        object_location="in_hand",
        vlm_context_hint="You have the duvet in hand. Separate the cover from the insert "
                         "by pulling the cover off.",
    ),

    # ── 11 ────────────────────────────────────────────────
    TaskStep.PLACE_DUVET_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_DUVET_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Fold the bare duvet insert and place it on the side table or chair.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.REMOVE_DUVET_COVER],
        next_step=TaskStep.PLACE_DUVET_COVER_ON_TABLE,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Place the duvet insert on the table or large flat surface.",
    ),

    # ── 12 ────────────────────────────────────────────────
    TaskStep.PLACE_DUVET_COVER_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_DUVET_COVER_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Fold the duvet cover and place it on the table beside the insert.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PLACE_DUVET_ON_TABLE],
        next_step=TaskStep.NAVIGATE_TO_BED_COVER,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Place the folded duvet cover on the table.",
    ),

    # ── 13 ────────────────────────────────────────────────
    TaskStep.NAVIGATE_TO_BED_COVER: EnhancedStepDefinition(
        step=TaskStep.NAVIGATE_TO_BED_COVER,
        phase=Phase.STRIPPING,
        description="Bed cover lies under the duvet. Navigate to the most accessible side.",
        action_type="navigate",
        needs_accessibility=True,
        needs_object="bed_cover",
        preconditions=[TaskStep.PLACE_DUVET_COVER_ON_TABLE],
        next_step=TaskStep.GRASP_BED_COVER,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="Navigate to the best accessible side to grab the bed cover / top sheet.",
    ),

    # ── 14 ────────────────────────────────────────────────
    TaskStep.GRASP_BED_COVER: EnhancedStepDefinition(
        step=TaskStep.GRASP_BED_COVER,
        phase=Phase.STRIPPING,
        description="Grasp the near edge of the bed cover (the side closest to you).",
        action_type="grasp",
        needs_accessibility=False,
        needs_object="bed_cover",
        preconditions=[TaskStep.NAVIGATE_TO_BED_COVER],
        next_step=TaskStep.PULL_BED_COVER_OFF,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="Find the near edge of the flat sheet/bed cover. "
                         "Describe where to grip it.",
    ),

    # ── 15 ────────────────────────────────────────────────
    TaskStep.PULL_BED_COVER_OFF: EnhancedStepDefinition(
        step=TaskStep.PULL_BED_COVER_OFF,
        phase=Phase.STRIPPING,
        description="Pull the bed cover fully off the bed. "
                    "Any tucked edges will need to be un-tucked first.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="bed_cover",
        preconditions=[TaskStep.GRASP_BED_COVER],
        next_step=TaskStep.PLACE_BED_COVER_ON_TABLE,
        preferred_sides=["foot", "left", "right"],
        object_location="in_hand",
        vlm_context_hint="Pull the bed cover off. If it is tucked, un-tuck before pulling.",
    ),

    # ── 16 ────────────────────────────────────────────────
    TaskStep.PLACE_BED_COVER_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_BED_COVER_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Fold the bed cover and place it on the table.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PULL_BED_COVER_OFF],
        next_step=TaskStep.NAVIGATE_TO_MATTRESS_PROTECTOR,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Place the folded bed cover on the table.",
    ),

    # ── 17 ────────────────────────────────────────────────
    TaskStep.NAVIGATE_TO_MATTRESS_PROTECTOR: EnhancedStepDefinition(
        step=TaskStep.NAVIGATE_TO_MATTRESS_PROTECTOR,
        phase=Phase.STRIPPING,
        description="Mattress protector is a fitted sheet under the bed cover. "
                    "Navigate to the most accessible corner.",
        action_type="navigate",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.PLACE_BED_COVER_ON_TABLE],
        next_step=TaskStep.UNCLIP_MATTRESS_PROT_CORNERS,
        preferred_sides=["foot", "left", "right"],
        object_location="mattress",
        vlm_context_hint="Navigate to the nearest accessible corner of the mattress "
                         "protector to begin un-clipping it.",
    ),

    # ── 18 ────────────────────────────────────────────────
    TaskStep.UNCLIP_MATTRESS_PROT_CORNERS: EnhancedStepDefinition(
        step=TaskStep.UNCLIP_MATTRESS_PROT_CORNERS,
        phase=Phase.STRIPPING,
        description="Un-tuck / un-clip all 4 corners of the mattress protector. "
                    "Work around accessible sides only. Skip blocked sides if necessary.",
        action_type="manipulate",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.NAVIGATE_TO_MATTRESS_PROTECTOR],
        next_step=TaskStep.REMOVE_MATTRESS_PROTECTOR,
        preferred_sides=["foot", "left", "right"],
        object_location="mattress",
        vlm_context_hint="Lift each mattress corner to un-clip the protector. "
                         "Only work from accessible sides. Describe which corner to start at.",
    ),

    # ── 19 ────────────────────────────────────────────────
    TaskStep.REMOVE_MATTRESS_PROTECTOR: EnhancedStepDefinition(
        step=TaskStep.REMOVE_MATTRESS_PROTECTOR,
        phase=Phase.STRIPPING,
        description="Grasp and pull the loosened mattress protector off the mattress.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="mattress_protector",
        preconditions=[TaskStep.UNCLIP_MATTRESS_PROT_CORNERS],
        next_step=TaskStep.PLACE_MATTRESS_PROT_ON_TABLE,
        preferred_sides=["foot", "left", "right"],
        object_location="mattress",
        vlm_context_hint="Pull the mattress protector off the mattress completely.",
    ),

    # ── 20 ────────────────────────────────────────────────
    TaskStep.PLACE_MATTRESS_PROT_ON_TABLE: EnhancedStepDefinition(
        step=TaskStep.PLACE_MATTRESS_PROT_ON_TABLE,
        phase=Phase.STRIPPING,
        description="Fold the mattress protector and place it on the table.",
        action_type="place",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.REMOVE_MATTRESS_PROTECTOR],
        next_step=TaskStep.STRIPPING_COMPLETE,
        preferred_sides=["left", "right"],
        object_location="side_table",
        vlm_context_hint="Place the folded mattress protector on the table.",
    ),

    # ── 21 ────────────────────────────────────────────────
    TaskStep.STRIPPING_COMPLETE: EnhancedStepDefinition(
        step=TaskStep.STRIPPING_COMPLETE,
        phase=Phase.TRANSITION,
        description="All bedding removed. Bare mattress visible. "
                    "Transition to making phase.",
        action_type="assess",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PLACE_MATTRESS_PROT_ON_TABLE],
        next_step=TaskStep.FIT_MATTRESS_PROT_HEAD_LEFT,
        preferred_sides=["foot", "left", "right"],
        object_location=None,
        vlm_context_hint="Confirm all bedding is removed. The mattress should be bare.",
    ),

    # ── 22 ────────────────────────────────────────────────
    TaskStep.FIT_MATTRESS_PROT_HEAD_LEFT: EnhancedStepDefinition(
        step=TaskStep.FIT_MATTRESS_PROT_HEAD_LEFT,
        phase=Phase.MAKING,
        description="Fit the HEAD-LEFT corner of the mattress protector under the mattress. "
                    "Approach from LEFT side if accessible, otherwise HEAD.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.STRIPPING_COMPLETE],
        next_step=TaskStep.FIT_MATTRESS_PROT_HEAD_RIGHT,
        preferred_sides=["left", "head"],
        object_location="head_left_corner",
        vlm_context_hint="You are fitting the HEAD-LEFT corner of the mattress protector. "
                         "Lift the mattress corner and tuck the elastic band underneath.",
    ),

    # ── 23 ────────────────────────────────────────────────
    TaskStep.FIT_MATTRESS_PROT_HEAD_RIGHT: EnhancedStepDefinition(
        step=TaskStep.FIT_MATTRESS_PROT_HEAD_RIGHT,
        phase=Phase.MAKING,
        description="Fit the HEAD-RIGHT corner. Approach from RIGHT side if accessible.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.FIT_MATTRESS_PROT_HEAD_LEFT],
        next_step=TaskStep.FIT_MATTRESS_PROT_FOOT_LEFT,
        preferred_sides=["right", "head"],
        object_location="head_right_corner",
        vlm_context_hint="Fit the HEAD-RIGHT corner elastic band under the mattress.",
    ),

    # ── 24 ────────────────────────────────────────────────
    TaskStep.FIT_MATTRESS_PROT_FOOT_LEFT: EnhancedStepDefinition(
        step=TaskStep.FIT_MATTRESS_PROT_FOOT_LEFT,
        phase=Phase.MAKING,
        description="Fit the FOOT-LEFT corner. Approach from LEFT or FOOT side.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.FIT_MATTRESS_PROT_HEAD_RIGHT],
        next_step=TaskStep.FIT_MATTRESS_PROT_FOOT_RIGHT,
        preferred_sides=["left", "foot"],
        object_location="foot_left_corner",
        vlm_context_hint="Fit the FOOT-LEFT corner of the mattress protector. "
                         "Tuck the elastic band under the mattress corner.",
    ),

    # ── 25 ────────────────────────────────────────────────
    TaskStep.FIT_MATTRESS_PROT_FOOT_RIGHT: EnhancedStepDefinition(
        step=TaskStep.FIT_MATTRESS_PROT_FOOT_RIGHT,
        phase=Phase.MAKING,
        description="Fit the FOOT-RIGHT corner. Approach from RIGHT or FOOT side.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object="mattress_protector",
        preconditions=[TaskStep.FIT_MATTRESS_PROT_FOOT_LEFT],
        next_step=TaskStep.SPREAD_BED_COVER,
        preferred_sides=["right", "foot"],
        object_location="foot_right_corner",
        vlm_context_hint="Fit the FOOT-RIGHT corner of the mattress protector.",
    ),

    # ── 26 ────────────────────────────────────────────────
    TaskStep.SPREAD_BED_COVER: EnhancedStepDefinition(
        step=TaskStep.SPREAD_BED_COVER,
        phase=Phase.MAKING,
        description="Unfold and spread the flat bed cover/top sheet evenly over the mattress. "
                    "Align top edge with the headboard, equal overhang left and right.",
        action_type="place",
        needs_accessibility=True,
        needs_object="bed_cover",
        preconditions=[TaskStep.FIT_MATTRESS_PROT_FOOT_RIGHT],
        next_step=TaskStep.TUCK_BED_COVER_LEFT,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="Spread the flat sheet evenly over the bed. "
                         "Describe where the edges are relative to the mattress.",
    ),

    # ── 27 ────────────────────────────────────────────────
    TaskStep.TUCK_BED_COVER_LEFT: EnhancedStepDefinition(
        step=TaskStep.TUCK_BED_COVER_LEFT,
        phase=Phase.MAKING,
        description="Tuck the LEFT edge of the bed cover under the mattress. "
                    "Only possible if LEFT side is accessible.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.SPREAD_BED_COVER],
        next_step=TaskStep.TUCK_BED_COVER_RIGHT,
        preferred_sides=["left"],
        object_location="left_edge",
        vlm_context_hint="Approach from the LEFT side. Lift the mattress edge slightly "
                         "and push the sheet underneath. Describe the motion.",
    ),

    # ── 28 ────────────────────────────────────────────────
    TaskStep.TUCK_BED_COVER_RIGHT: EnhancedStepDefinition(
        step=TaskStep.TUCK_BED_COVER_RIGHT,
        phase=Phase.MAKING,
        description="Tuck the RIGHT edge under the mattress from the RIGHT side.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.TUCK_BED_COVER_LEFT],
        next_step=TaskStep.TUCK_BED_COVER_FOOT,
        preferred_sides=["right"],
        object_location="right_edge",
        vlm_context_hint="Approach from the RIGHT side. Tuck the sheet under the mattress.",
    ),

    # ── 29 ────────────────────────────────────────────────
    TaskStep.TUCK_BED_COVER_FOOT: EnhancedStepDefinition(
        step=TaskStep.TUCK_BED_COVER_FOOT,
        phase=Phase.MAKING,
        description="Tuck the FOOT edge of the bed cover under the mattress. "
                    "Fold hospital corners if accessible.",
        action_type="tuck",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.TUCK_BED_COVER_RIGHT],
        next_step=TaskStep.PUT_DUVET_IN_COVER,
        preferred_sides=["foot"],
        object_location="foot_edge",
        vlm_context_hint="Approach from the FOOT side. Tuck the sheet under the mattress "
                         "at the foot end. Describe the tucking motion.",
    ),

    # ── 30 ────────────────────────────────────────────────
    TaskStep.PUT_DUVET_IN_COVER: EnhancedStepDefinition(
        step=TaskStep.PUT_DUVET_IN_COVER,
        phase=Phase.MAKING,
        description="Insert the duvet insert into the clean duvet cover. "
                    "Align all 4 corners. Shake to distribute evenly.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="duvet",
        preconditions=[TaskStep.TUCK_BED_COVER_FOOT],
        next_step=TaskStep.SPREAD_DUVET,
        preferred_sides=["foot", "left", "right"],
        object_location="in_hand",
        vlm_context_hint="Insert the duvet into its cover. Describe how to align "
                         "the corners and shake to distribute.",
    ),

    # ── 31 ────────────────────────────────────────────────
    TaskStep.SPREAD_DUVET: EnhancedStepDefinition(
        step=TaskStep.SPREAD_DUVET,
        phase=Phase.MAKING,
        description="Lay the dressed duvet over the bed, centred left-right. "
                    "Top edge aligned with the headboard or pillows.",
        action_type="place",
        needs_accessibility=True,
        needs_object="duvet",
        preconditions=[TaskStep.PUT_DUVET_IN_COVER],
        next_step=TaskStep.ALIGN_DUVET_EDGES,
        preferred_sides=["foot", "left", "right"],
        object_location="centre",
        vlm_context_hint="Spread the duvet evenly over the bed. "
                         "Is it centred? Does it reach the headboard?",
    ),

    # ── 32 ────────────────────────────────────────────────
    TaskStep.ALIGN_DUVET_EDGES: EnhancedStepDefinition(
        step=TaskStep.ALIGN_DUVET_EDGES,
        phase=Phase.MAKING,
        description="Smooth and align duvet edges from accessible sides. "
                    "Equal overhang on left and right. Smooth out wrinkles.",
        action_type="manipulate",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.SPREAD_DUVET],
        next_step=TaskStep.PUT_PILLOW_IN_COVER,
        preferred_sides=["left", "right", "foot"],
        object_location="centre",
        vlm_context_hint="Check duvet alignment from the side. "
                         "Describe any adjustments needed to centre and smooth it.",
    ),

    # ── 33 ────────────────────────────────────────────────
    TaskStep.PUT_PILLOW_IN_COVER: EnhancedStepDefinition(
        step=TaskStep.PUT_PILLOW_IN_COVER,
        phase=Phase.MAKING,
        description="Insert bare pillow into clean pillow cover. "
                    "Punch all corners in, close the open end.",
        action_type="manipulate",
        needs_accessibility=False,
        needs_object="pillow",
        preconditions=[TaskStep.ALIGN_DUVET_EDGES],
        next_step=TaskStep.PLACE_PILLOW_ON_BED,
        preferred_sides=["left", "right"],
        object_location="in_hand",
        vlm_context_hint="Insert the pillow into its cover. "
                         "Describe how to push the corners in fully.",
    ),

    # ── 34 ────────────────────────────────────────────────
    TaskStep.PLACE_PILLOW_ON_BED: EnhancedStepDefinition(
        step=TaskStep.PLACE_PILLOW_ON_BED,
        phase=Phase.MAKING,
        description="Place dressed pillow at the HEAD of the bed, centred. "
                    "Approach from LEFT or RIGHT side — not foot (too far).",
        action_type="place",
        needs_accessibility=True,
        needs_object=None,
        preconditions=[TaskStep.PUT_PILLOW_IN_COVER],
        next_step=TaskStep.TASK_COMPLETE,
        preferred_sides=["left", "right"],
        object_location="head",
        vlm_context_hint="Place the pillow at the head of the bed, centred against "
                         "the headboard. You must approach from the LEFT or RIGHT "
                         "side (foot is too far to reach the headboard).",
    ),

    # ── 35 ────────────────────────────────────────────────
    TaskStep.TASK_COMPLETE: EnhancedStepDefinition(
        step=TaskStep.TASK_COMPLETE,
        phase=Phase.MAKING,
        description="Bed fully made. All layers correctly placed and tucked.",
        action_type="assess",
        needs_accessibility=False,
        needs_object=None,
        preconditions=[TaskStep.PLACE_PILLOW_ON_BED],
        next_step=None,
        preferred_sides=[],
        object_location=None,
        vlm_context_hint="Final check: is the bed fully made with pillow, duvet, "
                         "bed cover, and mattress protector all in place?",
    ),
}


# ─────────────────────────────────────────────────────────
# Runtime task state
# ─────────────────────────────────────────────────────────

SIDE_PRIORITY = ["foot", "left", "right", "head"]   # default preference order

# Which sides are physically reachable for each action zone
REACH_RULES = {
    "head":             ["left", "right", "head"],      # head items: no reach from foot
    "foot":             ["left", "right", "foot"],      # foot items: no reach from head
    "centre":           ["foot", "left", "right"],
    "head_left_corner": ["left", "head"],
    "head_right_corner":["right", "head"],
    "foot_left_corner": ["left", "foot"],
    "foot_right_corner":["right", "foot"],
    "left_edge":        ["left"],
    "right_edge":       ["right"],
    "foot_edge":        ["foot"],
    "mattress":         ["foot", "left", "right"],
    "side_table":       ["left", "right"],
    "in_hand":          ["left", "right", "foot"],
    None:               ["foot", "left", "right", "head"],
}

ACCESSIBILITY_SCORE = {
    "free":             3,
    "partially_blocked":1,
    "blocked":          0,
    "unknown":          0,
}


@dataclass
class TaskState:
    current_step:      TaskStep = TaskStep.ANALYSE_ENVIRONMENT
    completed_steps:   List[TaskStep] = field(default_factory=list)
    accessibility:     Dict[str, str] = field(default_factory=dict)
    best_approach_side: Optional[str] = None
    detected_objects:  List[str] = field(default_factory=list)
    phase:             Phase = Phase.INIT
    notes:             List[str] = field(default_factory=list)

    # ── Core navigation ───────────────────────────────────

    def get_best_side_for_current_step(self) -> Optional[str]:
        """
        Returns the best side to approach from for the current step,
        considering both accessibility map AND reach rules.
        """
        defn = self.get_current_definition()
        allowed_by_reach = REACH_RULES.get(defn.object_location, SIDE_PRIORITY)

        # Intersect preferred_sides with reach rules
        candidates = [s for s in defn.preferred_sides if s in allowed_by_reach]
        if not candidates:
            candidates = allowed_by_reach  # fallback

        # Score by accessibility
        scored = []
        for side in candidates:
            acc = self.accessibility.get(side, "unknown")
            score = ACCESSIBILITY_SCORE.get(acc, 0)
            if score > 0:
                scored.append((score, side))

        if scored:
            scored.sort(key=lambda x: -x[0])
            return scored[0][1]

        # No fully accessible side — try partially blocked
        for side in candidates:
            if self.accessibility.get(side) == "partially_blocked":
                return side

        return candidates[0] if candidates else None

    def mark_complete(self, step: TaskStep, note: str = ""):
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        if note:
            self.notes.append(f"[{step.value}] {note}")
        defn = ENHANCED_STEP_REGISTRY[step]
        if defn.next_step:
            self.current_step = defn.next_step
            self.phase = ENHANCED_STEP_REGISTRY[defn.next_step].phase

    def set_accessibility(self, accessibility: Dict[str, str]):
        self.accessibility = accessibility
        for side in SIDE_PRIORITY:
            if accessibility.get(side) == "free":
                self.best_approach_side = side
                return
        for side in SIDE_PRIORITY:
            if accessibility.get(side) == "partially_blocked":
                self.best_approach_side = side
                return

    def is_complete(self) -> bool:
        return self.current_step == TaskStep.TASK_COMPLETE

    def get_current_definition(self) -> EnhancedStepDefinition:
        return ENHANCED_STEP_REGISTRY[self.current_step]

    def to_context_string(self) -> str:
        defn           = self.get_current_definition()
        completed_names= [s.value for s in self.completed_steps]
        acc_str        = ", ".join(f"{k}={v}" for k, v in self.accessibility.items()) or "unknown"
        obj_str        = ", ".join(self.detected_objects) or "none detected"
        best_side      = self.get_best_side_for_current_step()
        total          = len(ENHANCED_STEP_REGISTRY)
        done           = len(self.completed_steps)

        return f"""=== BED-MAKING TASK STATE ===
Current Phase      : {self.phase.value}
Current Step       : {self.current_step.value}  ({done + 1} of {total})
Step Goal          : {defn.description}
Action Type        : {defn.action_type}
Object Location    : {defn.object_location or 'n/a'}

Completed Steps    : {completed_names if completed_names else 'none yet'}

Accessibility      : {acc_str}
Best Side (global) : {self.best_approach_side or 'undetermined'}
Best Side (this step): {best_side or 'undetermined'}
Preferred Sides    : {defn.preferred_sides}
Detected Objects   : {obj_str}

VLM Context Hint   : {defn.vlm_context_hint}
Needs Side Info    : {'YES' if defn.needs_accessibility else 'NO'}
Target Object      : {defn.needs_object or 'none required'}
==========================="""


# ─────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    state = TaskState()
    state.set_accessibility({
        "head":  "blocked",
        "foot":  "free",
        "left":  "free",
        "right": "partially_blocked",
    })

    print("Enhanced State Machine — walkthrough\n")
    print(f"Total steps: {len(ENHANCED_STEP_REGISTRY)}")
    print(f"Initial step: {state.current_step.value}\n")

    for i in range(6):
        defn     = state.get_current_definition()
        best     = state.get_best_side_for_current_step()
        print(f"  Step {i+1:02d}: {state.current_step.value}")
        print(f"         action={defn.action_type}  "
              f"object_location={defn.object_location}  "
              f"best_side={best}")
        state.mark_complete(state.current_step, note="test")

    print(f"\n  ... fast-forward ...")
    print(f"\nContext string preview:\n")
    print(state.to_context_string())
