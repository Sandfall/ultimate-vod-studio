"""Fine-grained m_char classifier + a cascade that refines the character detector's output.

The detector recognizes DEFAULT-outfit m_chars directly (one class per m_char fighter type), but
CUSTOM m_chars (per-player faces/outfits) are unbounded and can't be a fixed class, and the
detector can also false-positive its m_char classes. So a 4-class portrait classifier (the three
m_char types + not_m_char) stays the authority on m_char type.

`refine_characters` does three things, all via that classifier:
  * a tile the detector FOUND (under any label) is reclassified: a confident m_char verdict sets/keeps
    its m_char type (recovers a custom m_char the detector missed or mislabeled as another fighter).
  * VETO: if the detector called a tile one of its m_char classes but the classifier confidently says
    not_m_char, the unverified m_char call is dropped (None): per-side voting decides instead.
  * a side the detector MISSED (None) is recovered by cropping the FIXED HUD box and classifying it,
    but only when the frame is anchored as a standard-layout in-match frame (at least one side was
    detected near its expected portrait position). The anchor guard is what stops us from
    hallucinating m_chars onto menus, replays, or repositioned-overlay feeds.

When the classifier is ABSENT, `refine_characters` is a no-op, so the detector's default-m_char classes
are trusted directly. A classifier verdict only overrides when its confidence clears a floor
(hallucination was 0% at plain argmax in CV, but the floor protects against out-of-distribution
tiles). The model is passed in by the caller (loaded once), like `characters.detect_characters`.
"""

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from vodparse.bbox import crop_frame
from vodparse.schema import BBox, CharacterDetection, SideCharacters

# Standard full-screen HUD portrait boxes, measured at 1920x1080 as [x1, y1, x2, y2]; scaled by
# w/1920 for other 16:9 resolutions. Each box's center x is the expected portrait position used by
# the standard-layout anchor check. (Same geometry the harvester uses for fixed-box extraction.)
FIXED_BOX_1080: dict[str, BBox] = {"left": [381, 884, 549, 1059],
                                   "right": [1120, 884, 1288, 1055]}
ANCHOR_TOL_1080 = 70  # a detection anchors the layout if its center x is within this of the box's

# The classifier's negative verdict: "this tile is not an m_char". Checkpoints have always
# named the negative class with a "not_" prefix, but the exact spelling has changed across
# dataset generations, so the loader normalizes any such label to this constant and all
# downstream logic (the positive/negative gate and the veto in _refine_side) keys on it
# structurally. No positive class ever uses the prefix.
NOT_M_CHAR_LABEL = "not_m_char"
_NEGATIVE_LABEL_PREFIX = "not_"

class _Classifier(Protocol):
    """Anything that maps a BGR crop to (label, confidence) and knows the pipeline character
    name for each of its m_char labels. `display_names` values must match the character
    detector's own names for those classes: _refine_side's veto/override checks compare
    detector names against them (see load_m_char_classifier on how the map travels in the
    checkpoint). Lets the cascade be unit-tested with a stub and keeps torch out of the
    import path for callers that don't load the model."""
    display_names: dict[str, str]
    def classify(self, crop_bgr: Any) -> tuple[str, float]: ...


class MCharClassifier:
    """A loaded 4-class portrait classifier. `classify(crop_bgr) -> (label, confidence)`;
    `display_names` maps each m_char label to the character name the pipeline reports."""

    def __init__(self, model: Any, classes: list[str], img_size: int, device: str,
                 display_names: dict[str, str]) -> None:
        import torchvision.transforms as T
        self.model = model
        self.classes = classes
        self.img_size = img_size
        self.device = device
        self.display_names = display_names
        self._tf = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def classify(self, crop_bgr: Any) -> tuple[str, float]:
        import torch
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return NOT_M_CHAR_LABEL, 0.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            x = self._tf(rgb).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)[0]
            i = int(probs.argmax())
            return self.classes[i], float(probs[i])


def load_m_char_classifier(path: str, device: str = "cpu") -> MCharClassifier | None:
    """Load the saved 4-class classifier, or None if the weights aren't present (so the pipeline
    runs unchanged without them, same as the optional timer/character models).

    The negative-class label is normalized to NOT_M_CHAR_LABEL (older checkpoints spell it
    differently), so a confident rejection can never be mistaken for a character name.
    ``display_names`` is not just presentation: _refine_side's veto/override checks compare
    the character detector's names against its values, so the two models' names must agree.
    The full names ride in the checkpoint (written at release packaging, or carried forward
    from the previous checkpoint by the trainer); a checkpoint without them still loads,
    reporting raw labels, but those checks degrade — hence the printed note."""
    if not Path(path).exists():
        return None
    import torch
    import torch.nn as nn
    import torchvision.models as M
    ckpt = torch.load(path, map_location=device)
    # Positions must stay put: classify() indexes the net's output by position in this list.
    classes: list[str] = [NOT_M_CHAR_LABEL if c.startswith(_NEGATIVE_LABEL_PREFIX) else c
                          for c in ckpt["classes"]]
    # m_char label -> the character name the pipeline reports (labels are the abbreviated
    # dataset-folder names). The fallback deliberately excludes the negative class: it must
    # never gain a display name, or _refine_side would report a rejection as a character.
    display_names: dict[str, str] | None = ckpt.get("display_names")
    if not display_names:
        display_names = {c: c for c in classes if c != NOT_M_CHAR_LABEL}
        print(f"  note: {path} carries no display_names metadata (fresh retrain?): m_char "
              f"types are reported under their raw labels, and the cascade checks that "
              f"compare the character detector's names against them are degraded. "
              f"Retraining with the bundled trainer carries the names forward "
              f"automatically when the previous checkpoint is present.")
    net = getattr(M, ckpt.get("arch", "resnet18"))(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)
    return MCharClassifier(net, classes, int(ckpt.get("img_size", 224)), device, display_names)


def _scale_box(box: BBox, scale: float) -> BBox:
    return [int(round(c * scale)) for c in box]


def _is_standard(entry: CharacterDetection | None, side: str, scale: float) -> bool:
    """True if this detection sits near its expected standard-layout portrait position."""
    if entry is None:
        return False
    box = FIXED_BOX_1080[side]
    cx = (entry["bbox"][0] + entry["bbox"][2]) / 2
    center = (box[0] + box[2]) / 2 * scale
    return abs(cx - center) <= ANCHOR_TOL_1080 * scale


def _refine_side(entry: CharacterDetection | None, side: str, frame: np.ndarray, scale: float,
                 anchored: bool, clf: _Classifier, conf_floor: float,
                 override_floor: float) -> CharacterDetection | None:
    """Reclassify a detected tile (or recover a missed side via the fixed box when anchored),
    and veto the detector's own m_char false-positives the classifier confidently rejects."""
    if entry is not None:
        box = entry["bbox"]
    elif anchored:
        box = _scale_box(FIXED_BOX_1080[side], scale)
    else:
        return None  # missed side on a non-anchored frame: don't fabricate a tile
    label, conf = clf.classify(crop_frame(frame, box))
    if label != NOT_M_CHAR_LABEL and conf >= conf_floor:
        # Flipping a tile the detector confidently called a NON-m_char fighter (a grey blocky
        # lookalike, say) into an m_char is the cascade's riskiest move: a single confident
        # classifier miss hallucinates an m_char over a correct detection. Demand a higher bar there:
        # clear override_floor AND out-confidence the detector. The cheap paths keep conf_floor:
        # refining the detector's OWN m_char call, or recovering a side it MISSED (entry is None),
        # where there's no confident fighter label to protect. (The two confidences come from
        # different models, so the > comparison is a heuristic guard, not a calibrated one.)
        # "Did the detector call this tile an m_char?" is answered by comparing its name against
        # display_names.values(): the two models' names agree per the loader's contract. With a
        # metadata-less checkpoint (raw-label fallback) they can't match, so every detected tile
        # takes this risky-path bar — degraded but conservative, and warned about at load.
        if entry is not None and entry["name"] not in clf.display_names.values():
            if not (conf >= override_floor and conf > entry["confidence"]):
                return entry  # the detector's confident non-m_char stands
        return {"name": clf.display_names.get(label, label),
                "confidence": round(conf, 3), "bbox": box}
    # The detector now has its own m_char classes and can false-positive them. If it called this
    # tile an m_char but the classifier confidently disagrees (not_m_char), drop the unverified call.
    if entry is not None and entry["name"] in clf.display_names.values() \
            and label == NOT_M_CHAR_LABEL and conf >= conf_floor:
        return None
    return entry  # not_m_char on a non-m_char detection, or low-confidence: keep the tile / stay None


def refine_characters(sides: SideCharacters | None, frame: np.ndarray, clf: _Classifier | None,
                      conf_floor: float = 0.6,
                      override_floor: float = 0.9) -> SideCharacters | None:
    """Refine a `detect_characters` result with the m_char classifier. No-op when the classifier is
    absent or nothing was detected. Returns a new SideCharacters; the input is not mutated.

    `conf_floor` gates the cheap reclassifications (the detector's own m_char call, a missed side);
    `override_floor` (higher) gates the one risky case: overruling a tile the detector confidently
    named as a non-m_char fighter: so a modest m_char verdict can't hallucinate over a correct
    fighter detection."""
    if clf is None or sides is None:
        return sides
    scale = frame.shape[1] / 1920.0
    anchored = (_is_standard(sides["left"], "left", scale)
                or _is_standard(sides["right"], "right", scale))
    return {
        "left": _refine_side(sides["left"], "left", frame, scale, anchored, clf,
                             conf_floor, override_floor),
        "right": _refine_side(sides["right"], "right", frame, scale, anchored, clf,
                              conf_floor, override_floor),
    }
