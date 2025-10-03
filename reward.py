import json
import re
import math
from typing import Any, Dict, List, Optional, Tuple

# ==================== Configuration ====================
DECISION_THRESHOLD = 0.85

# Smooth band (tolerance) — keep simple and target-centric
REL_TOL = 0.08            # ±8% around the target confidence
MIN_ABS_HALF_BAND = 0.03  # ensures the band isn't razor-thin for small targets

# Flat-top shape (two sigmoids product, normalized at the center)
SOFT_K = 0.04             # edge softness (higher = softer shoulders)
SOFT_GAMMA = 0.55         # <1 = flatter/wider top, >1 = sharper peak

CONF_DECAY_SCALE = 0.35   # exponential decay scale for confidence differences
CONF_PENALTY_WEIGHT = 0.05  # penalty weight when falling below decision threshold

# ==================== JSON Extraction ====================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from text (handles reasoning preamble)."""
    if not text:
        return None
    if isinstance(text, dict):
        return text
    text = text.strip()

    # Direct parse if it's clean JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find last valid JSON object in the text
    json_objects: List[Dict[str, Any]] = []
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                json_objects.append(obj)
        except Exception:
            continue
    if json_objects:
        return json_objects[-1]

    # Fallback: JSONDecoder from any '{' position
    decoder = json.JSONDecoder()
    positions = [m.start() for m in re.finditer(r'\{', text)]
    for pos in reversed(positions):
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                expected = [
                    "chapters", "selected_indices", "option_number",
                    "question_type", "updated_description", "suggested_code"
                ]
                if any(f in obj for f in expected):
                    return obj
        except Exception:
            continue
    return None

# ==================== Smooth Band + Soft-Top ====================

def _band_half(target_conf: float) -> float:
    """Half-width of the tolerance band."""
    return max(MIN_ABS_HALF_BAND, REL_TOL * max(1e-6, target_conf))

def calculate_confidence_band(target_conf: float, threshold: float) -> Tuple[float, float]:
    """
    Symmetric band centered at the target, clamped to [0,1].
    Keep purely target-centric (no threshold clamp) for simplicity/smoothness.
    """
    h = _band_half(target_conf)
    low = max(0.0, target_conf - h)
    high = min(1.0, target_conf + h)
    return low, high

def _sigmoid(x: float) -> float:
    """Numerically safe logistic."""
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def soft_top_score(pred_conf: float, low: float, high: float,
                   k: float = SOFT_K, gamma: float = SOFT_GAMMA) -> float:
    """
    Smooth 'flat-top' bump using product of sigmoids, normalized to 1 at band midpoint.
      score = ((σ((x-low)/k) * σ((high-x)/k))^gamma) / center_norm
    - Perfectly smooth (C∞), no kinks.
    - Wider/flatter top via smaller gamma (e.g., 0.55).
    - Softer edges via larger k (e.g., 0.04).
    """
    if not (0.0 <= low < high <= 1.0):
        return 0.0

    x = pred_conf
    s_left  = _sigmoid((x - low)  / max(1e-6, k))
    s_right = _sigmoid((high - x) / max(1e-6, k))
    bump = (s_left * s_right) ** max(1e-6, gamma)

    # Normalize to 1.0 at the midpoint
    mid = 0.5 * (low + high)
    sL_mid  = _sigmoid((mid - low)  / max(1e-6, k))
    sR_mid  = _sigmoid((high - mid) / max(1e-6, k))
    center = (sL_mid * sR_mid) ** max(1e-6, gamma)
    if center <= 0:
        return 0.0

    return max(0.0, min(1.0, bump / center))

# ==================== Task Evaluators ====================

def evaluate_select_chapters(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    tgt_list = targets.get("chapters", [])
    if not isinstance(tgt_list, list) or not tgt_list:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing targets.chapters"}

    gt_chapter = str(tgt_list[0].get("chapter", "")).strip()
    if not gt_chapter:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid gold chapter"}

    gold_conf_map: Dict[str, float] = {}
    for i, e in enumerate(tgt_list):
        ch = str(e.get("chapter", "")).strip()
        conf = e.get("confidence")
        if not ch or not isinstance(conf, (int, float)):
            return {"score": 0.0, "is_score_valid": False, "reason": f"Invalid targets.chapters entry at {i}"}
        gold_conf_map[ch] = float(conf)

    pred_list = assistant_data.get("chapters", [])
    if not isinstance(pred_list, list) or not pred_list:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'chapters' in assistant response"}

    pred_pairs: List[Tuple[str, float]] = []
    for i, e in enumerate(pred_list):
        ch = str(e.get("chapter", "")).strip()
        conf = e.get("confidence")
        if not ch or not isinstance(conf, (int, float)):
            return {"score": 0.0, "is_score_valid": False, "reason": f"Invalid assistant chapters entry at {i}"}
        pred_pairs.append((ch, float(conf)))

    pred_order = [c for c, _ in pred_pairs]
    if gt_chapter not in pred_order:
        # tiny soft floor if distribution shape roughly matches
        gold = gold_conf_map
        pred = dict(pred_pairs)
        diffs = [1.0 / (1.0 + abs(gold.get(k, 0.0) - pred.get(k, 0.0))) - 0.5 for k in gold]
        soft = 0.01 * (sum(diffs) / max(1, len(diffs)))
        return {"score": float(soft), "is_score_valid": True, "reason": f"GT {gt_chapter} not in predictions"}

    ranked = sorted(pred_pairs, key=lambda x: -x[1])
    labels = [c for c, _ in ranked]
    rank = labels.index(gt_chapter) + 1

    pred_conf_for_gt = dict(pred_pairs).get(gt_chapter, 0.0)
    target_conf_for_gt = gold_conf_map.get(gt_chapter, 0.0)

    low, high = calculate_confidence_band(target_conf_for_gt, threshold)
    soft = soft_top_score(pred_conf_for_gt, low, high)

    top1 = 1.0 if rank == 1 else 0.0
    recip_rank = 1.0 / rank

    # Simple, smooth, and stable: small weight on soft-top, mostly rank signals
    score = min(1.0, 0.75 * top1 + 0.15 * recip_rank + 0.10 * soft)

    return {
        "score": float(score),
        "is_score_valid": True,
        "reason": f"GT {gt_chapter} rank #{rank}; pred_conf={pred_conf_for_gt:.2f}, target_conf={target_conf_for_gt:.2f}, band=[{low:.2f},{high:.2f}]"
    }

def evaluate_select_candidates(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    gold_sel = targets.get("selected_indices", [])
    if not isinstance(gold_sel, list) or not gold_sel:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing targets.selected_indices"}

    try:
        gold_sel = [int(x) for x in gold_sel]
    except Exception:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid gold indices"}

    pred_sel = assistant_data.get("selected_indices", [])
    if not isinstance(pred_sel, list) or not pred_sel:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'selected_indices' in assistant response"}

    try:
        pred_sel = [int(x) for x in pred_sel]
    except Exception:
        return {"score": 0.0, "is_score_valid": False, "reason": "Non-integer index in assistant selected_indices"}

    gold_set = set(gold_sel)
    pred_set = set(pred_sel)
    correct = gold_set & pred_set
    coverage = len(correct) / max(1, len(gold_set))
    precision = len(correct) / max(1, len(pred_set))
    length_penalty = 1.0 / (1.0 + abs(len(pred_sel) - len(gold_sel)))

    if not correct:
        score = 0.0
        reason = f"No overlap with gold indices {gold_sel}"
    elif gold_set == pred_set:
        order_matches = sum(1 for i, val in enumerate(pred_sel[:len(gold_sel)]) if val == gold_sel[i])
        order_score = order_matches / len(gold_sel)
        score = min(1.0, 0.6 + 0.4 * order_score)
        reason = "Matched all indices"
        if order_matches != len(gold_sel):
            reason += f"; order alignment {order_score:.2f}"
    else:
        base = 0.45 * coverage + 0.35 * precision + 0.20 * length_penalty
        score = min(1.0, max(0.05, base))
        reason = (
            f"Partial overlap {sorted(correct)} with gold {gold_sel}; "
            f"precision={precision:.2f}, coverage={coverage:.2f}, length_penalty={length_penalty:.2f}"
        )

    return {"score": float(max(0.0, min(1.0, score))), "is_score_valid": True, "reason": reason}

def evaluate_score_candidate(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    cand = user_data.get("data", {}).get("candidate", {})
    cand_index = cand.get("index")
    if cand_index is None:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing candidate index"}

    try:
        gold_option = int(targets.get("option_number"))
        gold_conf = float(targets.get("confidence"))
    except Exception:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid targets"}

    option_num = assistant_data.get("option_number")
    pred_conf = assistant_data.get("confidence")
    if option_num is None or pred_conf is None:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing option_number or confidence"}

    try:
        option_num = int(option_num)
        pred_conf = float(pred_conf)
    except Exception:
        return {"score": 0.0, "is_score_valid": False, "reason": "Non-numeric values"}

    if option_num != cand_index:
        return {"score": 0.0, "is_score_valid": True, "reason": f"Wrong candidate: {option_num} != {cand_index}"}
    if option_num != gold_option:
        return {"score": 0.0, "is_score_valid": True, "reason": f"Index mismatch: {option_num} != {gold_option}"}

    diff = abs(pred_conf - gold_conf)
    decay = math.exp(-diff / max(CONF_DECAY_SCALE, 1e-6))
    band_low, band_high = calculate_confidence_band(gold_conf, threshold)
    in_band = band_low <= pred_conf <= band_high
    band_bonus = 0.1 if in_band else 0.0

    under_threshold_penalty = 0.0
    if gold_conf >= threshold and pred_conf < threshold:
        under_threshold_penalty = CONF_PENALTY_WEIGHT * max(0.0, threshold - pred_conf)

    score = decay - under_threshold_penalty + band_bonus
    score = max(0.01, min(1.0, score))

    reason_parts = [
        f"pred_conf={pred_conf:.2f}",
        f"target_conf={gold_conf:.2f}",
        f"band=[{band_low:.2f},{band_high:.2f}]",
        f"|diff|={diff:.2f}",
    ]
    if in_band:
        reason_parts.append("within band")
    if under_threshold_penalty > 0:
        reason_parts.append(f"under-threshold penalty={under_threshold_penalty:.2f}")

    return {
        "score": float(score),
        "is_score_valid": True,
        "reason": "; ".join(reason_parts),
    }

# ==================== Main Entry Point ====================

def evaluate(user_data: Dict, answer: str, targets: Dict) -> Dict:
    """Evaluate responses across tasks with a smooth, wide-top reward."""
    try:
        assistant_data = extract_json_from_text(answer)
        if not user_data:
            return {"score": 0.0, "is_score_valid": False, "reason": "Failed to parse user JSON"}
        if not assistant_data:
            return {"score": 0.0, "is_score_valid": False, "reason": "No assistant JSON to evaluate"}

        task = user_data.get("task")
        if not task:
            return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'task' in user data"}

        threshold = float(user_data.get("data", {}).get("confidence_threshold", DECISION_THRESHOLD))

        if task == "select_chapters":
            result = evaluate_select_chapters(user_data, assistant_data, targets, threshold)
        elif task == "select_candidates":
            result = evaluate_select_candidates(user_data, assistant_data, targets, threshold)
        elif task == "score_candidate":
            result = evaluate_score_candidate(user_data, assistant_data, targets, threshold)
        else:
            result = {"score": 0.0, "is_score_valid": False, "reason": f"Unknown task: {task}"}

        return {
            "score": float(result.get("score", 0.0)),
            "is_score_valid": bool(result.get("is_score_valid", False)),
            "reason": result.get("reason", "Unknown error")
        }

    except Exception as e:
        return {"score": 0.0, "is_score_valid": False, "reason": f"Error: {str(e)}"}
