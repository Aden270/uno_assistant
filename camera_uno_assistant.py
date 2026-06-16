from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
HUIZONG_DIR = PROJECT_ROOT / "huizong"
if str(HUIZONG_DIR) not in sys.path:
    sys.path.insert(0, str(HUIZONG_DIR))

from card import Card, Color, CardType
from game_state import GameState
from label_mapper import class_name_to_card
from rules import decide, format_hand, legal_plays


DEFAULT_DETECT_MODEL = ROOT / "detect_best.pt"
DEFAULT_CLASSIFY_MODEL = ROOT / "cls_best.pt"
DEFAULT_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


@dataclass
class FramePrediction:
    card: Card | None
    class_name: str | None
    detect_conf: float
    card_conf: float
    box: tuple[int, int, int, int] | None
    top2_conf: float = 0.0
    confidence_margin: float = 0.0
    is_stable: bool = False


@dataclass
class DisplayState:
    session_title: str
    phase: str
    turn_text: str
    top_text: str
    color_text: str
    penalty_text: str
    hand_text: str
    stable_text: str
    candidate_text: str
    stability_text: str
    recommendation_text: str
    pending_text: str
    message_text: str
    game_over: bool = False
    winner_text: str = ""


def play_stable_beep() -> None:
    def _worker() -> None:
        try:
            import winsound

            winsound.Beep(1180, 120)
            winsound.Beep(1480, 90)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop UNO camera assistant using detect+classify.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--classify-model", type=Path, default=DEFAULT_CLASSIFY_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--detect-conf", type=float, default=0.35)
    parser.add_argument("--classify-conf", type=float, default=0.62)
    parser.add_argument("--infer-every", type=int, default=4, help="Run recognition every N frames.")
    parser.add_argument("--initial-hand-size", type=int, default=5)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--my-index", type=int, default=0)
    parser.add_argument("--starting-player", type=int, default=0, help="Seat index of the first player to act after setup.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional path for annotated output video.")
    parser.add_argument("--display-backend", choices=["auto", "cv2", "tk"], default="tk")
    parser.add_argument("--min-margin", type=float, default=0.16, help="Require top1-top2 confidence gap at least this value.")
    parser.add_argument("--crop-padding", type=float, default=0.12, help="Expand detected box before classification.")
    parser.add_argument("--stable-frames", type=int, default=4, help="Frames needed to lock the first stable card.")
    parser.add_argument("--switch-frames", type=int, default=4, help="Frames needed to switch from one stable card to another.")
    parser.add_argument("--max-missed-frames", type=int, default=6, help="How long to keep the stable result when frames are uncertain.")
    parser.add_argument("--color-prior-strength", type=float, default=0.22, help="Boost same-color classes using a simple color prior.")
    parser.add_argument("--no-tta-rot180", dest="tta_rot180", action="store_false", help="Disable 180-degree flip test-time augmentation.")
    parser.add_argument("--one-four-margin", type=float, default=0.28, help="If same-color 1 and 4 are very close, force the result toward 1.")
    parser.add_argument("--red-one-two-margin", type=float, default=0.2, help="If red_1 and red_2 are very close, force the result toward red_1.")
    parser.add_argument("--blue-five-three-margin", type=float, default=1.2, help="If blue_5 and blue_3 are very close, force the result toward blue_5.")
    parser.add_argument("--blue-five-rescue-margin", type=float, default=0.68, help="If blue_3 leads blue_5 by only a small amount, actively rescue blue_5.")
    parser.set_defaults(tta_rot180=True)
    return parser


def build_player_labels(player_count: int, machine_index: int) -> list[str]:
    labels: list[str] = []
    human_counter = 1
    for index in range(player_count):
        if index == machine_index:
            labels.append("机器")
        else:
            labels.append(f"玩家{human_counter}")
            human_counter += 1
    return labels


def get_player_label(player_labels: list[str], seat_index: int) -> str:
    if 0 <= seat_index < len(player_labels):
        return player_labels[seat_index]
    return f"座位{seat_index}"


def get_other_count_index(state: GameState, seat_index: int) -> int | None:
    if seat_index == state.my_index:
        return None
    if seat_index < state.my_index:
        return seat_index
    return seat_index - 1


def set_initial_other_hand_counts(state: GameState, initial_hand_size: int) -> None:
    state.other_hand_counts = [max(0, initial_hand_size)] * max(0, state.player_count - 1)


def change_other_hand_count(state: GameState, seat_index: int, delta: int) -> None:
    other_index = get_other_count_index(state, seat_index)
    if other_index is None:
        return
    if other_index >= len(state.other_hand_counts):
        missing = other_index + 1 - len(state.other_hand_counts)
        state.other_hand_counts.extend([0] * missing)
    state.other_hand_counts[other_index] = max(0, state.other_hand_counts[other_index] + delta)


def format_ai_recommendation(hand: list[Card], decision) -> str:
    if decision.action == "draw":
        return "摸牌"
    if decision.card_index is None or not (0 <= decision.card_index < len(hand)):
        return "出牌"
    card = hand[decision.card_index]
    if decision.chosen_color:
        return f"出 {card}，并选{decision.chosen_color.value}色"
    return f"出 {card}"


def can_stack_penalty(card: Card, top: Card | None) -> bool:
    if top is None:
        return False
    if top.type == CardType.WILD_DRAW_FOUR:
        return card.type in (CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR)
    if top.type == CardType.DRAW_TWO:
        return card.type == CardType.DRAW_TWO
    return False


def get_winner_text(state: GameState, player_labels: list[str], my_hand: list[Card], phase: str) -> str | None:
    if phase != "对局":
        return None
    if len(my_hand) == 0:
        return f"{get_player_label(player_labels, state.my_index)} 获胜"
    for seat_index in range(state.player_count):
        if seat_index == state.my_index:
            continue
        other_index = get_other_count_index(state, seat_index)
        if other_index is not None and 0 <= other_index < len(state.other_hand_counts):
            if state.other_hand_counts[other_index] == 0:
                return f"{get_player_label(player_labels, seat_index)} 获胜"
    return None


def set_ultralytics_config_dir() -> None:
    config_dir = Path(tempfile.gettempdir()) / "uno_assistant_ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in DEFAULT_FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def put_text(frame: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int], size: int = 18) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=load_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    if padding_ratio <= 0:
        return box
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * padding_ratio))
    pad_y = int(round((y2 - y1) * padding_ratio))
    return clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width=width, height=height)


class TwoStageRecognizer:
    def __init__(
        self,
        detect_model: Path,
        classify_model: Path,
        device: str,
        imgsz: int,
        detect_conf: float,
        classify_conf: float,
        min_margin: float,
        crop_padding: float,
        color_prior_strength: float,
        tta_rot180: bool,
        one_four_margin: float,
        red_one_two_margin: float,
        blue_five_three_margin: float,
        blue_five_rescue_margin: float,
    ):
        set_ultralytics_config_dir()
        from ultralytics import YOLO

        self.detector = YOLO(str(detect_model))
        self.classifier = YOLO(str(classify_model))
        self.device = device
        self.imgsz = imgsz
        self.detect_conf = detect_conf
        self.classify_conf = classify_conf
        self.min_margin = min_margin
        self.crop_padding = crop_padding
        self.color_prior_strength = color_prior_strength
        self.tta_rot180 = tta_rot180
        self.one_four_margin = one_four_margin
        self.red_one_two_margin = red_one_two_margin
        self.blue_five_three_margin = blue_five_three_margin
        self.blue_five_rescue_margin = blue_five_rescue_margin

    @staticmethod
    def _extract_probs(result) -> np.ndarray:
        prob_values = result.probs.data
        if hasattr(prob_values, "detach"):
            return prob_values.detach().float().cpu().numpy().astype(np.float32)
        return np.asarray(prob_values, dtype=np.float32)

    @staticmethod
    def _class_name_list(names) -> list[str]:
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        return list(names)

    @staticmethod
    def _class_index_map(class_names: list[str]) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(class_names)}

    @staticmethod
    def _color_prefix(class_name: str) -> str:
        if class_name.startswith("wild"):
            return "black"
        return class_name.split("_", 1)[0]

    @staticmethod
    def _estimate_color_hint(crop: np.ndarray) -> tuple[str | None, float]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        color_mask = (s >= 55) & (v >= 55)
        if int(color_mask.sum()) == 0:
            dark_ratio = float((v < 60).mean())
            if dark_ratio >= 0.45:
                return "black", dark_ratio
            return None, 0.0

        counts = {
            "red": int((((h <= 10) | (h >= 165)) & color_mask).sum()),
            "yellow": int((((h >= 15) & (h <= 40)) & color_mask).sum()),
            "green": int((((h >= 41) & (h <= 90)) & color_mask).sum()),
            "blue": int((((h >= 91) & (h <= 140)) & color_mask).sum()),
        }
        best_color = max(counts, key=counts.get)
        total = max(1, int(color_mask.sum()))
        confidence = counts[best_color] / total

        dark_ratio = float((v < 50).mean())
        if dark_ratio >= 0.52 and confidence < 0.28:
            return "black", dark_ratio
        if confidence < 0.18:
            return None, confidence
        return best_color, confidence

    def _apply_color_prior(self, probs: np.ndarray, class_names: list[str], color_hint: str | None, confidence: float) -> np.ndarray:
        if color_hint is None or confidence <= 0.0 or self.color_prior_strength <= 0.0:
            return probs

        adjusted = probs.copy()
        same_color_gain = 1.0 + self.color_prior_strength * min(1.0, confidence / 0.45)
        other_color_penalty = 1.0 - self.color_prior_strength * 0.85 * min(1.0, confidence / 0.45)
        black_penalty = 1.0 - self.color_prior_strength * 1.05 * min(1.0, confidence / 0.45)
        black_gain = 1.0 + self.color_prior_strength * min(1.0, confidence / 0.55)

        for idx, class_name in enumerate(class_names):
            prefix = self._color_prefix(class_name)
            if color_hint == "black":
                adjusted[idx] *= black_gain if prefix == "black" else other_color_penalty
            else:
                if prefix == color_hint:
                    adjusted[idx] *= same_color_gain
                elif prefix == "black":
                    adjusted[idx] *= black_penalty
                else:
                    adjusted[idx] *= other_color_penalty

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _apply_green_rotation_preference(
        fused_probs: np.ndarray,
        base_probs: np.ndarray,
        rot_probs: np.ndarray | None,
        class_names: list[str],
    ) -> np.ndarray:
        if rot_probs is None or rot_probs.shape != fused_probs.shape:
            return fused_probs

        index_map = TwoStageRecognizer._class_index_map(class_names)
        idx_green1 = index_map.get("green_1")
        idx_green4 = index_map.get("green_4")
        if idx_green1 is None or idx_green4 is None:
            return fused_probs

        # If the rotated view gives green_1 materially stronger support than the
        # original view, lean into it. This matches the observed real-camera behavior.
        green1_gain = float(rot_probs[idx_green1] - base_probs[idx_green1])
        green4_gain = float(rot_probs[idx_green4] - base_probs[idx_green4])
        if green1_gain <= 0.02 and rot_probs[idx_green1] <= fused_probs[idx_green1]:
            return fused_probs

        adjusted = fused_probs.copy()
        adjusted[idx_green1] = max(
            adjusted[idx_green1],
            0.18 * base_probs[idx_green1] + 0.82 * rot_probs[idx_green1],
            rot_probs[idx_green1] * 1.18,
        )
        adjusted[idx_green4] = min(
            adjusted[idx_green4],
            0.5 * base_probs[idx_green4] + 0.5 * rot_probs[idx_green4],
        )

        # Only push when green_1 is not clearly worse than green_4 after rotation.
        if rot_probs[idx_green1] + 0.01 < rot_probs[idx_green4] and green4_gain > green1_gain:
            return fused_probs

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _apply_green_close_case_guard(
        probs: np.ndarray,
        base_probs: np.ndarray,
        rot_probs: np.ndarray | None,
        class_names: list[str],
    ) -> np.ndarray:
        if rot_probs is None or rot_probs.shape != probs.shape:
            return probs

        index_map = TwoStageRecognizer._class_index_map(class_names)
        idx_green1 = index_map.get("green_1")
        idx_green4 = index_map.get("green_4")
        if idx_green1 is None or idx_green4 is None:
            return probs

        fused_gap = float(probs[idx_green1] - probs[idx_green4])
        rot_gap = float(rot_probs[idx_green1] - rot_probs[idx_green4])
        base_gap = float(base_probs[idx_green1] - base_probs[idx_green4])

        # Only intervene in the exact troublesome zone:
        # fused result is still close or slightly losing, but rotated view leans to green_1.
        if rot_gap <= 0.015:
            return probs
        if fused_gap >= 0.08:
            return probs
        if fused_gap <= -0.22 and rot_gap < 0.08:
            return probs

        adjusted = probs.copy()
        green_strength = max(0.0, rot_gap) + max(0.0, rot_probs[idx_green1] - base_probs[idx_green1])
        if green_strength < 0.03:
            return probs

        adjusted[idx_green1] *= 1.22 + min(0.32, green_strength * 1.7)
        adjusted[idx_green4] *= 0.84 - min(0.18, max(0.0, rot_gap) * 0.9)

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _bias_close_one_vs_four_to_one(
        probs: np.ndarray,
        class_names: list[str],
        top_indices: np.ndarray,
        trigger_margin: float,
    ) -> np.ndarray:
        if len(top_indices) < 2:
            return probs

        first_name = class_names[int(top_indices[0])]
        second_name = class_names[int(top_indices[1])]
        first_parts = first_name.split("_")
        second_parts = second_name.split("_")
        if len(first_parts) != 2 or len(second_parts) != 2:
            return probs
        if first_parts[0] != second_parts[0]:
            return probs
        if {first_parts[1], second_parts[1]} != {"1", "4"}:
            return probs

        color_name = first_parts[0]
        color_margin = trigger_margin
        boost_one = 1.42
        suppress_four = 0.68
        if color_name == "green":
            color_margin = max(trigger_margin, 0.8)
            boost_one = 2.9
            suppress_four = 0.14
        elif color_name == "red":
            color_margin = max(trigger_margin, 0.36)
            boost_one = 1.68
            suppress_four = 0.54
        elif color_name in {"yellow", "blue"}:
            color_margin = min(trigger_margin, 0.24)
            boost_one = 1.28
            suppress_four = 0.74

        conf_gap = abs(float(probs[int(top_indices[0])]) - float(probs[int(top_indices[1])]))
        if conf_gap > color_margin:
            return probs

        adjusted = probs.copy()
        idx1 = int(top_indices[0]) if first_parts[1] == "1" else int(top_indices[1])
        idx4 = int(top_indices[0]) if first_parts[1] == "4" else int(top_indices[1])
        adjusted[idx1] *= boost_one
        adjusted[idx4] *= suppress_four

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _bias_close_red_one_vs_two_to_one(
        probs: np.ndarray,
        class_names: list[str],
        top_indices: np.ndarray,
        trigger_margin: float,
    ) -> np.ndarray:
        if len(top_indices) < 2:
            return probs

        first_name = class_names[int(top_indices[0])]
        second_name = class_names[int(top_indices[1])]
        if {first_name, second_name} != {"red_1", "red_2"}:
            return probs

        conf_gap = abs(float(probs[int(top_indices[0])]) - float(probs[int(top_indices[1])]))
        if conf_gap > trigger_margin:
            return probs

        adjusted = probs.copy()
        idx1 = int(top_indices[0]) if first_name == "red_1" else int(top_indices[1])
        idx2 = int(top_indices[0]) if first_name == "red_2" else int(top_indices[1])
        adjusted[idx1] *= 1.2
        adjusted[idx2] *= 0.84

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _bias_close_blue_five_vs_three_to_five(
        probs: np.ndarray,
        class_names: list[str],
        top_indices: np.ndarray,
        trigger_margin: float,
    ) -> np.ndarray:
        if len(top_indices) < 2:
            return probs

        first_name = class_names[int(top_indices[0])]
        second_name = class_names[int(top_indices[1])]
        if {first_name, second_name} != {"blue_5", "blue_3"}:
            return probs

        conf_gap = abs(float(probs[int(top_indices[0])]) - float(probs[int(top_indices[1])]))
        if conf_gap > trigger_margin:
            return probs

        adjusted = probs.copy()
        idx5 = int(top_indices[0]) if first_name == "blue_5" else int(top_indices[1])
        idx3 = int(top_indices[0]) if first_name == "blue_3" else int(top_indices[1])
        adjusted[idx5] *= 3.8
        adjusted[idx3] *= 0.1

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    @staticmethod
    def _rescue_blue_five_from_blue_three(
        probs: np.ndarray,
        class_names: list[str],
        trigger_margin: float,
    ) -> np.ndarray:
        index_map = TwoStageRecognizer._class_index_map(class_names)
        idx5 = index_map.get("blue_5")
        idx3 = index_map.get("blue_3")
        if idx5 is None or idx3 is None:
            return probs

        sorted_idx = np.argsort(probs)[::-1]
        top8 = set(int(i) for i in sorted_idx[:8])
        if idx3 not in top8 or idx5 not in top8:
            return probs

        p3 = float(probs[idx3])
        p5 = float(probs[idx5])
        if p3 <= p5:
            return probs

        gap = p3 - p5
        if gap > trigger_margin:
            return probs

        adjusted = probs.copy()
        # Strong rescue: blue_5 often under-ranked but still present among high candidates.
        adjusted[idx5] *= 4.4
        adjusted[idx3] *= 0.12

        total = float(adjusted.sum())
        if total > 0:
            adjusted /= total
        return adjusted

    def predict(self, frame: np.ndarray) -> FramePrediction:
        detect_results = self.detector.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.detect_conf,
            max_det=5,
            device=self.device,
            verbose=False,
        )
        boxes = list(detect_results[0].boxes) if detect_results[0].boxes is not None else []
        if not boxes:
            return FramePrediction(None, None, 0.0, 0.0, None)

        boxes = sorted(
            boxes,
            key=lambda box: float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0]),
            reverse=True,
        )
        best_box = boxes[0]
        xyxy = best_box.xyxy[0].tolist()
        h, w = frame.shape[:2]
        left, top, right, bottom = clip_box(xyxy[0], xyxy[1], xyxy[2], xyxy[3], width=w, height=h)
        left, top, right, bottom = expand_box((left, top, right, bottom), width=w, height=h, padding_ratio=self.crop_padding)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return FramePrediction(None, None, 0.0, 0.0, None)

        cls_results = self.classifier.predict(crop, device=self.device, verbose=False)
        base_prob_values = self._extract_probs(cls_results[0])
        class_names = self._class_name_list(cls_results[0].names)
        prob_values = base_prob_values.copy()
        rot_prob_values: np.ndarray | None = None

        if self.tta_rot180:
            rotated_crop = cv2.rotate(crop, cv2.ROTATE_180)
            rot_results = self.classifier.predict(rotated_crop, device=self.device, verbose=False)
            rot_prob_values = self._extract_probs(rot_results[0])
            if rot_prob_values.shape == prob_values.shape:
                prob_values = (base_prob_values + rot_prob_values) / 2.0
                prob_values = self._apply_green_rotation_preference(
                    fused_probs=prob_values,
                    base_probs=base_prob_values,
                    rot_probs=rot_prob_values,
                    class_names=class_names,
                )
                prob_values = self._apply_green_close_case_guard(
                    probs=prob_values,
                    base_probs=base_prob_values,
                    rot_probs=rot_prob_values,
                    class_names=class_names,
                )

        color_hint, color_hint_conf = self._estimate_color_hint(crop)
        prob_values = self._apply_color_prior(prob_values, class_names, color_hint, color_hint_conf)

        top_indices = np.argsort(prob_values)[::-1][:2]
        prob_values = self._bias_close_one_vs_four_to_one(
            probs=prob_values,
            class_names=class_names,
            top_indices=top_indices,
            trigger_margin=self.one_four_margin,
        )
        top_indices = np.argsort(prob_values)[::-1][:2]
        prob_values = self._bias_close_red_one_vs_two_to_one(
            probs=prob_values,
            class_names=class_names,
            top_indices=top_indices,
            trigger_margin=self.red_one_two_margin,
        )
        top_indices = np.argsort(prob_values)[::-1][:2]
        prob_values = self._bias_close_blue_five_vs_three_to_five(
            probs=prob_values,
            class_names=class_names,
            top_indices=top_indices,
            trigger_margin=self.blue_five_three_margin,
        )
        top_indices = np.argsort(prob_values)[::-1][:2]
        prob_values = self._rescue_blue_five_from_blue_three(
            probs=prob_values,
            class_names=class_names,
            trigger_margin=self.blue_five_rescue_margin,
        )
        top_indices = np.argsort(prob_values)[::-1][:2]
        top1_index = int(top_indices[0])
        top1_conf = float(prob_values[top1_index])
        top2_conf = float(prob_values[top_indices[1]]) if len(top_indices) > 1 else 0.0
        confidence_margin = top1_conf - top2_conf
        detect_conf = float(best_box.conf[0].item() if hasattr(best_box.conf[0], "item") else best_box.conf[0])
        if top1_conf < self.classify_conf or confidence_margin < self.min_margin:
            return FramePrediction(
                None,
                None,
                detect_conf,
                top1_conf,
                (left, top, right, bottom),
                top2_conf=top2_conf,
                confidence_margin=confidence_margin,
            )

        class_name = class_names[top1_index]
        card = class_name_to_card(class_name)
        return FramePrediction(
            card,
            class_name,
            detect_conf,
            top1_conf,
            (left, top, right, bottom),
            top2_conf=top2_conf,
            confidence_margin=confidence_margin,
        )


class PredictionStabilizer:
    def __init__(self, stable_frames: int, switch_frames: int, max_missed_frames: int):
        self.stable_frames = max(1, stable_frames)
        self.switch_frames = max(1, switch_frames)
        self.max_missed_frames = max(1, max_missed_frames)
        self.stable_prediction: FramePrediction | None = None
        self.candidate_prediction: FramePrediction | None = None
        self.candidate_count = 0
        self.missed_frames = 0

    def update(self, raw_prediction: FramePrediction) -> tuple[FramePrediction, str, bool]:
        if raw_prediction.card is None:
            self.missed_frames += 1
            if self.stable_prediction and self.missed_frames <= self.max_missed_frames:
                return self.stable_prediction, f"保持稳定结果，等待新帧 {self.missed_frames}/{self.max_missed_frames}", False
            self.stable_prediction = None
            self.candidate_prediction = None
            self.candidate_count = 0
            return raw_prediction, "当前结果不稳定，未锁定", False

        self.missed_frames = 0
        if self.stable_prediction and raw_prediction.class_name == self.stable_prediction.class_name:
            self.stable_prediction = replace(raw_prediction, is_stable=True)
            self.candidate_prediction = None
            self.candidate_count = 0
            return self.stable_prediction, f"稳定识别: {raw_prediction.class_name}", False

        threshold = self.stable_frames if self.stable_prediction is None else self.switch_frames
        if self.candidate_prediction and raw_prediction.class_name == self.candidate_prediction.class_name:
            self.candidate_prediction = raw_prediction
            self.candidate_count += 1
        else:
            self.candidate_prediction = raw_prediction
            self.candidate_count = 1

        if self.candidate_count >= threshold:
            self.stable_prediction = replace(raw_prediction, is_stable=True)
            self.candidate_prediction = None
            self.candidate_count = 0
            return self.stable_prediction, f"已锁定稳定结果: {self.stable_prediction.class_name}", True

        if self.stable_prediction:
            return (
                self.stable_prediction,
                f"保持 {self.stable_prediction.class_name}，候选 {raw_prediction.class_name} "
                f"{self.candidate_count}/{threshold}",
                False,
            )
        return raw_prediction, f"候选 {raw_prediction.class_name} {self.candidate_count}/{threshold}", False


def is_basic_legal(card: Card, state: GameState) -> bool:
    if state.under_attack:
        return can_stack_penalty(card, state.top_card)
    if card.is_wild:
        return True
    if state.active_color and card.color == state.active_color:
        return True
    top = state.top_card
    if top is None:
        return True
    if card.type == top.type and card.type != CardType.NUMBER:
        return True
    if card.type == CardType.NUMBER and top.type == CardType.NUMBER and card.value == top.value:
        return True
    return False


def do_play(state: GameState, card: Card, my_hand: list[Card], chosen_color: Color | None = None) -> None:
    state.update_top_card(card, chosen_color)
    state.my_hand = list(my_hand)


def advance_after_play(state: GameState, card: Card) -> None:
    state.advance_turn()
    if card.type == CardType.SKIP:
        state.advance_turn()


def apply_opening_top_card(state: GameState, card: Card, my_index: int) -> None:
    state.top_card = card
    state.current_color = card.color
    state.current_turn = my_index
    state.stack_penalty = 0
    if card.type == CardType.DRAW_TWO:
        state.stack_penalty = 2
    elif card.type == CardType.SKIP:
        state.advance_turn()


def update_message_after_play(state: GameState, my_hand: list[Card], base_message: str) -> str:
    if state.current_turn == state.my_index:
        decision = decide(my_hand, state)
        return f"{base_message} | AI建议: {format_ai_recommendation(my_hand, decision)}"
    return base_message


def color_key_to_color(key: int) -> Color | None:
    mapping = {
        ord("r"): Color.RED,
        ord("b"): Color.BLUE,
        ord("y"): Color.YELLOW,
        ord("g"): Color.GREEN,
    }
    return mapping.get(key)


def create_writer(path: Path, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 20.0, (frame_width, frame_height))


class FrameDisplay(Protocol):
    def show(self, frame: np.ndarray, state: DisplayState | None = None) -> int: ...

    def close(self) -> None: ...


def opencv_gui_available() -> bool:
    probe = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        cv2.imshow("opencv_gui_probe", probe)
        cv2.waitKey(1)
        cv2.destroyWindow("opencv_gui_probe")
        return True
    except cv2.error:
        return False


class OpenCVDisplay:
    def __init__(self, window_name: str):
        self.window_name = window_name

    def show(self, frame: np.ndarray, state: DisplayState | None = None) -> int:
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class TkDisplay:
    def __init__(self, window_name: str):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title(window_name)
        self.last_key = -1
        self.closed = False
        self.started = False
        self._default_bg = "#07111d"
        self._panel_bg = "#0d1b2a"
        self._panel_soft = "#13273b"
        self._accent = "#29f0b4"
        self._accent_2 = "#f7c65f"
        self._danger = "#ff6b6b"
        self._muted = "#9bb2c9"
        self._queue_button_keys: list[int] = []
        self._current_state: DisplayState | None = None
        self.root.configure(bg=self._default_bg)
        self.root.geometry("1520x920")
        self.root.minsize(1280, 760)
        self._build_styles()
        self._build_shell(window_name)
        self.last_key = -1
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Configure>", self._on_resize)
        self.root.focus_force()
        self._show_start_screen()

    def _build_styles(self) -> None:
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Uno.TFrame", background=self._default_bg)
        style.configure("Panel.TFrame", background=self._panel_bg)
        style.configure("SoftPanel.TFrame", background=self._panel_soft)
        style.configure(
            "Title.TLabel",
            background=self._default_bg,
            foreground="#f5f9ff",
            font=("Microsoft YaHei UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self._default_bg,
            foreground=self._muted,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=self._panel_bg,
            foreground="#f7fbff",
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "PanelText.TLabel",
            background=self._panel_bg,
            foreground="#d8e4f2",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Accent.TButton",
            background=self._accent,
            foreground="#04131e",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(10, 8),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#55ffd0"), ("pressed", "#1fd19e")],
            foreground=[("disabled", "#5a6f84")],
        )
        style.configure(
            "Secondary.TButton",
            background=self._panel_soft,
            foreground="#eff7ff",
            font=("Microsoft YaHei UI", 10),
            padding=(10, 8),
            borderwidth=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#1a3550"), ("pressed", "#0f2436")],
            foreground=[("disabled", "#5a6f84")],
        )
        style.configure(
            "Warn.TButton",
            background="#2f2230",
            foreground="#ffd1d1",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(10, 8),
            borderwidth=0,
        )
        style.map(
            "Warn.TButton",
            background=[("active", "#5a3030"), ("pressed", "#4a2525")],
        )

    def _build_shell(self, window_name: str) -> None:
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)

        self.video_panel = self.ttk.Frame(self.root, style="Panel.TFrame", padding=14)
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.video_panel.grid_rowconfigure(1, weight=1)
        self.video_panel.grid_columnconfigure(0, weight=1)

        header = self.ttk.Frame(self.video_panel, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.grid_columnconfigure(0, weight=1)
        self.video_title = self.ttk.Label(header, text=window_name, style="PanelTitle.TLabel")
        self.video_title.grid(row=0, column=0, sticky="w")
        self.phase_badge = self.ttk.Label(header, text="等待开始", style="PanelText.TLabel")
        self.phase_badge.grid(row=0, column=1, sticky="e")

        self.video_wrap = self.tk.Frame(self.video_panel, bg="#050b13", highlightthickness=1, highlightbackground="#1b3650")
        self.video_wrap.grid(row=1, column=0, sticky="nsew")
        self.video_wrap.grid_rowconfigure(0, weight=1)
        self.video_wrap.grid_columnconfigure(0, weight=1)
        self.label = self.tk.Label(self.video_wrap, bg="#050b13", bd=0, highlightthickness=0)
        self.label.grid(row=0, column=0, sticky="nsew")

        self.sidebar = self.ttk.Frame(self.root, style="Uno.TFrame", padding=(0, 18, 18, 18))
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self.start_card = self.ttk.Frame(self.sidebar, style="Panel.TFrame", padding=18)
        self.start_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.start_card.grid_columnconfigure(0, weight=1)
        self.start_title = self.ttk.Label(self.start_card, text="UNO 对弈控制台", style="Title.TLabel")
        self.start_title.grid(row=0, column=0, sticky="w")
        self.start_desc = self.ttk.Label(
            self.start_card,
            text="座位按 0,1,2... 编号；机器占你设置的座位，其余自动显示为玩家1、玩家2、玩家3。",
            style="Subtitle.TLabel",
            wraplength=420,
            justify="left",
        )
        self.start_desc.grid(row=1, column=0, sticky="w", pady=(8, 14))

        config_box = self.ttk.Frame(self.start_card, style="SoftPanel.TFrame", padding=14)
        config_box.grid(row=2, column=0, sticky="ew")
        for col in range(4):
            config_box.grid_columnconfigure(col, weight=1)

        self.players_var = self.tk.IntVar(value=2)
        self.my_index_var = self.tk.IntVar(value=0)
        self.starting_player_var = self.tk.IntVar(value=0)
        self.hand_size_var = self.tk.IntVar(value=5)
        self.source_var = self.tk.StringVar(value="0")
        self._build_labeled_spinbox(config_box, "玩家人数", self.players_var, 2, 8, 0, 0)
        self._build_labeled_spinbox(config_box, "机器座位", self.my_index_var, 0, 7, 0, 1)
        self._build_labeled_spinbox(config_box, "初始手牌", self.hand_size_var, 1, 20, 0, 2)
        self._build_labeled_entry(config_box, "视频源", self.source_var, 0, 3)
        self._build_labeled_spinbox(config_box, "首位出牌者", self.starting_player_var, 0, 7, 1, 0)

        button_row = self.ttk.Frame(self.start_card, style="Panel.TFrame")
        button_row.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)
        self.start_button = self.ttk.Button(button_row, text="开始游戏", style="Accent.TButton", command=self._start_from_menu)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.quit_button = self.ttk.Button(button_row, text="退出", style="Warn.TButton", command=self._on_close)
        self.quit_button.grid(row=0, column=1, sticky="ew")

        self.runtime_card = self.ttk.Frame(self.sidebar, style="Panel.TFrame", padding=18)
        self.runtime_card.grid(row=1, column=0, sticky="nsew")
        self.runtime_card.grid_columnconfigure(0, weight=1)
        for row in range(6):
            self.runtime_card.grid_rowconfigure(row, weight=0)
        self.runtime_card.grid_rowconfigure(5, weight=1)

        self.session_var = self.tk.StringVar(value="等待开始")
        self.status_var = self.tk.StringVar(value="点击左上方开始游戏，或直接用键盘操作。")
        self.stable_var = self.tk.StringVar(value="稳定识别: 无")
        self.candidate_var = self.tk.StringVar(value="候选识别: 无")
        self.turn_var = self.tk.StringVar(value="当前回合: -")
        self.top_var = self.tk.StringVar(value="顶牌: -")
        self.color_var = self.tk.StringVar(value="当前颜色: -")
        self.penalty_var = self.tk.StringVar(value="累计罚牌: 0")
        self.hand_var = self.tk.StringVar(value="机器手牌(0): 空")
        self.recommend_var = self.tk.StringVar(value="AI 建议: 暂无")
        self.pending_var = self.tk.StringVar(value="流程状态: 等待开始")
        self.winner_var = self.tk.StringVar(value="")
        self.return_menu_requested = False

        self._build_runtime_header()
        self._build_control_panel()
        self._build_help_panel()

    def _build_labeled_spinbox(self, parent, title: str, variable, start: int, end: int, row: int, column: int) -> None:
        card = self.ttk.Frame(parent, style="SoftPanel.TFrame", padding=10)
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        self.ttk.Label(card, text=title, style="PanelText.TLabel").grid(row=0, column=0, sticky="w")
        spin = self.tk.Spinbox(
            card,
            from_=start,
            to=end,
            textvariable=variable,
            bg="#07111d",
            fg="#f5f9ff",
            insertbackground="#f5f9ff",
            buttonbackground="#1a3550",
            relief="flat",
            justify="center",
            font=("Microsoft YaHei UI", 11, "bold"),
            width=6,
        )
        spin.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_labeled_entry(self, parent, title: str, variable, row: int, column: int) -> None:
        card = self.ttk.Frame(parent, style="SoftPanel.TFrame", padding=10)
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        self.ttk.Label(card, text=title, style="PanelText.TLabel").grid(row=0, column=0, sticky="w")
        entry = self.tk.Entry(
            card,
            textvariable=variable,
            bg="#07111d",
            fg="#f5f9ff",
            insertbackground="#f5f9ff",
            relief="flat",
            font=("Microsoft YaHei UI", 11, "bold"),
            justify="center",
        )
        entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_runtime_header(self) -> None:
        header = self.ttk.Frame(self.runtime_card, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        self.ttk.Label(header, text="对局总览", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(header, textvariable=self.session_var, style="PanelText.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.message_label = self.ttk.Label(
            header,
            textvariable=self.status_var,
            style="PanelText.TLabel",
            wraplength=420,
            justify="left",
        )
        self.message_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.winner_label = self.ttk.Label(
            header,
            textvariable=self.winner_var,
            style="PanelTitle.TLabel",
            wraplength=420,
            justify="left",
        )
        self.winner_label.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        info_box = self.ttk.Frame(self.runtime_card, style="SoftPanel.TFrame", padding=12)
        info_box.grid(row=1, column=0, sticky="ew", pady=(14, 12))
        info_box.grid_columnconfigure(0, weight=1)
        info_box.grid_columnconfigure(1, weight=1)
        labels = [
            self.turn_var,
            self.top_var,
            self.color_var,
            self.penalty_var,
            self.hand_var,
            self.recommend_var,
            self.stable_var,
            self.candidate_var,
            self.pending_var,
        ]
        for idx, variable in enumerate(labels):
            row = idx // 2
            col = idx % 2
            self.ttk.Label(
                info_box,
                textvariable=variable,
                style="PanelText.TLabel",
                wraplength=190,
                justify="left",
            ).grid(row=row, column=col, sticky="w", padx=4, pady=4)

    def _build_control_panel(self) -> None:
        controls = self.ttk.Frame(self.runtime_card, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.ttk.Label(controls, text="操作按钮", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        button_frame = self.ttk.Frame(controls, style="Panel.TFrame")
        button_frame.grid(row=1, column=0, sticky="ew")
        for col in range(3):
            button_frame.grid_columnconfigure(col, weight=1)

        buttons = [
            ("确认录牌 A", "Accent.TButton", "a"),
            ("确认出牌 S", "Accent.TButton", "s"),
            ("摸牌 D", "Secondary.TButton", "d"),
            ("下一回合 N", "Secondary.TButton", "n"),
            ("撤销 C", "Warn.TButton", "c"),
            ("主菜单 M", "Secondary.TButton", "m"),
            ("退出 Q", "Warn.TButton", "q"),
            ("红色 R", "Secondary.TButton", "r"),
            ("蓝色 B", "Secondary.TButton", "b"),
            ("黄色 Y", "Secondary.TButton", "y"),
            ("绿色 G", "Secondary.TButton", "g"),
        ]
        for idx, (text, style_name, key_char) in enumerate(buttons):
            row = idx // 3
            col = idx % 3
            self.ttk.Button(
                button_frame,
                text=text,
                style=style_name,
                command=lambda ch=key_char: self._queue_key(ch),
            ).grid(row=row, column=col, sticky="ew", padx=4, pady=4)

    def _build_help_panel(self) -> None:
        help_box = self.ttk.Frame(self.runtime_card, style="SoftPanel.TFrame", padding=14)
        help_box.grid(row=5, column=0, sticky="nsew")
        help_box.grid_columnconfigure(0, weight=1)
        self.ttk.Label(help_box, text="操作说明", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        guide_text = (
            "1. 开局先录入机器手牌，逐张给摄像头看，听到稳定提示音后按 A。\n"
            "2. 录完初始手牌后，把桌面顶牌放到镜头下，按 S 确认开局牌。\n"
            "3. 对局中轮到机器出牌时按 S，摸牌时按 D，若是摸到的牌需要录入则继续按 A。\n"
            "4. 万能牌或 +4 出现后，用 R/B/Y/G 记录选择的颜色。\n"
            "5. 误录手牌时按 C 撤销最后一张，按 M 回主菜单，按 Q 或 Esc 退出。"
        )
        self.ttk.Label(
            help_box,
            text=guide_text,
            style="PanelText.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 12))
        self.ttk.Label(help_box, text="简版 UNO 规则", style="PanelTitle.TLabel").grid(row=2, column=0, sticky="w")
        rules_text = (
            "同色、同数字或同功能牌可以出；万能牌可以在多数场景下直接出。"
            " 跳过牌会跳过下一位，+4 可以接 +4 或 +2，+2 只能接 +2；若轮到机器摸牌，先完成摸牌记录再继续回合。"
        )
        self.ttk.Label(
            help_box,
            text=rules_text,
            style="PanelText.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _show_start_screen(self) -> None:
        self.started = False
        self.video_title.configure(text="UNO 对弈预览")
        self.phase_badge.configure(text="准备阶段")
        self.session_var.set("尚未开始")
        self.status_var.set("在右侧设置玩家人数、机器座位、首位出牌者和初始手牌数量，然后点击“开始游戏”。")
        self.winner_var.set("")
        self.return_menu_requested = False
        self.start_card.grid()

    def _start_from_menu(self) -> None:
        try:
            players = max(2, int(self.players_var.get()))
            my_index = int(self.my_index_var.get())
            starting_player = int(self.starting_player_var.get())
            hand_size = max(1, int(self.hand_size_var.get()))
        except Exception:
            self.status_var.set("设置值无效，请检查玩家人数、机器座位、首位出牌者和初始手牌数量。")
            return
        if my_index < 0 or my_index >= players:
            self.status_var.set("机器座位必须落在 0 到 玩家人数-1 之间。")
            return
        if starting_player < 0 or starting_player >= players:
            self.status_var.set("首位出牌者必须落在 0 到 玩家人数-1 之间。")
            return
        self.players_var.set(players)
        self.my_index_var.set(my_index)
        self.starting_player_var.set(starting_player)
        self.hand_size_var.set(hand_size)
        self.started = True
        self.return_menu_requested = False
        self.start_card.grid_remove()
        self.video_title.configure(text="UNO 对弈实时画面")
        self.phase_badge.configure(text="识别运行中")
        self.status_var.set("对局已开始。把机器手牌放到摄像头前，稳定后按 A 录入。")
        self.winner_var.set("")
        self.root.focus_force()

    def apply_menu_settings(self, args) -> None:
        args.players = max(2, int(self.players_var.get()))
        args.my_index = min(max(0, int(self.my_index_var.get())), args.players - 1)
        args.starting_player = min(max(0, int(self.starting_player_var.get())), args.players - 1)
        args.initial_hand_size = max(1, int(self.hand_size_var.get()))
        args.source = self.source_var.get().strip() or args.source

    def _queue_key(self, key_char: str) -> None:
        if key_char.lower() == "m":
            self.return_menu_requested = True
        self._queue_button_keys.append(ord(key_char.lower()))
        self.root.focus_force()

    def _on_resize(self, _event=None) -> None:
        if self._current_state:
            self.phase_badge.configure(text=self._current_state.phase)

    def _on_key(self, event) -> None:
        if event.keysym == "Escape":
            self.last_key = ord("q")
            return
        if event.char:
            self.last_key = ord(event.char.lower())

    def _on_close(self) -> None:
        self.closed = True

    def show(self, frame: np.ndarray, state: DisplayState | None = None) -> int:
        if self.closed:
            return ord("q")
        if state is not None:
            self._current_state = state
            self._refresh_sidebar(state)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        win_w = max(1, self.video_wrap.winfo_width())
        win_h = max(1, self.video_wrap.winfo_height())
        src_w, src_h = image.size
        scale = min(win_w / src_w, win_h / src_h)
        if scale > 0:
            resized_w = max(1, int(round(src_w * scale)))
            resized_h = max(1, int(round(src_h * scale)))
            if resized_w != src_w or resized_h != src_h:
                image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        photo = self.ImageTk.PhotoImage(image=image)
        self.label.configure(image=photo)
        self.label.image = photo
        self.root.update_idletasks()
        self.root.update()
        if self._queue_button_keys:
            key = self._queue_button_keys.pop(0)
        else:
            key = self.last_key
            self.last_key = -1
        return key

    def _refresh_sidebar(self, state: DisplayState) -> None:
        self.phase_badge.configure(text=state.phase)
        self.session_var.set(state.session_title)
        self.status_var.set(state.message_text)
        self.turn_var.set(state.turn_text)
        self.top_var.set(state.top_text)
        self.color_var.set(state.color_text)
        self.penalty_var.set(state.penalty_text)
        self.hand_var.set(state.hand_text)
        self.recommend_var.set(state.recommendation_text)
        self.stable_var.set(state.stable_text)
        self.candidate_var.set(state.candidate_text)
        self.pending_var.set(state.pending_text or state.stability_text)
        self.winner_var.set(state.winner_text if state.game_over else "")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def build_display(backend: str) -> FrameDisplay:
    if backend == "cv2":
        if not opencv_gui_available():
            raise RuntimeError("当前 OpenCV 不支持 GUI 窗口，请改用 --display-backend tk")
        return OpenCVDisplay("UNO Desktop Assistant")
    if backend == "tk":
        return TkDisplay("UNO Desktop Assistant")
    if opencv_gui_available():
        return OpenCVDisplay("UNO Desktop Assistant")
    return TkDisplay("UNO Desktop Assistant")


def build_display_state(
    args,
    state: GameState,
    player_labels: list[str],
    my_hand: list[Card],
    phase: str,
    message: str,
    prediction: FramePrediction,
    raw_prediction: FramePrediction,
    stability_message: str,
    pending_wild_owner: int | None,
    draw_count: int,
    game_over: bool = False,
    winner_text: str = "",
) -> DisplayState:
    machine_label = get_player_label(player_labels, state.my_index)
    turn_text = f"当前回合: {get_player_label(player_labels, state.current_turn)}"
    top_text = f"顶牌: {state.top_card if state.top_card else '无'}"
    color_text = f"当前颜色: {state.active_color.value if state.active_color else '无'}"
    penalty_text = f"累计罚牌: {state.stack_penalty}"
    hand_text = f"{machine_label}手牌({len(my_hand)}): {format_hand(my_hand) if my_hand else '空'}"
    if prediction.card and prediction.is_stable:
        stable_text = (
            f"稳定识别: {prediction.card} | det={prediction.detect_conf:.2f} "
            f"cls={prediction.card_conf:.2f} gap={prediction.confidence_margin:.2f}"
        )
    else:
        stable_text = "稳定识别: 未锁定"
    if raw_prediction.card:
        candidate_text = (
            f"候选识别: {raw_prediction.card} | det={raw_prediction.detect_conf:.2f} "
            f"cls={raw_prediction.card_conf:.2f} gap={raw_prediction.confidence_margin:.2f}"
        )
    elif raw_prediction.box is not None:
        candidate_text = (
            f"候选识别: 低置信度框 | det={raw_prediction.detect_conf:.2f} "
            f"cls={raw_prediction.card_conf:.2f} gap={raw_prediction.confidence_margin:.2f}"
        )
    else:
        candidate_text = "候选识别: 无"

    if pending_wild_owner is not None:
        owner = get_player_label(player_labels, pending_wild_owner)
        pending_text = f"流程状态: 等待 {owner} 选择万能牌颜色（R/B/Y/G）"
    elif draw_count > 0:
        pending_text = f"流程状态: 摸牌确认中，还需录入 {draw_count} 张"
    else:
        pending_text = f"流程状态: {stability_message}"

    recommendation_text = "AI 建议: 暂无"
    if state.current_turn == state.my_index:
        try:
            recommendation_text = f"AI 建议: {format_ai_recommendation(my_hand, decide(my_hand, state))}"
        except Exception:
            recommendation_text = "AI 建议: 暂时无法生成"

    session_title = (
        f"{args.players} 人对局 | 机器座位 {machine_label} | 首位出牌 {get_player_label(player_labels, args.starting_player)} "
        f"| 开局手牌 {args.initial_hand_size} 张"
    )
    return DisplayState(
        session_title=session_title,
        phase=phase,
        turn_text=turn_text,
        top_text=top_text,
        color_text=color_text,
        penalty_text=penalty_text,
        hand_text=hand_text,
        stable_text=stable_text,
        candidate_text=candidate_text,
        stability_text=stability_message,
        recommendation_text=recommendation_text,
        pending_text=pending_text,
        message_text=message,
        game_over=game_over,
        winner_text=winner_text,
    )


def draw_overlay(
    frame: np.ndarray,
    state: GameState,
    player_labels: list[str],
    my_hand: list[Card],
    phase: str,
    message: str,
    prediction: FramePrediction,
    raw_prediction: FramePrediction,
    stability_message: str,
    pending_wild_owner: int | None,
    draw_count: int,
    game_over: bool = False,
    winner_text: str = "",
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    panel_h = min(220, max(180, h // 4))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (8, 18, 32), -1)
    out = cv2.addWeighted(overlay, 0.76, out, 0.24, 0)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (35, 95, 135), 2)
    cv2.line(out, (0, panel_h), (w, panel_h), (41, 240, 180), 2)

    y = 10
    out = put_text(out, f"阶段: {phase}", (10, y), (255, 220, 120), 20)
    y += 28
    turn_text = get_player_label(player_labels, state.current_turn)
    top_text = str(state.top_card) if state.top_card else "无"
    color_text = state.active_color.value if state.active_color else "无"
    out = put_text(out, f"当前回合: {turn_text}    顶牌: {top_text}    当前颜色: {color_text}    累计罚牌: {state.stack_penalty}", (10, y), (235, 235, 235), 16)
    y += 24
    machine_label = get_player_label(player_labels, state.my_index)
    out = put_text(out, f"{machine_label}手牌({len(my_hand)}): {format_hand(my_hand) if my_hand else '空'}", (10, y), (180, 255, 180), 15)
    y += 24

    if prediction.card and prediction.is_stable:
        pred_text = (
            f"稳定结果: {prediction.card} | det={prediction.detect_conf:.2f} "
            f"cls={prediction.card_conf:.2f} margin={prediction.confidence_margin:.2f}"
        )
        out = put_text(out, pred_text, (10, y), (90, 255, 120), 16)
    else:
        pred_text = "稳定结果: 未锁定"
        if raw_prediction.card:
            pred_text += (
                f" | 候选={raw_prediction.card} det={raw_prediction.detect_conf:.2f} "
                f"cls={raw_prediction.card_conf:.2f} margin={raw_prediction.confidence_margin:.2f}"
            )
        elif raw_prediction.box is not None:
            pred_text += (
                f" | 低置信度 det={raw_prediction.detect_conf:.2f} "
                f"cls={raw_prediction.card_conf:.2f} margin={raw_prediction.confidence_margin:.2f}"
            )
        out = put_text(out, pred_text, (10, y), (255, 180, 80), 16)
    y += 24
    out = put_text(out, stability_message, (10, y), (170, 220, 255), 15)
    y += 22

    if pending_wild_owner is not None:
        owner = get_player_label(player_labels, pending_wild_owner)
        out = put_text(out, f"等待 {owner} 的万能牌选色: R/B/Y/G", (10, y), (255, 180, 80), 16)
        y += 24
    elif draw_count > 0:
        out = put_text(out, f"正在摸牌流程，还需确认 {draw_count} 张", (10, y), (255, 180, 80), 16)
        y += 24
    else:
        out = put_text(out, message, (10, y), (255, 255, 110), 16)
    guide = "A=确认入库/抓牌  S=确认出牌  D=当前玩家摸牌  N=下家回合  C=撤销最后一张手牌  Q=退出"
    tip = "稳定识别后会有提示音，建议把牌正对摄像头并保持 1 秒。"
    out = put_text(out, tip, (10, h - 52), (120, 235, 255), 14)
    out = put_text(out, guide, (10, h - 28), (180, 180, 180), 14)

    display_box = raw_prediction.box if raw_prediction.box is not None else prediction.box
    if display_box is not None:
        x1, y1, x2, y2 = display_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (41, 240, 180), 3)
        label = raw_prediction.class_name if raw_prediction.class_name else "候选"
        cv2.putText(out, label, (x1 + 4, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (41, 240, 180), 2, cv2.LINE_AA)

    if game_over and winner_text:
        end_overlay = out.copy()
        cv2.rectangle(end_overlay, (0, 0), (w, h), (6, 12, 22), -1)
        out = cv2.addWeighted(end_overlay, 0.24, out, 0.76, 0)
        banner_w = min(w - 80, 760)
        banner_h = 160
        bx1 = max(40, (w - banner_w) // 2)
        by1 = max(40, (h - banner_h) // 2)
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (10, 28, 44), -1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (41, 240, 180), 3)
        out = put_text(out, "对局结束", (bx1 + 28, by1 + 24), (247, 198, 95), 30)
        out = put_text(out, winner_text, (bx1 + 28, by1 + 76), (245, 249, 255), 28)
        out = put_text(out, "按 M 返回主菜单，按 Q 退出程序", (bx1 + 28, by1 + 122), (160, 220, 255), 18)

    return out


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> int:
    args = build_parser().parse_args()
    display = build_display(args.display_backend)
    if not args.detect_model.exists():
        raise FileNotFoundError(f"检测模型不存在: {args.detect_model}")
    if not args.classify_model.exists():
        raise FileNotFoundError(f"分类模型不存在: {args.classify_model}")

    recognizer = TwoStageRecognizer(
        detect_model=args.detect_model,
        classify_model=args.classify_model,
        device=args.device,
        imgsz=args.imgsz,
        detect_conf=args.detect_conf,
        classify_conf=args.classify_conf,
        min_margin=args.min_margin,
        crop_padding=args.crop_padding,
        color_prior_strength=args.color_prior_strength,
        tta_rot180=args.tta_rot180,
        one_four_margin=args.one_four_margin,
        red_one_two_margin=args.red_one_two_margin,
        blue_five_three_margin=args.blue_five_three_margin,
        blue_five_rescue_margin=args.blue_five_rescue_margin,
    )

    while True:
        if isinstance(display, TkDisplay):
            display._show_start_screen()
            while not display.started and not display.closed:
                blank = np.zeros((720, 1080, 3), dtype=np.uint8)
                blank[:] = (8, 18, 32)
                blank = put_text(blank, "UNO CAMERA ASSISTANT", (48, 64), (245, 249, 255), 28)
                blank = put_text(blank, "右侧先设置对局参数，再点击开始游戏。", (48, 110), (140, 220, 255), 18)
                warm_state = DisplayState(
                    session_title="尚未开始",
                    phase="准备阶段",
                    turn_text="当前回合: -",
                    top_text="顶牌: -",
                    color_text="当前颜色: -",
                    penalty_text="累计罚牌: 0",
                    hand_text="机器手牌(0): 空",
                    stable_text="稳定识别: 未启动",
                    candidate_text="候选识别: 未启动",
                    stability_text="等待开始",
                    recommendation_text="AI 建议: 暂无",
                    pending_text="流程状态: 等待点击开始游戏",
                    message_text="设置好人数、机器座位和首位出牌者后，点击开始游戏。",
                )
                key = display.show(blank, warm_state)
                if key == ord("q"):
                    display.close()
                    return 0
            if display.closed:
                return 0
            display.apply_menu_settings(args)

        if args.players < 2:
            raise ValueError("--players 至少为 2")
        if not (0 <= args.my_index < args.players):
            raise ValueError("--my-index 必须在 [0, players) 范围内")
        if not (0 <= args.starting_player < args.players):
            raise ValueError("--starting-player 必须在 [0, players) 范围内")

        stabilizer = PredictionStabilizer(
            stable_frames=args.stable_frames,
            switch_frames=args.switch_frames,
            max_missed_frames=args.max_missed_frames,
        )
        capture = open_capture(str(args.source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频源: {args.source}")

        game_state = GameState(player_count=args.players, my_index=args.my_index)
        player_labels = build_player_labels(args.players, args.my_index)
        set_initial_other_hand_counts(game_state, args.initial_hand_size)
        my_hand: list[Card] = []
        phase = "初始手牌录入"
        message = f"请逐张展示机器手牌，按 A 确认，目标 {args.initial_hand_size} 张"
        frame_id = 0
        raw_prediction = FramePrediction(None, None, 0.0, 0.0, None)
        last_prediction = FramePrediction(None, None, 0.0, 0.0, None)
        stability_message = "等待稳定识别"
        pending_wild: Card | None = None
        pending_wild_owner: int | None = None
        draw_count = 0
        writer: cv2.VideoWriter | None = None
        winner_text: str | None = None
        return_to_menu = False

        try:
            while True:
                if isinstance(display, TkDisplay) and display.return_menu_requested:
                    display.return_menu_requested = False
                    return_to_menu = True
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                if frame_id % max(1, args.infer_every) == 0:
                    raw_prediction = recognizer.predict(frame)
                    last_prediction, stability_message, should_beep = stabilizer.update(raw_prediction)
                    if should_beep:
                        play_stable_beep()

                render = draw_overlay(
                    frame=frame,
                    state=game_state,
                    player_labels=player_labels,
                    my_hand=my_hand,
                    phase=phase,
                    message=message,
                    prediction=last_prediction,
                    raw_prediction=raw_prediction,
                    stability_message=stability_message,
                    pending_wild_owner=pending_wild_owner,
                    draw_count=draw_count,
                    game_over=winner_text is not None,
                    winner_text=winner_text or "",
                )

                if args.save_video:
                    if writer is None:
                        fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
                        writer = create_writer(args.save_video, render.shape[1], render.shape[0], fps)
                    writer.write(render)

                display_state = build_display_state(
                    args=args,
                    state=game_state,
                    player_labels=player_labels,
                    my_hand=my_hand,
                    phase=phase,
                    message=message,
                    prediction=last_prediction,
                    raw_prediction=raw_prediction,
                    stability_message=stability_message,
                    pending_wild_owner=pending_wild_owner,
                    draw_count=draw_count,
                    game_over=winner_text is not None,
                    winner_text=winner_text or "",
                )
                key = display.show(render, display_state)

                if key == ord("q"):
                    display.close()
                    return 0
                if key == ord("m"):
                    return_to_menu = True
                    break

                if winner_text is not None:
                    frame_id += 1
                    continue

                if key == ord("c"):
                    if my_hand:
                        removed = my_hand.pop()
                        game_state.my_hand = list(my_hand)
                        message = f"已撤销最后一张机器手牌: {removed}"
                    else:
                        message = "当前没有可撤销的机器手牌"

                elif key == ord("n") and phase == "对局":
                    game_state.advance_turn()
                    message = update_message_after_play(game_state, my_hand, f"已切到下一位出牌者: {get_player_label(player_labels, game_state.current_turn)}")

                elif pending_wild and pending_wild_owner is not None and color_key_to_color(key):
                    chosen = color_key_to_color(key)
                    do_play(game_state, pending_wild, my_hand, chosen)
                    if pending_wild_owner != game_state.my_index:
                        change_other_hand_count(game_state, pending_wild_owner, -1)
                    game_state.advance_turn()
                    owner = get_player_label(player_labels, pending_wild_owner)
                    message = update_message_after_play(game_state, my_hand, f"{owner} 出了 {pending_wild}，并选择 {chosen.value} 色")
                    pending_wild = None
                    pending_wild_owner = None

                elif key == ord("a"):
                    if last_prediction.card is None:
                        message = "当前没有稳定识别到牌，请调整角度或距离"
                    elif phase == "初始手牌录入":
                        my_hand.append(last_prediction.card)
                        game_state.my_hand = list(my_hand)
                        message = f"已录入机器手牌: {last_prediction.card} ({len(my_hand)}/{args.initial_hand_size})"
                        if len(my_hand) >= args.initial_hand_size:
                            phase = "等待开局顶牌"
                            message = f"机器手牌录入完成。请展示开局顶牌并按 S 确认，之后将从 {get_player_label(player_labels, args.starting_player)} 开始。"
                    elif phase == "对局" and draw_count > 0:
                        my_hand.append(last_prediction.card)
                        game_state.my_hand = list(my_hand)
                        draw_count -= 1
                        if draw_count == 0:
                            game_state.advance_turn()
                            message = update_message_after_play(game_state, my_hand, f"摸牌流程完成，刚录入 {last_prediction.card}")
                        else:
                            message = f"已录入摸到的牌: {last_prediction.card}，还需 {draw_count} 张"
                    elif phase == "对局":
                        my_hand.append(last_prediction.card)
                        game_state.my_hand = list(my_hand)
                        message = f"已将识别牌加入机器手牌: {last_prediction.card}"

                elif key == ord("s"):
                    if last_prediction.card is None:
                        message = "当前没有稳定识别到牌"
                    elif phase == "等待开局顶牌":
                        if last_prediction.card.is_wild:
                            message = "开局顶牌不建议直接用万能牌，请换一张再确认"
                        else:
                            apply_opening_top_card(game_state, last_prediction.card, args.my_index)
                            game_state.current_turn = args.starting_player
                            phase = "对局"
                            message = update_message_after_play(game_state, my_hand, f"开局顶牌已确认: {last_prediction.card}，由 {get_player_label(player_labels, game_state.current_turn)} 先出牌")
                    elif phase == "对局":
                        card = last_prediction.card
                        if game_state.current_turn == game_state.my_index:
                            found_index = next((i for i, hand_card in enumerate(my_hand) if hand_card == card), None)
                            if found_index is None:
                                message = f"{card} 不在机器手牌中，不能直接确认"
                            elif found_index not in legal_plays(my_hand, game_state):
                                message = f"{card} 当前不合法，请换牌或按 D 摸牌"
                            else:
                                my_hand.pop(found_index)
                                game_state.my_hand = list(my_hand)
                                if card.is_wild:
                                    pending_wild = card
                                    pending_wild_owner = game_state.my_index
                                    message = f"机器出了 {card}，请按 R/B/Y/G 选择颜色"
                                else:
                                    do_play(game_state, card, my_hand)
                                    advance_after_play(game_state, card)
                                    message = update_message_after_play(game_state, my_hand, f"机器出了 {card}")
                        else:
                            if not is_basic_legal(card, game_state):
                                message = f"检测到 {card}，但它与当前桌面状态不匹配"
                            elif card.is_wild:
                                pending_wild = card
                                pending_wild_owner = game_state.current_turn
                                message = f"{get_player_label(player_labels, game_state.current_turn)} 出了 {card}，请按 R/B/Y/G 记录选色"
                            else:
                                current_player = game_state.current_turn
                                change_other_hand_count(game_state, current_player, -1)
                                do_play(game_state, card, my_hand)
                                advance_after_play(game_state, card)
                                message = update_message_after_play(game_state, my_hand, f"{get_player_label(player_labels, current_player)} 出了 {card}")

                elif key == ord("d") and phase == "对局":
                    penalty = game_state.stack_penalty
                    if game_state.current_turn == game_state.my_index:
                        draw_count = max(1, penalty)
                        game_state.reset_penalty()
                        message = f"轮到机器摸牌，请逐张展示并按 A 确认，共 {draw_count} 张"
                    else:
                        current_player = game_state.current_turn
                        draw_cards = max(1, penalty)
                        change_other_hand_count(game_state, current_player, draw_cards)
                        game_state.reset_penalty()
                        game_state.advance_turn()
                        message = update_message_after_play(game_state, my_hand, f"{get_player_label(player_labels, current_player)} 摸牌并结束回合，共摸 {draw_cards} 张")

                winner_text = get_winner_text(game_state, player_labels, my_hand, phase)
                if winner_text is not None:
                    phase = "对局结束"
                    pending_wild = None
                    pending_wild_owner = None
                    draw_count = 0
                    message = f"{winner_text}。按 M 返回主菜单，或按 Q 退出程序。"

                frame_id += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not return_to_menu:
            break

    display.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
