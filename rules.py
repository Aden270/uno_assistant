"""
UNO 规则引擎 —— 兼容增强版
③ 合法性判定  |  ④ 出牌策略+局势判断  |  ⑤ 指令输出

目标：
1. 保持当前项目牌库与 GameState 兼容
2. 吸收“记牌器 + 威胁等级 + 动态权重”的策略思想
3. 不引入当前项目不存在的 REVERSE 等字段
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from card import Card, Color, CardType
from game_state import GameState


_COLOR_ORDER = [Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN]

# 当前项目的定制牌库：
# 4 色 * (1-5 各 2 张 + Skip 1 张 + Draw Two 1 张) + Wild 3 张 + Wild Draw Four 3 张
_INITIAL_COLOR_REMAINING = {
    Color.RED: 12,
    Color.BLUE: 12,
    Color.YELLOW: 12,
    Color.GREEN: 12,
}
_INITIAL_TYPE_REMAINING = {
    CardType.NUMBER: 40,
    CardType.SKIP: 4,
    CardType.DRAW_TWO: 4,
    CardType.WILD: 3,
    CardType.WILD_DRAW_FOUR: 3,
}

COLOR_REMAINING = dict(_INITIAL_COLOR_REMAINING)
TYPE_REMAINING = defaultdict(int, _INITIAL_TYPE_REMAINING)
PLAYED_COUNT = 0
_SEEN_SET: set[str] = set()
_LAST_TOP_CARD: Card | None = None


def _card_key(card: Card) -> str:
    return f"{card.color.name}|{card.type.name}|{card.value}"


def _auto_record(card: Card) -> None:
    """记牌：同一张顶牌只记录一次。"""
    global PLAYED_COUNT, _LAST_TOP_CARD

    key = _card_key(card)
    if key in _SEEN_SET:
        return

    _SEEN_SET.add(key)
    PLAYED_COUNT += 1
    _LAST_TOP_CARD = card

    if card.color != Color.BLACK and card.color in COLOR_REMAINING:
        COLOR_REMAINING[card.color] = max(0, COLOR_REMAINING[card.color] - 1)
    TYPE_REMAINING[card.type] = max(0, TYPE_REMAINING[card.type] - 1)


def _check_and_update(state: GameState) -> None:
    """只要发现顶牌变化，就自动记录。"""
    global _LAST_TOP_CARD
    if state.top_card and state.top_card != _LAST_TOP_CARD:
        _auto_record(state.top_card)


def reset_card_counter() -> None:
    global COLOR_REMAINING, TYPE_REMAINING, PLAYED_COUNT, _SEEN_SET, _LAST_TOP_CARD
    COLOR_REMAINING = dict(_INITIAL_COLOR_REMAINING)
    TYPE_REMAINING = defaultdict(int, _INITIAL_TYPE_REMAINING)
    PLAYED_COUNT = 0
    _SEEN_SET = set()
    _LAST_TOP_CARD = None


def get_counter_summary() -> str:
    colors = [f"{c.value}:{COLOR_REMAINING[c]}" for c in _COLOR_ORDER]
    return f"已记 {PLAYED_COUNT} 张 | 剩余颜色: {' '.join(colors)}"


def get_safest_color() -> Color:
    """选择剩余最少的颜色，尽量切到对手更不容易接的颜色。"""
    return min(_COLOR_ORDER, key=lambda c: COLOR_REMAINING[c])


def legal_plays(hand: list[Card], state: GameState) -> list[int]:
    """
    返回手牌中所有可合法打出的牌索引。

    当前项目的规则约定：
    1. 同色合法
    2. 同数字合法
    3. 同功能牌合法（Skip 接 Skip，+2 接 +2）
    4. Wild / Wild Draw Four 均视为可出
    5. 叠罚状态下：
       - 顶牌是 +4 时，可接 +4 或 +2
       - 顶牌是 +2 时，只可接 +2
    """
    _check_and_update(state)

    active = state.current_color
    top = state.top_card
    if active is None or top is None:
        return []

    legal: list[int] = []
    for i, card in enumerate(hand):
        if state.under_attack:
            if _can_stack(card, top):
                legal.append(i)
        else:
            if _is_playable(card, active, top):
                legal.append(i)
    return legal


def _is_playable(card: Card, active_color: Color, top: Card) -> bool:
    if card.is_wild:
        return True
    if card.color == active_color:
        return True
    if card.type == top.type and card.type != CardType.NUMBER:
        return True
    if card.type == CardType.NUMBER and top.type == CardType.NUMBER:
        return card.value == top.value
    return False


def _can_stack(card: Card, top: Card) -> bool:
    if top.type == CardType.WILD_DRAW_FOUR:
        return card.type in (CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR)
    if top.type == CardType.DRAW_TWO:
        return card.type == CardType.DRAW_TWO
    return False


@dataclass
class Decision:
    action: str
    card_index: int | None = None
    chosen_color: Color | None = None
    reason: str = ""

    def to_human(self) -> str:
        if self.action == "draw":
            return "摸牌"
        c = f"，选{self.chosen_color.value}色" if self.chosen_color else ""
        return f"出第{self.card_index + 1}张{c}  [{self.reason}]"


def decide(hand: list[Card], state: GameState) -> Decision:
    _check_and_update(state)
    legal_idx = legal_plays(hand, state)
    if not legal_idx:
        return Decision(action="draw", reason="无合法牌")

    legal_cards = [(i, hand[i]) for i in legal_idx]
    if len(legal_cards) == 1:
        i, c = legal_cards[0]
        return Decision(
            action="play",
            card_index=i,
            chosen_color=_pick_color(hand, c) if c.is_wild else None,
            reason="唯一合法",
        )

    threat_level = _get_threat_level(state)
    scored = [(i, c, _score_smart(c, hand, state, threat_level)) for i, c in legal_cards]
    scored.sort(key=lambda x: x[2], reverse=True)
    best_i, best_c, best_s = scored[0]

    threat_name = {1: "发育", 2: "警惕", 3: "警报"}[threat_level]
    return Decision(
        action="play",
        card_index=best_i,
        chosen_color=_pick_color(hand, best_c) if best_c.is_wild else None,
        reason=f"{threat_name} | 得分={best_s}",
    )


def _get_threat_level(state: GameState) -> int:
    counts = getattr(state, "other_hand_counts", None)
    if not counts:
        return 1
    min_cards = min(counts)
    if min_cards <= 1:
        return 3
    if min_cards <= 3:
        return 2
    return 1


def _score_smart(card: Card, hand: list[Card], state: GameState, threat_level: int) -> int:
    s = 0

    if threat_level == 1:
        weights = {
            CardType.NUMBER: 100,
            CardType.SKIP: 55,
            CardType.DRAW_TWO: 42,
            CardType.WILD: 30,
            CardType.WILD_DRAW_FOUR: 22,
        }
    elif threat_level == 2:
        weights = {
            CardType.NUMBER: 65,
            CardType.SKIP: 78,
            CardType.DRAW_TWO: 92,
            CardType.WILD: 55,
            CardType.WILD_DRAW_FOUR: 48,
        }
    else:
        weights = {
            CardType.NUMBER: 12,
            CardType.SKIP: 165,
            CardType.DRAW_TWO: 178,
            CardType.WILD: 105,
            CardType.WILD_DRAW_FOUR: 220,
        }
    s += weights.get(card.type, 0)

    # 记牌加成：对手可能更缺的颜色，适合作为切色目标或继续压制。
    if card.color != Color.BLACK:
        total_remaining = sum(COLOR_REMAINING.values())
        if total_remaining > 0:
            prob = COLOR_REMAINING[card.color] / total_remaining
            if prob < 0.15:
                s += 28
            elif prob < 0.25:
                s += 14

    # 手牌结构：前期尽量清孤色，后期适当保留主色连续出牌能力。
    if card.color != Color.BLACK:
        cnt = sum(1 for c in hand if c.color == card.color)
        if len(hand) > 5:
            if cnt <= 2:
                s += 15
            elif cnt >= 5:
                s -= 10
        else:
            if cnt >= 3:
                s += 10

    if state.stack_penalty > 0:
        if card.type == CardType.WILD_DRAW_FOUR:
            s += 50
        elif card.type == CardType.DRAW_TWO:
            s += 30

    # 只剩两张时，尽量优先清掉普通数字牌。
    if len(hand) == 2 and card.type == CardType.NUMBER:
        s += 100

    return s


def _pick_color(hand: list[Card], wild_card: Card) -> Color:
    counts = {c: 0 for c in _COLOR_ORDER}
    for card in hand:
        if card.color != Color.BLACK:
            counts[card.color] += 1

    my_best = max(_COLOR_ORDER, key=lambda c: counts[c])
    if counts[my_best] >= 2:
        return my_best
    return get_safest_color()


def format_hand(hand: list[Card]) -> str:
    return " | ".join(f"[{i+1}] {c}" for i, c in enumerate(hand))


def full_report(hand: list[Card], state: GameState) -> str:
    _check_and_update(state)
    legals = legal_plays(hand, state)
    decision = decide(hand, state)
    threat_level = _get_threat_level(state)
    threat_name = {1: "发育", 2: "警惕", 3: "警报"}[threat_level]
    return "\n".join(
        [
            f"记牌: {get_counter_summary()}",
            f"局势: {threat_name} | 场上: {state.active_color} | 顶牌: {state.top_card} | 累罚: {state.stack_penalty}",
            f"手牌: {format_hand(hand)}",
            f"合法: {[f'第{i+1}张' for i in legals] if legals else '无'}",
            f"决策: {decision.to_human()}",
        ]
    )
