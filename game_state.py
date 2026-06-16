"""
UNO 游戏状态 —— 模块②
维护一局游戏中的动态状态。初始化数据由外部（A模块/人工UI）喂入。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from card import Card, Color, CardType


@dataclass
class GameState:
    """
    一局 UNO 的完整运行时状态。

    初始化示例:
      gs = GameState(
          player_count=4,
          my_index=0,
          top_card=some_card,          # 开局翻开的第一张
          my_hand=[...],               # 初始手牌（A模块识别结果）
          other_hand_counts=[7,7,7],   # 其他人初始手牌数
      )
    """

    player_count: int
    my_index: int

    # ── 动态字段 ──
    current_color: Color | None = None      # None = 从 top_card 取（万能牌时等人工选）
    top_card: Card | None = None
    stack_penalty: int = 0                  # 当前累积抽牌罚数
    current_turn: int = 0                   # 当前轮到谁（索引 0 ~ player_count-1）
    my_hand: list[Card] = field(default_factory=list)
    other_hand_counts: list[int] = field(default_factory=list)

    _turn_count: int = field(default=0, repr=False)

    # ═══════════════
    #  初始化校验
    # ═══════════════

    def __post_init__(self):
        if self.current_color is None and self.top_card is not None:
            if not self.top_card.is_wild:
                self.current_color = self.top_card.color
        if not self.other_hand_counts:
            self.other_hand_counts = [0] * max(0, self.player_count - 1)

    # ═══════════════
    #  便捷查询
    # ═══════════════

    @property
    def is_my_turn(self) -> bool:
        return self.current_turn == self.my_index

    @property
    def active_color(self) -> Color | None:
        return self.current_color

    @property
    def under_attack(self) -> bool:
        """当前是否有叠加罚牌需要处理"""
        return self.stack_penalty > 0

    # ═══════════════
    #  状态变更
    # ═══════════════

    def update_top_card(self, card: Card, chosen_color: Color | None = None):
        """
        外部通知：牌堆顶更新（有人出牌后调用）。
        - card: 出的牌
        - chosen_color: 万能牌时指定的颜色
        """
        self.top_card = card
        self._turn_count += 1

        if card.is_wild:
            self.current_color = chosen_color
        else:
            self.current_color = card.color

        # 累积罚牌
        if card.type == CardType.DRAW_TWO:
            self.stack_penalty += 2
        elif card.type == CardType.WILD_DRAW_FOUR:
            self.stack_penalty += 4

    def reset_penalty(self):
        """玩家抽牌后调用：罚牌累计清零"""
        self.stack_penalty = 0

    def advance_turn(self):
        """轮到下一个人出牌"""
        self.current_turn = (self.current_turn + 1) % self.player_count

    def set_hand(self, cards: list[Card]):
        self.my_hand = list(cards)

    def update_other_hands(self, counts: list[int]):
        self.other_hand_counts = list(counts)

    def __repr__(self):
        me_flag = "(我)" if self.is_my_turn else ""
        return (
            f"GameState(#{self._turn_count} P{self.current_turn}{me_flag}, "
            f"色={self.current_color}, 顶={self.top_card}, "
            f"累罚={self.stack_penalty}, 手牌={len(self.my_hand)}张)"
        )
