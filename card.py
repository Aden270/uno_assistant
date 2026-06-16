"""
UNO 卡牌模型 —— 模块①
定制牌库：
  红/蓝/黄/绿: 数字1-5 各2张 + 跳过×1 + +2×1 = 每色12张
  黑: 改色×3, +4×3 = 6张
  共计 54 张
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class Color(Enum):
    RED = "红"
    BLUE = "蓝"
    YELLOW = "黄"
    GREEN = "绿"
    BLACK = "黑"


class CardType(Enum):
    NUMBER = "数字"
    SKIP = "跳过"
    DRAW_TWO = "+2"
    WILD = "改色"
    WILD_DRAW_FOUR = "+4"


@dataclass(frozen=True)
class Card:
    color: Color
    type: CardType
    value: int = 0

    def __post_init__(self):
        if self.type == CardType.NUMBER:
            if not (1 <= self.value <= 5):
                raise ValueError(f"数字牌取值 1-5，收到 {self.value}")
        if self.color == Color.BLACK and self.type not in (CardType.WILD, CardType.WILD_DRAW_FOUR):
            raise ValueError("黑色只能是万能牌")
        if self.color != Color.BLACK and self.type in (CardType.WILD, CardType.WILD_DRAW_FOUR):
            raise ValueError("万能牌颜色必须为黑色")

    @property
    def is_wild(self) -> bool:
        return self.type in (CardType.WILD, CardType.WILD_DRAW_FOUR)

    def __repr__(self):
        if self.type == CardType.NUMBER:
            return f"{self.color.value}{self.value}"
        return f"{self.color.value}{self.type.value}"


# ────────────────
# 牌库
# ────────────────

def build_deck() -> list[Card]:
    deck: list[Card] = []
    colors = [Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN]

    for c in colors:
        for v in range(1, 6):
            deck.append(Card(c, CardType.NUMBER, v))
            deck.append(Card(c, CardType.NUMBER, v))   # 每个数字两张
        deck.append(Card(c, CardType.SKIP))
        deck.append(Card(c, CardType.DRAW_TWO))

    for _ in range(3):
        deck.append(Card(Color.BLACK, CardType.WILD))
        deck.append(Card(Color.BLACK, CardType.WILD_DRAW_FOUR))

    return deck


if __name__ == "__main__":
    deck = build_deck()
    print(f"共计 {len(deck)} 张牌")
    for card in deck:
        print(f"  {card}")
