"""
YOLO class_id → UNO Card 映射层
对接 card.py 的 Card 模型
"""

from card import Card, Color, CardType

# YOLO11n-cls 的 30 类标签（来自 classes.json）
_CLASS_NAMES = [
    "blue_1", "blue_2", "blue_3", "blue_4", "blue_5",
    "blue_draw2", "blue_skip",
    "green_1", "green_2", "green_3", "green_4", "green_5",
    "green_draw2", "green_skip",
    "red_1", "red_2", "red_3", "red_4", "red_5",
    "red_draw2", "red_skip",
    "wild", "wild_draw4",
    "yellow_1", "yellow_2", "yellow_3", "yellow_4", "yellow_5",
    "yellow_draw2", "yellow_skip",
]

# 颜色名 → Color 枚举
_COLOR_MAP = {
    "red": Color.RED,
    "blue": Color.BLUE,
    "yellow": Color.YELLOW,
    "green": Color.GREEN,
    "wild": Color.BLACK,
}

# 类型名 → CardType 枚举
_TYPE_MAP = {
    "draw2": CardType.DRAW_TWO,
    "skip": CardType.SKIP,
    "wild": CardType.WILD,
    "wild_draw4": CardType.WILD_DRAW_FOUR,
}


def class_id_to_card(class_id: int) -> Card:
    """YOLO 输出的 class_id (0-29) → UNO Card 对象"""
    name = _CLASS_NAMES[class_id]
    parts = name.split("_")

    if parts[0] == "wild":
        if len(parts) > 1 and parts[1] == "draw4":
            return Card(Color.BLACK, CardType.WILD_DRAW_FOUR)
        return Card(Color.BLACK, CardType.WILD)

    color = _COLOR_MAP[parts[0]]
    try:
        value = int(parts[1])
        return Card(color, CardType.NUMBER, value)
    except ValueError:
        return Card(color, _TYPE_MAP[parts[1]])


def class_name_to_card(name: str) -> Card:
    """标签名（如 "red_skip"）→ UNO Card 对象"""
    idx = _CLASS_NAMES.index(name)
    return class_id_to_card(idx)


TOP5_THRESHOLD = 0.3  # Top-5 累加概率到这个阈值才认


if __name__ == "__main__":
    # 自测：打印全部映射
    for i, name in enumerate(_CLASS_NAMES):
        print(f"  {i:2d} {name:16s} → {class_id_to_card(i)}")
