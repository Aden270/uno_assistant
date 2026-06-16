"""
UNO 决策引擎 · 模拟对战 —— main.py
4人局：你(AI) vs 三个随机策略对手
"""

from __future__ import annotations
import random
from card import Card, Color, build_deck
from game_state import GameState
from rules import decide, format_hand, full_report


def deal(deck: list[Card], n_players: int, hand_size: int = 7):
    """发牌：每人 hand_size 张，返回手牌列表 + 剩余牌堆"""
    random.shuffle(deck)
    hands = [deck[i * hand_size:(i + 1) * hand_size] for i in range(n_players)]
    return hands, deck[n_players * hand_size:]


def first_card(draw_pile: list[Card]) -> tuple[Card, list[Card]]:
    """翻开第一张（跳过万能牌）"""
    for i, c in enumerate(draw_pile):
        if not c.is_wild:
            return c, draw_pile[:i] + draw_pile[i + 1:]
    return draw_pile[0], draw_pile[1:]  # 极端情况


def opponent_decide(hand: list[Card], state: GameState) -> tuple[int | None, Color | None]:
    """随机策略对手：合法牌里随机选"""
    from rules import legal_plays, _pick_color
    legals = legal_plays(hand, state)
    if not legals:
        return None, None  # 摸牌
    idx = random.choice(legals)
    card = hand[idx]
    color = _pick_color(hand, card) if card.is_wild else None
    return idx, color


def simulate(n_players: int = 4, verbose: bool = True):
    """跑一局"""
    # ── 初始化 ──
    deck = build_deck()
    hands, draw_pile = deal(deck, n_players, hand_size=7)
    top, draw_pile = first_card(draw_pile)
    discard_pile = [top]

    gs = GameState(
        player_count=n_players,
        my_index=0,  # 你是 P0
        top_card=top,
        my_hand=hands[0],
        other_hand_counts=[len(h) for h in hands[1:]],
    )

    p = lambda s: print(s) if verbose else None
    p(f"══════ 开局 ══════")
    p(f"顶牌: {top}  |  你的手牌: {format_hand(hands[0])}")
    p("")

    turn = 0
    max_turns = 200

    while turn < max_turns:
        pi = turn % n_players
        gs.current_turn = pi

        # 检查胜利
        for idx, h in enumerate(hands):
            if len(h) == 0:
                p(f"\n*** P{idx}{'（你）' if idx == 0 else ''} 赢了！***")
                return idx

        p(f"── 第 {turn + 1} 回合 · P{pi}{' ← 你' if pi == 0 else ''} ──")

        if pi == 0:
            # ═══ 你的回合：调用决策引擎 ═══
            gs.my_hand = hands[0]
            print(full_report(hands[0], gs))
            decision = decide(hands[0], gs)

            if decision.action == "draw":
                p("  → 摸牌")
                if draw_pile:
                    hands[0].append(draw_pile.pop())
                    p(f"  → 抽到 {hands[0][-1]}")
                gs.reset_penalty()
            else:
                card = hands[0].pop(decision.card_index)
                if card.is_wild and decision.chosen_color:
                    p(f"  → 出 {card}，选{decision.chosen_color.value}色")
                else:
                    p(f"  → 出 {card}")
                discard_pile.append(card)
                gs.update_top_card(card, decision.chosen_color)
        else:
            # ═══ 对手回合：随机策略 ═══
            idx, chosen = opponent_decide(hands[pi], gs)
            if idx is None:
                p(f"  → 摸牌")
                if draw_pile:
                    hands[pi].append(draw_pile.pop())
                gs.reset_penalty()
            else:
                card = hands[pi].pop(idx)
                if card.is_wild:
                    p(f"  → 出 {card}，选{chosen.value if chosen else '?'}色")
                else:
                    p(f"  → 出 {card}")
                discard_pile.append(card)
                gs.update_top_card(card, chosen)

        p(f"  手牌数: {' | '.join(f'P{i}={len(hands[i])}' for i in range(n_players))}")
        p(f"  累罚: {gs.stack_penalty}")

        # 回收弃牌堆当牌堆不够时
        if len(draw_pile) == 0 and len(discard_pile) > 1:
            top_save = discard_pile.pop()
            random.shuffle(discard_pile)
            draw_pile = discard_pile
            discard_pile = [top_save]
            p("  洗牌")

        p("")
        turn += 1
        gs.advance_turn()

    p("超时平局")
    return -1


if __name__ == "__main__":
    winner = simulate(n_players=4, verbose=True)
    print(f"\n── 胜者: P{winner} ──")
