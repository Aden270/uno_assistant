"""
UNO 桌面原型 · 完整流水线 v2
摄像头 → ONNX 识别 → 卡牌映射 → 决策引擎 → 屏幕显示

对局流程:
  1. 终端输入玩家人数
  2. 逐张扫描初始手牌(5张)，按 A 入库
  3. 扫描开局顶牌，按 S 确认（万能牌自动拒绝）
  4. 对局循环（默认从你开始出牌）

用法:
  python pipeline.py              # IVCam(索引1)
  python pipeline.py 0            # 指定摄像头索引
"""

from __future__ import annotations
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from card import Card, Color, CardType
from game_state import GameState
from rules import decide, legal_plays, format_hand, _pick_color
from label_mapper import class_id_to_card
from onnx_inference import ONNXClassifier

# ═══════════════════════════════
#  配置
# ═══════════════════════════════

MODEL_PATH = "best.onnx"  # 放在 pipeline.py 同目录下
CONFIDENCE_THRESHOLD = 0.05  # 手机实拍背景差异大，置信度偏低，降低阈值
INFER_EVERY_N = 15
INITIAL_HAND_SIZE = 5
DEBUG_TOP3 = True           # 调试：显示 Top-3 预测结果

# 万能牌选色按键映射
_COLOR_KEYS = {
    ord("r"): Color.RED,
    ord("b"): Color.BLUE,
    ord("y"): Color.YELLOW,
    ord("g"): Color.GREEN,
}

# ═══════════════════════════════
#  中文字体
# ═══════════════════════════════

_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
try:
    _FONT = ImageFont.truetype(_FONT_PATH, 18)
except Exception:
    try:
        _FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
        _FONT = ImageFont.truetype(_FONT_PATH, 18)
    except Exception:
        _FONT = ImageFont.load_default()


def _pil_text(frame: np.ndarray, text: str, pos: tuple, color: tuple, size: int = 18):
    x, y = pos
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        font = _FONT
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════
#  UI 绘制
# ═══════════════════════════════

def draw_overlay(frame: np.ndarray, gs: GameState, last_card: Card | None,
                 last_conf: float, top5: list, message: str, phase: str,
                 my_hand: list[Card], n_players: int):
    h, w = frame.shape[:2]

    # 顶部半透明背景
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 175), (30, 30, 30), -1)
    frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)

    y = 8

    label_color = (255, 200, 50) if phase != "对局" else (100, 255, 100)
    frame = _pil_text(frame, f"【{phase}】", (10, y), label_color, 18)
    y += 24

    if phase == "对局":
        cur = "你" if gs.current_turn == gs.my_index else f"P{gs.current_turn}"
    else:
        cur = "你"
    frame = _pil_text(frame, f"当前出牌人: {cur}", (10, y), (220, 220, 220), 16)
    y += 22

    frame = _pil_text(frame,
        f"上一张牌: {gs.top_card or '—'}  |  场上颜色: {gs.active_color or '—'}  |  累计罚牌: {gs.stack_penalty}",
        (10, y), (200, 200, 200), 15)
    y += 20

    frame = _pil_text(frame,
        f"我的手牌({len(my_hand)}张): {format_hand(my_hand)}",
        (10, y), (180, 255, 180), 14)
    y += 22

    # 识别结果
    if last_card:
        frame = _pil_text(frame, f"识别: {last_card} ({last_conf:.2f})",
                          (10, y), (0, 255, 100), 16)
        y += 18

    # 调试：显示 Top-3
    if DEBUG_TOP3 and top5:
        top_str = " | ".join(
            f"{class_id_to_card(cid)}({conf:.2f})" for cid, conf in top5[:3]
        )
        frame = _pil_text(frame, f"Top3: {top_str}", (10, y), (150, 150, 200), 13)
        y += 18

    msg_color = (100, 100, 255) if "警告" in message else (255, 255, 100)
    frame = _pil_text(frame, message, (10, y + 2), msg_color, 16)

    if "请选择颜色" in message:
        guide = "R=红  B=蓝  Y=黄  G=绿  —— 选色后自动继续"
    elif phase == "初始手牌":
        guide = "A=确认识别入库  |  Q=退出"
    elif phase == "开局翻牌":
        guide = "S=确认开局顶牌  |  Q=退出"
    else:
        guide = "A=抓牌入库  |  S=确认有人出牌  |  D=有人摸牌(无牌可出)  |  Q=退出"
    frame = _pil_text(frame, guide, (10, h - 28), (150, 150, 150), 15)

    return frame


# ═══════════════════════════════
#  基础合法性（不依赖手牌，用于验证对手出牌）
# ═══════════════════════════════

def is_basic_legal(card: Card, state: GameState) -> bool:
    """检查一张牌是否至少满足一条基本规则（不依赖手牌上下文）"""
    # 叠加罚牌中：只能出 +2 或 +4，无视颜色/数字
    if state.under_attack:
        return card.type in (CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR)

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


# ═══════════════════════════════
#  主流程
# ═══════════════════════════════

def main():
    # ── 终端输入 ──
    try:
        n_players = int(input("玩家人数（含你）: ").strip())
        if n_players < 2:
            print("至少 2 人")
            return
    except ValueError:
        print("请输入数字")
        return

    my_index = int(input(f"你是第几个人（0~{n_players - 1}，默认0）: ").strip() or "0")

    # ── 摄像头 ──
    source = sys.argv[1] if len(sys.argv) > 1 else "1"
    if source.startswith("http"):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(int(source))

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # ── 模型 ──
    try:
        clf = ONNXClassifier(MODEL_PATH)
        print(f"ONNX 模型加载成功 ({clf.input_w}x{clf.input_h})")
    except Exception as e:
        print(f"模型加载失败: {e}")
        cap.release()
        return

    # ── 游戏状态 ──
    gs = GameState(
        player_count=n_players,
        my_index=my_index,
    )
    my_hand: list[Card] = []

    last_card: Card | None = None
    last_conf: float = 0.0
    last_top5: list = []
    message = "将摄像头对准手牌，按 A 逐张入库"
    phase = "初始手牌"
    frame_count = 0
    warning_timer = 0
    pending_wild: Card | None = None   # 等待选择颜色的万能牌
    pending_wild_player: int = -1      # 出这张万能牌的人（索引）
    draw_count: int = 0                # 待摸牌数（>0 时 A 键逐张入库）

    print(f"\n{phase}: 逐张扫描 {INITIAL_HAND_SIZE} 张手牌，按 A 入库")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 定时推理 ──
        if frame_count % INFER_EVERY_N == 0:
            class_id, conf, top5 = clf.predict(frame)
            last_top5 = top5
            if conf >= CONFIDENCE_THRESHOLD:
                last_card = class_id_to_card(class_id)
                last_conf = conf
            else:
                last_card = None

        # ── 警告计时 ──
        if warning_timer > 0:
            warning_timer -= 1
            if warning_timer == 0 and "警告" in message:
                message = ""

        # ── 绘制 ──
        frame = draw_overlay(frame, gs, last_card, last_conf, last_top5, message, phase, my_hand, n_players)
        cv2.imshow("UNO 决策引擎", frame)

        key = cv2.waitKey(1) & 0xFF

        # ═══════════════════════
        #  全局按键
        # ═══════════════════════

        if key == ord("q"):
            break

        # ── 摸牌中拦截：只能按 A 入库，屏蔽其他操作 ──
        if draw_count > 0 and key != ord("a"):
            if key == ord("s"):
                message = f"请先完成摸牌！还需按 A {draw_count} 次"
                warning_timer = 45
            elif key == ord("d"):
                message = "已在摸牌中，请勿重复摸牌"
                warning_timer = 45
            frame_count += 1
            continue

        # ── A 键: 手牌入库 ──
        elif key == ord("a"):
            if last_card is None:
                message = "警告: 未识别到牌，请对准后重试"
                warning_timer = 45
            else:
                if phase == "初始手牌":
                    my_hand.append(last_card)
                    message = f"已入库: {last_card}  ({len(my_hand)}/{INITIAL_HAND_SIZE})"
                    print(f"  手牌 {len(my_hand)}/{INITIAL_HAND_SIZE}: {last_card}")
                    if len(my_hand) >= INITIAL_HAND_SIZE:
                        phase = "开局翻牌"
                        gs.my_hand = list(my_hand)
                        gs.current_turn = my_index
                        message = f"手牌已满！请扫描开局顶牌，按 S 确认"
                        print(f"\n{phase}: 扫描开局顶牌（不能是万能牌），按 S 确认")
                elif phase == "对局":
                    my_hand.append(last_card)
                    gs.my_hand = list(my_hand)
                    if draw_count > 0:
                        draw_count -= 1
                        if draw_count == 0:
                            # 摸完最后一张，推进回合
                            gs.advance_turn()
                            message = f"第{len(my_hand)}张入库: {last_card}  —— 摸牌完毕，轮到下家"
                            message = _update_msg_after_play(gs, message, my_hand, n_players)
                        else:
                            message = f"已入库({len(my_hand)}张): {last_card}  —— 还需摸 {draw_count} 张"
                    else:
                        message = f"已抓牌入库: {last_card}"
                    print(f"  抓牌: {last_card}")
                last_card = None

        # ── S 键: 确认出牌 ──
        elif key == ord("s"):
            if last_card is None:
                message = "警告: 未识别到牌"
                warning_timer = 45
            elif phase == "开局翻牌":
                if last_card.is_wild:
                    message = "警告: 开局顶牌不能是万能牌，请换一张"
                    warning_timer = 45
                else:
                    gs.top_card = last_card
                    gs.current_color = last_card.color
                    gs.current_turn = my_index
                    gs.my_hand = list(my_hand)
                    phase = "对局"
                    message = f"开局牌: {last_card}  轮到你了"
                    message = _update_msg_after_play(gs, message, my_hand, n_players)
                    print(f"\n对局开始！顶牌: {last_card}，轮到你了")
                last_card = None

            elif phase == "对局":
                is_my_turn = gs.current_turn == gs.my_index

                if is_my_turn:
                    # 我出牌：牌必须在手牌里且合法
                    found_idx = None
                    for i, c in enumerate(my_hand):
                        if c == last_card:
                            found_idx = i
                            break

                    if found_idx is None:
                        message = f"警告: {last_card} 不在你的手牌中！"
                        warning_timer = 45
                    elif found_idx not in legal_plays(my_hand, gs):
                        message = f"警告: {last_card} 不合法！请换一张或按D摸牌"
                        warning_timer = 45
                    else:
                        my_hand.pop(found_idx)
                        if last_card.is_wild:
                            # 万能牌：进入等待选色状态
                            pending_wild = last_card
                            pending_wild_player = my_index
                            message = f"你出了 {last_card}，请选择颜色: R=红 B=蓝 Y=黄 G=绿"
                        else:
                            _do_play(gs, last_card, my_hand, my_index)
                            gs.advance_turn()
                            if last_card.type == CardType.SKIP:
                                gs.advance_turn()  # 跳过下家
                            message = f"你出了 {last_card}"
                            message = _update_msg_after_play(gs, message, my_hand, n_players)
                    last_card = None
                else:
                    # 别人出牌：只检查基础合法性
                    if is_basic_legal(last_card, gs):
                        if last_card.is_wild:
                            # 万能牌：进入等待选色状态
                            pending_wild = last_card
                            pending_wild_player = gs.current_turn
                            who = f"P{gs.current_turn}"
                            message = f"{who} 出了 {last_card}，请选择颜色: R=红 B=蓝 Y=黄 G=绿"
                        else:
                            _do_play(gs, last_card, my_hand, gs.current_turn)
                            gs.advance_turn()
                            if last_card.type == CardType.SKIP:
                                gs.advance_turn()  # 跳过下家
                            prev = (gs.current_turn - 1) if gs.current_turn > 0 else (n_players - 1)
                            message = f"P{prev} 出了 {last_card}"
                            message = _update_msg_after_play(gs, message, my_hand, n_players)
                    else:
                        message = f"警告: {last_card} 不匹配场上颜色/数字！"
                        warning_timer = 45
                    last_card = None

        # ── R/B/Y/G 键: 万能牌选色 ──
        elif pending_wild and key in _COLOR_KEYS:
            chosen = _COLOR_KEYS[key]
            who = "你" if pending_wild_player == my_index else f"P{pending_wild_player}"
            _do_play(gs, pending_wild, my_hand, pending_wild_player, chosen_color=chosen)
            gs.advance_turn()
            message = f"{who} 出了 {pending_wild}（选{chosen.value}色）"
            message = _update_msg_after_play(gs, message, my_hand, n_players)
            pending_wild = None
            pending_wild_player = -1
            print(f"  {who} 选色 → {chosen.value}")

        # ── D 键: 有人摸牌 ──
        elif key == ord("d") and phase == "对局":
            is_me_drawing = gs.current_turn == gs.my_index
            who = "你" if is_me_drawing else f"P{gs.current_turn}"
            penalty = gs.stack_penalty
            gs.reset_penalty()

            if is_me_drawing:
                n = max(penalty, 1)  # 至少摸 1 张
                draw_count = n
                if penalty > 0:
                    message = f"罚牌 {penalty} 张！请逐张扫描按 A 入库（还需 {n} 张）"
                else:
                    message = f"无合法牌，请摸 1 张按 A 入库"
            else:
                gs.advance_turn()
                message = f"{who} 摸牌（罚{penalty}张），罚牌清零，轮到下家" if penalty else f"{who} 摸牌，轮到下家"
                message = _update_msg_after_play(gs, message, my_hand, n_players)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════
#  辅助函数
# ═══════════════════════════════

def _do_play(gs: GameState, card: Card, my_hand: list[Card], player_idx: int,
             chosen_color: Color | None = None):
    """更新 GameState 以反映一张牌被打出。chosen_color 由外部（选色按键）传入"""
    gs.update_top_card(card, chosen_color)
    gs.my_hand = list(my_hand)


def _update_msg_after_play(gs: GameState, base_msg: str, my_hand: list[Card], n_players: int) -> str:
    """出牌后追加：如果轮到我了，给 AI 建议"""
    if gs.current_turn == gs.my_index:
        d = decide(my_hand, gs)
        return base_msg + f"  |  AI建议: {d.to_human()}"
    return base_msg


if __name__ == "__main__":
    main()
