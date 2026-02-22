"""
托盘装箱求解器 (Pallet Packing Solver)
======================================
使用以下两种算法找到在托盘上堆放纸箱的最优方案：
  1. Guillotine 2D 切割装箱算法 —— 计算每一层的平面布局
  2. 无界背包动态规划 (Unbounded Knapsack DP) —— 纵向叠层优化

类说明：
  PalletSolver  — 单一 SKU 在一个托盘上的装箱求解
  Pallet        — 一个物理托盘的可变数据模型
  OrderSolver   — 将整张订单分配到多个托盘上
"""

import copy
import math


# ================================================================== #
#  单一 SKU 求解器
# ================================================================== #

class PalletSolver:
    """计算单一 SKU 在一个托盘上能放置的最大纸箱数，并生成 3D 布局数据。"""

    def __init__(self, pallet_w, pallet_l, max_h, item_w, item_l, item_h, tolerance=0.0):
        """
        参数:
            pallet_w   — 托盘宽度
            pallet_l   — 托盘长度
            max_h      — 可用堆叠高度（不含托盘底板高度）
            item_w     — 纸箱宽度
            item_l     — 纸箱长度
            item_h     — 纸箱高度
            tolerance  — 允许的悬挂量 (每侧)
        """
        self.pallet_w = float(pallet_w) + float(tolerance)  # 加上公差后的有效宽度
        self.pallet_l = float(pallet_l) + float(tolerance)  # 加上公差后的有效长度
        self.max_h    = float(max_h)
        self.item_w   = float(item_w)
        self.item_l   = float(item_l)
        self.item_h   = float(item_h)
        self.tol      = float(tolerance)

    # -------------------------------------------------------------- #
    #  公共接口
    # -------------------------------------------------------------- #

    def solve(self):
        """
        主入口：生成候选层 → 背包求解 → 按稳定性排序。
        返回: {"count": 总数, "layers": [层数据列表]}
        """

        # ---- 第 1 步：为每种纸箱摆放方向生成候选层 ----
        # 每种方向 = (名称, 该方向下的层高, 该方向下纸箱在层面的宽, 长)
        orientations = [
            ("flat", self.item_h, self.item_w, self.item_l),  # 正放（高度最矮）
            ("side", self.item_l, self.item_w, self.item_h),  # 侧放
            ("end",  self.item_w, self.item_l, self.item_h),  # 立放
        ]

        layers = []
        for name, height, w, l in orientations:
            if height > self.max_h:  # 该方向层高超出可用空间，跳过
                continue
            layout = self._pack2d(self.pallet_w, self.pallet_l, w, l)
            if layout["count"] > 0:
                layers.append({
                    "type":   name,
                    "height": height,
                    "count":  layout["count"],
                    "items":  layout["items"],
                    "dims":   {"w": w, "l": l, "h": height},
                    "area":   w * l,  # 单个纸箱占用的底面积（用于稳定性排序）
                })

        if not layers:
            return {"count": 0, "layers": []}

        # ---- 第 2 步：无界背包 DP，在纵向高度内最大化纸箱数 ----
        best = self._knapsack(layers)

        # ---- 第 3 步：按稳定性排序：密度高 / 底面积大的层放在底部 ----
        best["layers"].sort(key=lambda ly: (ly["count"], ly["area"]), reverse=True)
        return best

    # -------------------------------------------------------------- #
    #  空隙填充辅助方法（供 OrderSolver 调用）
    # -------------------------------------------------------------- #

    @staticmethod
    def recover_free_space(W, L, placed_items):
        """
        根据已放置的物品列表，重放 Guillotine 切割过程，
        重建剩余可用的自由矩形列表。

        参数:
            W, L          — 层平面的总宽度和长度
            placed_items  — 已放置物品列表 [{x, y, w, l, ...}, ...]
        返回:
            自由矩形列表 [{x, y, w, l}, ...]
        """
        free = [{"x": 0.0, "y": 0.0, "w": W, "l": L}]  # 初始为整个平面

        for item in placed_items:
            ix, iy = item["x"], item["y"]

            # 找到包含此物品原点的自由矩形
            found = -1
            for i, r in enumerate(free):
                if (ix >= r["x"] - 1e-5 and iy >= r["y"] - 1e-5
                        and ix < r["x"] + r["w"] and iy < r["y"] + r["l"]):
                    found = i
                    break

            if found != -1:
                r = free.pop(found)
                free.extend(PalletSolver._split(r, item["w"], item["l"]))
            # 若未找到匹配的自由矩形（浮点误差），安全跳过

        return free

    def try_pack_in_free_space(self, free_rects, specs, max_qty):
        """
        尝试将最多 max_qty 个物品塞入给定的自由矩形中。

        参数:
            free_rects — 自由矩形列表
            specs      — 物品规格 {"w", "l", "h", ...}
            max_qty    — 最多放置数量
        返回:
            {"items": [放置结果], "count": 成功放置数}
        """
        current_free = copy.deepcopy(free_rects)  # 深拷贝，不修改原始数据
        packed = []

        while len(packed) < max_qty:
            # 使用 "mixed" 策略（允许旋转），最灵活地填补空隙
            idx, rotated, _ = self._best_fit(current_free, specs["w"], specs["l"], "mixed")
            if idx == -1:  # 放不下了
                break

            r = current_free.pop(idx)
            pw, pl = (specs["l"], specs["w"]) if rotated else (specs["w"], specs["l"])
            packed.append({"x": r["x"], "y": r["y"], "w": pw, "l": pl, "rotated": rotated})
            current_free.extend(self._split(r, pw, pl))  # 切割剩余空间

        return {"items": packed, "count": len(packed)}

    # -------------------------------------------------------------- #
    #  无界背包 DP（最大化纸箱数量）
    # -------------------------------------------------------------- #

    def _knapsack(self, layers):
        """
        在可用高度内，通过无界背包 DP 选择最优的层组合。
        目标：最大化纸箱总数；相同数量时优先选底面积更大的（更稳定）。
        """
        SCALE = 1000  # 将浮点高度放大为整数，避免精度问题
        limit = int(math.floor(self.max_h * SCALE))

        # dp[h] = 高度恰好为 h 时的最优方案 {"count", "score", "layers"}
        dp = [None] * (limit + 1)
        dp[0] = {"count": 0, "score": 0, "layers": []}
        best = dp[0]

        for h in range(limit + 1):
            if dp[h] is None:
                continue
            for layer in layers:
                nh = h + int(math.floor(layer["height"] * SCALE))  # 新高度
                if nh > limit:
                    continue

                new_count = dp[h]["count"] + layer["count"]     # 累积纸箱数
                new_score = dp[h]["score"] + layer["area"]       # 累积底面积
                cur = dp[nh]

                # 更新条件：数量更多，或数量相同但稳定性更好
                if (cur is None
                        or new_count > cur["count"]
                        or (new_count == cur["count"] and new_score > cur["score"])):
                    dp[nh] = {
                        "count":  new_count,
                        "score":  new_score,
                        "layers": dp[h]["layers"] + [layer],
                    }
                    if (new_count > best["count"]
                            or (new_count == best["count"] and new_score > best["score"])):
                        best = dp[nh]

        return best

    # -------------------------------------------------------------- #
    #  2D Guillotine 切割装箱
    # -------------------------------------------------------------- #

    def _pack2d(self, W, L, w, l):
        """
        尝试三种策略（混合 / 不旋转 / 仅旋转），返回最优的 2D 布局。
        """
        best = {"count": -1, "items": []}
        for strategy in ("mixed", "normal", "rotated"):
            result = self._guillotine(W, L, w, l, strategy)
            if result["count"] > best["count"]:
                best = result
        return best

    def _guillotine(self, W, L, w, l, strategy):
        """
        Guillotine 切割装箱：贪心地将物品放入最佳匹配的自由矩形中，
        放置后将剩余空间切割成两个子矩形。
        """
        items = []
        free = [{"x": 0.0, "y": 0.0, "w": W, "l": L}]  # 可用空间列表

        while True:
            idx, rotated, _ = self._best_fit(free, w, l, strategy)
            if idx == -1:   # 没有足够大的自由矩形了
                break
            r = free.pop(idx)
            pw, pl = (l, w) if rotated else (w, l)
            items.append({"x": r["x"], "y": r["y"], "w": pw, "l": pl, "rotated": rotated})
            free.extend(self._split(r, pw, pl))  # 切割并添加新的自由矩形

        return {"count": len(items), "items": items}

    @staticmethod
    def _best_fit(free, w, l, strategy):
        """
        在自由矩形列表中找到「最短边残余最小」的矩形（Best Short Side Fit）。
        支持三种策略：
          - "normal"  : 不旋转
          - "rotated" : 旋转 90°
          - "mixed"   : 两种都尝试，取最优
        返回: (矩形索引, 是否旋转, 匹配分数)
        """
        best_idx, best_rot, best_score = -1, False, float("inf")
        for i, r in enumerate(free):
            candidates = []
            # 不旋转：宽 w ≤ 矩形宽，长 l ≤ 矩形长
            if strategy in ("mixed", "normal") and w <= r["w"] and l <= r["l"]:
                candidates.append((False, min(r["w"] - w, r["l"] - l)))
            # 旋转 90°：长 l ≤ 矩形宽，宽 w ≤ 矩形长
            if strategy in ("mixed", "rotated") and l <= r["w"] and w <= r["l"]:
                candidates.append((True, min(r["w"] - l, r["l"] - w)))
            for rot, score in candidates:
                if score < best_score:
                    best_idx, best_rot, best_score = i, rot, score
        return best_idx, best_rot, best_score

    @staticmethod
    def _split(r, pw, pl):
        """
        在自由矩形 r 的原点放置一个 pw×pl 的物品后，
        将剩余空间切割成两个子矩形。
        切割方向选择「使较大子矩形面积最大化」的方式。
        """
        rw = r["w"] - pw   # 右侧剩余宽度
        rl = r["l"] - pl   # 上方剩余长度
        children = []

        # 比较两种切割方向，选择能产生更大子矩形的方向
        if max(r["w"] * rl, rw * pl) > max(rw * r["l"], pw * rl):
            # 水平切割：上方子矩形取满宽
            if rl > 0:
                children.append({"x": r["x"], "y": r["y"] + pl, "w": r["w"], "l": rl})
            if rw > 0:
                children.append({"x": r["x"] + pw, "y": r["y"], "w": rw, "l": pl})
        else:
            # 垂直切割：右侧子矩形取满长
            if rw > 0:
                children.append({"x": r["x"] + pw, "y": r["y"], "w": rw, "l": r["l"]})
            if rl > 0:
                children.append({"x": r["x"], "y": r["y"] + pl, "w": pw, "l": rl})

        return children


# ================================================================== #
#  托盘数据模型
# ================================================================== #

class Pallet:
    """表示一个物理托盘及其上面已放置的层。"""

    def __init__(self, pW, pL, bH, maxH, tol):
        """
        参数:
            pW, pL  — 托盘宽度、长度
            bH      — 托盘底板高度
            maxH    — 允许的最大总高度（含底板）
            tol     — 公差 / 悬挂量
        """
        self.pW = pW
        self.pL = pL
        self.bH = bH
        self.maxH = maxH
        self.tol = tol
        self.layers: list[dict] = []      # 已放置的层列表
        self.skus: dict[str, bool] = {}   # 此托盘包含的所有 SKU 名称 (通过字典保持插入顺序)
        self.height = bH                  # 当前已使用的总高度

    def remaining_height(self):
        """返回剩余可用高度。"""
        return self.maxH - self.height

    def add_layer(self, sku_name, layer_data, count_override=None):
        """
        添加一个完整（或部分）层到托盘顶部。

        参数:
            sku_name        — SKU 名称
            layer_data      — PalletSolver 产出的层数据
            count_override  — 若不为 None，则只取前 N 个物品（用于最后一层不满的情况）
        """
        layer = copy.deepcopy(layer_data)  # 深拷贝，防止修改原缓存

        # 如果实际需要的数量少于该层满载量，截取前 N 个
        if count_override is not None and count_override < layer["count"]:
            layer["items"] = layer["items"][:count_override]
            layer["count"] = count_override

        # 给每个物品打上 SKU 标签（用于渲染时按 SKU 着色）
        for item in layer["items"]:
            item["sku"] = sku_name

        self.layers.append({
            "sku":    sku_name,
            "height": layer["height"],
            "dims":   layer["dims"],
            "items":  layer["items"],
            "count":  layer["count"],
        })
        self.height += layer["height"]
        self.skus[sku_name] = True

    def add_to_top_layer(self, sku_name, new_items, new_h):
        """
        将额外的物品注入到现有的顶层中（空隙填充）。
        如果新物品的高度大于顶层高度，则扩展顶层高度。
        """
        if not self.layers:
            return

        top = self.layers[-1]

        # 给新物品打上 SKU 标签
        for item in new_items:
            item["sku"] = sku_name

        top["items"].extend(new_items)
        top["count"] += len(new_items)

        # 如果新物品使层变高，更新层高和托盘总高
        if new_h > top["height"]:
            self.height += new_h - top["height"]
            top["height"] = new_h

        self.skus[sku_name] = True


# ================================================================== #
#  订单求解器（多 SKU、多托盘）
# ================================================================== #

class OrderSolver:
    """将整张订单的所有 SKU 分配到多个托盘上，使用自下而上的装箱策略。"""

    def __init__(self, pW, pL, bH, maxH, tol, max_mixed):
        """
        参数:
            pW, pL     — 托盘宽度、长度
            bH         — 托盘底板高度
            maxH       — 最大允许总高度
            tol        — 公差
            max_mixed  — 单个托盘上允许的最多 SKU 种类数
        """
        self.pW  = float(pW)
        self.pL  = float(pL)
        self.bH  = float(bH)
        self.maxH = float(maxH)
        self.tol  = float(tol)
        self.max_mixed = int(max_mixed)
        self.usable_h  = self.maxH - self.bH  # 可用于堆叠纸箱的净高度

        # 布局缓存：sku_name → 最优单层布局（避免重复计算）
        self._layout_cache: dict[str, dict | None] = {}

    def solve(self, order_items):
        """
        主入口：将订单物品列表装箱到多个托盘中。

        参数:
            order_items — [{name, qty, w, l, h}, ...] 按订单顺序排列
        返回:
            托盘字典列表
        """
        # 策略：按原订单顺序装箱
        stream = [dict(x) for x in order_items]
        sku_specs = {x["name"]: x for x in order_items}  # SKU 名称 → 规格

        pallets: list[dict] = []
        current: Pallet | None = None

        def _new_pallet():
            return Pallet(self.pW, self.pL, self.bH, self.maxH, self.tol)

        idx = 0
        while idx < len(stream):
            item = stream[idx]
            name = item["name"]
            qty  = item["qty"]
            specs = sku_specs.get(name)
            if not specs:  # 找不到规格，跳过
                idx += 1
                continue

            while qty > 0:
                # 确保有一个当前在用的托盘
                if current is None:
                    current = _new_pallet()

                # 检查混装限制：如果当前托盘已达到最大 SKU 种类数且此 SKU 是新的
                if name not in current.skus and len(current.skus) >= self.max_mixed:
                    pallets.append(self._to_dict(current, len(pallets) + 1))
                    current = _new_pallet()

                # ==== 步骤 A：尝试填充顶层空隙 ====
                qty = self._try_fill_voids(current, name, specs, qty)
                if qty <= 0:
                    break  # 全部通过空隙填充完成

                # ==== 步骤 B：标准整层装箱（使用完整多层方案） ====
                plan = self._get_full_plan(name, specs, current.remaining_height())
                if plan is None:
                    if current.height == self.bH:
                        # 即使空托盘都放不下 → 跳过此 SKU
                        print(f"跳过 {name}：尺寸超出托盘限制。")
                        qty = 0
                        break
                    else:
                        # 当前托盘放不下更多层了 → 封板，开新托盘
                        pallets.append(self._to_dict(current, len(pallets) + 1))
                        current = _new_pallet()
                        continue

                # 按方案中的每层逐层添加（可能包含不同朝向的层）
                placed_any = False
                for plan_layer in plan:
                    if qty <= 0:
                        break
                    if plan_layer["height"] > current.remaining_height():
                        break  # 后续层放不下了
                    actual = min(qty, plan_layer["count"])
                    current.add_layer(name, plan_layer, count_override=actual)
                    qty -= actual
                    placed_any = True

                if not placed_any:
                    # 一层都放不进 → 封板
                    pallets.append(self._to_dict(current, len(pallets) + 1))
                    current = _new_pallet()
                    continue

            idx += 1

        # 将最后一个未封板的托盘加入结果
        if current and current.layers:
            pallets.append(self._to_dict(current, len(pallets) + 1))

        return pallets

    # -------------------------------------------------------------- #
    #  内部辅助方法
    # -------------------------------------------------------------- #

    def _try_fill_voids(self, pallet, sku_name, specs, qty):
        """
        尝试将物品塞入托盘顶层的空隙中。
        返回填充后剩余的数量。
        """
        if not pallet.layers:
            return qty

        top = pallet.layers[-1]

        # 用辅助 Solver 来调用静态/实例方法
        helper = PalletSolver(self.pW, self.pL, 100, 1, 1, 1, self.tol)

        # 重建顶层的自由矩形列表
        free = helper.recover_free_space(
            self.pW + self.tol, self.pL + self.tol, top["items"])

        # 计算如果加入新物品后，层高是否会增加
        potential_h = max(top["height"], specs["h"])
        height_increase = potential_h - top["height"]

        # 检查高度增加是否会超过托盘限制
        if pallet.height + height_increase > self.maxH:
            return qty

        # 尝试在自由空间中放置物品
        result = helper.try_pack_in_free_space(free, specs, qty)
        if result["count"] > 0:
            pallet.add_to_top_layer(sku_name, result["items"], potential_h)
            qty -= result["count"]

        return qty

    def _get_full_plan(self, name, specs, remaining_h):
        """
        获取一个能在 remaining_h 内放置的最优多层方案（背包 DP 结果）。
        优先使用缓存的最优方案；若放不下，则根据实际剩余高度重新计算。
        返回: 层列表 [{height, count, items, dims, ...}, ...] 或 None
        """
        # 首次计算时缓存完整的多层方案
        if name not in self._layout_cache:
            ps = PalletSolver(self.pW, self.pL, self.usable_h,
                              specs["w"], specs["l"], specs["h"], self.tol)
            sol = ps.solve()
            self._layout_cache[name] = sol["layers"] if sol["layers"] else None

        opt = self._layout_cache[name]
        if opt:
            # 检查缓存方案的总高度是否能放进剩余空间
            total_plan_h = sum(ly["height"] for ly in opt)
            if total_plan_h <= remaining_h:
                return opt  # 完整方案能放进剩余空间

        # 自适应：用实际剩余高度重新求解（可能得到不同的层组合）
        ps = PalletSolver(self.pW, self.pL, remaining_h,
                          specs["w"], specs["l"], specs["h"], self.tol)
        sol = ps.solve()
        return sol["layers"] if sol["layers"] else None

    @staticmethod
    def _to_dict(p, pid):
        """将 Pallet 对象转换为可序列化的字典。"""
        return {
            "id":          f"Pallet-{pid}",
            "type":        "Mixed" if len(p.skus) > 1 else "Full",
            "layers":      p.layers,
            "skus":        list(p.skus.keys()),
            "total_count": sum(ly["count"] for ly in p.layers),
        }
