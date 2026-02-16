"""
Pallet Simulator — Streamlit Web App + Plotly 3D Visualization
===============================================================
Two modes:
  1. Single SKU Simulator — Calculate optimal box placement for one carton type
  2. Whole Order Planning — Import CSV order, distribute multiple SKUs across pallets
"""

import csv
import io
import random
import hashlib

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from solver import PalletSolver, OrderSolver

# ------------------------------------------------------------------ #
#  Page Configuration
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Pallet Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS for premium look
# ------------------------------------------------------------------ #

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4A90D9, #67B7DC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .result-card h2 {
        margin: 0;
        font-size: 2.5rem;
        color: #67B7DC;
    }
    .result-card p {
        margin: 0.3rem 0 0;
        color: #aaa;
        font-size: 0.9rem;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

DEFAULTS = {
    "palletL": 48.0,   "palletW": 40.0,
    "baseH":    6.0,   "maxH":   54.0,
    "tolerance": 0.0,
    "itemL":  27.5,    "itemW":  17.5,    "itemH": 12.5,
}

CSV_REQUIRED_COLUMNS = ["Item Name", "Total Case", "Length", "Width", "Height"]

CSV_TEMPLATE = "Item Name,Total Case,Length,Width,Height\nSample Item A,10,12,8,6\nSample Item B,5,10,10,10\n"

# ------------------------------------------------------------------ #
#  Helper: Stable color per SKU
# ------------------------------------------------------------------ #

# Pastel palette for SKU coloring
SKU_PALETTE = [
    "#6EC6FF", "#FFB74D", "#81C784", "#E57373", "#BA68C8",
    "#4DD0E1", "#FFD54F", "#AED581", "#FF8A65", "#9575CD",
    "#4FC3F7", "#FFF176", "#A5D6A7", "#EF9A9A", "#CE93D8",
]


def color_for_sku(sku: str) -> str:
    """Generate a stable color for a given SKU name."""
    idx = int(hashlib.md5(sku.encode()).hexdigest(), 16) % len(SKU_PALETTE)
    return SKU_PALETTE[idx]


# ------------------------------------------------------------------ #
#  Helper: 3D Plotly Visualization
# ------------------------------------------------------------------ #

def _box_mesh(x, y, z, dx, dy, dz, color, opacity=0.9, name=""):
    """Create a Mesh3d trace for a 3D box."""
    # 8 vertices of a box
    vx = [x, x+dx, x+dx, x,    x,    x+dx, x+dx, x]
    vy = [y, y,    y+dy, y+dy, y,    y,    y+dy, y+dy]
    vz = [z, z,    z,    z,    z+dz, z+dz, z+dz, z+dz]

    # 12 triangles (2 per face)
    i = [0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 0, 3]
    j = [1, 2, 1, 5, 5, 6, 3, 2, 5, 6, 4, 7]
    k = [2, 3, 5, 4, 6, 7, 4, 6, 6, 7, 7, 4]

    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        color=color, opacity=opacity,
        flatshading=True,
        name=name,
        hoverinfo="name" if name else "skip",
        showlegend=bool(name),
    )


def _box_edges(x, y, z, dx, dy, dz, color="#555"):
    """Create wireframe edges for a 3D box."""
    corners = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz],
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex += [corners[a][0], corners[b][0], None]
        ey += [corners[a][1], corners[b][1], None]
        ez += [corners[a][2], corners[b][2], None]

    return go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color=color, width=1.5),
        showlegend=False, hoverinfo="skip",
    )


def draw_pallet_3d(pW, pL, bH, layers, tol, color_mode="single", height_limit=None):
    """
    Build a Plotly figure showing the pallet + stacked layers.

    Args:
        color_mode: "single" = uniform color | "sku" = per-SKU coloring
    """
    traces = []

    # --- Pallet base ---
    traces.append(_box_mesh(0, 0, 0, pW, pL, bH, "#6B3A1F", 0.85, name="Pallet Base"))
    traces.append(_box_edges(0, 0, 0, pW, pL, bH, "#3E2112"))

    offset = -tol / 2.0
    z = bH
    legend_added = set()

    for layer in layers:
        lh = layer["height"]
        layer_sku = layer.get("sku", "default")

        for item in layer["items"]:
            iw = item.get("w", layer["dims"]["w"])
            il = item.get("l", layer["dims"]["l"])

            if color_mode == "single":
                color = "#D4A574"
                name = ""
            else:
                sku = item.get("sku", layer_sku)
                color = color_for_sku(sku)
                # Show in legend only once per SKU
                name = sku if sku not in legend_added else ""
                if sku not in legend_added:
                    legend_added.add(sku)

            bx = item["x"] + offset
            by = item["y"] + offset

            traces.append(_box_mesh(bx, by, z, iw, il, lh, color, 0.92, name=name))
            traces.append(_box_edges(bx, by, z, iw, il, lh, "#444"))

        z += lh

    # --- Layout ---
    max_dim = max(pW + tol, pL + tol, z if z > bH else bH + 10)
    z_max = height_limit if height_limit and height_limit > z else max(z, bH + 10)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Width (in)", range=[-tol - 1, pW + tol + 1],
                       backgroundcolor="#1a1a2e", gridcolor="#333", showbackground=True),
            yaxis=dict(title="Length (in)", range=[-tol - 1, pL + tol + 1],
                       backgroundcolor="#16213e", gridcolor="#333", showbackground=True),
            zaxis=dict(title="Height (in)", range=[0, z_max + 2],
                       backgroundcolor="#0d1117", gridcolor="#333", showbackground=True),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        paper_bgcolor="#0d1117",
        legend=dict(
            font=dict(color="#ccc"),
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#30475e",
        ),
    )
    return fig


def draw_empty_pallet(pW, pL, bH):
    """Draw only the empty pallet base."""
    return draw_pallet_3d(pW, pL, bH, [], 0)


# ------------------------------------------------------------------ #
#  Sidebar: Pallet Settings (shared by both tabs)
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 📐 Pallet Settings")
    st.caption("Configure the pallet dimensions (in inches)")

    pallet_L = st.number_input(
        "Pallet Length (in)", value=DEFAULTS["palletL"], min_value=1.0, step=1.0,
        help="Length of the pallet base, e.g. 48 inches",
    )
    pallet_W = st.number_input(
        "Pallet Width (in)", value=DEFAULTS["palletW"], min_value=1.0, step=1.0,
        help="Width of the pallet base, e.g. 40 inches",
    )
    base_H = st.number_input(
        "Base Height (in)", value=DEFAULTS["baseH"], min_value=0.0, step=0.5,
        help="Height of the empty pallet (base board), e.g. 6 inches",
    )
    max_H = st.number_input(
        "Max Height (in)", value=DEFAULTS["maxH"], min_value=1.0, step=1.0,
        help="Maximum total height including pallet base, e.g. 54 inches",
    )
    tolerance = st.number_input(
        "Tolerance / Overhang (in)", value=DEFAULTS["tolerance"], min_value=0.0, step=0.25,
        help="Allowed overhang per side (tolerance), e.g. 0.5 inches",
    )


# ------------------------------------------------------------------ #
#  Header
# ------------------------------------------------------------------ #

st.markdown('<div class="main-header">📦 Pallet Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Optimize carton placement on pallets — Interactive 3D Visualization</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Tabs
# ------------------------------------------------------------------ #

tab_single, tab_order = st.tabs(["🔲 Single SKU Simulator", "📋 Whole Order Planning"])


# ================================================================== #
#  TAB 1: Single SKU Simulator
# ================================================================== #

with tab_single:
    col_input, col_viz = st.columns([1, 2.5], gap="large")

    with col_input:
        st.markdown("### 📏 Carton Dimensions")
        item_L = st.number_input(
            "Length (in)", value=DEFAULTS["itemL"], min_value=0.1, step=0.5,
            key="single_itemL", help="Length of a single carton",
        )
        item_W = st.number_input(
            "Width (in)", value=DEFAULTS["itemW"], min_value=0.1, step=0.5,
            key="single_itemW", help="Width of a single carton",
        )
        item_H = st.number_input(
            "Height (in)", value=DEFAULTS["itemH"], min_value=0.1, step=0.5,
            key="single_itemH", help="Height of a single carton",
        )

        st.markdown("---")
        calculate = st.button("🚀 Calculate Packing", use_container_width=True, type="primary")

    with col_viz:
        if calculate:
            # Validate
            if max_H <= base_H:
                st.error("❌ Max Height must be greater than Base Height.")
            else:
                usable_h = max_H - base_H
                solver = PalletSolver(pallet_W, pallet_L, usable_h, item_W, item_L, item_H, tolerance)
                result = solver.solve()

                # Result summary
                total = result["count"]
                num_layers = len(result["layers"])

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{total}</h2>
                        <p>Total Items</p>
                    </div>
                    """, unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{num_layers}</h2>
                        <p>Layers</p>
                    </div>
                    """, unsafe_allow_html=True)
                with r3:
                    used_h = base_H + sum(ly["height"] for ly in result["layers"])
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{used_h:.1f}"</h2>
                        <p>Total Height</p>
                    </div>
                    """, unsafe_allow_html=True)

                # 3D visualization
                fig = draw_pallet_3d(pallet_W, pallet_L, base_H, result["layers"], tolerance,
                                     color_mode="single", height_limit=max_H)
                st.plotly_chart(fig, use_container_width=True, key="single_3d")

                # Layer detail table
                if result["layers"]:
                    st.markdown("#### 📊 Layer Details")
                    layer_data = []
                    for i, ly in enumerate(result["layers"], 1):
                        layer_data.append({
                            "Layer": i,
                            "Orientation": ly["type"].capitalize(),
                            "Items": ly["count"],
                            "Height (in)": ly["height"],
                            "Box Dims": f'{ly["dims"]["w"]} × {ly["dims"]["l"]} × {ly["dims"]["h"]}',
                        })
                    st.dataframe(pd.DataFrame(layer_data), use_container_width=True, hide_index=True)
        else:
            # Show empty pallet as placeholder
            fig = draw_empty_pallet(pallet_W, pallet_L, base_H)
            st.plotly_chart(fig, use_container_width=True, key="single_empty")
            st.info("👆 Set carton dimensions and click **Calculate Packing** to see results.")


# ================================================================== #
#  TAB 2: Whole Order Planning
# ================================================================== #

with tab_order:
    col_left, col_right = st.columns([1.2, 2.5], gap="large")

    with col_left:
        st.markdown("### ⚙️ Order Settings")

        max_mixed = st.number_input(
            "Max Mixed SKUs per Pallet", value=4, min_value=1, step=1,
            help="Maximum number of different SKU types allowed on a single mixed pallet",
        )

        st.markdown("---")

        # CSV template download
        st.download_button(
            label="📥 Download CSV Template",
            data=CSV_TEMPLATE,
            file_name="Order_Template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # CSV file upload
        uploaded_file = st.file_uploader(
            "📤 Import Order CSV",
            type=["csv"],
            help="Upload a CSV file with columns: Item Name, Total Case, Length, Width, Height",
        )

        # Parse uploaded CSV
        order_data = []
        if uploaded_file is not None:
            try:
                raw = uploaded_file.read().decode("utf-8-sig")
                # Normalize all line-ending styles (\r\n, \r, \n) to \n
                content = raw.replace("\r\n", "\n").replace("\r", "\n")
                reader = csv.DictReader(io.StringIO(content))

                # Validate headers
                if reader.fieldnames:
                    headers = [h.strip() for h in reader.fieldnames]
                    missing = [col for col in CSV_REQUIRED_COLUMNS if col not in headers]
                    if missing:
                        st.error(f"❌ Missing required columns: **{', '.join(missing)}**\n\n"
                                 "Please use the template for the correct format.")
                    else:
                        for row_num, raw_row in enumerate(reader, start=2):
                            row = {k.strip(): v for k, v in raw_row.items()}
                            try:
                                name   = row.get("Item Name", "Unknown")
                                qty    = int(row.get("Total Case", 0))
                                length = float(row.get("Length", 0))
                                width  = float(row.get("Width", 0))
                                height = float(row.get("Height", 0))
                            except ValueError:
                                continue
                            if qty <= 0 or length <= 0 or width <= 0 or height <= 0:
                                continue
                            order_data.append({
                                "name": name, "qty": qty,
                                "l": length, "w": width, "h": height,
                            })

                        if order_data:
                            st.success(f"✅ Loaded **{len(order_data)}** SKUs from CSV")
                            # Show preview
                            preview_df = pd.DataFrame([
                                {"Item": d["name"], "Qty": d["qty"],
                                 "Dims": f'{d["l"]}×{d["w"]}×{d["h"]}'}
                                for d in order_data
                            ])
                            st.dataframe(preview_df, use_container_width=True, hide_index=True, height=200)
                        else:
                            st.warning("⚠️ No valid data rows found in CSV.")
                else:
                    st.error("❌ CSV file is empty or missing headers.")
            except Exception as exc:
                st.error(f"❌ Import error: {exc}")

        st.markdown("---")

        plan_clicked = st.button(
            "📦 Plan Order",
            use_container_width=True,
            type="primary",
            disabled=len(order_data) == 0,
        )

    with col_right:
        if plan_clicked and order_data:
            if max_H <= base_H:
                st.error("❌ Max Height must be greater than Base Height.")
            else:
                try:
                    solver = OrderSolver(pallet_W, pallet_L, base_H, max_H, tolerance, max_mixed)
                    pallets = solver.solve(order_data)

                    if not pallets:
                        st.warning("No pallets generated. Check your input dimensions.")
                    else:
                        # Store in session state for pallet selection
                        st.session_state["plan_results"] = pallets

                        # Summary metrics
                        total_items = sum(p["total_count"] for p in pallets)
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>{len(pallets)}</h2>
                                <p>Pallets Required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>{total_items}</h2>
                                <p>Total Items Packed</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with m3:
                            full = sum(1 for p in pallets if p["type"] == "Full")
                            mixed = len(pallets) - full
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>{full}F / {mixed}M</h2>
                                <p>Full / Mixed Pallets</p>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as exc:
                    st.error(f"❌ Planning error: {exc}")
                    st.session_state.pop("plan_results", None)

        # Display results if available
        if "plan_results" in st.session_state and st.session_state["plan_results"]:
            pallets = st.session_state["plan_results"]

            st.markdown("---")

            # Pallet selector
            pallet_options = []
            for i, p in enumerate(pallets):
                sku_label = ", ".join(p["skus"][:3])
                if len(p["skus"]) > 3:
                    sku_label += f" +{len(p['skus'])-3} more"
                pallet_options.append(
                    f"Pallet {i+1} — {p['type']} — {p['total_count']} items — {sku_label}"
                )

            selected_idx = st.selectbox(
                "🔍 Select Pallet to Visualize",
                range(len(pallet_options)),
                format_func=lambda x: pallet_options[x],
            )

            pallet = pallets[selected_idx]

            # Pallet detail table
            st.markdown(f"#### 📋 Pallet {selected_idx + 1} Contents")
            sku_counts = {}
            for layer in pallet["layers"]:
                for item in layer.get("items", []):
                    sku = item.get("sku", layer.get("sku", "Unknown"))
                    sku_counts[sku] = sku_counts.get(sku, 0) + 1

            detail_df = pd.DataFrame([
                {"SKU": sku, "Quantity": qty, "Color": color_for_sku(sku)}
                for sku, qty in sku_counts.items()
            ])

            # Show without color column (it's for internal use)
            st.dataframe(
                detail_df[["SKU", "Quantity"]],
                use_container_width=True, hide_index=True,
            )

            # 3D visualization
            fig = draw_pallet_3d(
                pallet_W, pallet_L, base_H,
                pallet["layers"], tolerance,
                color_mode="sku", height_limit=max_H,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"order_3d_{selected_idx}")

        elif not plan_clicked:
            # Placeholder
            fig = draw_empty_pallet(pallet_W, pallet_L, base_H)
            st.plotly_chart(fig, use_container_width=True, key="order_empty")
            st.info("👆 Upload an order CSV and click **Plan Order** to generate the palletization plan.")


# ------------------------------------------------------------------ #
#  Footer
# ------------------------------------------------------------------ #

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.85rem;">'
    '📦 Pallet Simulator • Built with Streamlit & Plotly'
    '</div>',
    unsafe_allow_html=True,
)
