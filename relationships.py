# app.py
import re
import colorsys
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

# Louvain community detection (python-louvain package)
import community as community_louvain

st.set_page_config(layout="wide", page_title="Product Recommendation Network")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_csv(upload):
	return pd.read_csv(upload)

def extract_dm_id(url: str):
	"""Extract DM_###### id from a URL-like string."""
	if pd.isna(url):
		return None
	m = re.search(r"(DM_\d+)", str(url))
	return m.group(1) if m else None

def clean_price(val):
	"""Parse '$12.34' or '12.34' -> float; handle None/NaN/garbage."""
	if pd.isna(val):
		return np.nan
	s = re.sub(r"[^\d\.\-]", "", str(val))
	try:
		return float(s) if s != "" else np.nan
	except:
		return np.nan

def pick_price(row):
	"""Choose product_price_1 else product_non_member_price (as float)."""
	primary = clean_price(row.get("product_price_1"))
	if not np.isnan(primary):
		return primary
	return clean_price(row.get("product_non_member_price"))

def build_full_name(row):
	name = str(row.get("product_name", "") or "").strip()
	sub  = str(row.get("product_subname", "") or "").strip()
	return (name + (" " + sub if sub else "")).strip() or name or "Unknown"

def find_rec_columns(df):
	"""
	Return:
	  - URL cols (preferred: *_url; else any recommended_product_* that isn't *_name/_title/_subtitle)
	  - mapping index->name col if present (recommended_product_{i}_name or title/subtitle pair)
	"""
	url_cols = [c for c in df.columns if c.startswith("recommended_product_") and c.endswith("_url")]
	if not url_cols:
		url_cols = [c for c in df.columns
					if c.startswith("recommended_product_")
					and not c.endswith("_name")
					and not c.endswith("_title")
					and not c.endswith("_subtitle")]
	name_cols = {}
	for i in range(1, 501):
		c = f"recommended_product_{i}_name"
		if c in df.columns:
			name_cols[i] = c
	for i in range(1, 501):
		t = f"recommended_product_{i}_title"
		s = f"recommended_product_{i}_subtitle"
		if t in df.columns and s in df.columns and i not in name_cols:
			name_cols[i] = (t, s)
	return url_cols, name_cols

# ----------------------------
# Graph building (with contextual filtering)
# ----------------------------
def build_graph(df: pd.DataFrame,
				brand_filter: list[str] | None,
				type_filter: list[str] | None,
				name_query: str | None,
				name_exact: list[str] | None) -> nx.DiGraph:
	"""
	Build the full directed graph, then apply contextual filters:
	  - brand/type filters: keep selected nodes + neighbours
	  - product name filter: keep name-matched nodes + neighbours
	If multiple filters provided, the intersection of selected sets is used before adding neighbours.
	"""
	df = df.copy()

	# Base attributes
	df["product_id"] = df["link"].apply(extract_dm_id)
	df["price"]      = df.apply(pick_price, axis=1)
	df["full_name"]  = df.apply(build_full_name, axis=1)
	df["brand_name"] = df.get("brand_name", "")
	df["type"]       = df.get("type", "")

	rec_url_cols, rec_name_cols = find_rec_columns(df)
	for col in rec_url_cols:
		df[col + "_id"] = df[col].apply(extract_dm_id)

	G = nx.DiGraph()

	# Main product nodes
	for _, row in df.iterrows():
		pid = row["product_id"]
		if not pd.isna(pid):
			G.add_node(
				pid,
				name=row["full_name"],
				price=row["price"],
				brand=row.get("brand_name", ""),
				ptype=row.get("type", "")
			)

	# Recommendation edges + ensure rec nodes exist
	for _, row in df.iterrows():
		src = row["product_id"]
		if pd.isna(src):
			continue
		for col in rec_url_cols:
			rec_id = row.get(col + "_id")
			if pd.isna(rec_id) or rec_id is None:
				rec_url = row.get(col)
				rec_id = extract_dm_id(rec_url)
			if rec_id:
				if rec_id not in G:
					# Try to construct a sensible name from *_name or title/subtitle
					rec_name = None
					m = re.search(r"recommended_product_(\d+)", col)
					if m:
						idx = int(m.group(1))
						name_col = rec_name_cols.get(idx)
						if isinstance(name_col, tuple):
							tcol, scol = name_col
							t = str(row.get(tcol, "") or "").strip()
							s = str(row.get(scol, "") or "").strip()
							rec_name = (t + (" " + s if s else "")).strip() or None
						elif isinstance(name_col, str):
							rec_name = str(row.get(name_col, "") or "").strip() or None
					G.add_node(rec_id,
							   name=rec_name if rec_name else rec_id,
							   price=np.nan,
							   brand="",
							   ptype="")
				G.add_edge(src, rec_id)

	# ---- Build selected set from filters (intersection first) ----
	selected = set(G.nodes())

	# brand/type selection
	if (brand_filter and len(brand_filter) > 0) or (type_filter and len(type_filter) > 0):
		by_brand_type = {
			n for n, data in G.nodes(data=True)
			if ((not brand_filter) or (data.get("brand", "") in brand_filter))
			and ((not type_filter) or (data.get("ptype", "") in type_filter))
		}
		selected = selected & by_brand_type

	# Name filter (exact list preferred; else substring query)
	if name_exact and len(name_exact) > 0:
		names_map = nx.get_node_attributes(G, "name")
		by_name = {n for n, nm in names_map.items() if nm in set(name_exact)}
		selected = selected & by_name
	elif name_query and name_query.strip():
		q = name_query.strip().lower()
		names_map = nx.get_node_attributes(G, "name")
		by_query = {n for n, nm in names_map.items() if q in str(nm).lower()}
		selected = selected & by_query

	# If any filter applied, include neighbours for context
	any_filter = (
		(brand_filter and len(brand_filter) > 0) or
		(type_filter and len(type_filter) > 0) or
		(name_exact and len(name_exact) > 0) or
		(name_query and name_query.strip())
	)
	if any_filter:
		context = set()
		for n in selected:
			context.update(list(G.predecessors(n)))
			context.update(list(G.successors(n)))
		keep = selected | context
		G = G.subgraph(keep).copy()

	return G

# ----------------------------
# Top-N hubs + neighbours
# ----------------------------
def top_n_with_neighbours(G: nx.DiGraph, top_n: int) -> nx.DiGraph:
	if G.number_of_nodes() == 0:
		return G.copy()
	indeg = dict(G.in_degree())
	N = min(top_n, len(indeg))
	hubs = [n for n, _ in sorted(indeg.items(), key=lambda x: x[1], reverse=True)[:N]]
	keep = set(hubs)
	for n in hubs:
		keep.update(list(G.predecessors(n)))
		keep.update(list(G.successors(n)))
	return G.subgraph(keep).copy()

# ----------------------------
# Price -> colour mapping (with explicit cmin/cmax on LOG)
# ----------------------------
def colour_array_and_colourbar(prices_array: np.ndarray, mode: str, clip_pct: float = 95.0):
	"""
	Return (colour_values, colourbar_dict, colourscale, cmin, cmax)
	"""
	p = np.array(prices_array, dtype=float)
	p[~np.isfinite(p)] = np.nan

	positive = p[np.isfinite(p) & (p > 0)]
	if positive.size == 0:
		p = np.ones_like(p)  # avoid empty log ranges
		min_pos = 1.0
	else:
		min_pos = float(np.nanmin(positive))
		p = np.where(np.isfinite(p) & (p > 0), p, min_pos)

	# Linear
	if mode == "Linear":
		vals = p.copy()
		cbar = dict(title=dict(text="Price ($)", side="right"), thickness=15, xanchor="left")
		return vals, cbar, "RdBu", float(np.min(vals)), float(np.max(vals))

	# Clipped linear (percentile)
	if mode == "Clipped (percentile)":
		hi = np.percentile(p, clip_pct)
		vals = np.clip(p, None, hi)
		cbar = dict(
			title=dict(text=f"Price ($) (clipped @ {clip_pct:.1f}th pct)", side="right"),
			thickness=15, xanchor="left"
		)
		return vals, cbar, "RdBu", float(np.min(vals)), float(np.max(vals))

	# Log (base 10) — normalise with cmin/cmax so full colour range is used
	vals = np.log10(p)
	vmin = float(np.min(vals))
	vmax = float(np.max(vals))

	# Human-friendly tick labels showing real $ on log scale
	canonical = np.array([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000], dtype=float)
	lo_p, hi_p = float(10**vmin), float(10**vmax)
	tick_prices = canonical[(canonical >= lo_p) & (canonical <= hi_p)]
	if tick_prices.size == 0:
		tick_prices = np.array([lo_p, hi_p], dtype=float)
	tickvals = np.log10(tick_prices)
	ticktext = [f"${int(tp) if tp >= 10 else tp:.0f}" if tp >= 10 else f"${tp:.2f}" for tp in tick_prices]

	cbar = dict(
		title=dict(text="Price ($, log scale)", side="right"),
		thickness=15,
		xanchor="left",
		tickvals=tickvals.tolist(),
		ticktext=ticktext
	)
	return vals, cbar, "RdBu", vmin, vmax

# ----------------------------
# SQRT bubble sizing (in-degree)
# ----------------------------
def sqrt_sizes(deg_values, min_size: int, max_size: int):
	d = np.array(deg_values, dtype=float)
	if d.size == 0:
		return np.array([])
	vals = np.sqrt(d)
	lo, hi = float(np.min(vals)), float(np.max(vals))
	if hi == lo:
		return np.full_like(vals, (min_size + max_size) / 2.0)
	norm = (vals - lo) / (hi - lo)
	return min_size + norm * (max_size - min_size)

# ----------------------------
# Layouts
# ----------------------------
def radial_by_indegree_pos(G: nx.DiGraph, rings=6, seed=42):
	"""
	Concentric rings by in-degree percentile; highest in-degree at centre.
	Returns dict: node -> (x,y)
	"""
	if G.number_of_nodes() == 0:
		return {}
	rng = np.random.default_rng(seed)
	indeg = dict(G.in_degree())
	vals = np.array(list(indeg.values()), dtype=float)
	percentiles = np.percentile(vals, np.linspace(0, 100, rings))
	shells = [[] for _ in range(rings)]
	nodes = list(G.nodes())
	for n in nodes:
		v = indeg.get(n, 0)
		idx = np.searchsorted(percentiles, v, side="right") - 1
		idx = int(np.clip(idx, 0, rings - 1))
		shells[idx].append(n)

	pos = {}
	for r, ring_nodes in enumerate(shells):
		if not ring_nodes:
			continue
		R = (r + 1) / rings  # (0..1]
		if r == rings - 1:
			for i, n in enumerate(ring_nodes):
				jitter = 0.02 * rng.normal(size=2)
				pos[n] = (0.02 * i + jitter[0], 0.02 * i + jitter[1])
		else:
			k = len(ring_nodes)
			angles = np.linspace(0, 2*np.pi, k, endpoint=False)
			rng.shuffle(angles)
			for a, n in zip(angles, ring_nodes):
				pos[n] = (R * np.cos(a), R * np.sin(a))
	return pos

def compute_positions(G: nx.DiGraph, layout_choice: str):
	if layout_choice == "Radial by in-degree (bullseye)":
		return radial_by_indegree_pos(G, rings=6, seed=42)
	elif layout_choice == "Kamada–Kawai":
		return nx.kamada_kawai_layout(G)
	else:  # Spring
		return nx.spring_layout(G, k=0.3, seed=42)

# ----------------------------
# Community colours & hulls
# ----------------------------
def pastel_hsl_palette(n):
	"""Return n distinct pastel colours as hex."""
	cols = []
	for i in range(n):
		h = i / max(1, n)
		s = 0.45
		l = 0.80
		r, g, b = colorsys.hls_to_rgb(h, l, s)
		cols.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
	return cols

def community_hulls_traces(G: nx.Graph, pos: dict, min_size: int = 15, opacity: float = 0.2):
	"""
	Build translucent polygon traces for communities (size >= min_size).
	Uses a simple convex hull per community (2+ nodes handled with small circle).
	"""
	if G.number_of_nodes() == 0:
		return []

	# Undirected for community finding
	parts = community_louvain.best_partition(G.to_undirected())
	comm_to_nodes = {}
	for n, cid in parts.items():
		comm_to_nodes.setdefault(cid, []).append(n)

	# Colours
	comm_ids = sorted([cid for cid, nodes in comm_to_nodes.items() if len(nodes) >= min_size])
	palette = pastel_hsl_palette(len(comm_ids))

	traces = []
	for colour, cid in zip(palette, comm_ids):
		nodes = comm_to_nodes[cid]
		if len(nodes) < min_size:
			continue
		pts = np.array([pos[n] for n in nodes if n in pos], dtype=float)
		if pts.shape[0] == 0:
			continue
		if pts.shape[0] == 1:
			# single node: draw tiny circle
			x0, y0 = pts[0]
			circle = go.Scatter(
				x=[x0 + 0.02*np.cos(t) for t in np.linspace(0, 2*np.pi, 40)],
				y=[y0 + 0.02*np.sin(t) for t in np.linspace(0, 2*np.pi, 40)],
				fill="toself", mode="lines", line=dict(width=0),
				hoverinfo="none",
				name=f"Community {cid}",
				marker=dict(color=colour),
				opacity=opacity
			)
			traces.append(circle)
			continue

		# Convex hull
		try:
			from scipy.spatial import ConvexHull
			hull = ConvexHull(pts)
			hull_pts = pts[hull.vertices]
			hull_trace = go.Scatter(
				x=np.append(hull_pts[:, 0], hull_pts[0, 0]),
				y=np.append(hull_pts[:, 1], hull_pts[0, 1]),
				fill="toself",
				mode="lines",
				line=dict(width=0),
				hoverinfo="none",
				name=f"Community {cid}",
				marker=dict(color=colour),
				opacity=opacity
			)
			traces.append(hull_trace)
		except Exception:
			# Fallback: bounding box
			xmin, ymin = pts.min(axis=0)
			xmax, ymax = pts.max(axis=0)
			xpoly = [xmin, xmax, xmax, xmin, xmin]
			ypoly = [ymin, ymin, ymax, ymax, ymin]
			traces.append(go.Scatter(
				x=xpoly, y=ypoly, fill="toself", mode="lines", line=dict(width=0),
				hoverinfo="none",
				name=f"Community {cid}", marker=dict(color=colour), opacity=opacity
			))
	return traces

# ----------------------------
# Plot graph (with highlight overlays & communities)
# ----------------------------
def plot_graph(H: nx.DiGraph,
			   label_top_percent: int = 10,
			   colour_mode: str = "Clipped (percentile)",
			   clip_pct: float = 95.0,
			   min_size: int = 8,
			   max_size: int = 50,
			   height: int = 900,
			   highlight_nodes: set[str] | None = None,
			   layout_choice: str = "Spring",
			   show_communities: bool = False,
			   community_min_size: int = 15,
			   community_opacity: float = 0.2):
	if H.number_of_nodes() == 0:
		return go.Figure()

	pos = compute_positions(H, layout_choice)
	indeg = dict(H.in_degree())

	# Price -> colour
	prices = {n: H.nodes[n].get("price", np.nan) for n in H.nodes()}
	price_vals = np.array([prices[n] for n in H.nodes()], dtype=float)
	colour_values, colourbar_dict, colourscale_name, cmin, cmax = colour_array_and_colourbar(
		price_vals, colour_mode, clip_pct=clip_pct
	)

	# Labels for top % by indegree
	counts = np.array(list(indeg.values()), dtype=float)
	cutoff = np.percentile(counts, 100 - label_top_percent) if len(counts) else 0
	top_label_nodes = {n for n, v in indeg.items() if v >= cutoff}

	# Edge coords (base vs highlighted)
	base_edge_x, base_edge_y = [], []
	hi_edge_x, hi_edge_y = [], []
	highlight_nodes = highlight_nodes or set()

	for u, v in H.edges():
		x0, y0 = pos[u]; x1, y1 = pos[v]
		if (u in highlight_nodes) or (v in highlight_nodes):
			hi_edge_x += [x0, x1, None]; hi_edge_y += [y0, y1, None]
		else:
			base_edge_x += [x0, x1, None]; base_edge_y += [y0, y1, None]

	base_edge_trace = go.Scatter(
		x=base_edge_x, y=base_edge_y,
		line=dict(width=0.5, color="#BBBBBB"),
		hoverinfo="none",
		mode="lines",
		showlegend=False
	)
	hi_edge_trace = go.Scatter(
		x=hi_edge_x, y=hi_edge_y,
		line=dict(width=1.5, color="#666666"),
		hoverinfo="none",
		mode="lines",
		showlegend=False
	)

	# Nodes
	nodes = list(H.nodes())
	node_x = [pos[n][0] for n in nodes]
	node_y = [pos[n][1] for n in nodes]
	deg_values = [indeg.get(n, 0) for n in nodes]
	base_sizes = sqrt_sizes(deg_values, min_size=min_size, max_size=max_size)

	base_text  = []
	base_hover = []
	for n in nodes:
		price_display = prices[n]
		if isinstance(price_display, float) and np.isnan(price_display):
			price_display = "N/A"
		full_name = H.nodes[n].get("name", n)
		base_hover.append(
			f"ID: {n}<br>"
			f"Name: {full_name}<br>"
			f"Price: {price_display}<br>"
			f"In-degree: {indeg.get(n, 0)}"
		)
		base_text.append(full_name if n in top_label_nodes else "")

	base_node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode="markers+text",
		text=base_text,
		textposition="top center",
		hovertext=base_hover,
		hoverinfo="text",
		marker=dict(
			showscale=True,
			colorscale=colourscale_name,   # Plotly key must be 'colorscale'
			reversescale=True,             # low price -> red, high price -> blue
			color=colour_values,           # Plotly key must be 'color'
			cmin=cmin,
			cmax=cmax,
			size=base_sizes,
			colorbar=colourbar_dict,       # Plotly key must be 'colorbar'
			line=dict(width=1, color="#333")
		),
		name="Products"
	)

	# Highlight overlay
	hi_node_trace = None
	if highlight_nodes:
		hi_nodes = [n for n in nodes if n in highlight_nodes]
		if hi_nodes:
			node_index = {n: i for i, n in enumerate(nodes)}
			hi_x = [pos[n][0] for n in hi_nodes]
			hi_y = [pos[n][1] for n in hi_nodes]
			hi_sizes = [base_sizes[node_index[n]] * 1.2 for n in hi_nodes]
			hi_colours = [colour_values[node_index[n]] for n in hi_nodes]
			hi_text = [H.nodes[n].get("name", n) for n in hi_nodes]
			hi_hover = []
			for n in hi_nodes:
				price_display = prices[n]
				if isinstance(price_display, float) and np.isnan(price_display):
					price_display = "N/A"
				hi_hover.append(
					f"ID: {n}<br>"
					f"Name: {H.nodes[n].get('name', n)}<br>"
					f"Price: {price_display}<br>"
					f"In-degree: {indeg.get(n, 0)}"
				)
			hi_node_trace = go.Scatter(
				x=hi_x, y=hi_y,
				mode="markers+text",
				text=hi_text,
				textposition="top center",
				hovertext=hi_hover,
				hoverinfo="text",
				marker=dict(
					showscale=False,
					colorscale=colourscale_name,
					reversescale=True,
					color=hi_colours,
					cmin=cmin,
					cmax=cmax,
					size=hi_sizes,
					line=dict(width=3, color="#000")
				),
				name="Highlighted"
			)

	# Community hulls (drawn beneath edges/nodes)
	traces = []
	if show_communities:
		hulls = community_hulls_traces(H, pos, min_size=community_min_size, opacity=community_opacity)
		traces.extend(hulls)

	traces.extend([base_edge_trace, hi_edge_trace, base_node_trace])
	if hi_node_trace is not None:
		traces.append(hi_node_trace)

	fig = go.Figure(
		data=traces,
		layout=go.Layout(
			title=dict(text="Product Recommendation Network (Top-N hubs + neighbours)", font=dict(size=20)),
			showlegend=False,
			hovermode="closest",
			margin=dict(b=20, l=5, r=5, t=50),
			xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
			yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
			height=height
		)
	)
	return fig

# ----------------------------
# Export helpers
# ----------------------------
def export_nodes_edges(H: nx.DiGraph):
	indeg = dict(H.in_degree())
	rows_nodes = []
	for n, data in H.nodes(data=True):
		rows_nodes.append({
			"product_id": n,
			"name": data.get("name", ""),
			"price": data.get("price", ""),
			"brand": data.get("brand", ""),
			"type": data.get("ptype", ""),
			"in_degree": indeg.get(n, 0)
		})
	nodes_df = pd.DataFrame(rows_nodes)
	edges_df = pd.DataFrame([{"source": u, "target": v} for u, v in H.edges()])
	return nodes_df, edges_df

def to_excel_bytes(nodes_df, edges_df):
	out = BytesIO()
	with pd.ExcelWriter(out, engine="openpyxl") as writer:
		nodes_df.to_excel(writer, index=False, sheet_name="Nodes")
		edges_df.to_excel(writer, index=False, sheet_name="Edges")
	return out.getvalue()

# ----------------------------
# UI
# ----------------------------
st.title("🍷 Product Recommendation Network")

uploaded = st.file_uploader("Upload your product CSV", type=["csv"])
if not uploaded:
	st.info("Upload a CSV exported from your scrape to begin.")
	st.stop()

df = load_csv(uploaded)

# Sidebar controls — Filters
st.sidebar.header("Filters (contextual)")
brand_vals = sorted(df.get("brand_name", pd.Series(dtype=str)).dropna().unique().tolist())
type_vals  = sorted(df.get("type", pd.Series(dtype=str)).dropna().unique().tolist())

brand_filter = st.sidebar.multiselect("brand_name", brand_vals)
type_filter  = st.sidebar.multiselect("type", type_vals)

# Product name filter for graph content
st.sidebar.markdown("---")
name_query = st.sidebar.text_input("Product name contains (case-insensitive)", "")
df["_full_name"] = (df.get("product_name", "").fillna("").astype(str).str.strip() + " " +
					df.get("product_subname", "").fillna("").astype(str).str.strip()).str.strip()
suggestions = []
if name_query.strip():
	suggestions = sorted(df.loc[df["_full_name"].str.contains(name_query.strip(), case=False, na=False), "_full_name"]
						 .dropna().unique().tolist())[:200]
name_exact = st.sidebar.multiselect("Or pick exact product(s)", suggestions)

# Display controls
st.sidebar.header("Display")
top_n = st.sidebar.slider("Top-N hubs (by in-degree)", min_value=10, max_value=1000, value=100, step=10)
label_top_percent = st.sidebar.slider("Label top % nodes (by in-degree)", min_value=1, max_value=50, value=10, step=1)

# Layout selector
layout_choice = st.sidebar.selectbox("Layout", ["Radial by in-degree (bullseye)", "Kamada–Kawai", "Spring"])

# Price colour options + percentile slider for clipping
colour_mode = st.sidebar.selectbox("Colour scale for price", ["Clipped (percentile)", "Log (base 10)", "Linear"])
clip_pct = 95.0
if colour_mode == "Clipped (percentile)":
	clip_pct = st.sidebar.slider("Clip at percentile", min_value=1.0, max_value=99.9, value=95.0, step=0.1)

graph_height = st.sidebar.slider("Graph height (px)", 500, 1600, 900, step=50)

# SQRT sizing controls
st.sidebar.markdown("---")
min_size = st.sidebar.slider("Min bubble size (px)", 4, 30, 8)
max_size = st.sidebar.slider("Max bubble size (px)", 20, 160, 50)

# Communities
st.sidebar.markdown("---")
show_communities = st.sidebar.checkbox("Show community hulls (Louvain)", value=False)
community_min_size = st.sidebar.slider("Min community size to draw hull", 3, 200, 15, step=1)
community_opacity = st.sidebar.slider("Community hull opacity", 0.05, 0.6, 0.2, step=0.05)

# ----------------------------
# Search Highlight (overlay)
# ----------------------------
st.sidebar.header("Search highlight")
hl_name_query = st.sidebar.text_input("Highlight: name contains", "")
hl_suggestions = []
if hl_name_query.strip():
	hl_suggestions = sorted(df.loc[df["_full_name"].str.contains(hl_name_query.strip(), case=False, na=False), "_full_name"]
							.dropna().unique().tolist())[:200]
hl_name_exact = st.sidebar.multiselect("Highlight: pick exact product(s)", hl_suggestions)
hl_ids_raw = st.sidebar.text_input("Highlight: DM IDs (comma/space)", "")
hl_ids = set()
if hl_ids_raw.strip():
	for tok in re.split(r"[,\s]+", hl_ids_raw.strip()):
		if tok:
			hl_ids.add(tok.strip())

# Build full graph (contextual filter applied here, including product name)
G_full = build_graph(
	df,
	brand_filter=brand_filter,
	type_filter=type_filter,
	name_query=name_query,
	name_exact=name_exact
)

# Build Top-N subgraph: hubs + neighbours
H = top_n_with_neighbours(G_full, top_n=top_n)

# Build highlight set mapped into H (by name & ID)
highlight_nodes = set()
if H.number_of_nodes() > 0:
	names_map_H = nx.get_node_attributes(H, "name")
	if hl_name_query and hl_name_query.strip():
		q = hl_name_query.strip().lower()
		highlight_nodes |= {n for n, nm in names_map_H.items() if q in str(nm).lower()}
	if hl_name_exact and len(hl_name_exact) > 0:
		exact_set = set(hl_name_exact)
		highlight_nodes |= {n for n, nm in names_map_H.items() if nm in exact_set}
	if hl_ids:
		highlight_nodes |= {n for n in H.nodes() if n in hl_ids}

st.write(f"**Filtered graph:** {G_full.number_of_nodes()} nodes / {G_full.number_of_edges()} edges")
st.write(f"**Displayed subgraph (Top-{top_n} hubs + neighbours):** {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")
if highlight_nodes:
	st.write(f"**Highlighted nodes:** {len(highlight_nodes)}")

fig = plot_graph(
	H,
	label_top_percent=label_top_percent,
	colour_mode=colour_mode,
	clip_pct=clip_pct,
	min_size=min_size,
	max_size=max_size,
	height=graph_height,
	highlight_nodes=highlight_nodes,
	layout_choice=layout_choice,
	show_communities=show_communities,
	community_min_size=community_min_size,
	community_opacity=community_opacity
)
st.plotly_chart(fig, use_container_width=True)

# Downloads
st.subheader("📥 Download displayed subgraph")
nodes_df, edges_df = export_nodes_edges(H)
st.download_button("Download Nodes (CSV)", nodes_df.to_csv(index=False).encode("utf-8"),
				   file_name="nodes.csv", mime="text/csv")
st.download_button("Download Edges (CSV)", edges_df.to_csv(index=False).encode("utf-8"),
				   file_name="edges.csv", mime="text/csv")

excel_bytes = to_excel_bytes(nodes_df, edges_df)
st.download_button("Download Nodes+Edges (Excel)", excel_bytes,
				   file_name="subgraph.xlsx",
				   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
