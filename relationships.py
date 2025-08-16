# app.py
import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

st.set_page_config(layout="wide", page_title="Product Recommendation Network (Option B)")

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
	s = str(val)
	s = re.sub(r"[^\d\.\-]", "", s)  # strip $, commas, spaces, etc.
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
	  - url cols (preferred: *_url; else any recommended_product_* that isn't *_name/_title/_subtitle)
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
	for i in range(1, 201):
		c = f"recommended_product_{i}_name"
		if c in df.columns:
			name_cols[i] = c
	for i in range(1, 201):
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
	  - Brand/type filters (Option B): keep selected nodes + neighbors
	  - Product name filter (Option B): keep name-matched nodes + neighbors
	If multiple filters provided, the intersection of selected sets is used before adding neighbors.
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

	# Brand/type selection
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

	# If any filter applied, include neighbors for context (Option B)
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
# Top-N hubs + neighbors (Option B)
# ----------------------------
def top_n_with_neighbors(G: nx.DiGraph, top_n: int) -> nx.DiGraph:
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
# Price -> color mapping (with explicit cmin/cmax on LOG)
# ----------------------------
def color_array_and_colorbar(prices_array: np.ndarray, mode: str, clip_pct: float = 95.0):
	"""
	Return (color_values, colorbar_dict, colorscale, cmin, cmax)
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

	# Log (base 10) — normalize with cmin/cmax so full color range is used
	vals = np.log10(p)
	vmin = float(np.min(vals))
	vmax = float(np.max(vals))

	# Human-friendly tick labels showing real $ on log scale; keep within [vmin, vmax]
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
# Plot graph (with highlight overlays)
# ----------------------------
def plot_graph(H: nx.DiGraph,
			   label_top_percent: int = 10,
			   color_mode: str = "Log (base 10)",
			   clip_pct: float = 95.0,
			   min_size: int = 8,
			   max_size: int = 50,
			   height: int = 900,
			   highlight_nodes: set[str] | None = None):
	if H.number_of_nodes() == 0:
		return go.Figure()

	pos = nx.spring_layout(H, k=0.3, seed=42)
	indeg = dict(H.in_degree())

	# Price -> color
	prices = {n: H.nodes[n].get("price", np.nan) for n in H.nodes()}
	price_vals = np.array([prices[n] for n in H.nodes()], dtype=float)
	color_values, colorbar_dict, colorscale_name, cmin, cmax = color_array_and_colorbar(
		price_vals, color_mode, clip_pct=clip_pct
	)

	# Labels for top % by indegree
	counts = np.array(list(indeg.values()), dtype=float)
	cutoff = np.percentile(counts, 100 - label_top_percent) if len(counts) else 0
	top_label_nodes = {n for n, v in indeg.items() if v >= cutoff}

	# Edge coords (base)
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

	# Base node trace (all nodes)
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
			colorscale=colorscale_name,
			reversescale=True,      # low price -> red, high price -> blue
			color=color_values,     # linear/clip/log values
			cmin=cmin,
			cmax=cmax,
			size=base_sizes,
			colorbar=colorbar_dict,
			line=dict(width=1, color="#333")
		),
		name="Products"
	)

	# Highlight node overlay (subset only, thicker outline + slightly larger)
	if highlight_nodes:
		hi_nodes = [n for n in nodes if n in highlight_nodes]
		if hi_nodes:
			hi_x = [pos[n][0] for n in hi_nodes]
			hi_y = [pos[n][1] for n in hi_nodes]
			hi_sizes = []
			hi_colors = []
			hi_text = []
			hi_hover = []

			node_index = {n: i for i, n in enumerate(nodes)}
			for n in hi_nodes:
				idx = node_index[n]
				hi_sizes.append(base_sizes[idx] * 1.2)  # 20% larger
				hi_colors.append(color_values[idx])      # keep same colormap value
				price_display = prices[n]
				if isinstance(price_display, float) and np.isnan(price_display):
					price_display = "N/A"
				full_name = H.nodes[n].get("name", n)
				hi_hover.append(
					f"ID: {n}<br>"
					f"Name: {full_name}<br>"
					f"Price: {price_display}<br>"
					f"In-degree: {indeg.get(n, 0)}"
				)
				# Always show label for highlighted, plus if already in top_label_nodes
				hi_text.append(full_name)

			hi_node_trace = go.Scatter(
				x=hi_x, y=hi_y,
				mode="markers+text",
				text=hi_text,
				textposition="top center",
				hovertext=hi_hover,
				hoverinfo="text",
				marker=dict(
					showscale=False,           # avoid duplicate colorbar
					colorscale=colorscale_name,
					reversescale=True,
					color=hi_colors,
					cmin=cmin,
					cmax=cmax,
					size=hi_sizes,
					line=dict(width=3, color="#000")  # thick black outline
				),
				name="Highlighted"
			)
		else:
			hi_node_trace = None
	else:
		hi_node_trace = None

	traces = [base_edge_trace, hi_edge_trace, base_node_trace]
	if hi_node_trace is not None:
		traces.append(hi_node_trace)

	fig = go.Figure(
		data=traces,
		layout=go.Layout(
			title=dict(text="Product Recommendation Network (Top-N hubs + neighbors)", font=dict(size=20)),
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
st.title("🍷 Product Recommendation Network — Option B (Contextual)")

uploaded = st.file_uploader("Upload your product CSV", type=["csv"])
if not uploaded:
	st.info("Upload a CSV exported from your scrape to begin.")
	st.stop()

df = load_csv(uploaded)

# Sidebar controls — Filters
st.sidebar.header("Filters (Contextual)")
brand_vals = sorted(df.get("brand_name", pd.Series(dtype=str)).dropna().unique().tolist())
type_vals  = sorted(df.get("type", pd.Series(dtype=str)).dropna().unique().tolist())

brand_filter = st.sidebar.multiselect("brand_name", brand_vals)
type_filter  = st.sidebar.multiselect("type", type_vals)

# Product name filter for graph content (Option B)
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

# Color scale options + percentile slider for clipping
color_mode = st.sidebar.selectbox("Color scale for price", ["Log (base 10)", "Linear", "Clipped (percentile)"])
clip_pct = 95.0
if color_mode == "Clipped (percentile)":
	clip_pct = st.sidebar.slider("Clip at percentile", min_value=80.0, max_value=99.9, value=95.0, step=0.1)

graph_height = st.sidebar.slider("Graph height (px)", 500, 1600, 900, step=50)

# SQRT sizing controls
st.sidebar.markdown("---")
min_size = st.sidebar.slider("Min bubble size (px)", 4, 30, 8)
max_size = st.sidebar.slider("Max bubble size (px)", 20, 160, 50)

# ----------------------------
# Search Highlight (overlay)
# ----------------------------
st.sidebar.header("Search Highlight")
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

# Build Option-B Top-N subgraph: hubs + neighbors
H = top_n_with_neighbors(G_full, top_n=top_n)

# Build highlight set mapped into H (by name & ID)
highlight_nodes = set()
if H.number_of_nodes() > 0:
	# Map node -> name in H
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
st.write(f"**Displayed subgraph (Top-{top_n} hubs + neighbors):** {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")
if highlight_nodes:
	st.write(f"**Highlighted nodes:** {len(highlight_nodes)}")

fig = plot_graph(
	H,
	label_top_percent=label_top_percent,
	color_mode=color_mode,
	clip_pct=clip_pct,
	min_size=min_size,
	max_size=max_size,
	height=graph_height,
	highlight_nodes=highlight_nodes
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
