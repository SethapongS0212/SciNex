"""
visualizer.py
-------------
Generates an interactive HTML visualization of the knowledge graph using pyvis.
Opens in any browser — nodes are draggable, zoomable, filterable.

For regular KGs:
    High-degree nodes are larger and colored differently.
    Edge labels show the predicate (relation).

For fused KGs (paper + entity nodes):
    Entity nodes: circles, colored by degree (blue/orange/red)
    Paper nodes:  squares, colored by role (red=root, purple=cited_by, dark-purple=citing)
    mentions edges are hidden (too many) — only cites and KG edges shown
    A legend is shown in the bottom-right corner.
"""

import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def build_viz(
    graph: nx.MultiDiGraph,
    output_path: str | Path,
    paper_title: str = "",
    max_nodes: int = 200,
    max_edges: int = 500,
):
    """
    Render the KG as an interactive pyvis HTML file.
    Automatically detects fused graphs (containing paper:: nodes) and
    applies a different visual style to paper nodes vs entity nodes.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis not installed. Skipping visualization. Run: pip install pyvis")
        return None

    output_path = Path(output_path)

    # Detect fused graph
    is_fused = any(str(n).startswith("paper::") for n in graph.nodes())

    # For fused graphs, hide mentions edges — they connect every paper to
    # every entity which creates a hairball with no useful structure
    if is_fused:
        render_graph = nx.MultiDiGraph()
        for n, d in graph.nodes(data=True):
            render_graph.add_node(n, **d)
        hidden = 0
        for u, v, d in graph.edges(data=True):
            if d.get("relation") == "mentions":
                hidden += 1
            else:
                render_graph.add_edge(u, v, **d)
        logger.info(f"Fused viz: hiding {hidden} mentions edges for clarity")
    else:
        render_graph = graph

    # Trim very large graphs
    if render_graph.number_of_nodes() > max_nodes:
        logger.info(f"Trimming to top {max_nodes} nodes by degree.")
        top_nodes = sorted(render_graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        render_graph = render_graph.subgraph([n for n, _ in top_nodes]).copy()

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
        cdn_resources="in_line"
    )

    # Looser physics for fused graphs to separate paper and entity clusters
    if is_fused:
        net.barnes_hut(
            gravity=-12000,
            central_gravity=0.15,
            spring_length=220,
            spring_strength=0.02,
            damping=0.12
        )
    else:
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=120,
            spring_strength=0.04,
            damping=0.09
        )

    # Add nodes
    for node_id, data in render_graph.nodes(data=True):
        degree = render_graph.degree(node_id)
        label  = data.get("label", str(node_id))
        display_label = label if len(label) < 50 else label[:47] + "..."
        is_paper = str(node_id).startswith("paper::")

        if is_paper:
            relation = data.get("relation", "")
            if not relation:
                color = "#e74c3c"   # red — root paper
                size  = 28
            elif relation == "cited_by":
                color = "#8e44ad"   # purple — papers that cite this work
                size  = 18
            else:
                color = "#6c3483"   # dark purple — papers this work cites
                size  = 18

            year       = data.get("year", "")
            cite_count = data.get("citation_count", "")
            tooltip    = f"<b>{label}</b>"
            if year:
                tooltip += f"<br>Year: {year}"
            if cite_count:
                tooltip += f"<br>Citations: {cite_count}"

            net.add_node(
                node_id,
                label=display_label,
                title=tooltip,
                size=size,
                color=color,
                shape="square",
                font={"size": 10, "color": "#ffffff"}
            )
        else:
            # Entity node — circle, colored by degree
            if degree <= 2:
                color = "#4a90d9"
            elif degree <= 6:
                color = "#f5a623"
            else:
                color = "#d0021b"
            size = 12 if degree == 1 else 18 if degree <= 3 else 26 if degree <= 7 else 38

            net.add_node(
                node_id,
                label=display_label,
                title=f"<b>{label}</b><br>Connections: {degree}",
                size=size,
                color=color,
                shape="dot",
                font={"size": 11, "color": "#ffffff"}
            )

    # Add edges
    edges_added = 0
    for src, dst, data in render_graph.edges(data=True):
        if edges_added >= max_edges:
            break
        relation = data.get("relation", "")
        sentence = data.get("sentence", "")

        if relation == "cites":
            net.add_edge(
                src, dst,
                label="",
                title="<b>cites</b>",
                arrows="to",
                color={"color": "#27ae60", "highlight": "#2ecc71"},
                width=1.2,
            )
        else:
            tooltip = f"<b>{relation}</b>"
            if sentence:
                tooltip += f"<br><i>{sentence[:120]}...</i>"
            net.add_edge(
                src, dst,
                label=relation,
                title=tooltip,
                arrows="to",
                color={"color": "#7f8c8d", "highlight": "#f5a623"},
                width=1.5,
                font={"size": 9, "color": "#bdc3c7", "strokeWidth": 0}
            )
        edges_added += 1

    title_html = (
        f"<h3 style='font-family:sans-serif;color:#e0e0e0;text-align:center;margin:8px 0'>"
        f"Knowledge Graph — {paper_title}</h3>"
    ) if paper_title else ""

    legend_html = ""
    if is_fused:
        legend_html = """
        <div style="position:fixed;bottom:20px;right:20px;background:#2c2c4e;
                    padding:12px 18px;border-radius:8px;font-family:sans-serif;
                    font-size:12px;color:#e0e0e0;z-index:999;line-height:1.9">
          <b>Legend</b><br>
          <span style="color:#d0021b">&#9679;</span> Hub entity &nbsp;
          <span style="color:#f5a623">&#9679;</span> Mid entity &nbsp;
          <span style="color:#4a90d9">&#9679;</span> Leaf entity<br>
          <span style="color:#e74c3c">&#9632;</span> Root paper &nbsp;
          <span style="color:#8e44ad">&#9632;</span> Cited-by &nbsp;
          <span style="color:#6c3483">&#9632;</span> Citing<br>
          <span style="color:#27ae60">&#9135;&#9135;</span> cites &nbsp;
          <span style="color:#7f8c8d">&#9135;&#9135;</span> KG relation
        </div>"""

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": {"enabled": true}
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.15},
        "font": {"align": "middle"}
      }
    }
    """)

    net.save_graph(str(output_path))

    html = output_path.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>{title_html}{legend_html}", 1)
    output_path.write_text(html, encoding="utf-8")

    logger.info(f"Visualization saved → {output_path}")
    return output_path