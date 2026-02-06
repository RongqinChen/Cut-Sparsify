from networkx import Graph as NxGraph


class Node:
    _points = [
        [0, 0, 0, 0, 0],
        [0, 0.5, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0.5, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0.5, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0.5, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    _style = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;points={_points}"

    def __init__(self, id, value, width, height, x, y):
        self.id = f"node-{id}"
        self.value = value  # conntent displayed in the node
        self.width = width  # width of the node
        self.height = height  # height of the node
        self.x = x  # x coordinate of the node
        self.y = y  # y coordinate of the node
        # connecting points of the node

    def get_drawio_cell_lines(self):
        lines = [
            f'<mxCell id="{self.id}" value="{self.value}" style="{self._style}" vertex="1" parent="1">',
            f'<mxGeometry x="{self.x}" y="{self.y}" width="{self.width}" height="{self.height}" as="geometry" />',
            "</mxCell>",
        ]
        return lines


class Edge:
    _style = "endArrow=none;html=1;rounded=0;entryX=0.5;entryY=0.5;entryDx=0;entryDy=0;entryPerimeter=0;exitX=0.5;exitY=0.5;exitDx=0;exitDy=0;exitPerimeter=0;"

    def __init__(self, id, value, source: Node, target: Node):
        self.id = f"edge-{id}"
        self.value = value  # content displayed in the edge
        self.source = source  # source node
        self.target = target  # target node

    def get_drawio_cell_lines(self):
        lines = [
            f'<mxCell id="{self.id}" value="{self.value}" style="{self._style}" edge="1" parent="1" source="{self.source.id}" target="{self.target.id}">',
            '<mxGeometry width="50" height="50" relative="1" as="geometry">',
            f'<mxPoint x="{self.source.x}" y="{self.source.y}" as="sourcePoint" />',
            f'<mxPoint x="{self.target.x}" y="{self.target.y}" as="targetPoint" />',
            "</mxGeometry>",
            "</mxCell>",
        ]
        return lines


def _output_project_head(filepath):
    line = '<mxfile host="Electron" agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/26.0.9 Chrome/128.0.6613.186 Electron/32.2.5 Safari/537.36" version="26.0.9">'  # noqa
    with open(filepath, "w") as f:
        f.write(line)


def _output_project_tail(filepath):
    line = "</mxfile>"
    with open(filepath, "a") as f:
        f.write(line)
        f.write("\n")


def _output_page_head(filepath, idx=1):
    lines = [
        f'<diagram name="page-{idx}" id="page-{idx}">',
        '<mxGraphModel dx="830" dy="478" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">', # noqa
        "<root>",
        '<mxCell id="0" />',
        '<mxCell id="1" parent="0" />',
    ]
    with open(filepath, "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _output_page_tail(filepath):
    lines = [
        "</root>",
        "</mxGraphModel>",
        "</diagram>",
    ]
    with open(filepath, "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _output_edges(filepath, edges: list[Edge]):
    lines = []
    for edge in edges:
        lines.extend(edge.get_drawio_cell_lines())
    with open(filepath, "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _output_nodes(filepath, nodes: list[Node]):
    lines = []
    for node in nodes:
        lines.extend(node.get_drawio_cell_lines())
    with open(filepath, "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def draw_graph_drawio(
    nxg: NxGraph, pos: dict, node_labels=None, edge_labels=None, node_size=20, filepath=None
):
    assert filepath is not None, "filepath must be specified"
    assert filepath.endswith(".drawio"), "filepath must end with .drawio"

    # if node_labels is None, use the node index as the label
    if node_labels is None:
        node_labels = {idx: str(idx) for idx in nxg.nodes}
    # if edge_labels is None, the label is empty
    if edge_labels is None:
        edge_labels = {idx: "" for idx in range(nxg.number_of_edges())}

    # create nodes and edges
    nodes = [
        Node(idx, node_labels[idx], node_size, node_size, pos[idx][0]-node_size/2, pos[idx][1]-node_size/2)
        for idx in nxg.nodes
    ]
    edges = [
        Edge(idx, edge_labels[idx], nodes[edge[0]], nodes[edge[1]])
        for idx, edge in enumerate(nxg.edges)
    ]
    # output to drawio format
    _output_project_head(filepath)
    _output_page_head(filepath)
    _output_edges(filepath, edges)
    _output_nodes(filepath, nodes)
    _output_page_tail(filepath)
    _output_project_tail(filepath)
