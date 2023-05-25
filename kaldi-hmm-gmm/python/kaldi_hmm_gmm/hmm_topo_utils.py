import _kaldi_hmm_gmm


def draw_hmm_topology(hmm: _kaldi_hmm_gmm.HmmTopology, phone: int) -> "Digraph":
    """
    Please refer to https://graphviz.readthedocs.io/en/stable/manual.html
    for how to use the returned value.

    For instance,

        print(dot)

    And copy the output to https://dreampuf.github.io/GraphvizOnline/
    you will see the figure.
    """
    try:
        import graphviz
    except Exception:
        print("Please run")
        print("  pip install graphviz")
        print("before calling this function.")
        raise

    graph_attr = {
        "rankdir": "LR",
        "size": "8.5,11",
        "center": "1",
        "orientation": "Portrait",
        "ranksep": "0.4",
        "nodesep": "0.25",
        "label": f"Topology for phone {phone} (pdf_class/transition_prob)",
    }

    topo = hmm.topology_for_phone(phone)

    default_node_attr = {
        "shape": "circle",
        "style": "bold",
        "fontsize": "14",
    }

    final_state_attr = {
        "shape": "doublecircle",
        "style": "bold",
        "fontsize": "14",
    }

    dot = graphviz.Digraph(name="HMM topology", graph_attr=graph_attr)

    # first draw state
    for src_state, hmm_state in enumerate(topo):
        if len(hmm_state.transitions) > 0:
            dot.node(str(src_state), label=str(src_state), **default_node_attr)
        else:
            dot.node(str(src_state), label=str(src_state), **final_state_attr)

    for src_state, hmm_state in enumerate(topo):
        for dst_state, prob in hmm_state.transitions:
            if src_state == dst_state:
                pdf_class = hmm_state.forward_pdf_class
            else:
                pdf_class = hmm_state.self_loop_pdf_class

            dot.edge(str(src_state), str(dst_state), label=f"{pdf_class}/{prob}")
    return dot
