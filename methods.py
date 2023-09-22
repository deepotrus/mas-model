import networkx as nx
from matplotlib import pyplot as plt

def plot_graph(schedule):
    G = schedule.net
    pos = nx.spring_layout(G)
    labels = get_graphlabels(schedule._agents)

    plt.figure(figsize=(8, 5))
    s = nx.draw_networkx_nodes(
        G,
        pos,
        node_size = 200,
        node_color = list(dict(nx.degree(G)).values()),
        alpha = 1,
        cmap = plt.cm.BuGn
    )

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=6, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    #show the colorbar on the right side
    cbar = plt.colorbar(s)
    cbar.ax.set_ylabel('Degree', size=12)

    plt.axis('off')
    plt.show();
