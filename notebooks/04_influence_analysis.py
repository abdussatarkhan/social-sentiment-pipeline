# %% [markdown]
# # 04: Social Interaction Graph & PageRank Detractor Amplification
# ### Real-Time Social Listening & Brand Sentiment Pipeline
# 
# **Author:** Network Science & Graph Analytics  
# **Techniques:** NetworkX Directed Graphs, PageRank Centrality, Sentiment Diffusion, Amplification Risk Index  
# 
# ---
# ### Objectives:
# In social crises, not all negative posts exert equal damage. A negative tweet from an isolated user with 0 followers dies out quickly; the same criticism from a central industry figure triggers cascade reactions across hundreds of engineers.
# 
# In this notebook, we:
# 1. Build a directed interaction network from replies, mentions, and quotes.
# 2. Compute PageRank centrality scores to identify authoritative voices.
# 3. Formulate the **Amplification Risk Score** to identify dangerous detractors.
# 4. Visualize the network structure with sentiment-colored node topologies.

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Add repo to sys.path
sys.path.append(str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from scripts.influence_analysis import SocialInfluenceAnalyzer

sns.set_theme(style="white")
# %matplotlib inline

# %% [markdown]
# ## 1. Constructing the Social Interaction Graph
# We simulate a 50-node scale-free social discussion network centered around target brand `ApexCloud`.

# %%
analyzer = SocialInfluenceAnalyzer()
edges_df, nodes_df = analyzer.generate_synthetic_graph(num_nodes=50, seed=101)

G = analyzer.build_network_graph(edges_df, nodes_df)
print(f"Graph Nodes (Users): {G.number_of_nodes()}")
print(f"Graph Edges (Interactions): {G.number_of_edges()}")
print(f"Is Directed: {G.is_directed()}")

# %% [markdown]
# ## 2. Calculating Centralities & Amplification Risk Index
# We define the **Amplification Risk Score**:
# $$\text{Risk}(u) = \text{PageRank}(u) \cdot \left(1 + \log(1 + \text{NegPosts}(u))\right) \cdot \left(\frac{\text{NegPosts}(u)}{\text{TotalPosts}(u)}\right)^{1.2} \times 1000$$

# %%
df_influence = analyzer.compute_influence_metrics()

print("--- Top 10 High-Risk Brand Detractor Amplifiers ---")
display_cols = ["author_handle", "platform", "pagerank_score", "in_degree", "negative_posts", "total_posts", "amplification_risk_score"]
print(df_influence[display_cols].head(10).to_string(index=False))

# %% [markdown]
# ## 3. Centrality Comparison: PageRank vs Degree Centrality
# Why PageRank matters over raw degree count: PageRank accounts for the quality and influence of who is engaging with the author.

# %%
plt.figure(figsize=(8, 5), dpi=150)
scatter = plt.scatter(
    df_influence["in_degree"],
    df_influence["pagerank_score"],
    c=df_influence["amplification_risk_score"],
    cmap="YlOrRd",
    s=df_influence["negative_posts"] * 8 + 40,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.5,
)
cbar = plt.colorbar(scatter)
cbar.set_label("Amplification Risk Index", fontsize=9, fontweight="bold")

plt.title("Structural Authority: PageRank vs In-Degree Centrality", fontsize=11, fontweight="bold", pad=12)
plt.xlabel("In-Degree (Incoming Replies & Mentions)", fontsize=9)
plt.ylabel("PageRank Authority Score", fontsize=9)

# Annotate top 3 high-risk influencers
for _, row in df_influence.head(3).iterrows():
    plt.annotate(
        row["author_handle"],
        (row["in_degree"], row["pagerank_score"]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=8,
        fontweight="bold",
        color="#9B2C2C",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Network Visualization of Sentiment Propagation
# - **Node Size:** PageRank centrality
# - **Node Color:** Mean sentiment (Red = Detractor, Blue/Green = Promoter)
# - **Edge Arrows:** Directed conversation flows

# %%
plt.figure(figsize=(11, 9), dpi=150)
pos = nx.spring_layout(G, k=0.35, iterations=50, seed=42)

# Node attributes
node_colors = [G.nodes[n].get("mean_sentiment", 0.0) for n in G.nodes()]
node_sizes = [G.nodes[n].get("total_posts", 1) * 20 + 60 for n in G.nodes()]

# Draw Edges
nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#A0AEC0", arrowsize=10, width=0.8)

# Draw Nodes
nodes_plot = nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    cmap="coolwarm_r",  # Red = negative, Blue = positive
    node_size=node_sizes,
    alpha=0.9,
    edgecolors="black",
    linewidths=0.8,
)

# Label top 5 riskiest nodes
top_risk_users = set(df_influence.head(5)["author_id"].tolist())
labels = {n: G.nodes[n].get("handle", n) for n in G.nodes() if n in top_risk_users}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", font_color="#1A202C")

cbar = plt.colorbar(nodes_plot, fraction=0.03, pad=0.04)
cbar.set_label("Sentiment Polarity Score (Red = Negative, Blue = Positive)", fontsize=8)

plt.title("Brand Sentiment Conversation Graph Topology", fontsize=12, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Community Detection & Echo Chamber Segregation

# %%
from networkx.algorithms.community import greedy_modularity_communities

# Undirected projection for community detection
G_undir = G.to_undirected()
communities = list(greedy_modularity_communities(G_undir))

print(f"Detected {len(communities)} distinct conversational communities.")
for idx, comm in enumerate(communities[:4], 1):
    sub_df = df_influence[df_influence["author_id"].isin(comm)]
    avg_sent = sub_df["negative_posts"].sum() / max(1, sub_df["total_posts"].sum())
    print(f"  Community #{idx}: {len(comm)} members | Negative Post Ratio: {avg_sent*100:.1f}%")

# %% [markdown]
# ### Operational Summary:
# 1. Rather than treating all critical comments uniformly, PR and Developer Relations teams can triage outreach by prioritizing nodes with **Amplification Risk Score > 50.0**.
# 2. High-in-degree negative nodes serve as contagion catalysts; rapid intervention on their threads significantly dampens secondary retweet/reply propagation.
