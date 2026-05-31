# Do Professional CS2 Players Have Distinct Playstyles? An Unsupervised Learning Approach

*A data science deep-dive into what the stats actually say about player archetypes at the top of professional Counter-Strike 2.*

---
<div align="center">
  <img src="images/cs_thumpnail.jpg" alt="Counter Strike Thumpnail">
</div>

## Introduction

I have been playing Counter-Strike for longer than I care to admit. It started on a family computer in CS 1.6 – the era of pixelated smokes and 64-tick servers – then progressed through CS:GO's ranked matchmaking and global competitive scene, and finally into CS2 in 2023, which rebuilt the engine on Source 2 with volumetric smokes and a proper subtick system. Same DNA, new body.

Playing through all three eras gives you an intuition for the game that is hard to fake. You understand why a player holds a particular angle, why a team might stack one bombsite over another, and why an AWPer having a bad day can unravel an entire strategy. That intuition is exactly what I wanted to test with data.

---

### The Competitive Scene and HLTV

Counter-Strike is one of the world's most-watched esports, structured around Valve's biannual **Major Championships** and a year-round circuit of third-party events including ESL Pro League, BLAST Premier, and IEM. The format is 5-versus-5, with teams alternating between Terrorist and Counter-Terrorist sides across a 24-round match.

[**HLTV.org**](https://www.hltv.org) is the de facto statistics platform for professional Counter-Strike, founded in 2000 and tracking data across every era of the game. For this project, I scraped player data directly from HLTV covering the **last 3 months** (February–May 2026), filtered to matches against top-20 ranked opponents. The result is a dataset of **98 professional players** and 45 features per player.

---

### Understanding the Statistics

**Rating 3.0** is HLTV's composite performance metric, combining kill frequency, survival rate, multi-kill rounds, and impact. A rating of 1.00 is the professional average; above 1.15 is elite. In our dataset, ratings ranged from **0.84** (karrigan, a veteran IGL) to **1.41** (donk).

**KAST** (Kill, Assist, Survive, or Trade) is the percentage of rounds in which a player contributed in at least one of those ways. It measures consistency rather than raw output, with the professional average sitting around 72–74%.

**ADR** (Average Damage per Round) captures damage output regardless of whether it results in a kill – a player who deals 99 damage to an enemy finished by a teammate still contributes to ADR. Elite fraggers typically post ADR above 80; support players often sit in the 65–72 range.

**KPR** and **DPR** (Kills and Deaths per Round) are straightforward. DPR is role-dependent – entry fraggers are expected to die more than AWPers or lurkers.

**T-Rating and CT-Rating** split performance by side. AWPers are often dominant on CT holding angles, while aggressive riflers tend to flip this balance on T side.

**Round Swing** measures how correlated a player's performance is with whether their team wins the round – a strongly positive value means the player is consistently the difference-maker.

**Headshot Percentage** varies sharply by weapon. AWP users have low HS% because the weapon one-shots to any upper-body zone; riflers can push past 60% through precise aim.

**Impact Rating** captures clutch contributions – multi-kill rounds and 1vX situations. **Grenade Damage per Round** reflects utility usage: flashes, HEs, and molotovs that deal damage without earning a kill credit.

For weapon-specific analysis, I collected each player's kill share across four categories: **rifle**, **sniper**, **pistol**, and **SMG**.

---

### Roles in Counter-Strike 2

Professional teams organize around five roles, though boundaries blur in practice.

The **AWPer** holds the team's \$4,750 one-shot sniper rifle, locking down angles on defense and picking off opponents before engagements start. Losing the AWP is an economic disaster, so AWPers must be selective with duels.

The **Entry Fragger** pushes into a site first, creating space for teammates even at the cost of their own life. Their job is to open a round favorably – not to top the scoreboard.

The **Lurker** operates away from the main execution, cutting off rotations or catching defenders out of position for high-impact individual plays.

The **Support** player flashes, smokes, and throws molotovs so teammates can frag. Their value shows up in their teammates' stats more than their own.

The **IGL** (In-Game Leader) calls strategy and sacrifices personal stats to do it. Their statistical signature is typically a below-average rating paired with high grenade damage and assist rates – karrigan, Aleksib, and HooXi are textbook examples.

---

### What This Analysis Is Trying to Find

The central question is simple: **do the statistics tell us anything about how players play, or just how good they are?**

To find out, I applied four unsupervised learning techniques to the dataset:

- **PCA / SVD** – to identify which combinations of features capture the most variance, revealing the true underlying dimensions of player performance.
- **K-Means Clustering** – to partition players into statistically similar groups, with the elbow method and silhouette scores guiding the choice of k.
- **Hierarchical Clustering** – to build a bottom-up tree of player similarities, producing a dendrogram that reveals how players relate at different levels of granularity.

The goal is not to definitively label players with roles – that would require positional and timing data HLTV does not expose publicly. The goal is to see what naturally emerges from the numbers: are there genuine statistical archetypes among professional CS2 players, and if so, what defines them?

---
## Theoretical Background

This section covers the three unsupervised learning methods used in this analysis: Principal Components Analysis (PCA), K-Means Clustering, and Hierarchical Clustering. All methods are discussed in James et al. [1].

---

### Principal Components Analysis (PCA)

PCA finds a low-dimensional representation of a high-dimensional dataset by identifying the directions of greatest variance. The **first principal component** is the normalized linear combination of features that captures the most variance:

$$Z_1 = \phi_{11}X_1 + \phi_{21}X_2 + \cdots + \phi_{p1}X_p$$

where the loading vector $\phi_1 = (\phi_{11}, \ldots, \phi_{p1})^T$ is constrained to $\sum_j \phi_{j1}^2 = 1$. Each subsequent component is orthogonal to the previous ones and maximizes the remaining variance. Geometrically, the first $M$ principal components span the $M$-dimensional subspace closest to the data in terms of average squared Euclidean distance.

**Tuning:** The number of components $M$ is selected using a **scree plot** of the proportion of variance explained (PVE) per component:

$$\text{PVE}_m = \frac{\sum_{i=1}^n z_{im}^2}{\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}$$

One retains enough components to explain a meaningful share of total variance, looking for an "elbow" in the scree plot, or go by a desired threshold such as 85% or 90%.

**Interpretation:** Loading vectors indicate which features contribute to each component. Observations are interpreted through their scores projected onto the principal component axes.

**Limitations:** PCA assumes linear relationships and is sensitive to variable scaling – features must be standardized before application. Components can also be difficult to interpret when loadings are spread across many variables.

---

### K-Means Clustering

K-Means partitions $n$ observations into $K$ non-overlapping clusters by minimizing total within-cluster variation. Formally, the objective is:

$$\underset{C_1, \ldots, C_K}{\text{minimize}} \sum_{k=1}^K \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2$$

The algorithm iterates two steps: (1) assign each observation to the nearest cluster centroid, and (2) recompute centroids as the mean of all assigned observations. This is repeated until assignments stabilize. The algorithm finds a local optimum, so it should be run multiple times with different random initializations, selecting the solution with the lowest objective value.

**Tuning:** $K$ is the primary hyperparameter, selected using the **elbow method** (inertia vs. $K$) and the **silhouette score**, which measures how similar each point is to its own cluster relative to the nearest competing cluster. Silhouette scores range from $-1$ to $1$; higher is better.

**Interpretation:** Each cluster is summarized by its centroid. Cluster profiles are examined by comparing mean feature values across groups.

**Limitations:** K-Means requires $K$ to be specified in advance, assumes roughly spherical clusters of similar size, and is sensitive to outliers. Because it finds local optima, results can vary across runs.

---

### Hierarchical Clustering

Hierarchical clustering builds a nested sequence of groupings without requiring $K$ to be specified in advance. The agglomerative (bottom-up) approach begins with each observation as its own cluster and iteratively merges the two most similar clusters:

1. Compute all $\binom{n}{2}$ pairwise dissimilarities.
2. Fuse the two most similar clusters.
3. Recompute inter-cluster dissimilarities and repeat until all observations are in one cluster.

The result is a **dendrogram** – a tree diagram where the height of each fusion reflects the dissimilarity between merged groups. Clusters are extracted by cutting the dendrogram at a chosen height.

**Linkage:** The choice of linkage method determines how inter-cluster dissimilarity is computed. **Complete linkage** uses the maximum pairwise distance between clusters; **average linkage** uses the mean; **single linkage** uses the minimum. Complete and average linkage are preferred, as single linkage tends to produce elongated, chain-like clusters.

**Tuning:** Key choices are the linkage method and dissimilarity measure (Euclidean distance is standard). The cut height is chosen by visual inspection of the dendrogram.

**Limitations:** Hierarchical clustering assumes a nested cluster structure, which may not reflect reality. Early merges cannot be undone, so errors propagate upward. It is also computationally more expensive than K-Means at large $n$.

---

## Methodology

### Data Collection and Processing

Player statistics were scraped directly from HLTV.org using browser automation, covering the period February 19 – May 19, 2026, filtered to matches against top-20 ranked opponents with a minimum of 10 maps played. This produced a dataset of **98 professional players**. Performance statistics (Rating 3.0, KAST, ADR, KPR, DPR, T/CT rating, round swing, impact rating, multi-kill rate, headshot percentage, grenade damage, save-related metrics, and weapon kill distribution) were collected in one scraping pass, resulting in **40 numeric features** per player.

Distribution inspection revealed that four sniper-related columns – `sniper_kills`, `sniper_pct`, `awp_kills`, and `ssg08_kills` – were heavily zero-inflated, since the majority of players are riflers who rarely touch the AWP. These were replaced with log-transformed equivalents using `log1p` to compress the right tail and prevent the AWPer signal from being distorted by extreme raw values. Two low-information columns (`top_weapon` and `top_weapon_pct`) were dropped.

All 40 features were standardized to zero mean and unit variance using `StandardScaler` before any further analysis. This is essential for both PCA and clustering because without it, features with large absolute scales (e.g. total kills in the hundreds) would dominate those on smaller scales (e.g. assists per round near 0.2).

---

### Principal Component Analysis (PCA)
PCA was applied to the standardized feature matrix using `sklearn.decomposition.PCA` with `n_components=0.9`, which retains the minimum number of components necessary to explain 90% of the total variance. A combined scree plot – showing both individual and cumulative proportion of variance explained – was produced to validate this threshold and examine the decomposition structure. 

The PCA-reduced output was used as input for K-Means. Hierarchical clustering was run on the full 40-feature standardized matrix instead, to preserve fine-grained pairwise similarity structure that dimensionality reduction might discard.

---

### K-Means Clustering

K-Means was applied to the PCA-reduced data with $k$ ranging from 2 to 10. Each model was run with `n_init=20` random initializations to reduce sensitivity to starting conditions, selecting the solution that minimized within-cluster inertia at each $k$. Two metrics guided the choice of $k$:

- **Inertia (elbow plot):** Total within-cluster sum of squares as a function of $k$. A distinct elbow indicates diminishing returns from adding further clusters.
- **Silhouette score:** Ranges from $-1$ to $1$ and measures how much more similar each point is to its own cluster than to the nearest alternative. Higher values indicate more compact, well-separated clusters.

Both metrics were plotted across the full range of $k$ and inspected together to select the final number of clusters. 

---

### Hierarchical Clustering

Agglomerative hierarchical clustering was applied to the full standardized feature matrix using `sklearn.cluster.AgglomerativeClustering` with `distance_threshold=0` and `n_clusters=None`, producing a complete dendrogram for each linkage method. Four linkage strategies were evaluated by visual inspection of their dendrograms. Silhouette scores were then computed on the full standardized feature matrix for Ward clustering across $k = 2$ to $6$, and a silhouette plot was produced. The dendrogram was cut at $k = 2$, $k = 3$, and $k = 4$ using `scipy.cluster.hierarchy.cut_tree`, and each solution was profiled by comparing mean feature values across clusters. The final number of clusters was selected based on both the silhouette scores and the interpretability of the resulting groupings.

---

## Results







