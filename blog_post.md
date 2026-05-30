# Do Professional CS2 Players Have Distinct Playstyles? An Unsupervised Learning Approach

*A data science deep-dive into what the stats actually say about player archetypes at the top of professional Counter-Strike 2.*

---

## Introduction

### A Game That Grew Up With Us

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
- **Gaussian Mixture Models (GMM)** – to assign probabilistic cluster memberships, allowing borderline players to carry partial membership in multiple archetypes.
- **Hierarchical Clustering** – to build a bottom-up tree of player similarities, producing a dendrogram that reveals how players relate at different levels of granularity.

The goal is not to definitively label players with roles – that would require positional and timing data HLTV does not expose publicly. The goal is to see what naturally emerges from the numbers: are there genuine statistical archetypes among professional CS2 players, and if so, what defines them?

---

*Data was scraped from HLTV.org via browser automation, covering February 19 – May 19, 2026, filtered to top-20 opponents with a minimum of 10 maps played. Code and data are in the accompanying repository.*
