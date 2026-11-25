# Notation & LaTeX Style Guide

This guide is for an agent that writes or edits LaTeX for this project (BlueROV2 / MarineGym, depth- and sonar-based RL navigation, comparing a **DeFM** encoder with a **VAE** encoder). Follow it exactly. When the guide and the existing text disagree, follow the guide and list the disagreement in your preview report (§7).

---

## 0. Workflow (mandatory)

1. **Read the project first.** Get parameter values, dimensions, and loss weights from the code and configs (YAML, Python, URDF/xacro, `*.json`). **Never invent numbers.** If a value can't be found, write `\todo{?}` (or `??`) and list it in the report.
2. **Check every symbol against the registry (§3)** before using it. Don't introduce a symbol that's already reserved. A new symbol must be added to the registry in the same change.
3. **Write the LaTeX** following §4–§6.
4. **Build a preview before committing** (§7). Show it to the user and **wait for approval**. Don't `git commit` or push until the user says so.

---

## 1. General conventions

| Kind | Style | Example |
|---|---|---|
| Scalars | italic | $m$, $u$, $\psi$ |
| Vectors | bold lowercase (`\mathbf{}`; `\boldsymbol{}` for Greek) | $\mathbf{a}_t$, $\boldsymbol{\nu}$, $\boldsymbol{\eta}$ |
| Matrices | bold uppercase | $\mathbf{M}$, $\mathbf{D}(\boldsymbol{\nu})$, $\mathbf{J}_\Theta(\boldsymbol{\eta})$ |
| Images / 2-D arrays | italic uppercase, non-bold | $D_t$, $D^{\mathrm{col}}_t$ |
| Sets / spaces | calligraphic | $\mathcal{O}$, $\mathcal{A}$, $\mathcal{B}$ |
| Losses | `\Lc{name}` → $\mathcal{L}_{\mathrm{name}}$ | $\mathcal{L}_{\mathrm{KL}}$ |
| Word-like sub/superscripts | upright `\mathrm{}` | $d_{\mathrm{VAE}}$, $r_{\max}$ |
| Index sub/superscripts | italic | $s_{t,i}$, $\lambda_k$ |
| Frames | superscript, Fossen style | $\mathbf{p}^{n}$, $\mathbf{v}^{b}$ |

* State a symbol's space and size the first time it appears: $D_t \in \mathbb{R}_{\ge 0}^{H \times W}$.
* Time index is always a subscript $t$ and comes first: $s_{t,i}$, not $s_{i,t}$.
* Never use a multi-letter italic name such as `$obs$`. Write `$\mathrm{obs}$` or choose a symbol.
* Use `\top` for transpose: `\mathbf{x}^\top`.
* Units always use `siunitx`: `\si{\metre\per\second}`, `\SI{12}{\metre}`.

### 1.2 Citations and cross-references

* `\textcite{key}` when the authors are part of the sentence: "\textcite{kulkarni2024dce} train …".
* `\parencite{key}` otherwise, placed before the full stop: "… for aerial robots \parencite{kulkarni2024dce}."
* Never use `\cite`, `\citep`, `\citet` or bare `[key]` brackets. Never write the author names and `\textcite` together ("Hochreiter and Schmidhuber \textcite{…}").
* A key must exist in `references.bib` and have a row in `sources/ratings.md` before it is cited.
* Cross-references always carry their type with a non-breaking space: `Section~\ref{}`, `Figure~\ref{}`, `Table~\ref{}`, `Equation~\eqref{}`. No hard-coded numbers ("Sec. 2.2"); `cleveref` is not loaded.
* Every `\section` has a label: `sec:introduction`, `sec:background`, `sec:related-work`, `sec:method`, `sec:results`, `sec:discussion`, `sec:conclusion`.

### 1.1 Notation sources

Each topic takes its notation from one reference. Write that topic the way its reference does, and cite the reference where the notation is introduced in the project.

| Topic | Reference | Key in `references.bib` | What it sets |
|---|---|---|---|
| UUV kinematics and dynamics | Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control* (with SNAME naming) | **missing — add** | Frames $\{n\}$, $\{b\}$; $\boldsymbol{\eta}$, $\boldsymbol{\nu}$, $\boldsymbol{\tau}$; $\mathbf{M}$, $\mathbf{C}(\boldsymbol{\nu})$, $\mathbf{D}(\boldsymbol{\nu})$, $\mathbf{g}(\boldsymbol{\eta})$; hydrodynamic derivatives; sign convention. Everything in §2 and §3.1. |
| MDP basics | Sutton and Barto, *Reinforcement Learning: An Introduction* | `sutton2018reinforcement` | The agent–environment loop, state, action, reward, return, value function, discount $\gamma$. |
| POMDP formulation | Song, Romero, Müller, Koltun and Scaramuzza, *Reaching the limit in autonomous racing* | `song2023reaching` | The split between the state and the observation $o_t$, and a policy that acts on observations instead of the full state. |
| Depth images and the DeFM encoder | DeFM | **missing — add** | Depth image $D_t$ and its size $H\times W$, the log-scaled 3-channel preprocessing $\Gamma(\cdot)$, the embedding $\mathbf{z}^{\mathrm{DeFM}}_t$. |
| Collision images and the VAE encoder | Deep Collision Encoding (DCE) | **missing — add** | Collision image $D^{\mathrm{col}}_t$, its reconstruction, the VAE loss in §4. |
| Problem formulation, observation groups, policy features | Deowan and Kelasidi, *Unified Planning-Learning Framework for Robust UUV Navigation Under Partial Observability* | `deowan2026unified` | Partially observable control problem with $\pi(\mathbf{a}_t\mid o_{\le t})$; observation groups $o^{(\cdot)}_t$; feature groups $\xi^{(\cdot)}_t$ in the policy input. Their symbols that clash with Fossen ($\pi_\theta$, $\varphi_t$, $r_t$, $x_t$, $\nu_t$, $\omega_t$, $w_\cdot$) are replaced by the registry symbols below. |
| PPO and GAE | Schulman et al., *Proximal Policy Optimization Algorithms*; Schulman et al., GAE | `schulman2017ppo`; GAE **missing — add** | Clipped objective, clip range $\epsilon$, advantage $\hat A_t$, TD residual $\delta_t$, $\lambda_{\mathrm{GAE}}$. |
| Recurrent networks (RNN, LSTM, GRU) | Hochreiter and Schmidhuber (LSTM); Cho et al. (GRU); figures after Olah, *Understanding LSTM Networks* | `hochreiter1997long`; `cho2014learning`; `olah2015lstm` | Hidden state $\mathbf{h}_t$, LSTM cell state $\mathbf{c}_t$ and gates $\mathbf{f}_t,\mathbf{i}_t,\mathbf{o}_t$, candidates $\tilde{\mathbf{c}}_t,\tilde{\mathbf{h}}_t$, GRU gates $\boldsymbol{\upsilon}_t,\boldsymbol{\chi}_t$. |

**When two sources collide, Fossen wins.** Fossen symbols are never renamed. A symbol from another source that clashes with Fossen is renamed, and the rename is recorded in the last column of §3.2. For example, Song et al. write the policy as $\pi_\theta(\mathbf{a}_t\mid\mathbf{o}_t)$. The structure still follows the source; only clashing letters change. **One deliberate exception (2026-11-25):** the policy weights are $\theta$, as in Song et al. and Deowan and Kelasidi, written only as the subscript of $\pi_\theta$ (and in $\max_\theta$, $\mathrm{GRU}(\cdot;\theta)$), while pitch always carries its time index $\theta_t$ or sits in $\boldsymbol\eta$. The Method states this once.

A new topic needs a row in this table before its notation is used.

---

## 2. Vehicle dynamics — Fossen notation

Use Fossen (*Handbook of Marine Craft Hydrodynamics and Motion Control*) with SNAME naming for **all** UUV dynamics. Frames: $\{n\}$ is NED and $\{b\}$ is body.

```latex
\boldsymbol{\eta} = [x, y, z, \phi, \theta, \psi]^\top \in \mathbb{R}^3 \times \mathbb{S}^3,
\qquad
\boldsymbol{\nu} = [u, v, w, p, q, r]^\top \in \mathbb{R}^6
```

```latex
\dot{\boldsymbol{\eta}} = \mathbf{J}_\Theta(\boldsymbol{\eta})\,\boldsymbol{\nu}
```

```latex
\mathbf{M}\dot{\boldsymbol{\nu}}
 + \mathbf{C}(\boldsymbol{\nu})\boldsymbol{\nu}
 + \mathbf{D}(\boldsymbol{\nu})\boldsymbol{\nu}
 + \mathbf{g}(\boldsymbol{\eta})
 = \boldsymbol{\tau},
\qquad
\mathbf{M} = \mathbf{M}_{RB} + \mathbf{M}_A,\;
\mathbf{C} = \mathbf{C}_{RB} + \mathbf{C}_A,\;
\mathbf{D}(\boldsymbol{\nu}) = \mathbf{D}_L + \mathbf{D}_{NL}(\boldsymbol{\nu})
```

```latex
\mathbf{M}_A = -\operatorname{diag}(X_{\dot u}, Y_{\dot v}, Z_{\dot w}, K_{\dot p}, M_{\dot q}, N_{\dot r}),
\qquad
\mathbf{D}_L = -\operatorname{diag}(X_u, Y_v, Z_w, K_p, M_q, N_r),

\mathbf{D}_{NL}(\boldsymbol{\nu}) = -\operatorname{diag}(X_{u|u|}|u|,\ Y_{v|v|}|v|,\ Z_{w|w|}|w|,\
  K_{p|p|}|p|,\ M_{q|q|}|q|,\ N_{r|r|}|r|),

W = mg,\qquad B = \rho g \nabla
```

**Sign convention (important).** In Fossen, the hydrodynamic derivatives $X_{\dot u}, X_u, X_{u|u|}, \dots$ are **negative**. The simulator config usually stores **positive magnitudes**. In tables, list the magnitudes and say so in the caption (see §5). In equations, keep Fossen's minus signs as written above. Never mix the two conventions.

**Restoring geometry.** Write CG and CB as $\mathbf{r}^b_{bg}=[x_g,y_g,z_g]^\top$ and $\mathbf{r}^b_{bb}=[x_b,y_b,z_b]^\top$, and the vertical metacentric distance as $\overline{BG}_z = z_g - z_b$. Because $z$ points down, a CB *above* the CG gives $\overline{BG}_z > 0$.

---

## 3. Symbol registry

Fossen symbols are **reserved**. ML symbols have been renamed so they don't collide with them. Don't reuse a symbol for another meaning.

### 3.1 Reserved by dynamics (Fossen / SNAME)

| Symbol | Meaning | Space |
|---|---|---|
| $x,y,z$ | NED position | $\mathbb{R}$ |
| $\phi,\theta,\psi$ | roll, pitch, yaw | $\mathbb{S}$ |
| $\boldsymbol{\Theta}$ | Euler angles $[\phi,\theta,\psi]^\top$ | $\mathbb{S}^3$ |
| $u,v,w,p,q,r$ | body linear / angular velocities | $\mathbb{R}$ |
| $\boldsymbol{\eta},\ \boldsymbol{\nu}$ | pose, velocity | $\mathbb{R}^3\times\mathbb{S}^3,\ \mathbb{R}^6$ |
| $\boldsymbol{\tau}$ | generalized forces | $\mathbb{R}^6$ |
| $\mathbf{M},\mathbf{C}(\boldsymbol{\nu}),\mathbf{D}(\boldsymbol{\nu}),\mathbf{g}(\boldsymbol{\eta})$ | inertia, Coriolis, damping, restoring | $\mathbb{R}^{6\times6}$, $\mathbb{R}^6$ |
| $\mathbf{J}_\Theta(\boldsymbol{\eta})$ | kinematic transform | $\mathbb{R}^{6\times6}$ |
| $m,\ I_x,I_y,I_z$ | mass, moments of inertia | $\mathbb{R}_{>0}$ |
| $\nabla,\ \rho,\ g,\ W,\ B$ | displaced volume, water density, gravity, weight, buoyancy | $\mathbb{R}_{>0}$ |
| $X_\cdot, Y_\cdot, Z_\cdot, K_\cdot, M_\cdot, N_\cdot$ | SNAME hydrodynamic derivatives | $\mathbb{R}$ |
| $\overline{BG}_z$ | $z_g - z_b$ | $\mathbb{R}$ |
| $\mathbf{u}$ | thruster control inputs | $\mathbb{R}^{n_u}$ |
| $\mathbf{T}$ | thrust configuration matrix, $\boldsymbol{\tau}=\mathbf{T}\mathbf{K}\mathbf{u}$ | $\mathbb{R}^{6\times n_u}$ |

### 3.2 Problem, sensors, learning

| Symbol | Meaning | Space | Renamed from (reason) |
|---|---|---|---|
| $\mathbf{s}_t$ | agent internal state | $\mathbb{R}^{n_s}$ | — |
| $\mathcal{S},\ \mathcal{A}$ | state / action space (Background, MDP) | — | — |
| $\mathcal{P},\ \mathcal{P}_0$ | transition probability / initial-state distribution | — | $\rho_0$ (water density $\rho$) |
| $R(\cdot,\cdot)$ | reward function, $\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ | — | $r$ (yaw rate) |
| $\mathbf{a}_t$ | action | $\mathbb{R}^{n_a}$ | — |
| $\mathbf{p}^n_{\mathrm{goal}}$ | goal position | $\mathbb{R}^3$ | $g$ (collides with $\mathbf{g}(\boldsymbol\eta)$ and gravity $g$) |
| $\mathcal{B}=\{\mathcal{B}_k\}_{k=1}^{N_{\mathcal{B}}}$ | obstacle set | $\mathcal{B}_k\subset\mathbb{R}^3$ | $K$ (collides with roll moment $K$) |
| $D_t$ | depth image, range per pixel divided by the \SI{10}{\metre} max range | $[0,1]^{H\times W}$, $H{\times}W = 72{\times}128$ | — (kept to match DeFM) |
| $r_{\mathrm{robot}}$ | half-edge of the robot cube in the collision image | \SI{0.3}{\metre} | — |
| $R^{\mathrm{img}}_t$ | raw range image | $\mathbb{R}_{\ge0}^{72\times128}$ | — |
| $\tilde{\mathbf{r}}_t$, $\mathbf{r}_t$ | noisy / clean sonar ranges | $[0,r_{\max}]^{N_r}$, $N_r=64$ | $\mathbf{s}_t$ (collides with state) |
| $\alpha_i$ | bearing of sonar ray $i$ | $[0,2\pi)$ | $\beta_i$ / $\psi_i$ (KL weight / yaw) |
| $\sigma_r,\ \zeta_{t,i}$ | sonar noise std. and noise sample | $\mathbb{R}_{>0}$, $\mathbb{R}$ | $\epsilon$, $\nu$ (PPO clip / velocity) |
| $p_{\mathrm{drop}},\ p_{\mathrm{spur}}$ | sonar drop / spurious probabilities | $[0,1]$ | — |
| $o_t=[o^{\mathrm{sonar}}_t,o^{\mathrm{depth}}_t,o^{\mathrm{imu}}_t,o^{\mathrm{vel}}_t,o^{\mathrm{goal}}_t]$ | raw observation, grouped as in `deowan2026unified` | $\mathcal{O}$ | — |
| $\xi^{\mathrm{sonar}}_t$ | sonar features: 16 sector minima, front minimum, front TTC risk | $\mathbb{R}^{18}$ | — (Deowan's $\xi^{\mathrm{sonar}}_t$) |
| $d_{s,t},\ d_{f,t}$ | sonar sector minimum, front minimum ($\pm30^\circ$) | $[0,r_{\max}]$ | — |
| $c^{\mathrm{TTC}}_t$ | front TTC risk, $\operatorname{clip}(1-d_{f,t}/(t_{\mathrm{ttc}}\max(u_t,u_{\min})),0,1)$ | $[0,1]$ | — (Deowan uses TTC features without a formula) |
| $\xi^{\mathrm{depth}}_t=\boldsymbol{\mu}^{\mathrm{VAE}}_t$ | depth features (VAE latent mean); takes the place of Deowan's $\xi^{\mathrm{wm}}_t$ | $\mathbb{R}^{64}$ | — |
| $\Delta\mathbf{p}^b_{\mathrm{goal},t}$ | goal relative to the vehicle, body frame | $\mathbb{R}^3$ | Deowan's lookahead $\Delta\mathbf{p}^{\mathrm{lh}}_t$ (no planner here) |
| $e^\psi_t$ | yaw error towards the goal | $\mathbb{S}$ | — (Deowan) |
| $\boldsymbol{\nu}^{*}_t,\ \mathbf{S}$ | velocity reference $[u^*_t,v^*_t,w^*_t,r^*_t]^\top=\mathbf{S}\mathbf{a}_t$, action scaling | $\mathbb{R}^4$, $\mathbb{R}^{4\times4}$ | — |
| $e\in\{\mathrm{DeFM},\mathrm{VAE}\}$ | encoder/method index | — | $m$ (mass) |
| $E^{e}$ | depth encoder of method $e$ | $\mathbb{R}^{H\times W}\!\to\mathbb{R}^{d_e}$ | — |
| $\mathbf{z}^{e}_t$ | depth embedding / latent | $\mathbb{R}^{d_e}$ | bold to separate from heave position $z$ |
| $\Gamma(\cdot)$ | DeFM log-scaled 3-channel preprocessing | $\mathbb{R}^{3\times H\times W}$ | — |
| $\boldsymbol{\mu}^{\mathrm{VAE}}_t,\boldsymbol{\sigma}^{\mathrm{VAE}}_t$ | VAE latent mean / std. | $\mathbb{R}^{d_{\mathrm{VAE}}}$ | — |
| $q(\mathbf{z}\mid D_t;\mathbf{w}_{\mathrm{enc}})$ | VAE encoder | — | $q_\phi$ ($\phi$ = roll) |
| $\mathrm{Dec}(\mathbf{z};\mathbf{w}_{\mathrm{dec}})$ | VAE decoder | — | $p_\vartheta$ ($p$ = roll rate) |
| $D^{\mathrm{col}}_t,\ \hat D^{\mathrm{col}}_t$ | collision image / reconstruction | $\mathbb{R}^{H\times W}$ | $C_t$ (Coriolis $\mathbf{C}$) |
| $\Omega_t$ | valid-pixel mask | $\{0,1\}^{H\times W}$ | $M_t$ (mass matrix $\mathbf{M}$) |
| $\beta$ | KL weight | $\mathbb{R}_{>0}$ | — |
| $\mathbf{x}_t$ | policy input $[\Delta\mathbf{p}^b_{\mathrm{goal},t},\sin e^\psi_t,\cos e^\psi_t,\mathbf{v}^b_t,r_t,z_t,\phi_t,\theta_t,\xi^{\mathrm{sonar}}_t,\xi^{\mathrm{depth}}_t]$ | $\mathbb{R}^{94}$ | Deowan's $\varphi_t$ ($\phi$ = roll); bold to separate from north position $x$ |
| $\pi_\theta(\mathbf{a}_t\mid o_{\le t})$, $\pi_\theta(\mathbf{a}_t\mid\mathbf{h}_t)$ | policy (history form / recurrent form) | — | $\pi_\theta$ ($\theta$ = pitch) |
| $V(\mathbf{x}_t;\mathbf{w}_V)$ | critic (MLP, no memory) | $\mathbb{R}$ | — |
| $\theta$ (as subscript of $\pi_\theta$), $\theta_{\mathrm{old}}$ | policy weights, following `deowan2026unified` (user decision 2026-11-25) | — | EXCEPTION to the Fossen rule: pitch is $\theta$ too. Pitch always carries its time index ($\theta_t$) or sits in $\boldsymbol\eta$; the policy weights only appear as the subscript of $\pi$, in $\max_\theta$ and in $\mathrm{GRU}(\cdot;\theta)$. |
| $\mathbf{w}_V,\mathbf{w}_{\mathrm{enc}},\mathbf{w}_{\mathrm{dec}},\mathbf{w}_{\mathrm{DeFM}}$ | critic / encoder / decoder / DeFM weights | — | $\phi,\vartheta,\omega$ (clash with Fossen) |
| $\mathbf{h}_t$ | GRU hidden state | $\mathbb{R}^{n_h}$ | — |
| $\mathbf{c}_t,\ \tilde{\mathbf{c}}_t$ | LSTM cell state / candidate (Background only) | $\mathbb{R}^{n_h}$ | $C_t$ (Coriolis $\mathbf{C}$) |
| $\mathbf{f}_t,\ \mathbf{i}_t,\ \mathbf{o}_t$ | LSTM forget / input / output gates (Background only) | $(0,1)^{n_h}$ | — |
| $\boldsymbol{\upsilon}_t,\ \boldsymbol{\chi}_t,\ \tilde{\mathbf{h}}_t$ | GRU update / reset gates, candidate state | $(0,1)^{n_h}$, $\mathbb{R}^{n_h}$ | $\mathbf{z}_t$, $\mathbf{r}_t$ (embedding $\mathbf{z}^e_t$, sonar $\mathbf{r}_t$) |
| $R_t$ | reward, $R_t=-w_{\mathrm{alive}}+w_p\Delta d_t-w_zZ_t-w_cC_t-w_{\mathrm{ttc}}T_t-w_vV_t-w_{\mathrm{tilt}}\Theta_t-w_{\mathrm{eff}}E_t+w_s\iota^{\mathrm{succ}}_t-w_f\iota^{\mathrm{fail}}_t$ (form of `deowan2026unified`, Eq. 14) | $\mathbb{R}$ | $r_t$ (yaw rate $r$, sonar $\mathbf{r}$) |
| $w_p,w_z,w_c,w_{\mathrm{ttc}},w_v,w_{\mathrm{tilt}},w_{\mathrm{eff}},w_s,w_f,w_{\mathrm{alive}}$ | reward weights (always with a word or letter subscript) | $\mathbb{R}_{>0}$ | Deowan's $w_\cdot$; replaces $\lambda_{\mathrm{prog}},\dots$ (heave $w_t$ carries a time index) |
| $\Delta d_t, Z_t, C_t, T_t, V_t, \Theta_t, E_t$ | reward terms: progress, depth error, clearance, TTC, overspeed, tilt, effort (scalars, non-bold) | $\mathbb{R}$ | Deowan's symbols; scalar and time-indexed, unlike bold $\mathbf{C}$, $\mathbf{T}$, $\boldsymbol\Theta$ (Fossen) |
| $\gamma,\ \lambda_{\mathrm{GAE}}$ | discount, GAE parameter | $[0,1)$, $[0,1]$ | — |
| $\hat A_t,\ \hat G_t$ | advantage, return target | $\mathbb{R}$ | — |
| $\kappa_t$ | PPO probability ratio | $\mathbb{R}_{>0}$ | $\rho_t$ (water density $\rho$) |
| $\epsilon$ | PPO clip range | $\mathbb{R}_{>0}$ | — |
| $\lambda_V,\lambda_H$ | PPO loss weights (always subscripted) | $\mathbb{R}_{>0}$ | — |
| $\mathbf{p}^n_t=[x_t,y_t,z_t]^\top$, $\mathbf{v}^b_t=[u_t,v_t,w_t]^\top$ | vehicle position in $\{n\}$, linear velocity in $\{b\}$ | $\mathbb{R}^3$ | — (Fossen; parts of $\boldsymbol\eta$, $\boldsymbol\nu$) |
| $\{w\}$, $\mathbf{p}^w_t$, $\mathbf{p}^w_{\mathrm{goal}}$ | simulator world frame ($z$ up) and positions in it; Method only. Body frame $\{b\}$ in the Method is forward-left-up (Isaac Sim, as in `deowan2026unified`); Background's $\{b\}$ is Fossen's forward-starboard-down. They differ by \SI{180}{\degree} about $x$. | $\mathbb{R}^3$ | — |
| $U_t,\ U_{\max}$ | speed $\lVert\mathbf{v}^b_t\rVert_2$ and its soft limit | $\mathbb{R}_{\ge0}$ | — (Fossen's speed $U$) |
| $d^{\mathrm{goal}}_t,\ d^{\mathrm{obs}}_t$ | distance to the goal / to the nearest obstacle (word superscript; subscripted $d_e$ stays a dimension) | $\mathbb{R}_{\ge0}$ | — (code: `dist`, `min_clear`) |
| $d_{\mathrm{safe}},\ d_{\mathrm{succ}}$ | soft-clearance and success distances | $\mathbb{R}_{>0}$ | — |
| $t_{\mathrm{ttc}},\ u_{\min}$ | time-to-collision horizon, minimum surge speed in the TTC term | $\mathbb{R}_{>0}$ | — |
| $\iota^{\mathrm{succ}}_t,\ \iota^{\mathrm{fail}}_t$ | success / failure indicators | $\{0,1\}$ | Deowan's $\iota$; was $\mathbb{I}$ |
| $\delta_t$ | TD residual in GAE | $\mathbb{R}$ | — (Schulman et al., GAE) |
| $V_t,\ V^{\mathrm{clip}}_t$ | critic value $V(\mathbf{x}_t;\mathbf{w}_V)$ and its clipped version | $\mathbb{R}$ | — (PPO value clipping) |
| $\ell_{\mathrm{H}},\ \mathcal{H}$ | Huber loss, entropy | — | — |

---

## 4. Equations

* Display equations use the tinted boxes: `beqn` for one equation, `balign` for aligned groups. Don't use plain `equation` or `align`.
* Every displayed equation gets a `\label{eq:<group>-<name>}`. Groups: `dyn`, `kin`, `obs`, `sonar`, `vae`, `defm`, `ppo`, `reward`. Examples: `eq:dyn-6dof`, `eq:vae-kl`, `eq:ppo-clip`.
* Refer to equations with `\eqref{}` only.
* Use `\Lc{...}` for losses, and $D_{\mathrm{KL}}$ (upright KL) for the divergence, so it isn't read as the depth image $D_t$.
* Always write a matrix with its argument when it has one: $\mathbf{D}(\boldsymbol{\nu})$, never bare $\mathbf{D}$. This keeps it apart from the depth image $D_t$.

Reference losses (keep these forms):

```latex
\begin{beqn}
  \mathcal{L}^{\mathrm{VAE}}(\mathbf{w}_{\mathrm{enc}},\mathbf{w}_{\mathrm{dec}})
  = \underbrace{\bigl\lVert \Omega_t \odot (D^{\mathrm{col}}_t - \hat D^{\mathrm{col}}_t)\bigr\rVert_2^2}_{\Lc{rec}}
  + \beta\,\underbrace{D_{\mathrm{KL}}\!\bigl(q(\mathbf{z}\mid D_t)\,\Vert\,\mathcal{N}(\mathbf{0},\mathbf{I})\bigr)}_{\Lc{KL}}
  \label{eq:vae-total}
\end{beqn}

\begin{balign}
  \mathcal{L}^{\mathrm{PPO}}(\theta,\mathbf{w}_V)
    &= \Lc{clip} + \lambda_V\Lc{V} - \lambda_H\Lc{H}, \label{eq:ppo-total}\\
  \Lc{clip} &= -\mathbb{E}_t\bigl[\min\bigl(\kappa_t\hat A_t,\
     \operatorname{clip}(\kappa_t,1-\epsilon,1+\epsilon)\hat A_t\bigr)\bigr], \label{eq:ppo-clip}\\
  \kappa_t &= \frac{\pi_\theta(\mathbf{a}_t\mid\mathbf{x}_t)}
                   {\pi_{\theta_{\mathrm{old}}}(\mathbf{a}_t\mid\mathbf{x}_t)} \label{eq:ppo-ratio}
\end{balign}
```

---

## 5. Tables

Template (adapt the rows; keep the structure):

```latex
\begin{table}[H]
  \centering
  \caption{BlueROV2 Heavy model parameters used in MarineGym. Hydrodynamic
    coefficients are listed as magnitudes; in Fossen's sign convention they enter
    with a negative sign (e.g.\ $X_{\dot u} = -\SI{5.5}{\kilogram}$).}
  \label{tab:bluerov-model}
  \begin{tabular}{@{}llll@{}}
    \toprule
    Parameter & Symbol & Value & Unit \\
    \midrule
    ...
    \bottomrule
  \end{tabular}
\end{table}
```

Rules:
* The column spec must match the number of columns used. `{llll}` for four columns, not `{lllll}`.
* Use `booktabs` only. No vertical rules and no `\hline`.
* Group related rows with `\midrule`: rigid body, then added mass, then linear damping, then quadratic damping, then thrusters.
* Lists of axis-wise values: `5.5, 12.7, 14.57` in the order $(x,y,z)$ or $(\phi,\theta,\psi)$ / $(u,v,w)$ / $(p,q,r)$.
* Units: `\si{}` in the Unit column, and the same unit for every value in that row.
* Caption above the table. Label as `tab:<kebab-name>`.
* Copy values verbatim from the project config. Don't round unless the config does. Put the source file path in a LaTeX comment above the table: `% source: marinegym/cfg/bluerov2_heavy.yaml`.
* For a notation table, use the columns `Symbol & Space & Description`, filled from §3.

---

## 6. Required preamble

Make sure these lines exist in `main.tex` (or in `setup.sty`) **before** `\begin{document}`. Don't load a package twice with different options.

```latex
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{float}          % [H]
\usepackage{siunitx}
\usepackage{tcolorbox}
\tcbuselibrary{skins,theorems}
\tcbset{
  eqbox/.style={
    enhanced, colback=blue!4, colframe=blue!4, boxrule=0pt, arc=2pt,
    left=4pt, right=4pt, top=-2pt, bottom=2pt,
    before skip=6pt, after skip=6pt
  }
}
\newtcolorbox{beqn}{eqbox, ams equation}
\newtcolorbox{balign}{eqbox, ams align}
\providecommand{\Lc}{}
\renewcommand{\Lc}[1]{\mathcal{L}_{\mathrm{#1}}}
```

---

## 7. Preview before committing (mandatory)

1. Write each new or changed equation and table into a standalone file `preview/preview.tex`. Use `\documentclass{article}` and the §6 preamble, and copy the snippets verbatim.
2. Compile it: `latexmk -pdf -interaction=nonstopmode -outdir=preview preview/preview.tex`, or run `pdflatex` twice. The build must have **zero errors** and no `Overfull \hbox` warnings caused by the new content.
3. Render it: `pdftoppm -png -r 130 preview/preview.pdf preview/page`. Look at the images yourself to check spacing, overflow, and that the tinted boxes appear.
4. Show the user:
   * the rendered PNG page(s),
   * a diff of the `.tex` files you changed,
   * a report listing: values taken from the code (with file paths), any `\todo{?}` left, new symbols added to the registry, and any conflicts with §3 found in the existing text.
5. **Stop and wait for approval.** Commit only after an explicit "ok"/"commit". Use a message such as `docs(notation): add BlueROV2 parameter table`. Don't commit the `preview/` folder; add it to `.gitignore`.
