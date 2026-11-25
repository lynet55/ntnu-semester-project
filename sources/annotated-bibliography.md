# Annotated bibliography: memory in learned navigation

Detailed notes per paper. Ratings (can I cite it?) live in `sources/ratings.md`; this file says what each paper does. Layout and editing rules: `sources/README.md`.

Thesis context: proposed underwater stack = raw sonar + depth image → frozen encoder latent → GRU + MLP policy, trained by RL (underwater analogue of Kulkarni & Alexis DCE).

Each entry: domain · perception → memory · what to cite it for · caveat. "Check" = not verified from the full text.

1. **Ha & Schmidhuber 2018** (`ha2018recurrent`) — simulated Gym/game environments · VAE compresses frames, a recurrent model (MDN-RNN) predicts in latent space, a small controller acts · Origin of the "compressed latent + recurrent memory" pattern. · Not robotics; the controller is trained by evolution, not gradient RL.

2. **Hoeller et al. 2021** (`hoeller2021learning`, RA-L) — ANYmal quadruped, depth camera · VAE → 32-d latent per frame; LSTM fuses latent sequence + camera trajectory into a belief state; MLP policy gets belief + goal; no explicit map; sim-to-real · Quadruped example; contrast to DCE. · Recurrence sits in the representation module, not the policy.

3. **Miki et al. 2022** (`miki2022learning`, Sci. Robot.) — ANYmal, elevation map + proprioception · "Attention-based recurrent encoder that integrates proprioceptive and exteroceptive input", trained end-to-end (check: teacher-student distillation, gate details) · Recurrent belief is standard in perceptive locomotion; example of explicit map + implicit memory hybrid; notes 2.5D maps are unreliable. · Locomotion, not navigation.

4. **Agarwal et al. 2023** (`agarwal2023legged`, CoRL) — legged robot, egocentric depth camera · Two variants: monolithic GRU policy on ConvNet depth latent + proprioception (trained by DAgger from a scandots teacher); RMA variant where a vision GRU estimates a terrain latent for a frozen MLP base policy · Why memory is needed with a narrow field of view; shows both "memory in policy" and "memory in estimator" in one paper. · Memory trained by distillation, not RL.

5. **Lee et al. 2024** (`lee2024learning`, Sci. Robot.) — wheeled-legged robot, urban navigation · Hierarchical RL. Navigation policy is FEEDFORWARD: memory via explicit position history (20 visited positions, 1D CNN), three stacked height scans, previous outputs, and the locomotion controller's hidden state. Locomotion student uses Miki's GRU architecture. · Example of explicit history instead of recurrence at the navigation level. · Do NOT cite as a recurrent navigation policy.

6. **Kulkarni & Alexis 2024** (`kulkarni2024dce`, ICRA) — aerial robot, depth camera · DCE latent + robot state → MLP + 64-unit GRU → velocity/yaw-rate commands; POMDP formulation; "local navigation … without access to a global map" · Anchor architecture. · Bounded rooms, real goals up to ~20 m; no GRU ablation; GRU not justified by citation.

7. **Yang et al. 2026** (`yang2026spatially`, IJRR; arXiv 2506.05997) — mapless long-range navigation, legged/ground · VAE-pretrained depth encoder → self/cross-attention → recurrent unit (SRU/LSTM/GRU) → MLP velocity head; end-to-end PPO, no distillation or auxiliary losses; SRU +23.5% over existing RNNs, also beats explicit-mapping baseline · (a) limitation: plain RNNs fail at spatial memorization; (b) related work states recurrent architectures are used in locomotion and navigation; (c) closest structural relative of DCE outside NTNU. · Quotes checked against arXiv v2; confirm in journal version.

### Underwater / marine (checked against full text, 2026-11-23)

8. **Havenstrøm et al. 2021** (`havenstrom2021deep`, Front. Robot. AI 7:566037) — AUV, 3D path following + collision avoidance · Simulated FLS 15×15 range grid → max-pooled 8×8 → flattened 64 values + 14 state values → Stable Baselines MLP (2×64, tanh), PPO; outputs rudder/elevator fins, PI thrust · Learned underwater policy on sonar/range input; precedent for feeding (pooled) sonar directly; baseline "no memory" case. · VERIFIED: feedforward, no RNN, no frame stacking; simulation only; assumes full state incl. ocean current.

9. **Yuan et al. 2023** (`yuan2023auv`, JMSE 11(12):2258) — AUV, active sonar (120 beams → 25 groups, min distance) + relative target + previous heading rate · DDPG; actor and critic "implemented" with GRU, chosen over LSTM for fewer parameters · Underwater GRU-in-policy precedent; GRU-vs-LSTM argument. · UNRESOLVED: text never says whether the GRU runs over time or over the 25 sonar groups; replay buffer stores single-step (s,a,r,s') tuples with batch 64 and no sequence sampling → likely NOT cross-step memory. Cite as "a GRU-based actor-critic" unless Fig. 7 shows temporal recurrence. Simulation only.

9b. **Yuan et al. 2021** (`yuan2021auv`, JMSE 9(11):1166; Yuan, Wang, Zhang, Lin, Yu, Li) — AUV, active sonar 256 beams → 13 groups, sonar is the only input · D-DQN-LSTM: LSTM replaces the MLP as part of the Q-network; motivated by decisions "based on the state of sequential observations"; TD loss · Earliest underwater LSTM-in-policy precedent found; same group as 9. · Check: sequence length / whether the LSTM consumes a history of steps not stated in the readable text; simulation only.

10. **Yang et al. 2022** (`yang2022monocular`, ISRR; arXiv 2212.04373) — low-cost AUV, monocular camera (DPT-estimated depth) + single-beam echo sounder · PPO; observations stacked over k = 5 steps (0.5 s each): 3-conv CNN on stacked depth → 512-d, concatenated with stacked goal (96), sonar (32), past actions (64) → FC → policy + value; mapless; domain randomization; sim + real · Closest prior work on depth + sonar → learned underwater navigation. · VERIFIED: frame stacking, no recurrence; CNN trained end-to-end (no frozen latent).

11. **Larsen et al. 2024** (`larsen2024variational`, arXiv 2404.00623) — SURFACE vessel (simulated Cybership II), 180-beam planar range sensor · 1D circular-conv VAE → 12-d latent; encoder frozen ("locked") in best config; PPO (SB3); policy architecture not described; no RNN, no frame stacking · Marine precedent for frozen pretrained exteroceptive encoder + RL = DCE pipeline minus the GRU; also the counter-choice to feeding range data raw. · Not underwater; simulation only; preprint.

12. **Singh et al. 2025** (`singh2025ariel`, arXiv 2507.10003) — underwater robot Ariel, 5 cameras + IMU, field-tested in Trondheim dry dock · NO learned navigation policy: classical graph-based planner (GBPlanner; VE + GVI modes). GRUs only in DeepVL, a supervised GRU-ensemble velocity predictor (motor commands, IMU, battery voltage → body velocity + covariance) fused into ReAqROVIO · Recurrence used underwater for state estimation, not control. · Do NOT cite as a recurrent policy.

13. **Shafer & Gal 2026** (`shafer2026endtoend`, arXiv 2606.08513, Univ. Haifa) — AUV in HoloOcean, single 84×84 RGB frame + 3 stacked 100×100 FLS images + proprioception · Hierarchical: HL policy (2 Hz, RLPD/SERL, pretrained ResNet-10 image encoder, DrQ augmentation) outputs subgoals; LL policy (10 Hz, SAC+HER) on proprioception → thrusters · Most recent camera + imaging-sonar learned underwater navigation; pretrained image encoder. · VERIFIED: no recurrence, temporal context only from sonar frame stacking; simulation only; trained with prior demonstrations.

14. **Deowan & Kelasidi 2026** (`deowan2026unified`, arXiv 2608.05365) — UUV in Isaac Sim, 256-ray 360° sonar (12 m) + 128×72 depth image · Voronoi/RRT global planner + PPO local policy (MLP 256-256, ELU) on path-relative features; learned latent world model (encoder f_ω, one-step latent dynamics g_ψ, decoder; latent 64) trained by reconstruction + latent-dynamics loss; BT distillation variants · MOST LIKELY REVIEWER COMPARISON: same sensors (sonar + depth), same sim ecosystem. · Differences to our stack: memory/latent trained by predictive objective (group B), one-step predictor with no stated recurrence, feedforward MLP policy, planner-guided rather than mapless; encoder architecture unspecified.

15. **Wang, Xiang, Duan & Yang 2024** (`wang2024adversarial`, Eng. Appl. of AI 130:107728) — underactuated AUV, depth tracking control · Adversarial DRL; "the LSTM and MLP combined control agent" (highlights) · Underwater recurrent DRL agent in a strong venue. · Depth control, not navigation; abstract/highlights only.

Optional (same-group follow-ups / extra domain): Malczyk et al. 2026 (arXiv 2602.01266, GRU 64, active perception, ablates occupancy grid not GRU); Sawant et al. 2026 (arXiv 2603.22182, cross-modal depth+grayscale latent + GRU, degraded depth); REASAN 2025 (arXiv 2512.09537, legged, supervised transformer estimator on raw lidar + LSTM in each policy). Deowan, Yousha, Hossain, Hassan & Marxer, "Optimizing underwater robot navigation: a study of DRL algorithms and multi-modal sensor fusion" (ICRA 2025, `deowan2025optimizing`; real-time sensor data + historical actions; check architecture).

### Image encoders (VAE / CNN / pretrained) in the underwater domain (added 2026-11-24, for sec:rw-representation)

16. **Manderson et al. 2018** (`manderson2018vision`, IROS 2018) — Aqua hexapod AUV, monocular camera, coral reef · CNN trained by supervised learning on human-labelled images → collision avoidance + reef target selection; ~1000 m collision-free in open ocean · Earliest real-world learned underwater visual navigation (CNN, imitation). · Not RL, no latent pretraining.

17. **Manderson et al. 2020** (`manderson2020vision`, RSS 2020; arXiv 2006.16235) — same robot · ResNet-18 + concrete dropout, single image + 2D goal → discrete yaw/pitch (7 classes each); behaviour cloning + hindsight relabelling from VO; ~1 km, ~40 waypoints in Caribbean · Goal-conditioned CNN policy underwater. · Imitation, not RL; pretrained backbone not stated.

18. **Gao et al. 2024** (`gao2024robotic`, Biomimetics 9(4); Wang, Wang, Zhao, Gong, Zuo, Zheng) — robotic fish, FPV RGB in Unreal/AirSim · Cross-modal VAE (image encoder → latent, decoders for image + next-gate pose) → behaviour-cloned velocity policy · Only found underwater(-ish) VAE latent used for a navigation policy. · Simulation only; imitation not RL; latent dim not given; gate-passing, not general collision avoidance.

19. **Yang, Le, Gong & Pu 2025** (`yang2025duvin`, DUViN, arXiv 2509.02983, Univ. Melbourne) — BlueROV2 Heavy, monocular RGB (3 frames) · Frozen pretrained Depth Anything V2 ViT-S encoder → 256-d feature → transformer → diffusion policy (4-DoF velocities); phase 2 adapts encoder by self-supervised underwater depth estimation; trained in-air sim, tested underwater sim + tank · Frozen pretrained depth-feature encoder underwater; argues optical depth sensors are ineffective underwater. · Imitation (MPC expert), not RL.

20. **Panisilvam et al. 2022** (`panisilvam2022conditioned`, OCEANS 2022 Hampton Roads) — multibeam imaging sonar · Conditioned VAE translates simulated sonar images to real-sonar texture for dataset generation · VAE underwater used for sim-to-real data generation, not control; notes imaging sonar "has only seen limited use in autonomous navigation".

21. **SCOPE 2025** (arXiv 2511.13922) — imaging-sonar video · VAE-based compression + artifact correction for low-bandwidth streaming · VAE underwater for compression/transmission. · Authors/bib key to fill in.

22. **Pucci & Martinel 2024** (`pucci2024cevae`, CE-VAE, arXiv 2406.01294) — underwater RGB · Capsule-enhanced VAE for joint image enhancement + compression · VAE underwater for image enhancement.

Also relevant: Larsen et al. IFAC (risk-based convolutional perception for ASV collision avoidance, cited as [3] in entry 11) — CNN encoder on range data, surface vessel. Sonar-based deep learning survey arXiv 2412.11840 mentions VAEs/GANs used for OOD detection on sonar.

Finding: underwater VAEs are used for perception (sim-to-real sonar synthesis, compression, enhancement, OOD), not as frozen latents for RL policies. Learned underwater navigation uses end-to-end CNNs (Manderson 2018/2020 imitation; Yang 2022 RL) or large pretrained encoders (Shafer 2026 ResNet-10; DUViN Depth Anything). Only VAE-latent policy found: robotic-fish CM-VAE with behaviour cloning in sim. No found work: frozen task-driven depth VAE + RL underwater.

## Classification: where memory lives and what trains it

| Where memory lives | Trained by RL reward | Trained by distillation | Trained by own predictive/reconstruction/supervised objective |
|---|---|---|---|
| Inside the policy (encoder → RNN → action head) | Kulkarni 2024, Malczyk 2026, Sawant 2026, Yang 2026, REASAN, DD-PPO/Mirowski (end-to-end CNN); underwater: Yuan 2021 (LSTM-Q), Yuan 2023 (GRU-DDPG) — temporal memory unverified in both; Wang 2024 (LSTM+MLP, depth control) | Agarwal 2023 (monolithic) | — |
| Recurrent belief encoder inside policy, fusing sensors | — | Miki 2022, Lee 2024 locomotion student | — |
| Separate estimator / latent model, feedforward policy or planner | — | Agarwal 2023 (RMA variant) | Hoeller 2021, Ha & Schmidhuber 2018; underwater: Deowan 2026 (one-step latent world model, no stated recurrence), Singh 2025 (velocity estimator → classical planner) |
| No recurrence (explicit history / frame stacking) | Lee 2024 navigation policy; Mnih 2015; underwater: Yang 2022 (k=5), Shafer 2026 (3 sonar frames) | — | — |
| No memory at all (single observation) | Havenstrøm 2021 (underwater); Larsen 2024 (surface, frozen VAE) | — | — |

Zhang 2025 (NMI): CNN + GRU in policy, trained by differentiable physics (not RL).
DCE's distinctive combination: frozen, task-driven encoder + memory learned purely from RL reward, no distillation or auxiliary loss. Shared with Sawant/Malczyk; closest outside NTNU: Yang 2026 and REASAN.

Underwater finding (as of 2026-11-23, not exhaustive): recurrence inside an RL navigation policy appears only in sonar-only DQN/DDPG work by Yuan et al. (2021, 2023), where it's unclear whether memory spans time steps. Depth/camera + sonar works (Yang 2022, Shafer 2026, Deowan 2026) use frame stacking, a feedforward policy, or a non-recurrent latent model. Gap: no found work combines a frozen pretrained depth latent + sonar in a recurrent policy trained purely by RL, i.e. the aerial/legged "group A" design has not been carried over underwater.
