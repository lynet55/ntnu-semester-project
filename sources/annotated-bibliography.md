# Annotated bibliography: memory in learned navigation

Detailed notes per paper. Ratings (can I cite it?) live in `sources/ratings.md`; this file says what each paper does. Layout and editing rules: `sources/README.md`.

Project context: proposed underwater stack = raw sonar + depth image → frozen encoder latent → GRU + MLP policy, trained by RL (underwater analogue of Kulkarni & Alexis DCE).

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

### Underwater sensing: 3D information from sonar and cameras (verified quotes, 2026-11-25, for Introduction steps 3-4 and 6)

Quotes below were copied from the full text (PDF) and checked word for word on 2026-11-25. Section headings and pages as stated.

23. **Massot-Campos & Oliver-Codina 2015** (`massotcampos2015optical`, Sensors 15(12):31525–31557, doi:10.3390/s151229864) — survey of optical sensors and methods for underwater 3D reconstruction · Use for: infrared depth cameras fail underwater; light absorption and scattering limit optical range.
    - Sec. 4.4 *Structured Light*, p. 31536: "The use of infrared projectors, such as Kinect, has also been tested underwater [75]. The attempt confirmed that the absorption of the infrared spectrum is too strong to reach distances greater than a few centimeters."
    - Sec. 2 *The Underwater Environment*, p. 31525: "(3) Insufficient illumination and wavelength-dependent light absorption, producing dim and monotone images [15]."
    - Sec. 2, p. 31525: "(5) Significant scattering and light diffusion that limits the operational distance of the systems."
    - Caveat: the Kinect result is a single cited test of an infrared *projector* camera; do not generalise it to all depth cameras.

24. **Westman, Gkioulekas & Kaess 2020** (`westman2020volumetric`, "A Volumetric Albedo Framework for 3D Imaging Sonar Reconstruction", ICRA 2020; PDF https://www.cs.cmu.edu/~kaess/pub/Westman20icra.pdf) — 3D reconstruction from imaging sonar · Use for: FLS measures azimuth and range but not elevation; cameras have limited range in turbid water.
    - Sec. I *Introduction*, PDF p. 1: "These acoustic sensors offer long-range imaging capabilities in turbid waters that are often encountered in the field, where standard optical cameras typically have a limited sensing range."
    - Sec. I, PDF p. 1: "Imaging sonars are quite analogous to optical cameras in that they provide 2D image measurements of a 3D environment."
    - Sec. I, PDF p. 1: "In contrast, each pixel in an FLS image corresponds to discrete azimuth and range bins, with the elevation angle remaining ambiguous."
    - Caveat: the phrase "the fundamental challenge of 3D imaging sonar reconstruction" is NOT in this paper (it is from a different Westman et al. paper on Fermat paths). Proceedings page numbers not yet looked up.

25. **McConnell, Martin & Englot 2020** (`mcconnell2020fusing`, "Fusing Concurrent Orthogonal Wide-aperture Sonar Images for Dense Underwater 3D Reconstruction", IROS 2020; arXiv 2007.10407) — two orthogonal imaging sonars resolve elevation · Use for: elevation ambiguity; recovering elevation needs extra sonars, multiple viewpoints/motion, or assumptions.
    - Abstract: "We propose a novel approach to handling the ambiguity in elevation angle associated with the observations of a forward looking multi-beam imaging sonar"
    - Fig. 2 caption (Sec. I): "the elevation angle ϕ is not captured in the resulting 2D sonar image"
    - Sec. II-A *Estimating Elevation Angle*: "these methods rely on several assumptions that may often be violated"
    - Sec. II-B *Carving Out Low-Intensity Background Pixels*: "require the objects of interest to be observed from multiple vantage points"
    - Sec. II-C *Acoustic Structure From Motion*: "the perception system requiring a series of frames to recover 3D information, rather than a single timestep"
    - Caveat: checked against the arXiv HTML (v1), not the IEEE version; page numbers still to add.

### Reinforcement learning as the approach (verified quotes, 2026-11-25, for Introduction step 5)

Quotes copied word for word from the full text on 2026-11-25. arXiv versions have no page numbers; check pages in the journal versions before quoting directly in the project.

26. **Kober, Bagnell & Peters 2013** (`kober2013reinforcement`, "Reinforcement Learning in Robotics: A Survey", IJRR, doi:10.1177/0278364913495721) — classic survey of RL in robotics · Use for: RL lets you obtain behaviour that is hard to engineer by hand.
    - Abstract (authors' preprint, https://www.ri.cmu.edu/pub_files/2013/7/Kober_IJRR_2013.pdf, p. 1): "Reinforcement learning offers to robotics a framework and set of tools for the design of sophisticated and hard-to-engineer behaviors."
    - Caveat: read from the authors' preprint, not the typeset IJRR article.

27. **Tang, Abbatematteo, Hu, Chandra, Martín-Martín & Stone** (`tang2025deep`, "Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes", accepted to Annu. Rev. Control Robot. Auton. Syst.; arXiv 2408.03539 v3) — survey of real-world DRL successes · Use for: DRL has real-world successes, most mature in quadruped locomotion; caveat that most milestones are in simulation.
    - Sec. 1 *Introduction*: "recent advances have enabled DRL to succeed at some real-world robotic tasks."
    - Sec. 1 *Introduction*: "For instance, DRL has enabled champion-level drone racing"
    - Sec. 1 *Introduction* (caveat): "Yet, the most notable milestones of DRL so far have been achieved in simulation or game environments"
    - Sec. 2 *Why Another Survey on RL for Robotics?*: "DRL, which is behind the most notable real-world successes of RL in robotics"
    - Sec. 4.1.1 *Quadruped Locomotion*: "Quadruped locomotion is one of the robotic domains where DRL has provided mature real-world solutions."
    - Caveat: Sec. 4.2 (Navigation) not read word for word. Published year, volume and DOI to confirm.

28. **Ha, Lee, van de Panne, Xie, Yu & Khadiv** (`ha2025learning`, "Learning-based legged locomotion: state of the art and future perspectives", IJRR, doi:10.1177/02783649241312698; arXiv 2406.01152 v2) — survey of learning-based legged locomotion · Use for: why RL (direct sensory input, no separate state estimator), depth as policy input, history and memory for partial observability.
    - Abstract: "In recent years, however, a number of factors have dramatically accelerated progress in learning-based methods"
    - Sec. 1 *Introduction*: "including reliable and affordable hardware, significant improvements in simulation environments, and scalable learning algorithms for high-dimensional continuous control problems."
    - Sec. 1.3 *Control and Learning Algorithms*: "Currently, the two dominant approaches to controlling legged robots in multi-contact scenarios are optimal control (OC) and reinforcement learning (RL) methods."
    - Sec. 1.3: "it is simple to directly incorporate arbitrary sensory data as input to the policy, including high-dimensional modalities such as vision."
    - Sec. 1.3: "This largely eliminates the need for a separate state estimation module"
    - Sec. 3.2.1 *Proprioception*: "a short history buffer is almost always employed"
    - Sec. 3.2.2 *Exteroception*: "Instead of feeding the sensory values directly into the policy, a compressed representation of them can be learned."
    - Sec. 3.2.2: "used raw sensory readings directly as input to the policy, such as depth images or point clouds"
    - Sec. 4.4 *Privileged Learning*: "Robotic tasks in the real world are inherently partially observable"
    - Sec. 4.4: belief states are built by "stacking a sequence of previous observations" or "using models with memory such as RNN or TCN"
    - Caveat: RL is one of TWO dominant approaches; do not write that RL dominates. Checked against arXiv v2; confirm in the IJRR version.

29. **Hong, Liu, Peng & Zhang 2025** (`hong2025control`, "Control of Marine Robots in the Era of Data-Driven Intelligence", arXiv 2506.21063 v1, formatted for Annu. Rev. Control Robot. Auton. Syst.) — review of model-based vs data-driven control of marine robots · Use for: underwater control relied on models; RL learns unknown dynamics without an explicit model; limitations (sample inefficiency, simplified training environments).
    - Abstract: "The control of marine robots has long relied on model-based methods grounded in classical and modern control theory."
    - Sec. 3.2 *Model-Free Data-Driven Control Methods*: "rely on simulation or experimental data and reward functions instead of relying on explicit system models."
    - Sec. 3.3 *Integration of Model-Based and Data-Driven Control Methods*: "their effectiveness depends heavily on the accuracy of system models"
    - Sec. 3.3: "data-driven approaches, especially RL, excel at learning unknown dynamics and coping with unstructured environments"
    - Sec. 3.2.1.1 *Value-Based Methods* (limitation): "many studies rely on fixed or overly simplified training environments, limiting the real-world applicability"
    - Sec. 3.2.1.2 *Policy-Based Methods* (limitation): "standard PPO may suffer from sample inefficiency, poor performance under sparse rewards"
    - Caveat: preprint; Sec. 6 (conclusions) not read word for word.

30. **Xiao, Liu, Warnell & Stone 2022** (`xiao2022motion`, Autonomous Robots 46(5):569–597; arXiv 2011.13112 v2) — survey of ML for mobile robot navigation · Use for (Related Work 2.1): end-to-end vs subsystem learning; hybrids.
    - Sec. 3.2 *Learning Navigation Subsystems*: "Most learning approaches in this category have focused on the local planning subsystem"
    - Sec. 3.4 *Analysis*, Observation 1: "The majority of machine learning for navigation work focuses on end-to-end approaches"
    - Sec. 3.4, Observation 2: "an end-to-end learning architecture cannot completely replace a map-based path planner"
    - Sec. 3.4, Observation 4: "Approaches that seek to learn subsystems have focused predominately on local planning rather than global planning."
    - Sec. 3.4: "leads us to propose a hybrid classical and learning navigation paradigm as a very promising direction for future research"

Not yet read (do not cite until checked): Lai et al. 2026, "A Review on Path Planning for Autonomous Underwater Vehicles: From Models, Classical Methods, and Learning-Based Intelligence Perspectives", J. Field Robotics 43(3):2366–2389, doi:10.1002/rob.70143 (paywalled; candidate Strong anchor for RL in underwater navigation).

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
