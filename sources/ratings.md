# Source ratings (master list)

Canonical list of sources and whether they can be cited. Rules and column meanings: `sources/README.md`. Paper details: `sources/annotated-bibliography.md`.

Merged 2026-11-25 from the 2026-11-24 underwater sources sheet and the annotated bibliography; venues of entries already in `references.bib` checked against the bib the same day. Rows marked `venue only` have not had citations checked.

Sections: Intro = Introduction; 2.1 navigation; 2.2 representation; 2.3 memory; 2.4 underwater; 2.5 gap.

## 1. Surveys and framing (general robotics / RL)

| Key | Work | Venue | Cit. | Rating | Basis | Read | Bib | Used in | Note | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| xiao2022motion | Xiao, Liu, Warnell & Stone 2022, "Motion Planning and Control for Mobile Robot Navigation Using Machine Learning: A Survey" | Autonomous Robots | – | Strong | venue only | notes | Y | 2.1, Intro | Learned policies can't yet replace global planners, hence hybrids. | https://doi.org/10.1007/s10514-022-10039-8 |
| ha2025learning | Ha et al. 2025, "Learning-Based Legged Locomotion: State of the Art and Future Perspectives" | IJRR | – | Strong | venue only | notes | Y | 2.1, 2.3, Intro | Common policy structure; history or sequence models for partial observability. | https://doi.org/10.1177/02783649241312698 |
| almahamid2022autonomous | AlMahamid & Grolinger 2022, "Autonomous UAV Navigation using RL: A Systematic Review" | Eng. Appl. of AI | – | Strong | venue only | notes | Y | 2.1 | | https://doi.org/10.1016/j.engappai.2022.105321 |
| lesort2018state | Lesort et al. 2018, state representation learning for control | Neural Networks | – | Strong | venue only | notes | Y | 2.2 | Representation shaped by policy vs fixed beforehand. | https://doi.org/10.1016/j.neunet.2018.07.006 |
| morad2023popgym | Morad et al. 2023, POPGym | ICLR 2023 | – | Strong | venue only | notes | Y | 2.3, Intro | Benchmarks memory models in partially observable RL. | https://arxiv.org/abs/2303.01859 |
| heess2015memory | Heess, Hunt, Lillicrap & Silver 2015, "Memory-Based Control with Recurrent Neural Networks" | NIPS 2015 Deep RL Workshop | – | Borderline | venue only | notes | Y | 2.3 | Workshop paper. | https://arxiv.org/abs/1512.04455 |
| mnih2015human | Mnih et al. 2015, DQN | Nature | – | Strong | venue only | — | Y | 2.2 | | https://doi.org/10.1038/nature14236 |
| raffin2019decoupling | Raffin et al. 2019, "Decoupling Feature Extraction from Policy Learning" | ICLR 2019 Workshop (SPiRL) | – | Borderline | venue only | — | Y | 2.2 | Rated 2026-11-25 from bib venue (workshop). | — |
| patel2026defm | Patel et al. 2026, "DeFM: Learning Foundation Representations from Depth for Robotics" | preprint (`@misc`) | – | Preprint | — | — | Y | 2.2, Method | Rated 2026-11-25 from bib. Encoder used in the thesis; update if published. | — |

## 2. Representation and memory in learned navigation (aerial / legged / ground)

| Key | Work | Venue | Cit. | Rating | Basis | Read | Bib | Used in | Note | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| kulkarni2024dce | Kulkarni & Alexis 2024, Deep Collision Encoder | ICRA 2024 | – | Strong | venue only | full | Y | 2.2, 2.3, Intro | Anchor architecture: frozen DCE latent + MLP + GRU(64), RL. No GRU ablation. | https://arxiv.org/abs/2402.03947 |
| kulkarni2023task | Kulkarni & Alexis 2023, "Task-Driven Compression for Collision Encoding Based on Depth Images" | ISVC 2023 (LNCS) | – | Borderline | venue only | notes | Y | 2.2 | Rated 2026-11-25 from bib venue. | https://arxiv.org/abs/2309.05289 |
| kulkarni2023semantically | Kulkarni, Nguyen & Alexis 2023, "Semantically-Enhanced Deep Collision Prediction" | IROS 2023 | – | Strong | venue only | notes | Y | 2.2 | Rated 2026-11-25 from bib venue. | — |
| hoeller2021learning | Hoeller et al. 2021 | RA-L | – | Strong | venue only | full | Y | 2.2, 2.3, Intro | Depth VAE + LSTM belief; memory in estimator, not policy. | https://doi.org/10.1109/LRA.2021.3068639 |
| ha2018recurrent | Ha & Schmidhuber 2018, recurrent world models | NeurIPS 2018 | – | Strong | venue only | notes | Y | 2.3 | Not robotics; controller trained by evolution. | https://arxiv.org/abs/1809.01999 |
| miki2022learning | Miki et al. 2022 | Science Robotics | – | Strong | venue only | notes | Y | 2.2, 2.3 | Recurrent belief encoder; locomotion, not navigation. | https://doi.org/10.1126/scirobotics.abk2822 |
| agarwal2023legged | Agarwal et al. 2023 | CoRL 2022 | – | Strong | venue only | notes | Y | 2.2, 2.3 | GRU memory trained by distillation, not RL. | https://arxiv.org/abs/2211.07638 |
| lee2024learning | Lee et al. 2024 | Science Robotics | – | Strong | venue only | notes | Y | 2.3 | Navigation policy is feedforward with explicit history. Don't cite as recurrent. | https://arxiv.org/abs/2405.01792 |
| loquercio2018dronet | Loquercio et al. 2018, DroNet | RA-L | – | Strong | venue only | — | Y | 2.2 | | https://doi.org/10.1109/LRA.2018.2795643 |
| yang2026spatially | Yang et al. 2026, spatially enhanced recurrent memory | IJRR | – | Strong | venue only | full (arXiv v2) | Y | 2.3, Intro | Plain RNNs struggle with spatial memory; SRU beats them. Confirm quotes in journal version. | https://arxiv.org/abs/2506.05997 |
| zhang2025learning | Zhang et al. 2025 | Nature Machine Intelligence | – | Strong | venue only | notes | Y | 2.3 | CNN + GRU, trained by differentiable physics, not RL. | https://arxiv.org/abs/2407.10648 |
| wijmans2023emergence | Wijmans et al. 2023, emergence of maps in memories of blind agents | ICLR 2023 | – | Strong | venue only | notes | Y | 2.3 | | https://arxiv.org/abs/2301.13261 |
| malczyk2026active | Malczyk, Kulkarni & Alexis 2026, "RL for Active Perception in Autonomous Navigation" | preprint (`@misc`; arXiv says accepted, venue unknown) | – | Preprint | — | notes | Y | 2.3 | Add venue when found. GRU 64; ablates occupancy grid, not GRU. | https://arxiv.org/abs/2602.01266 |
| sawant2026crossmodal | Sawant et al. 2026 | arXiv | – | Preprint | — | notes | Y | 2.3 | Cross-modal latent + GRU. | https://arxiv.org/abs/2603.22182 |
| yuan2025reasan | Yuan et al. 2025, REASAN | arXiv | – | Preprint | — | notes | Y | 2.2, 2.3 | Legged; transformer estimator + LSTM. | https://arxiv.org/abs/2512.09537 |

## 3. Underwater: sensing constraints

| Key | Work | Venue | Cit. | Rating | Basis | Read | Bib | Used in | Note | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| paull2014auv | Paull et al. 2014, AUV navigation and localization review | IEEE JOE | ~1500 | Strong | S2 2026-11-24 | notes | – | Intro, Background | GPS/RF attenuation; reduced camera range, scattering, lighting. | https://doi.org/10.1109/JOE.2013.2278891 |
| massotcampos2015optical | Massot-Campos & Oliver-Codina 2015 | Sensors | ~180 | Strong | S2 2026-11-24 | notes | – | Intro, Background | RGB-D infrared range a few centimetres underwater (Sec. 4.4). | https://doi.org/10.3390/s151229864 |
| gonzalezsabbagh2023survey | González-Sabbagh & Robles-Kelly 2023, underwater computer vision survey | ACM Computing Surveys | – | Strong | venue only | abstract | – | Intro (optional) | Depth and shape recovery. | https://doi.org/10.1145/3578516 |
| koser2020challenges | Köser & Frese 2020, challenges in underwater visual navigation and SLAM | Book chapter | 41 | Borderline | S2 2026-11-24 | abstract | – | Optional | | https://doi.org/10.1007/978-3-030-30683-0_11 |

## 4. Underwater: planning pipeline and the turn to learning (surveys)

| Key | Work | Venue | Cit. | Rating | Basis | Read | Bib | Used in | Note | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| sha2021path | Sha et al. 2021, AUV path planning and obstacle avoidance review | Ocean Engineering | ~370 | Strong | S2 2026-11-24 | abstract | – | 2.4, Intro | Global vs local planning split. No quote that classical planners "dominate". | https://doi.org/10.1016/j.oceaneng.2021.109355 |
| zeng2015survey | Zeng et al. 2015, path planning for persistent autonomy | Ocean Engineering | ~250 | Strong | S2 2026-11-24 | not read | – | Optional | Paywalled. | https://doi.org/10.1016/j.oceaneng.2015.10.007 |
| an2023intelligent | An et al. 2023, intelligent path planning for underwater vehicles review | JIRS | 49 | Borderline | S2 2026-11-24 | abstract | – | 2.4, Intro | ML methods "gradually applied"; traditional methods adapt poorly to dynamic, unstructured settings. Pair with a Strong source. | https://doi.org/10.1007/s10846-022-01794-y |
| zhu2025machinery | Zhu & Cui 2025, "From Machinery to Biology: A Review on Mapless Autonomous Underwater Navigation" | JMSE | 5 | Weak | S2 2026-11-24 | notes | Y | Optional | DRL planners "emerged as a powerful alternative" (Sec. 2.3.2). No GRU/LSTM. | https://doi.org/10.3390/jmse13112202 |
| tian2025reinforcement | Tian et al. 2025, RL in UUVs review | J. Marine Sci. Appl. | 6 | Do not cite | S2 2026-11-24 | abstract | Y | — | 2026-11-25: journal mirror page mentions a retraction; not confirmed on Springer. Check the DOI before any use; currently cited in RelatedWork.tex (Sec. 2.3, 2.4). | https://doi.org/10.1007/s11804-025-00723-3 |
| panda2020comprehensive | Panda & Das 2020, AUV path planning review | Int. J. Automation and Computing | 177 | Strong (by citations) | S2 2026-11-24 | full | – | — | Dropped: no direct classical-vs-learned comparison. | https://doi.org/10.1007/s11633-019-1204-9 |
| zhu2022motion | Zhu, Yan & Yang 2022, UUV motion planning and tracking control review | Intelligence & Robotics | ~13 | Weak | S2 2026-11-24 | abstract | – | — | Dropped. | https://doi.org/10.20517/ir.2022.13 |
| hsu2020selected | Hsu et al., RL-based control for AUVs review | FITEE | – | — | — | abstract | – | — | Dropped: low-level control only, excludes path planning. | https://arxiv.org/abs/1911.11991 |
| hong2025control | Hong et al. 2025, control of marine robots in the era of data-driven intelligence | arXiv (formatted for Annu. Rev. Control Robot. Auton. Syst.) | – | Preprint | — | abstract | – | — | Dropped until publication is confirmed. | https://arxiv.org/abs/2506.21063 |
| vishwakarma2026survey | Vishwakarma et al. 2026, sensor-based planning and control for UUVs | arXiv | – | Preprint | — | notes | – | — | Dropped; doesn't claim classical dominance. | https://arxiv.org/abs/2604.05003 |

## 5. Underwater: learned navigation and control (primary work)

| Key | Work | Venue | Cit. | Rating | Basis | Read | Bib | Used in | Note | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| yang2022monocular | Yang et al. 2022, monocular depth + echo sounder PPO navigation | ISRR 2022 | – | Strong | venue only | full | Y | 2.2, 2.4, Intro | Frame stacking k=5, no recurrence; real robot. Intro says end-to-end methods "have emerged". | https://arxiv.org/abs/2212.04373 |
| havenstrom2021deep | Havenstrøm, Rasheed & San 2021 | Front. Robot. AI | 58 | Borderline | S2 2026-11-24 | full | Y | 2.4 | Pooled FLS + state → MLP, PPO; no memory; sim only. | https://doi.org/10.3389/frobt.2020.566037 |
| yuan2021auv | Yuan et al. 2021, AUV obstacle avoidance with DRL | JMSE | 57 | Borderline | S2 2026-11-24 | notes | Y | 2.4 | LSTM as part of Q-network; sonar only; temporal memory unverified. | https://doi.org/10.3390/jmse9111166 |
| yuan2023auv | Yuan et al. 2023, AUV collision avoidance with DDPG | JMSE | 10 | Weak | S2 2026-11-24 | full | Y | 2.4 | GRU actor/critic; sonar only; single-step replay suggests no cross-step memory. | https://doi.org/10.3390/jmse11122258 |
| wang2024adversarial | Wang, Xiang, Duan & Yang 2024, adversarial DRL depth tracking | Eng. Appl. of AI | 49 | Strong (venue) | S2 2026-11-24 | abstract | – | 2.4 | LSTM + MLP agent. Depth control, not navigation. | https://doi.org/10.1016/j.engappai.2023.107728 |
| manderson2018vision | Manderson et al. 2018, vision-based reef navigation | IROS 2018 | – | Strong | venue only | notes | Y | 2.2 (optional) | Supervised CNN; not RL. | — |
| manderson2020vision | Manderson et al. 2020, "Vision-Based Goal-Conditioned Policies for Underwater Navigation in the Presence of Obstacles" | RSS 2020 | – | Strong | venue only | notes | Y | 2.2 | Rated 2026-11-25 from bib venue. Imitation, not RL. | https://arxiv.org/abs/2006.16235 |
| deowan2025optimizing | Deowan, Yousha, Hossain, Hassan & Marxer 2025, "Optimizing Underwater Robot Navigation: A Study of DRL Algorithms and Multi-Modal Sensor Fusion" | ICRA 2025 | – | Strong | venue only | not read | Y | 2.4 (todo) | Authors from bib, 2026-11-25. Check stacking vs recurrence. | — |
| deowan2026unified | Deowan & Kelasidi 2026, unified planning-learning framework | arXiv | – | Preprint | — | full | Y | 2.4, 2.5, Intro | Closest work. Map + one-step world model; feedforward policy. | https://arxiv.org/abs/2608.05365 |
| shafer2026endtoend | Shafer & Gal 2026, end-to-end AUV motion planning with RL | arXiv | – | Preprint | — | full | Y | 2.2, 2.4 | ResNet-10 encoder; 3 stacked FLS frames; no recurrence; sim. | https://arxiv.org/abs/2606.08513 |
| larsen2024variational | Larsen et al. 2024, VAE range latent for surface vessel | arXiv (fix to OMAE 2024) | – | Preprint | — | full | Y | 2.2 (optional) | Surface vessel, not underwater. Re-rate after fixing bib entry. | https://arxiv.org/abs/2404.00623 |
| singh2025ariel | Singh et al. 2025, Ariel | arXiv | – | Preprint | — | full | Y | 2.4 (planner side) | Classical GBPlanner; GRU only for velocity estimation. Not a recurrent policy. | https://arxiv.org/abs/2507.10003 |
| yang2025duvin | Yang et al. 2025, DUViN | arXiv | – | Preprint | — | notes | Y | 2.2 | Frozen Depth Anything encoder + diffusion policy; imitation. | https://arxiv.org/abs/2509.02983 |
| gao2024robotic | Gao et al. 2024, robotic fish cross-modal VAE | Biomimetics (MDPI) | – | Weak | venue only | notes | – | 2.2 (optional) | Only underwater-ish VAE latent policy found; BC, sim. | — |
| panisilvam2022conditioned | Panisilvam et al. 2022, conditioned VAE sonar sim-to-real | OCEANS 2022 | – | Borderline | venue only | notes | Y | 2.2 (optional) | Perception, not control. | — |
| pucci2024cevae | Pucci & Martinel 2024, CE-VAE | arXiv | – | Preprint | — | notes | Y | 2.2 (optional) | Image enhancement. | https://arxiv.org/abs/2406.01294 |
| scope2025 | SCOPE 2025, sonar video compression | arXiv | – | Preprint | — | notes | – | 2.2 (optional) | Authors and key to fill in. | https://arxiv.org/abs/2511.13922 |

## Open items

- Confirm retraction status of tian2025reinforcement at the DOI; it is still cited in `Sections/RelatedWork.tex`.
- Check citation counts for `venue only` rows used in the Introduction (xiao2022motion, ha2025learning, kulkarni2024dce, hoeller2021learning, yang2026spatially, morad2023popgym).
- Identify the PPO + CNN + GRU UUV path-planning paper cited (empty `\textcite{}`) in `Sections/RelatedWork.tex` Sec. 2.3 (not found; dropped from draft).
- Add to bib (rows with Bib = –) before citing: sha2021path, an2023intelligent, wang2024adversarial, massotcampos2015optical, paull2014auv; optional: zeng2015survey, gonzalezsabbagh2023survey, koser2020challenges, gao2024robotic, scope2025.
- Avoid "recurrent units in learned navigation remain unexplored" (contradicted by Yuan 2021/2023, Kulkarni 2024). Defensible: recurrent policies combining depth images with sonar remain unexplored underwater.

## Decisions log

- 2026-11-24: Sec. 2.4 draft trend sentence uses sha2021path + an2023intelligent + yang2022monocular, worded as gradual adoption (no Strong source quantifies growth). "Recurring architecture" framed as recurring across cited works, not "most common".
- 2026-11-25: Folder restructured; this file is the master list. Moved into the git repo (master); claude.ai project keeps a mirror. Ratings for raffin2019decoupling, patel2026defm, kulkarni2023task, kulkarni2023semantically, manderson2020vision filled in from bib venues; deowan2025optimizing authors taken from bib.
