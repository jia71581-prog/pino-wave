# v22 supersession note

The original `r16_dscp_v22_wide128_radial_guard` artifact is retained for audit history, but it is not the production-faithful candidate. Its module projected the sixteen coefficient maps onto the registered low/middle radial subspace, yet omitted the frozen v18 full-time parent-energy keep mask during final correction materialization.

It is superseded by `r16_dscp_v22b_wide128_radial_guard_confined`, which preserves all v18 deployment constraints and adds the radial guard:

- exact v18 Wide128 topology and checkpoint state (`25,266` parameters; checkpoint SHA-256 `51c0f0f0e0959e82f117ba9dd58c4fcd5bab50dc5d8fb0730473f4a17c887dfc`);
- label-free deployment inputs;
- registered two-dimensional radial high-band projection on the coefficient maps;
- exact C1 causality and free-surface enforcement;
- frozen parent-energy keep mask with bitwise identity to the parent outside the mask.

The v22b self-test completed at `2026-08-27T15:21:10Z` with status `passed`. Its bound module SHA-256 is `98bac1a4810958273d84b4a5b1af2732a5322f0dc73405ad996fa9aa39749054`, script SHA-256 is `c1780417cf6856287d216e3513a44e39082e647d94fbad516e6e2c58a3f346af`, and spec SHA-256 is `09e214ca447d8d9d0ccbe65b3902759158b1c5500ee97a936f3dd7f12c8593c2`.

No v22 files are deleted or overwritten. Future deployment references must point to v22b.
