# Changelog

## [0.0.14](https://github.com/jejjohnson/somax/compare/somax-v0.0.13...somax-v0.0.14) (2026-09-12)


### Features

* **core:** add ScaledModel to integrate in transformed coordinates ([5a2da78](https://github.com/jejjohnson/somax/commit/5a2da789e42919aa56f5221cd8990d7e7bb058f0))
* **core:** add Scales and StateAffine ([293ee0a](https://github.com/jejjohnson/somax/commit/293ee0a6d3c015c2e4fe335b1b7db55a66a1f83f))
* **core:** constrained parameters via paramax ([e7d0201](https://github.com/jejjohnson/somax/commit/e7d0201237cba4ae607eac62ecfeb313576881a5))
* **core:** constrained parameters via paramax ([69eca61](https://github.com/jejjohnson/somax/commit/69eca6194180a7ac5b9792c9a80ecc829ab5e1a3)), closes [#172](https://github.com/jejjohnson/somax/issues/172)
* **core:** optional somax[flows] extra — flowjax bridge ([19ecfca](https://github.com/jejjohnson/somax/commit/19ecfca1e3d86560fda00e48d76cfcdbd1a1a240))
* **core:** optional somax[flows] extra — flowjax bridge ([34fe399](https://github.com/jejjohnson/somax/commit/34fe399488942dfcbf1092b229b5155418c8df26)), closes [#178](https://github.com/jejjohnson/somax/issues/178)
* **core:** ScaledModel — integrate in transformed coordinates ([69fd299](https://github.com/jejjohnson/somax/commit/69fd299408e350fa4c15bcedb9b7dee9148dc869)), closes [#175](https://github.com/jejjohnson/somax/issues/175)
* **core:** Scales and StateAffine ([975aae0](https://github.com/jejjohnson/somax/commit/975aae05007b1d95aa045e2bf5c07f0a3c1b2eb9)), closes [#171](https://github.com/jejjohnson/somax/issues/171)
* **da,io,cli:** StateAffine hooks for DA, export and configs ([46030ca](https://github.com/jejjohnson/somax/commit/46030ca05be6c0166338889956c2edb60ee3b86d)), closes [#179](https://github.com/jejjohnson/somax/issues/179)
* **da,io,cli:** thread StateAffine through DA, export and configs ([0a09bd7](https://github.com/jejjohnson/somax/commit/0a09bd7049fe87a92e8c441f791349913f2a3550))
* **models:** add BarotropicQG.from_nondimensional with boundary-layer guards ([f8784a0](https://github.com/jejjohnson/somax/commit/f8784a07f9a9aae6ade50e6deea4899f06894009))
* **models:** add spherical shallow water and barotropic QG models ([3c01fc0](https://github.com/jejjohnson/somax/commit/3c01fc08b8428d8f075be691f729223b37dafd0c))
* **models:** BarotropicQG.from_nondimensional with boundary-layer guards ([1e68251](https://github.com/jejjohnson/somax/commit/1e6825165803c16196dc351df7f3371cd1ec61ef)), closes [#173](https://github.com/jejjohnson/somax/issues/173)
* **models:** from_nondimensional for pde1d and pde2d ([9f9f0f6](https://github.com/jejjohnson/somax/commit/9f9f0f6bc47c2a5f123293674bfa45190d8b6fa9))
* **models:** from_nondimensional for pde1d and pde2d ([e7148de](https://github.com/jejjohnson/somax/commit/e7148de65d1645c64c42debc38cc1ea684e0b4c3)), closes [#176](https://github.com/jejjohnson/somax/issues/176)
* **models:** from_nondimensional for the layered models ([03f1a74](https://github.com/jejjohnson/somax/commit/03f1a749a6b2258935466bdad4058594a613d9ed))
* **models:** from_nondimensional for the layered models ([e40e928](https://github.com/jejjohnson/somax/commit/e40e92814df8d79afa85334e87f25d243cd80519)), closes [#174](https://github.com/jejjohnson/somax/issues/174)
* **models:** from_nondimensional for the spherical models ([f3f2867](https://github.com/jejjohnson/somax/commit/f3f28677b952b2a1bea2695da960e9c30d0237a2))
* **models:** from_nondimensional for the spherical models ([d02a34a](https://github.com/jejjohnson/somax/commit/d02a34a49478f7dc549b8b9919287c8664f5eddc))
* **models:** spherical shallow water and barotropic QG models ([8761c82](https://github.com/jejjohnson/somax/commit/8761c82bfda6ac0f60a06f8570f6c2eae72e8323))


### Bug Fixes

* **cli:** accept a legacy SI artifact on restart ([f492a3b](https://github.com/jejjohnson/somax/commit/f492a3b0bacd401e736e25b1df0d98de313828c6))
* **cli:** mark the coordinate system, and track the generated config ([809a4a3](https://github.com/jejjohnson/somax/commit/809a4a323380fb91e6688c9a8b370ff3ef77d6c1))
* **core,models:** correct dt_from_cfl, and keep the convection direction ([2ecf5af](https://github.com/jejjohnson/somax/commit/2ecf5af7de48a5602fee77f4614218ff53955851))
* **core,models:** wrappers in the term-model factories, and reject infinities ([fc4cf9c](https://github.com/jejjohnson/somax/commit/fc4cf9c2d2d55a4424004706a6e88bf10ec05d26))
* **core:** distinguish generated moment pairs, and validate the sample floor ([aef5dcd](https://github.com/jejjohnson/somax/commit/aef5dcdd2d01b714f18a550b157b9c66ecffb313))
* **core:** per-state field metadata, and keep the groups under nondimensional() ([a4c40e2](https://github.com/jejjohnson/somax/commit/a4c40e2e9b2f6fc6ead9cb22234dc5e9723b1d33))
* **core:** preserve the inner model's term structure in ScaledModel ([c90e58b](https://github.com/jejjohnson/somax/commit/c90e58bf47c5d0e171b65849726d17349e0e4f45))
* **core:** reject a degenerate ScaledModel time scale ([75a9ef7](https://github.com/jejjohnson/somax/commit/75a9ef70b61ec776b05092097fd487c41eb4fe1d))
* **core:** reject non-finite beta and resolution thresholds ([d3788d6](https://github.com/jejjohnson/somax/commit/d3788d647715bf6edfd4856463e11ce9929e3af1))
* **core:** sum the directional Courant contributions ([b5ceeee](https://github.com/jejjohnson/somax/commit/b5ceeee7d472218dc42c0b68055dbf192f51f2d7))
* **core:** unwrap on the public API, and make constrained params usable ([dc49cc4](https://github.com/jejjohnson/somax/commit/dc49cc4ce1aca70b69612b559855f45ff20dcae4))
* **da,cli,io:** thread the transform and the nondim block through ([9706773](https://github.com/jejjohnson/somax/commit/9706773db775ccd2915a0982a2713ddeb1c5289b))
* **flows,ci,docs:** keep the affine fixed, and actually run these tests ([d1cc348](https://github.com/jejjohnson/somax/commit/d1cc348082813b7905528f80b48d8aff20c460c4))
* **models,cli:** make the spherical models usable from the CLI ([f6d63e5](https://github.com/jejjohnson/somax/commit/f6d63e56fe1e568ad299ab5b2e97360054af0e05))
* **models,cli:** zonal spacing in the guards, and free them from the CLI extras ([33de16b](https://github.com/jejjohnson/somax/commit/33de16b1920bedfcf932d01f43752a88db17fa90))
* **models:** finish freeing the guards from the CLI extras ([3dfa6c0](https://github.com/jejjohnson/somax/commit/3dfa6c0c4e7852cd52c81a91ce88e340651c65fd))
* **models:** interface gravity in the layered scales, and tighten the factories ([8e8202d](https://github.com/jejjohnson/somax/commit/8e8202d1b9d096cd96a42c151964475ceb1698e9))
* **paramax:** mask only arrays, bound the interval span, checkpoint wrappers ([d6d027d](https://github.com/jejjohnson/somax/commit/d6d027da25cc9584ba61d9a1417d4e3108e366b6))
* **qg:** share the interface gravity, and normalise the default wind curl ([9aeff8c](https://github.com/jejjohnson/somax/commit/9aeff8ca58d3ca9e0e636887097f407e04413c22))
* **resolution:** reject a negative n_cells_min ([0382ed0](https://github.com/jejjohnson/somax/commit/0382ed085406ebbfe3b9e03eadb201677a5da133))
* **scales:** overflow-safe direction norm, and scope the nondim contract ([db72460](https://github.com/jejjohnson/somax/commit/db724608f000bff8d851346786a2c99f1e942cf6))
* **spherical:** address second-round review on the spherical stack ([10e5bf7](https://github.com/jejjohnson/somax/commit/10e5bf7249ae86d353c26e7a05c66e9fb59c3e4d))
* **spherical:** forcing shape and scaling, masked sources, wall ghosts ([c78fdec](https://github.com/jejjohnson/somax/commit/c78fdec07511d0fac6814fc68671cc216b2b684e))
* **transforms:** tracer scaling, finite overrides, wider moment accumulators ([5644b10](https://github.com/jejjohnson/somax/commit/5644b103a8cdc706af29cf27d767c79e02c30f8e))


### Documentation

* **scales:** describe the directional advective CFL bound ([a67e8be](https://github.com/jejjohnson/somax/commit/a67e8be676b53b66f846b503ce6cc18cdc3dd107))

## [0.0.13](https://github.com/jejjohnson/somax/compare/somax-v0.0.12...somax-v0.0.13) (2026-06-19)


### Documentation

* **gallery:** double-gyre bring-up methodology + barotropic QG page ([#154](https://github.com/jejjohnson/somax/issues/154)) ([cbf0459](https://github.com/jejjohnson/somax/commit/cbf045988ef810a6de1b59179faf824e122e223f))

## [0.0.12](https://github.com/jejjohnson/somax/compare/somax-v0.0.11...somax-v0.0.12) (2026-06-08)


### Features

* **core:** reduced-order forcing basis + geonnax-wired forcing bank ([#143](https://github.com/jejjohnson/somax/issues/143)) ([0bf7933](https://github.com/jejjohnson/somax/commit/0bf79339fdf678154a2027388a628986c3b6591a))

## [0.0.11](https://github.com/jejjohnson/somax/compare/somax-v0.0.10...somax-v0.0.11) (2026-06-03)


### Documentation

* **book:** generated strict-myst api reference ([#54](https://github.com/jejjohnson/somax/issues/54)) ([#135](https://github.com/jejjohnson/somax/issues/135)) ([629db0a](https://github.com/jejjohnson/somax/commit/629db0a67aef8b8e7f2fe6cc7baf08e050c39b02))
* **book:** phase 0 foundations chapters — grids, operators, boundary conditions ([#52](https://github.com/jejjohnson/somax/issues/52)) ([#132](https://github.com/jejjohnson/somax/issues/132)) ([540f71f](https://github.com/jejjohnson/somax/commit/540f71f8bad58cf9e8d6e5f146849b9d8dfd56ed))
* **book:** phase 1-4 theory & practice chapters ([#53](https://github.com/jejjohnson/somax/issues/53)) ([#134](https://github.com/jejjohnson/somax/issues/134)) ([fc606e6](https://github.com/jejjohnson/somax/commit/fc606e6b191b871f172e6a7aa46eb8db05f5674d))

## [0.0.10](https://github.com/jejjohnson/somax/compare/somax-v0.0.9...somax-v0.0.10) (2026-06-02)


### Features

* **observe:** diagnostics, monitors & fail-fast observability ([#129](https://github.com/jejjohnson/somax/issues/129)) ([92b28d6](https://github.com/jejjohnson/somax/commit/92b28d6b2d63d84d44260a0269c294af5756e66a))


### Bug Fixes

* **solvers:** matrix-free IMEX solver to avoid dense-Jacobian OOM ([#55](https://github.com/jejjohnson/somax/issues/55)) ([#131](https://github.com/jejjohnson/somax/issues/131)) ([ef1f323](https://github.com/jejjohnson/somax/commit/ef1f3232161b0d928f2e6db22e8e3b064ab3bf95))

## [0.0.9](https://github.com/jejjohnson/somax/compare/somax-v0.0.8...somax-v0.0.9) (2026-06-02)


### Features

* **da:** filterax ensemble-filter integration (Phase 4a) ([#127](https://github.com/jejjohnson/somax/issues/127)) ([71c16e9](https://github.com/jejjohnson/somax/commit/71c16e9a9eeea1b2837f2ee379c63aa3d6f9bd60))
* **da:** vardax variational 4DVar integration (Phase 4b) ([#128](https://github.com/jejjohnson/somax/issues/128)) ([d7ee90d](https://github.com/jejjohnson/somax/commit/d7ee90d490b3acb737b3ed2bb0cdb01fedd91f42))
* **eval:** reference-free evaluation metrics on the model grid (Phase 3) ([#125](https://github.com/jejjohnson/somax/issues/125)) ([0b32990](https://github.com/jejjohnson/somax/commit/0b32990bfcd97746460802c9db704f44f1e377b5))

## [0.0.8](https://github.com/jejjohnson/somax/compare/somax-v0.0.7...somax-v0.0.8) (2026-04-20)


### Features

* **cli:** crash-recovery checkpointing for long simulations ([#98](https://github.com/jejjohnson/somax/issues/98)) ([5ecb447](https://github.com/jejjohnson/somax/commit/5ecb447cc05c7142a91bc043e62e73b94def3670))


### Documentation

* **cli:** basin data pipeline scaffold + decisions ([#74](https://github.com/jejjohnson/somax/issues/74), [#75](https://github.com/jejjohnson/somax/issues/75)) ([#101](https://github.com/jejjohnson/somax/issues/101)) ([09dd292](https://github.com/jejjohnson/somax/commit/09dd292b7df47778e8c11f1f634394dfbbb652fc))

## [0.0.7](https://github.com/jejjohnson/somax/compare/somax-v0.0.6...somax-v0.0.7) (2026-04-20)


### Bug Fixes

* **models:** thread Mask1D/Mask2D through every model ([#95](https://github.com/jejjohnson/somax/issues/95)) ([e40e228](https://github.com/jejjohnson/somax/commit/e40e2281ccfef5fe3c49c4217455c5156800dd21))

## [0.0.6](https://github.com/jejjohnson/somax/compare/somax-v0.0.5...somax-v0.0.6) (2026-04-10)


### Features

* somax-sim CLI + DVC pipelines for reference simulations ([#71](https://github.com/jejjohnson/somax/issues/71)) ([a0220b2](https://github.com/jejjohnson/somax/commit/a0220b277b76b41ffe09c7eb4e26f00b4ac6d3b0))

## [0.0.5](https://github.com/jejjohnson/somax/compare/somax-v0.0.4...somax-v0.0.5) (2026-04-09)


### Features

* add baroclinic (multilayer) quasi-geostrophic model ([#67](https://github.com/jejjohnson/somax/issues/67)) ([d993046](https://github.com/jejjohnson/somax/commit/d993046687ce7553b046acfcce174b0baf2b212f))
* add multilayer SWM and reparameterized QG models ([#69](https://github.com/jejjohnson/somax/issues/69)) ([0a52fbd](https://github.com/jejjohnson/somax/commit/0a52fbd1afaa63957d5961c6c88827b1326ee857))

## [0.0.4](https://github.com/jejjohnson/somax/compare/somax-v0.0.3...somax-v0.0.4) (2026-04-09)


### Features

* add StratificationProfile and refactor ModalTransform ([#66](https://github.com/jejjohnson/somax/issues/66)) ([62e1da8](https://github.com/jejjohnson/somax/commit/62e1da81b1bb722f135a43630de30d89ac852506))
* phase 1-2 PDE models + 13 Steps to Navier-Stokes tutorials ([#62](https://github.com/jejjohnson/somax/issues/62)) ([9621419](https://github.com/jejjohnson/somax/commit/962141943d5cae9cf2c3c7bb3e10f3d5d783cef7))
* phase 4 GFD planar models — shallow water + quasi-geostrophic ([#65](https://github.com/jejjohnson/somax/issues/65)) ([0a6182c](https://github.com/jejjohnson/somax/commit/0a6182c78262618addde38f5a5dcf5a98790e1c5))

## [0.0.3](https://github.com/jejjohnson/somax/compare/somax-v0.0.2...somax-v0.0.3) (2026-04-08)


### Features

* **core:** add SeasonalWindForcing and InterpolatedForcing ([#60](https://github.com/jejjohnson/somax/issues/60)) ([2f66ab2](https://github.com/jejjohnson/somax/commit/2f66ab24b27d1fb6d8c73895b2c6f6cef24bd293))
* migrate Lorenz models to SomaxModel + tutorial notebooks ([#30](https://github.com/jejjohnson/somax/issues/30), [#33](https://github.com/jejjohnson/somax/issues/33)) ([0290b62](https://github.com/jejjohnson/somax/commit/0290b6241bcca8cfbea1d1e94c6ee303505aef87))
* **models:** migrate Lorenz models to SomaxModel + tutorial notebooks ([#59](https://github.com/jejjohnson/somax/issues/59)) ([0290b62](https://github.com/jejjohnson/somax/commit/0290b6241bcca8cfbea1d1e94c6ee303505aef87))

## [0.0.2](https://github.com/jejjohnson/somax/compare/somax-v0.0.1...somax-v0.0.2) (2026-04-08)


### Features

* **core:** add ModalTransform, HelmholtzCache, and SimulationCheckpointer ([247bfe7](https://github.com/jejjohnson/somax/commit/247bfe77d32fa7bf56438e6e4d2225730ef935c8))


### Bug Fixes

* address PR review comments ([9ba7381](https://github.com/jejjohnson/somax/commit/9ba7381a4ca10164109d1c8ceb7f30ecf19062e3))
* gitHub actions configuration ([53a453a](https://github.com/jejjohnson/somax/commit/53a453a9ae1fe640c332f225a44d71dd2afd1fb9))
* pin CI git dependencies and ty behavior ([3b00537](https://github.com/jejjohnson/somax/commit/3b00537aff87fde391eff6eb2d9fcca17e1598b7))
* pin spectraldiffx dependency source ([f56dce7](https://github.com/jejjohnson/somax/commit/f56dce7e0bdb749fa348d620fffc110c26302c5a))
* stabilize uv and ty CI configuration ([fd71bc4](https://github.com/jejjohnson/somax/commit/fd71bc4d4792d5fc3bd380bf344a1a90902e4cf9))
* **typecheck:** add ty rules for optional deps and eqx.Module callables ([39ab3a0](https://github.com/jejjohnson/somax/commit/39ab3a0cb3a24266f176c2976faf9b7abc8ed882))


### Documentation

* rewrite README for modernized tooling and project structure ([52954cb](https://github.com/jejjohnson/somax/commit/52954cb85b768a858ac0425265177b8279999fd2))
