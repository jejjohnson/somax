# Changelog

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
