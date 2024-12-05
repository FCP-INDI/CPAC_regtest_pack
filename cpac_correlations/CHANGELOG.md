<!-- Copyright (C) 2022-2024  C-PAC Developers

This file is part of C-PAC.

C-PAC is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

C-PAC is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with C-PAC. If not, see <https://www.gnu.org/licenses/>. -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0]

### Changed

* [Coerce `n_cpus` to int](https://github.com/FCP-INDI/CPAC_regtest_pack/commit/418310362e714157fede0a25cca3f52b2ad609c5)
* [Make `args` optional](github.com/FCP-INDI/CPAC_regtest_pack/commit/a90cfd41506ce6dca6104868358e46fca9892dad) to facilitate calling as a function or from the CLI
* Document and update some typehints
* [Migrate from deprecated CircleCI image](https://circleci.com/docs/next-gen-migration-guide/)
* Update some style rules to match C-PAC code style
* Apply the subject-session labels to the correlation coefficients instead of averaging by site

## [1.0.128] - 2023-11-14

### Changed

* Packaged `cpac_correlations`

[1.1.0]: https://github.com/FCP-INDI/CPAC_regtest_pack/releases/tag/cpac_correlations/v1.1.0
[1.0.128]: https://github.com/FCP-INDI/CPAC_regtest_pack/releases/tag/cpac_correlations/v1.0.128
