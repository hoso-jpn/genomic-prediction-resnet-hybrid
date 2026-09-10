# Producer integration fixture

These tiny, wholly synthetic panels were emitted by the actual upstream
`bin/build_gs_panel.py` CLI and `build_gs_panel_manifest.build_manifest`
from [adzuki-snp-pipeline at 3158ca5](https://github.com/hoso-jpn/adzuki-snp-pipeline/tree/3158ca50c2c13c31bdc80db302c7df4bbb5670bf).
They cover a populated and an empty panel, phased calls, missing calls,
opaque sample IDs (`001`, `NA`), and manifest schema v2.

Rebuild with `python tests/fixtures/gs_panel_producer_v2/regenerate.py /path/to/producer`.
The script requires that exact producer commit. It contains the full synthetic
VCF input. The matrix and metadata come directly from the producer CLI;
the manifest comes from its public builder with fixture-only provenance.
The fixed run IDs/timestamp and `fixture/unused:synthetic` container identities
are placeholders, not evidence of a Nextflow or container run. No real data,
variant-calling result, performance claim, or biological validation is included.

The fixtures are checked into this repository so CI does not fetch or execute
external code. Historical schema v1 is separately covered by unit fixtures.
