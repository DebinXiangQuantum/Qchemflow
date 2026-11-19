# Molecule names (G3)

Load the `drug` pack and add `ligands_from_names` with `names=["aspirin"]`.
Its `LigandLibrary` works with the existing filters, descriptors, conformers and
docking operators. Names are ambiguous: **check the returned PubChem CID** before
using a structure for scientific work.

Each entry contains RDKit-canonical SMILES and the InChIKey computed from that
SMILES, plus `source="pubchem"`, `cid`, `retrieved_at` (the original fetch time in
UTC ISO format), and `cached`. Structures that fail parsing or disagree with
PubChem's InChIKey are refused with CF-E060 and are not cached.

The resolver uses PubChem PUG REST over HTTPS with a 15-second timeout. Exact
names key a JSON cache at `CHEMFLOW_NAME_CACHE`, defaulting to
`~/.chemflow/name_cache.json`. Successful batches are written by atomic file
replacement; a failed batch returns no partial library and writes no new records.
Concurrent writers may overwrite each other's additions; atomic replacement
prevents a torn JSON file, not lost updates. Cache hits preserve the original
fetch timestamp and do not contact PubChem. Unreadable/malformed cache files or
incomplete entries are treated as misses; inconsistent cached structures are
refused rather than trusted. An offline miss fails with CF-E080 naming the
molecule. A writable cache location is required for newly fetched records.

This operator has the `network` effect. Signed tasks (including agent tools)
must include `"network"` in `allowed_effects`, even on cache hits. Missing
permission is CF-E120 before any node executes. Unsigned `Executor.run` may run
it without a task permission. Executor memoisation never stores network nodes;
the name cache is their persistent store. The resource forecast is `cpu_numeric`,
with unknown `wall_s` and all other resource dimensions zero.
