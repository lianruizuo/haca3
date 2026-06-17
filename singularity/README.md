# Building the HACA3 Singularity image

These files rebuild the HACA3 Singularity/Apptainer image from the repository
source. The rebuilt image is identical to the published one except that it also
includes the new theta-only encoder CLI **`haca3-encode`** (no other behavior is
changed).

- `haca3.def` — the image definition (same PyTorch base and `pip install .` as
  the project `Dockerfile`).
- `build_singularity.sh` — convenience wrapper around `singularity build`.

## Requirements
- A **Linux** machine with **sudo** access (macOS cannot build Singularity
  images natively).
- **Singularity** or **Apptainer** installed (the script auto-detects either).
- Internet access (to pull the PyTorch base image and pip dependencies).

## Build
From the repository root, after pulling the branch you want to package:

```bash
sudo bash singularity/build_singularity.sh
```

This writes `haca3.sif` to the repo root. To choose a different output path:

```bash
sudo bash singularity/build_singularity.sh /data/images/haca3_encode.sif
```

The image is built from your **local checkout**, so make sure the code you want
is checked out before building.

## Verify
```bash
singularity exec haca3.sif haca3-encode --help
```

## Use the new theta encoder
```bash
singularity exec --nv haca3.sif haca3-encode \
  --in-path image.nii.gz \
  --harmonization-model harmonization.pt \
  --out-dir theta_out/
```

This runs only the theta encoder (no harmonization) and writes per-slice and
per-volume theta CSVs into `theta_out/`. The existing `haca3-test` and
`haca3-train` commands continue to work exactly as before.
