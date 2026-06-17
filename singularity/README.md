# Building the HACA3 Singularity image

These files rebuild the HACA3 Singularity/Apptainer image from the repository
source. The rebuilt image is identical to the published one except that it also
includes the new theta-only encoder CLI **`haca3-encode`** (no other behavior is
changed).

- `haca3.def` — the image definition (same PyTorch base and `pip install .` as
  the project `Dockerfile`).
- `build_singularity.sh` — convenience wrapper around `singularity build`.

## Requirements
- A **Linux** machine (macOS cannot build Singularity images natively).
- **Singularity** or **Apptainer** installed (the script auto-detects either).
- Internet access (to pull the PyTorch base image and pip dependencies).

## Build
From the repository root, after pulling the branch you want to package:

```bash
bash singularity/build_singularity.sh
```

By default this does a **rootless `--fakeroot` build** (no `sudo`). This is the
mode that works when the repo lives on **root-squashed network storage** (e.g. a
cluster/NFS path like `/valiant02/...`), where a `sudo` build fails with
`Permission denied` because the root user is mapped to "nobody" and cannot read
your files.

This writes `haca3.sif` to the repo root. To choose a different output path:

```bash
bash singularity/build_singularity.sh /data/images/haca3_encode.sif
```

The image is built from your **local checkout**, so make sure the code you want
is checked out before building.

### If `--fakeroot` is not available at your site
Some sites disable unprivileged user namespaces. In that case, copy the repo to
**local disk** (which root *can* read) and build there with `sudo`:

```bash
cp -a /path/to/haca3 /tmp/haca3-build
cd /tmp/haca3-build
HACA3_BUILD_MODE=sudo bash singularity/build_singularity.sh /tmp/haca3.sif
mv /tmp/haca3.sif /path/to/haca3/haca3.sif
```

> Do **not** run `sudo bash singularity/build_singularity.sh` directly from
> root-squashed NFS — the build reads your source as root and will fail with
> `Permission denied`.

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
