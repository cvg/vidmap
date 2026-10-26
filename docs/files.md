# Files and folders

## Input images

JPG, JPEG and PNG are supported. Image names are relative to the input root,
e.g. `cam0/1.png`.

| Input | `--input_data` points to |
|---|---|
| Uncalibrated monocular | A directory containing `cam0/`, or an MP4/MOV video |
| Calibrated monocular | A directory containing `cam0/` |
| Physical rig | A directory containing `cam0/`, `cam1/`, … |
| Panorama | A directory containing `cam0/` with 360° equirectangular images |

Uncalibrated monocular images use natural filename order (`frame2.png` before
`frame10.png`). Rigs, calibrated monocular and panoramas resolve numeric filenames
automatically: `000001.png`, `1.100.png` or `1.100-left.png`. Numbers are seconds;
every `camN/` folder must contain exactly one image per shared timestamp.

### Frame association

If filenames cannot establish associations, set `frames: frames.csv` inside your
frontend config’s `rig` section. For `--input_data /data/images`:

```text
/data/images/
├── frames.csv
├── cam0/
│   ├── a.png
│   └── b.png
└── cam1/
    ├── left_a.png
    └── left_b.png
```

`/data/images/frames.csv`:

```csv
frame_id,timestamp_ns,camera_id,image_name
10,1000000000,0,cam0/a.png
10,1000000000,1,cam1/left_a.png
20,1100000000,0,cam0/b.png
20,1100000000,1,cam1/left_b.png
```

## Configuration files

| Files | Select with |
|---|---|
| `vidmap/configs/frontend/<name>.yaml` | `--frontend-conf <name>` |
| `vidmap/configs/mapping/<name>.yaml` | `--mapping-conf <name>` |

Use names without `.yaml`, e.g. `uncalib/base`. Inherit presets with
`defaults: [calib/rig]`; see [camera examples](cameras.md).

## Run outputs

Local runs write under `--output`:

```text
/results/run/
├── frontend_config.yaml     # Resolved frontend settings
├── mapping_config.yaml      # Resolved mapping settings
├── mapper_inputs/           # Reusable frontend results (below)
├── database_complete.db     # Mapping's working database copy
└── rec/                     # Final COLMAP reconstruction
```

Optional outputs:

| Path | Contents |
|---|---|
| `video_frames/<video-id>/cam0/` | Decoded video images |
| `prepared_capture/images/` | Projected panorama views or undistorted rig images |
| `full_depth_maps.h5` | Full depth maps retained with `--cache-depth-maps` |
| `vidmap-viewer-embedded.html` | Standalone viewer, after [export](../README.md#browser-viewer) |

Original image folders remain at their input paths. Use a new output directory
when changing sources, calibration or projection.

### Mapper inputs

Keep this directory together to rerun mapping without extraction:

| File in `mapper_inputs/` | Contents |
|---|---|
| `database_complete.db` | Cameras, rigs, frames, keypoints and matches |
| `depth_maps.h5` | Depth priors and validity |
| `lc_masks.json` | Sequential / loop-closure match labels |
| `manifest.json` | Input identity, frame order and file checksums |
| `local_input.json` | Source image paths for local runs |
| `geocalib_per_image.h5`, `geocalib_batch.h5` | GeoCalib priors, when generated |
| `vgc_filtered_pairs.json` | Pairs excluded from calibration, when generated |

[Rerun mapping](../README.md#execution) with `--mapper-inputs` and the original
frontend config. External mapper inputs are neither modified nor copied into
the output folder.

Regenerate older flat-layout datasets in a new root to update both images and COLMAP paths.
