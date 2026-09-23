# Bundled assets

Weights and `labels.map` are unchanged copies of the files in
`container/lib/` of https://github.com/soffiafdz/hvr_validation. The two
reference grids are the same volumes with the `:ident` header attribute
removed and the file repacked (`minc_modify_header -delete :ident`,
`h5repack`); voxels and geometry are identical to the originals.

| file | md5 | what |
|---|---|---|
| `ensemble_hcvc.pth` | f73ffc2f5295894f07737895b6cf0d07 | model `simple`: L/R hippocampus + temporal-horn CSF |
| `ensemble_hcvc-ag.pth` | 7a449d26b282a44ca68df6f342388506 | model `detailed`: HC head/body/tail, VC head/body/tail, amygdala |
| `ref_hcvc.mnc` | 938ebf1cb2d8e1cc479236b2a4e75dd5 | sampling grid for `simple` |
| `ref_hcvc-ag.mnc` | e29fd08403fa983ce8f8a5207dd4fbb5 | sampling grid for `detailed` |
| `mni_icbm152_t1_tal_nlin_sym_09c.mnc` | 30a4c9e46e4f7cdf15f5a08b96a8181e | ICBM 152 2009c nonlinear symmetric T1, 1 mm: registration target and intensity reference for `native` / `assemblynet` input. Original 16-bit values (an 8-bit copy changed native-mode volumes by 3.5 %); `:ident` and `:history` removed |
| `mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc` | 7441c4bd209577287c4150b97236fc0a | its brain mask, byte, unchanged values |
| `labels.map` | a7bdc784b5637c24d047601fdda3b1d6 | colour lookup for QC images |

Not copied: `ensemble_hvr_extra.pth` and `ref_extra.mnc` are byte-identical
to the `-ag` files (same md5).

## Facts read from the files

- Weights are PyTorch zip-format **full-object pickles** (not state_dicts),
  produced with `torch.save(EnsembleModel)`.
  They reference `model.ensemble.EnsembleModel`, `model.vae2.VAE_UNET2`,
  `model.basic2.InceptionModule_`, `model.util.PreprocessModule`, so
  `src/model/` must be importable as top-level `model`.
- Tensors are stored with device tag `cpu`; loading needs no GPU.
- Loading requires `torch.load(path, map_location="cpu", weights_only=False)`
  on torch >= 2.6.
- Both reference grids: 119 x 96 x 140 voxels (x,y,z), 1 mm, starts
  x=-63, y=-69, z=-53, MNI/ICBM152 stereotaxic coordinates. One hemisphere
  is processed at a time; the left side is X-flipped onto this grid.
- The network normalises its input with a fixed mean/sd
  (`PreprocessModule`), so input intensities must be on the training scale:
  linear-normalised to the ICBM152 2009c template (`volume_pol`), roughly
  0-100 in brain tissue.

## Label values in outputs

- `simple`: 11 L-HC, 12 L-VC, 21 R-HC, 22 R-VC
- `detailed`: 111/112/113 L-HC tail/body/head, 121/122/123 L-VC
  tail/body/head, 130 L-amygdala; 211...230 same on the right. Order
  verified from the anterior-posterior position of each label (x13 is
  adjacent to the amygdala); the old code had head and tail swapped.
