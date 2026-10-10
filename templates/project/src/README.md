# src

The `__pkg__` package: code that configs instantiate with `_target_`.
Entry points live in `scripts/`, not here.

```
__pkg__/
  data/        DataModule, datasets, transforms, collator
  model/       network modules; the LightningModule in model/lightning.py
  utils/       callbacks and the rest
```

A new class plugs in without editing `scripts/train.py`: a file in the group
`configs/<group>/` with `_target_: __pkg__.<module>.<Class>` and the
constructor's arguments. An object that is not needed right away (optimizer,
scheduler) gets `_partial_: true`: the constructor receives a function and
passes the missing arguments itself.

`data/datamodule.py`, `model/mlp.py` and `Regressor` in `model/lightning.py`
are stubs on synthetic data, so that the skeleton trains right away. Replace
them with your own and update `configs/data/` and `configs/model/`.
