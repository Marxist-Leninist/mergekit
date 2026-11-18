# Custom state-dict layer remapping

`mergekit.layer_remap.remap_layers(state, order, prefix="blocks")` creates an independent copy with donor layers placed in the requested order. It preserves depth and requires matching tensor names, shapes and dtypes. Example: `[1, 0]` swaps a two-layer model; `[0, 0]` repeats its first layer in both slots. Other weights are copied unchanged. It does not change configurations, infer architectural compatibility, or guarantee improved quality.

Pass the result to `mergekit.merge_methods.merge_state_dicts` to interpolate it with the original. Always preserve the original, check identity round trips, choose configurations using validation data only, and compare matched held-out examples. Independent seeds need not have aligned neurons, so direct averaging can reduce quality. No automatic deployment or upload occurs.
