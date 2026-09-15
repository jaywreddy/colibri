# Mask writing — CMU recipe

The process recipe for writing and developing the mask at CMU's cleanroom,
kept as its own page since it changes on a different schedule than the
design decisions in `docs/decisions.md` (this is "how to run the tool",
not "what number the design uses").

## Exposure

- **Writer:** Heidelberg MLA 150 (maskless aligner)
- **Dose:** 125 (MLA 150 dose units)

## Develop

- **Developer:** AZ 400K : DI water, 1:4
- **Time:** 80 s

## Chrome etch

- **Etchant:** Cr etch 1020 (equivalently CR-7)
- **Time:** 80 s

## Resist strip

- **Stripper:** Nanostrip
- **Temperature:** 60 °C
- **Time:** 20 min

## Layers to write

Only three GDS layers matter to the writer; everything else in a plate's
GDS (assembly verniers, tick-code IDs, etc.) is a sub-feature drawn on one
of these, not a separate write pass.

| Layer/datatype | Name | What it is |
|---|---|---|
| **10/0** | `LAYER_FRONT` (`witness_geom.LAYER_FRONT`) | the written data — WRITE THIS. Darkfield, positive resist: this is the CLEAR data, where chrome comes off. Every die's actual pattern (garland, region-art centrepiece, halftone screen, bench cell) lives here. This is the only layer that goes through exposure/develop/etch above. |
| **1/0** | `LAYER_OUTLINE` (`witness_geom.LAYER_OUTLINE`) | cell/die outline boxes — reference geometry only, never chrome. Used for alignment and to bound each cell/die when reading the plate back; do not expose. |
| **2/0** | `LAYER_DICE` (`export_witness.LAYER_DICE`) | the saw's street centrelines (dicing cut lines) — annotation only, never chrome. `DICING.md` and the map SVG's red dashes are generated from this layer's data in the manifest, not from the layer itself. |

`LAYER_PAIR` (21/0, outline-only, marks the back die of a two-layer
experiment cell) exists in the code for the hidden two-ply exemplars but
carries no production geometry — nothing on the current plate uses it.

So: **write 10/0 only.** 1/0 and 2/0 are there for humans and downstream
tooling (dicing, verification) to read, not for the aligner to expose.
