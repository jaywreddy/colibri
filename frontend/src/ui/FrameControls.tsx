import { useStore } from '../store';
import { log } from '../logger';
import type { FaceId, FrameSpec } from '../api';
import { Button, NumberRow, Section, SelectRow, SliderRow } from './kit';

/**
 * Frame dials for one face — algorithm picker, theme picker, density/bloom/
 * foliage sliders, frame band (auto or explicit), and the seed control. All
 * edits go to `patchFaceFrame`; App debounces regeneration of the box manifest.
 * (The one-click seed shuffle lives next to "Apply to all faces" in FaceEditor.)
 *
 * Every dial carries a title explaining what it changes on the plate; the copy
 * is taken from the generators' own docstrings (frames/algorithms/wreath.py,
 * colonize.py) and plates.py::_carrier_recipe_data for the seed.
 */
export default function FrameControls({ faceId }: { faceId: FaceId }) {
  const face = useStore((s) => s.boxSpec.faces[faceId]);
  const patchFaceFrame = useStore((s) => s.patchFaceFrame);

  if (!face) {
    return <div style={{ opacity: 0.6, padding: 12 }}>No frame on this face.</div>;
  }
  const f = face.frame;

  const setNum = <K extends keyof FrameSpec>(key: K, value: number) => {
    log('frame_param_changed', { faceId, key, value });
    patchFaceFrame(faceId, { [key]: value } as Partial<FrameSpec>);
  };

  // band_um === null is a real API value (plates.FrameSpecBody band_um is
  // `float | None`), so auto is reachable again after any drag — the slider
  // alone can only ever write a concrete width.
  const setBandAuto = () => {
    log('frame_param_changed', { faceId, key: 'band_um', value: null });
    patchFaceFrame(faceId, { band_um: null });
  };

  return (
    <Section title="Frame" testId="frame-controls" persistId="frame">
      <div title="Wreath lays an ordered garland along a guiding vine (deliberate, symmetric); space colonization grows an organic scattered mat into the band.">
        <SelectRow
          label="Algorithm"
          value={f.algorithm}
          options={[
            { value: 'wreath', label: 'Laurel wreath' },
            { value: 'colonize', label: 'Space colonization' },
          ]}
          onChange={(v) =>
            patchFaceFrame(faceId, { algorithm: v as FrameSpec['algorithm'] })
          }
        />
      </div>

      {f.algorithm === 'wreath' && (
        <div title="Composition preset: rank spacing, leaf size/overlap, bloom frequency and whether the fill is a continuous rank or discrete rosettes.">
          <SelectRow
            label="Wreath style"
            value={f.wreath_style}
            options={[
              { value: 'garland2', label: 'Garland (lush tropical)' },
              { value: 'laurel', label: 'Laurel (classic rank)' },
              { value: 'garland', label: 'Garland (loose mixed)' },
              { value: 'clusters', label: 'Clusters (spaced rosettes)' },
            ]}
            onChange={(v) =>
              patchFaceFrame(faceId, { wreath_style: v as FrameSpec['wreath_style'] })
            }
          />
        </div>
      )}

      <div title="Motif species mix for the band. Esmeralda is the only theme shipped: Colombian flora — orchid-weighted blooms with coffee/heliconia/anthurium, fern and palm fronds against broad blades.">
        <SelectRow
          label="Theme"
          value={f.theme}
          options={[{ value: 'esmeralda', label: 'Esmeralda (Colombian)' }]}
          onChange={(v) => patchFaceFrame(faceId, { theme: v as FrameSpec['theme'] })}
        />
      </div>

      <div title="How tightly the foliage packs: leaf-station spacing along the wreath vine (colonize: the attractor field). Higher = busier, denser gold in the band.">
        <SliderRow
          label="Density"
          value={f.density}
          min={0.4}
          max={1.8}
          step={0.05}
          decimals={2}
          onChange={(v) => setNum('density', v)}
        />
      </div>

      <div title="Flower presence — how often and how large the blooms (orchid/coffee/heliconia) sit in the rank and the corner medallions.">
        <SliderRow
          label="Bloom"
          value={f.bloom}
          min={0}
          max={1.6}
          step={0.05}
          decimals={2}
          onChange={(v) => setNum('bloom', v)}
        />
      </div>

      <div title="Leaf mass — leaf length and the small-leaf understory pass, i.e. greenery versus bare vine.">
        <SliderRow
          label="Foliage"
          value={f.foliage}
          min={0}
          max={1.4}
          step={0.05}
          decimals={2}
          onChange={(v) => setNum('foliage', v)}
        />
      </div>

      <div title="Width of the decorated perimeter band, inward from the plate rim. Auto = ~12% of the shorter face side (13% for the wreath), so it tracks box-dimension changes.">
        <SliderRow
          label={f.band_um == null ? 'Frame band (auto)' : 'Frame band'}
          value={f.band_um ?? Math.round(0.12 * Math.min(face.width_um, face.height_um))}
          min={500}
          max={6000}
          step={100}
          unit="μm"
          decimals={0}
          onChange={(v) => setNum('band_um', v)}
        />
        <div style={{ marginBottom: 8 }}>
          <Button
            testId="frame-band-auto"
            title="Back to auto width — the band re-derives from the face size on every box-dimension change"
            active={f.band_um == null}
            disabled={f.band_um == null}
            onClick={setBandAuto}
          >
            Auto
          </Button>
        </div>
      </div>

      <div title="Deterministic RNG seed for this face's frame: vine routing, leaf placement, corner compositions — and the moiré carrier base angle ((seed × 17) mod 180°), which is why every face reads differently. Same seed = same frame.">
        <NumberRow
          label="Seed"
          value={f.seed}
          onChange={(v) => setNum('seed', Math.floor(v))}
          testId="frame-seed"
        />
      </div>
    </Section>
  );
}
