import { useStore } from '../store';
import { log } from '../logger';
import type { FaceId, FrameSpec } from '../api';
import { NumberRow, Section, SelectRow, SliderRow } from './kit';

/**
 * Frame dials for one face — algorithm picker, theme picker, density/bloom/
 * foliage sliders, frame band, and the seed control. All edits go to
 * `patchFaceFrame`; App debounces regeneration of the box manifest. (The
 * one-click seed shuffle lives next to "Apply to all faces" in FaceEditor.)
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

  return (
    <Section title="Frame" testId="frame-controls" persistId="frame">
      <SelectRow
        label="Algorithm"
        value={f.algorithm}
        options={[
          { value: 'wreath', label: 'Laurel wreath' },
          { value: 'colonize', label: 'Space colonization' },
        ]}
        onChange={(v) => patchFaceFrame(faceId, { algorithm: v as FrameSpec['algorithm'] })}
      />

      {f.algorithm === 'wreath' && (
        <SelectRow
          label="Wreath style"
          value={f.wreath_style}
          options={[
            { value: 'garland2', label: 'Garland (lush tropical)' },
            { value: 'laurel', label: 'Laurel (classic rank)' },
            { value: 'garland', label: 'Garland (loose mixed)' },
            { value: 'clusters', label: 'Clusters (spaced rosettes)' },
          ]}
          onChange={(v) => patchFaceFrame(faceId, { wreath_style: v as FrameSpec['wreath_style'] })}
        />
      )}

      <SelectRow
        label="Theme"
        value={f.theme}
        options={[{ value: 'esmeralda', label: 'Esmeralda (Colombian)' }]}
        onChange={(v) => patchFaceFrame(faceId, { theme: v as FrameSpec['theme'] })}
      />

      <SliderRow
        label="Density"
        value={f.density}
        min={0.4}
        max={1.8}
        step={0.05}
        decimals={2}
        onChange={(v) => setNum('density', v)}
      />

      <SliderRow
        label="Bloom"
        value={f.bloom}
        min={0}
        max={1.6}
        step={0.05}
        decimals={2}
        onChange={(v) => setNum('bloom', v)}
      />

      <SliderRow
        label="Foliage"
        value={f.foliage}
        min={0}
        max={1.4}
        step={0.05}
        decimals={2}
        onChange={(v) => setNum('foliage', v)}
      />

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

      <NumberRow
        label="Seed"
        value={f.seed}
        onChange={(v) => setNum('seed', Math.floor(v))}
        testId="frame-seed"
      />
    </Section>
  );
}
