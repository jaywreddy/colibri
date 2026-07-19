import { useStore } from '../store';
import { log } from '../logger';
import type { FaceId } from '../api';
import { CheckRow, Section, SelectRow, SliderRow } from './kit';

/**
 * Central-pattern parameter controls for one face of the box. Edits write to
 * the face's pattern_params in the store; App debounces the box regenerate.
 */
export default function ParameterPanel({ faceId }: { faceId: FaceId }) {
  const face = useStore((s) => s.boxSpec.faces[faceId]);
  const catalog = useStore((s) => s.catalog);
  const patchFace = useStore((s) => s.patchFace);

  if (!face) return null;
  const descriptor = catalog.find((c) => c.slug === face.pattern_slug);
  if (!descriptor || descriptor.params.length === 0) return null;

  const setParam = (name: string, value: unknown) => {
    log('face_param_changed', { faceId, name, value });
    patchFace(faceId, {
      pattern_params: { ...face.pattern_params, [name]: value },
    });
  };

  return (
    <Section title="Pattern parameters" testId="parameter-panel" persistId="pattern-params">
      {descriptor.params.map((p) => {
        const v =
          (face.pattern_params[p.name] as number | string | boolean | undefined) ??
          (p.default as number | string | boolean);
        if (p.type === 'choice') {
          return (
            <SelectRow
              key={p.name}
              label={p.unit ? `${p.label} (${p.unit})` : p.label}
              value={String(v)}
              options={(p.choices ?? []).map((c) => ({ value: c, label: c }))}
              onChange={(c) => setParam(p.name, c)}
            />
          );
        }
        if (p.type === 'bool') {
          return (
            <CheckRow
              key={p.name}
              label={p.label}
              checked={Boolean(v)}
              onChange={(checked) => setParam(p.name, checked)}
            />
          );
        }
        return (
          <SliderRow
            key={p.name}
            label={p.label}
            value={Number(v)}
            min={p.min ?? 0}
            max={p.max ?? 1}
            step={p.step ?? (p.type === 'int' ? 1 : 0.01)}
            unit={p.unit}
            decimals={p.type === 'int' ? 0 : 2}
            onChange={(raw) => setParam(p.name, p.type === 'int' ? Math.round(raw) : raw)}
          />
        );
      })}
    </Section>
  );
}
