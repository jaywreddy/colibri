import { defaultBoxSpec, FOIL_TAPE_PRESETS_UM, type FoilFinish } from '../api';
import { copperTapeLengthCm, cutList, keepoutUm, overlapUm, FOIL_COLORS } from '../assembly';
import { log } from '../logger';
import { useStore } from '../store';
import { FACE_LABELS } from './FacesPanel';
import {
  Button,
  ChipRow,
  Disclosure,
  KIT,
  NumberRow,
  Section,
  SliderRow,
  Swatch,
  TextRow,
} from './kit';

const mm1 = (um: number): string => (um / 1000).toFixed(1);

/**
 * One-click box size presets (outer dims, um).
 *
 * `production` is the box that ships — the same numbers api.ts::defaultBoxSpec
 * and backend boxes.default_box_spec carry, so the rail opens with a preset
 * selected instead of reading as a custom size. The rest are exploratory.
 *
 * The retired 'Mini (wafer)' preset sized the box so its six plates packed onto
 * one 4-inch Si wafer; that flow went with export_wafer.py, and the plates are
 * diced out of a 5-inch mask now.
 */
const SIZE_PRESETS = [
  { id: 'production', label: 'Production', w: 32000, d: 32000, h: 35000 },
  { id: 'ring-box', label: 'Ring box', w: 50000, d: 50000, h: 40000 },
  { id: 'compact', label: 'Compact', w: 45000, d: 45000, h: 35000 },
  { id: 'pendant', label: 'Pendant', w: 40000, d: 40000, h: 55000 },
] as const;

/** Brass hinge tube OD presets (um). */
const TUBE_PRESETS = [
  { label: '3/32″', um: 2400 },
  { label: '1/8″', um: 3175 },
] as const;

const FINISHES: FoilFinish[] = ['bright', 'copper', 'gold', 'rose', 'patina', 'gunmetal'];

/**
 * Left "Build" rail — the box's physical envelope: 1 size (presets + dims +
 * glass), 2 solder joints, 3 hinge, then the view-only preview controls and
 * the live cut list + copper-tape estimate. Everything here is computed
 * client-side from src/assembly.ts so it updates instantly.
 *
 * The old "4 · Grating pitch" section is gone. The fabricated carrier pitch is
 * a PROCESS constant now (production.CARRIER_UM, 65.5 µm — the period
 * that subtends 0.75 arcmin at 300 mm so the lines stay invisible in hand),
 * stamped onto every face by defaultBoxSpec; offering a slider for it invited
 * re-pitching all six gratings of a designed part from a preview. The
 * barrier-switch tilt readout went with it: no production face is a barrier
 * interlace any more, so the number it printed described nothing on the box.
 *
 * Validation errors render in a non-collapsible strip at the very top: regen
 * halts while the spec is invalid (App.tsx::regen), so that message must never
 * be hideable behind a collapsed Section.
 */
export default function BuildPanel({ validationErrors }: { validationErrors: string[] }) {
  const boxSpec = useStore((s) => s.boxSpec);
  const setBoxSpec = useStore((s) => s.setBoxSpec);
  const patchBoxSpec = useStore((s) => s.patchBoxSpec);
  const patchGlass = useStore((s) => s.patchGlass);
  const patchFoil = useStore((s) => s.patchFoil);
  const patchHinge = useStore((s) => s.patchHinge);
  const patternScale = useStore((s) => s.patternScale);
  const setPatternScale = useStore((s) => s.setPatternScale);

  const ko = keepoutUm(boxSpec);
  const ov = overlapUm(boxSpec);
  // A PINNED rim (spec.art_rim_um — the production box keeps the 3.64 mm the
  // plate was written with) wins over the derived foil rim when it is larger.
  const derivedRim = ko;
  const artRim = Math.max(derivedRim, boxSpec.art_rim_um ?? 0);
  const cuts = cutList(boxSpec);
  const tapeCm = copperTapeLengthCm(boxSpec);

  const activeSizePreset = SIZE_PRESETS.find(
    (p) =>
      p.w === boxSpec.width_um && p.d === boxSpec.depth_um && p.h === boxSpec.height_um
  )?.id;
  const activeTube = TUBE_PRESETS.find((p) => p.um === boxSpec.hinge.tube_od_um)?.um;

  return (
    <div data-testid="build-panel">
      {validationErrors.length > 0 && (
        <div
          data-testid="validation-errors"
          role="alert"
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 2,
            margin: 12,
            padding: '8px 10px',
            border: `1px solid ${KIT.error}`,
            borderRadius: 4,
            // Sticky over a scrolling rail, so the backing has to be opaque:
            // the red tint sits on top of the panel color, not on the sections.
            background: `linear-gradient(rgba(255, 136, 136, 0.08), rgba(255, 136, 136, 0.08)), ${KIT.panel}`,
            color: KIT.error,
            fontSize: 11,
            lineHeight: 1.5,
          }}
        >
          <div style={{ letterSpacing: 1.5, marginBottom: 4 }}>
            INVALID SPEC — REGEN PAUSED
          </div>
          {validationErrors.map((e, i) => (
            <div key={i}>⚠ {e}</div>
          ))}
        </div>
      )}

      <Section title="1 · Size" testId="section-box" persistId="box">
        {/* Kept as its own testid'd wrapper: the size presets used to be a
            separate Section and selectors may still target them by that id. */}
        <div data-testid="section-size-presets">
          <ChipRow
            label="Presets"
            chips={SIZE_PRESETS.map((p) => ({
              value: p.id,
              label: p.label,
              testId: `size-preset-${p.id}`,
            }))}
            value={activeSizePreset}
            onSelect={(id) => {
              const p = SIZE_PRESETS.find((x) => x.id === id)!;
              log('size_preset_applied', { preset: id });
              patchBoxSpec({ width_um: p.w, depth_um: p.d, height_um: p.h });
            }}
          />
        </div>
        <SliderRow
          label="Width (X)"
          value={boxSpec.width_um / 1000}
          min={10}
          max={80}
          step={0.5}
          unit="mm"
          withNumber
          onChange={(mm) => patchBoxSpec({ width_um: Math.round(mm * 1000) })}
          testId="box-width"
        />
        <SliderRow
          label="Depth (Z)"
          value={boxSpec.depth_um / 1000}
          min={10}
          max={80}
          step={0.5}
          unit="mm"
          withNumber
          onChange={(mm) => patchBoxSpec({ depth_um: Math.round(mm * 1000) })}
          testId="box-depth"
        />
        <SliderRow
          label="Height (Y)"
          value={boxSpec.height_um / 1000}
          min={10}
          max={80}
          step={0.5}
          unit="mm"
          withNumber
          onChange={(mm) => patchBoxSpec({ height_um: Math.round(mm * 1000) })}
          testId="box-height"
        />
        <SliderRow
          label="Ply thickness"
          value={boxSpec.glass.thickness_um / 1000}
          min={0.3}
          // 3.0, not 2.0: the production box is 2.25 mm fused-quartz plies,
          // which the old ceiling could not even express.
          max={3.0}
          step={0.05}
          unit="mm"
          decimals={2}
          onChange={(mm) => patchGlass({ thickness_um: Math.round(mm * 1000) })}
          testId="glass-thickness"
        />
        <SliderRow
          label="Refractive index n"
          value={boxSpec.glass.n}
          min={1.3}
          max={1.8}
          step={0.01}
          onChange={(n) => patchGlass({ n })}
          testId="glass-n"
        />
        <TextRow
          label="Material"
          value={boxSpec.glass.material}
          onChange={(material) => patchGlass({ material })}
          testId="glass-material"
        />
        <ChipRow
          label="Litho metal (preview)"
          chips={(
            [
              { value: 'gold', label: 'Gold' },
              { value: 'chrome', label: 'Chrome' },
              { value: 'chrome-ar', label: 'Chrome AR' },
            ] as const
          ).map((m) => ({ ...m, testId: `metal-${m.value}` }))}
          value={boxSpec.metal ?? 'gold'}
          onSelect={(metal) => {
            log('metal_changed', { metal });
            patchBoxSpec({ metal: metal as 'gold' | 'chrome' | 'chrome-ar' });
          }}
        />
      </Section>

      <Section title="2 · Solder joints" testId="section-foil" persistId="foil">
        <ChipRow
          label="Copper foil tape"
          chips={FOIL_TAPE_PRESETS_UM.map((p) => ({
            value: p.um,
            label: p.label,
            testId: `foil-tape-${p.um}`,
          }))}
          value={boxSpec.foil.tape_width_um}
          onSelect={(um) => {
            log('foil_tape_changed', { tape_width_um: um });
            patchFoil({ tape_width_um: um });
          }}
        />
        <Disclosure label="Advanced" testId="foil-advanced">
          <NumberRow
            label="Tape width (custom)"
            value={boxSpec.foil.tape_width_um}
            min={2000}
            max={12000}
            step={1}
            unit="μm"
            onChange={(v) => patchFoil({ tape_width_um: v })}
            testId="foil-tape-custom"
          />
          <SliderRow
            label="Safety margin"
            value={boxSpec.foil.safety_um}
            min={0}
            max={2000}
            step={50}
            unit="μm"
            onChange={(v) => patchFoil({ safety_um: v })}
            testId="foil-safety"
          />
        </Disclosure>
        <SliderRow
          label="Solder bead Ø (preview)"
          value={boxSpec.foil.bead_um / 1000}
          min={0.5}
          max={4.0}
          step={0.1}
          unit="mm"
          onChange={(mm) => patchFoil({ bead_um: Math.round(mm * 1000) })}
          testId="foil-bead"
        />
        <div style={{ fontSize: 12, marginBottom: 4 }}>Finish</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          {FINISHES.map((f) => (
            <Swatch
              key={f}
              color={FOIL_COLORS[f]}
              label={f}
              active={boxSpec.foil.finish === f}
              onClick={() => patchFoil({ finish: f })}
              testId={`finish-swatch-${f}`}
            />
          ))}
        </div>
      </Section>

      <Section title="3 · Hinge" testId="section-hinge" persistId="hinge">
        <ChipRow
          label="Tube OD"
          chips={TUBE_PRESETS.map((p) => ({
            value: p.um,
            label: `${p.label} (${p.um} μm)`,
            testId: `hinge-tube-${p.um}`,
          }))}
          value={activeTube}
          onSelect={(um) => patchHinge({ tube_od_um: um })}
        />
        <Disclosure label="Custom tube" testId="hinge-tube-advanced">
          <NumberRow
            label="Tube OD (custom)"
            value={boxSpec.hinge.tube_od_um}
            min={1500}
            max={5000}
            step={50}
            unit="μm"
            onChange={(v) => patchHinge({ tube_od_um: v })}
            testId="hinge-tube-od"
          />
        </Disclosure>
        <NumberRow
          label="Rod OD"
          value={boxSpec.hinge.rod_od_um}
          min={800}
          max={3000}
          step={50}
          unit="μm"
          onChange={(v) => patchHinge({ rod_od_um: v })}
          testId="hinge-rod-od"
        />
        <ChipRow
          label="Segments"
          chips={[3, 5, 7].map((n) => ({
            value: n,
            label: String(n),
            testId: `hinge-seg-${n}`,
          }))}
          value={boxSpec.hinge.segments}
          onSelect={(n) => patchHinge({ segments: n })}
        />
        <SliderRow
          label="Coverage"
          value={boxSpec.hinge.coverage * 100}
          min={30}
          max={95}
          step={5}
          unit="%"
          decimals={0}
          onChange={(pct) => patchHinge({ coverage: pct / 100 })}
          testId="hinge-coverage"
        />
      </Section>

      <Section title="Preview (view only)" testId="section-preview" persistId="preview">
        <ChipRow
          label="Pattern scale"
          chips={[1, 2, 4, 8].map((s) => ({
            value: s,
            label: `${s}×`,
            testId: `pattern-scale-${s}`,
          }))}
          value={patternScale}
          onSelect={(s) => {
            log('pattern_scale_changed', { scale: s });
            setPatternScale(s as number);
          }}
        />
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 2 }}>
          Magnifies the procedurally drawn carrier/louvre family on the two-ply
          exemplar faces (preview only, no effect on the fab masks). Every
          production wall is a LITERAL raster of its own fabricated chrome, so
          this does nothing to it — zoom the camera instead.
        </div>
        <div
          data-testid="pattern-scale-centerpiece-note"
          style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 6 }}
          title="plate.frag::uPatternScale — the centerpiece barrier pitch is excluded from the scale multiplier on purpose."
        >
          The exemplar centerpiece barrier and its interlace lanes are{' '}
          <b>not</b> scaled: the T/n parallax gap does not scale with the
          barrier pitch p, so magnifying p would magnify the swap angle
          atan((p/4)/(T/n)) by the same factor and the preview would answer
          &ldquo;does a hand tilt perform the swap?&rdquo; wrong by 2&ndash;8&times;.
          They stay at exact fab pitch, so at 1&times; they are sub-pixel and
          deliberately fade to flat gold — only <b>camera zoom</b> (scroll in) or
          a higher device pixel ratio resolves them.
        </div>
      </Section>

      <Section title="Cut list" testId="section-cutlist" persistId="cutlist">
        <table
          data-testid="cut-list"
          style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}
        >
          <thead>
            <tr style={{ opacity: 0.6, fontSize: 10, textAlign: 'left' }}>
              <th style={{ padding: '2px 4px' }}>PLATE</th>
              <th style={{ padding: '2px 4px', textAlign: 'right' }}>W (mm)</th>
              <th style={{ padding: '2px 4px', textAlign: 'right' }}>H (mm)</th>
            </tr>
          </thead>
          <tbody>
            {cuts.map((c) => (
              <tr
                key={c.face}
                data-testid={`cut-${c.face}`}
                style={{ borderTop: '1px solid #1a1e25' }}
              >
                <td style={{ padding: '3px 4px' }}>{FACE_LABELS[c.face]}</td>
                <td style={{ padding: '3px 4px', textAlign: 'right' }}>
                  {c.width_mm.toFixed(1)}
                </td>
                <td style={{ padding: '3px 4px', textAlign: 'right' }}>
                  {c.height_mm.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div
          data-testid="keepout-readout"
          style={{ marginTop: 8, fontSize: 11, opacity: 0.7, lineHeight: 1.5 }}
        >
          Pattern keep-out: <b>{mm1(artRim)} mm</b> per edge
          <br />
          (foil overlap {mm1(ov)} mm + safety {mm1(boxSpec.foil.safety_um)} mm
          {(boxSpec.art_rim_um ?? 0) > derivedRim
            ? `; pinned at ${mm1(boxSpec.art_rim_um!)} mm, the rim the plate was written with`
            : ''}
          )
        </div>
        <div
          data-testid="tape-length"
          style={{ marginTop: 4, fontSize: 11, opacity: 0.7, lineHeight: 1.5 }}
        >
          Copper tape ≈ <b>{tapeCm.toFixed(1)} cm</b> (sum of plate perimeters)
        </div>
      </Section>

      <div style={{ padding: 12 }}>
        <Button
          testId="reset-defaults"
          onClick={() => {
            log('box_reset');
            setBoxSpec(defaultBoxSpec());
          }}
          style={{ width: '100%' }}
        >
          Reset to defaults
        </Button>
      </div>
    </div>
  );
}
