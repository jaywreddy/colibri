import { defaultBoxSpec, FOIL_TAPE_PRESETS_UM, type FoilFinish } from '../api';
import { copperTapeLengthCm, cutList, keepoutUm, overlapUm, FOIL_COLORS } from '../assembly';
import { log } from '../logger';
import { useStore } from '../store';
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

/** One-click box size presets (outer dims, um). */
const SIZE_PRESETS = [
  { id: 'ring-box', label: 'Ring box', w: 50000, d: 50000, h: 40000 },
  { id: 'compact', label: 'Compact', w: 45000, d: 45000, h: 35000 },
  { id: 'pendant', label: 'Pendant', w: 40000, d: 40000, h: 55000 },
  // Largest box whose 6 plates pack onto one 4-inch (100 mm) Si wafer with
  // 300 um dicing streets — the fab wafer-layout target (see
  // backend/app/export_wafer.py::solve_max_scale). H/W keeps the 0.8 ratio.
  { id: 'mini-wafer', label: 'Mini (wafer)', w: 28900, d: 28900, h: 23120 },
] as const;

/** Brass hinge tube OD presets (um). */
const TUBE_PRESETS = [
  { label: '3/32″', um: 2400 },
  { label: '1/8″', um: 3175 },
] as const;

const FINISHES: FoilFinish[] = ['bright', 'copper', 'gold', 'rose', 'patina', 'gunmetal'];

/** Grating-pitch presets (μm) for the fabricated back carrier + leaf louvre. */
const GRATING_PITCH_PRESETS = [4, 10, 20, 40, 60] as const;
/** Litho floor (2 μm line + 2 μm gap) and a sane upper bound for the custom field. */
const GRATING_PITCH_MIN_UM = 4;
const GRATING_PITCH_MAX_UM = 200;

/**
 * Left "Build" rail — the design toolbox: size presets, box dims + glass,
 * solder-joint (foil) settings, hinge settings, and the live cut list +
 * copper-tape estimate — all computed client-side from src/assembly.ts so
 * they update instantly.
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

  const gratingPitch = boxSpec.carrier_pitch_um;
  const activePitchPreset = GRATING_PITCH_PRESETS.find((p) => p === gratingPitch);

  const ko = keepoutUm(boxSpec);
  const ov = overlapUm(boxSpec);
  const cuts = cutList(boxSpec);
  const tapeCm = copperTapeLengthCm(boxSpec);

  const activeSizePreset = SIZE_PRESETS.find(
    (p) =>
      p.w === boxSpec.width_um && p.d === boxSpec.depth_um && p.h === boxSpec.height_um
  )?.id;
  const activeTube = TUBE_PRESETS.find((p) => p.um === boxSpec.hinge.tube_od_um)?.um;

  return (
    <div data-testid="build-panel">
      <Section title="Size presets" testId="section-size-presets">
        <ChipRow
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
      </Section>

      <Section title="Preview" testId="section-preview">
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
          Magnifies every procedural period on screen (preview only — no effect
          on the fab masks). The centerpiece switch/comb (60 μm) and ripple lanes
          (15 μm) are exact-fab and sub-pixel at 1×, reading as flat gold — raise
          the scale or zoom in to resolve them. The leaf/carrier fringes track the
          Grating pitch below.
        </div>
      </Section>

      <Section title="Pattern" testId="section-pattern">
        <ChipRow
          label="Grating pitch"
          chips={GRATING_PITCH_PRESETS.map((p) => ({
            value: p,
            label: `${p} μm`,
            testId: `grating-pitch-${p}`,
          }))}
          value={activePitchPreset}
          onSelect={(um) => {
            log('grating_pitch_changed', { carrier_pitch_um: um });
            patchBoxSpec({ carrier_pitch_um: um as number });
          }}
        />
        <Disclosure label="Advanced" testId="grating-pitch-advanced">
          <NumberRow
            label="Pitch (custom)"
            value={gratingPitch}
            min={GRATING_PITCH_MIN_UM}
            max={GRATING_PITCH_MAX_UM}
            step={1}
            unit="μm"
            onChange={(v) => {
              const clamped = Math.max(
                GRATING_PITCH_MIN_UM,
                Math.min(GRATING_PITCH_MAX_UM, v)
              );
              log('grating_pitch_changed', { carrier_pitch_um: clamped });
              patchBoxSpec({ carrier_pitch_um: clamped });
            }}
            testId="grating-pitch-custom"
          />
        </Disclosure>
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 2 }}>
          Sets the fringe spacing + tilt sensitivity of the REAL fabricated part
          (finer = livelier, with diffraction onset below ~5 μm). 4 μm = 2 μm
          lines at the litho floor. Drives the leaf / back-carrier family only —
          the switch & comb faces keep their own 60 μm barrier architecture.
        </div>
      </Section>

      <Section title="Box" testId="section-box">
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
          label="Glass thickness"
          value={boxSpec.glass.thickness_um / 1000}
          min={0.3}
          max={2.0}
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
      </Section>

      <Section title="Solder joints" testId="section-foil">
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

      <Section title="Hinge" testId="section-hinge">
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

      <Section title="Cut list" testId="section-cutlist">
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
                <td style={{ padding: '3px 4px', textTransform: 'capitalize' }}>{c.face}</td>
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
          Pattern keep-out: <b>{mm1(ko)} mm</b> per edge
          <br />
          (foil overlap {mm1(ov)} mm + safety {mm1(boxSpec.foil.safety_um)} mm)
        </div>
        <div
          data-testid="tape-length"
          style={{ marginTop: 4, fontSize: 11, opacity: 0.7, lineHeight: 1.5 }}
        >
          Copper tape ≈ <b>{tapeCm.toFixed(1)} cm</b> (sum of plate perimeters)
        </div>
        {validationErrors.length > 0 && (
          <div
            data-testid="validation-errors"
            style={{ marginTop: 8, color: KIT.error, fontSize: 11, lineHeight: 1.4 }}
          >
            {validationErrors.map((e, i) => (
              <div key={i}>⚠ {e}</div>
            ))}
          </div>
        )}
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
