import {
  defaultBoxSpec,
  FACE_IDS,
  FOIL_TAPE_PRESETS_UM,
  type BoxManifest,
  type FaceId,
  type FoilFinish,
} from '../api';
import { bondedArtKeepoutUm, copperTapeLengthCm, cutList, keepoutUm, overlapUm, FOIL_COLORS } from '../assembly';
import { log } from '../logger';
import { useStore } from '../store';
import { FACE_LABELS } from './FacesPanel';
import {
  Button,
  CheckRow,
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
 * Fab centerpiece barrier pitch (μm) — mirrors backend
 * plates.FAB_CENTER_PERIOD_UM. LAST-RESORT default only, for a manifest that
 * carries `switch_interlace` but neither period field; the readout otherwise
 * reports the number the plate itself published.
 */
const FAB_CENTER_PERIOD_UM = 60;

type SwitchTilt = {
  face: FaceId;
  /** Exterior tilt (deg) at which the A↔B swap peaks, at the DESIGN period. */
  deg: number;
  /** Design barrier/comb period (μm) the plate published for that face. */
  periodUm: number;
  /**
   * Period (μm) actually baked into front.svg when the lattice budget forced a
   * coarser barrier lattice (recipe_data `svg_bake_barrier_period_um`), and the
   * tilt that follows from it — null when the bake matched the design or has
   * not run yet for this plate. A coarsened comb swaps at a LARGER angle, so
   * this is the number a fab shop's part obeys.
   */
  bakedPeriodUm: number | null;
  bakedDeg: number | null;
  /** Paraxial back-plane depth T/n (μm). */
  gapUm: number;
  thicknessUm: number;
  n: number;
};

/** Exterior tilt (deg) that walks the back plane by p/4 across a T/n gap. */
const tiltDegFor = (periodUm: number, gapUm: number): number =>
  (Math.atan(periodUm / 4 / gapUm) * 180) / Math.PI;

/**
 * Effective barrier-switch tilt for the CURRENT box, or null when no composed
 * face is a barrier interlace.
 *
 * Physics: both images live in the back layer as alternating lanes of pitch
 * p/2, so the swap peaks once the back plane has walked half a lane — p/4 —
 * and the back gold sits at the paraxial T/n air gap under the front comb
 * (BoxScene puts the inner plane at exactly `outerZ - T/n`). Hence the tilt
 * that performs the swap is `atan((p/4) / (T/n))`. Defaults (p 60 μm, T 500 μm,
 * n 1.46 → gap 342.5 μm) give 2.5°, which is the number the shader comment on
 * uPatternScale quotes and the reason Pattern Scale must not touch p.
 *
 * `switch_interlace` is READ from recipe_data, never inferred from the slug.
 * SWITCH_INTERLACE_SLUGS lives in plates.py, and a second copy of that set on
 * this side is precisely how the preview lost its barrier registration before
 * (see plate.frag::uSwitchBarrierPhaseUm). The price is that the readout only
 * appears once the first box manifest has landed — which is also what makes it
 * honest: it describes a plate that exists, not a spec being typed.
 *
 * Face choice: whichever interlace face the user is most plausibly asking
 * about — the selected one, else the front, else the first in FACE_IDS order.
 */
function switchTilt(manifest: BoxManifest | null, selectedFaceId: FaceId): SwitchTilt | null {
  if (!manifest) return null;
  const face = [selectedFaceId, 'front' as FaceId, ...FACE_IDS].find((fid) =>
    Boolean(manifest.faces[fid]?.recipe_data?.switch_interlace)
  );
  if (!face) return null;
  const fm = manifest.faces[face]!;
  const rd = fm.recipe_data ?? {};
  // Same precedence the shader binds (BoxScene uCenterPeriodUm): on an
  // interlace face the comb and the lanes share ONE period by construction, and
  // switch_interlace_period_um is its canonical name.
  const periodUm =
    Number(rd.switch_interlace_period_um ?? rd.fab_center_period_um) || FAB_CENTER_PERIOD_UM;
  // What the fab SVG actually baked, if ensure_plate_svg had to snap the barrier
  // to a coarser lattice (plates.py::_SVG_BAKE_KEYS). Absent on a freshly
  // generated box — the bake runs on export — which is exactly why it is
  // reported as a SEPARATE number rather than silently replacing the design.
  const bakedRaw = Number(rd.svg_bake_barrier_period_um);
  const bakedPeriodUm =
    Number.isFinite(bakedRaw) && bakedRaw > 0 && Math.abs(bakedRaw - periodUm) > 0.01 * periodUm
      ? bakedRaw
      : null;
  // Per-face substrate, falling back to the box glass. The `?.` is defensive
  // against a manifest cached before substrate was stamped: this runs inside a
  // render, so a throw here would blank the whole Build rail.
  const glass = manifest.spec.glass;
  const thicknessUm = Number(fm.substrate?.thickness_um) || glass.thickness_um;
  const n = Number(fm.substrate?.n) || glass.n;
  const gapUm = thicknessUm / Math.max(1, n);
  return {
    face,
    deg: tiltDegFor(periodUm, gapUm),
    periodUm,
    bakedPeriodUm,
    bakedDeg: bakedPeriodUm === null ? null : tiltDegFor(bakedPeriodUm, gapUm),
    gapUm,
    thicknessUm,
    n,
  };
}

/**
 * Left "Build" rail — the design toolbox, ordered to match README's design
 * workflow: 1 size (presets + dims + glass), 2 solder joints, 3 hinge,
 * 4 grating pitch, then the view-only preview controls and the live cut list +
 * copper-tape estimate. Everything here is computed client-side from
 * src/assembly.ts so it updates instantly. Steps 5-6 of the workflow live
 * elsewhere (per-face patterns in the Faces rail, export in the header) — the
 * hint under step 4 points there.
 *
 * The "Preview (view only)" section also carries the one physics readout on
 * this rail: the effective barrier-switch tilt (see switchTilt), because the
 * question Pattern Scale invites — "will a hand tilt actually perform the
 * swap?" — has a number, and that number must come from the composed plate
 * rather than from the magnified thing on screen.
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
  // Read for the switch-tilt readout only — the effective swap angle comes from
  // the COMPOSED plate's recipe_data, not from the spec being typed.
  const boxManifest = useStore((s) => s.boxManifest);
  const selectedFaceId = useStore((s) => s.selectedFaceId);

  const gratingPitch = boxSpec.carrier_pitch_um;
  const tilt = switchTilt(boxManifest, selectedFaceId);
  const activePitchPreset = GRATING_PITCH_PRESETS.find((p) => p === gratingPitch);

  const ko = keepoutUm(boxSpec);
  const ov = overlapUm(boxSpec);
  // Bonded: the front art starts at the inner ply's window when that is
  // further in than the foil rim (mirrors assembly.bonded_art_keepout_um).
  const artRim = boxSpec.bonded ? bondedArtKeepoutUm(boxSpec) : ko;
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
          label={boxSpec.bonded ? 'Ply thickness' : 'Glass thickness'}
          value={boxSpec.glass.thickness_um / 1000}
          min={0.3}
          // 3.0, not 2.0: the production box is bonded 2.25 mm fused-quartz
          // plies, which the old ceiling could not even express.
          max={3.0}
          step={0.05}
          unit="mm"
          decimals={2}
          onChange={(mm) => patchGlass({ thickness_um: Math.round(mm * 1000) })}
          testId="glass-thickness"
        />
        <CheckRow
          label="Bonded 2-ply walls (bevel-step corners)"
          checked={!!boxSpec.bonded}
          onChange={(bonded) => patchBoxSpec({ bonded })}
          testId="glass-bonded"
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

      <Section title="4 · Grating pitch" testId="section-pattern" persistId="pattern-pitch">
        <ChipRow
          label="Pitch"
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
          Box-wide fringe spacing + tilt sensitivity of the REAL fabricated part
          (finer = livelier, with diffraction onset below ~5 μm). 4 μm = 2 μm
          lines at the litho floor. Drives the leaf / back-carrier family only —
          the switch & comb faces keep their own 60 μm barrier architecture.
        </div>
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 6 }}>
          Next: pick a pattern per face in the <b>Faces</b> rail (right) → check
          the open-lid preview under the scene → <b>Export fab bundle</b> (top
          right).
        </div>
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
          Magnifies the leaf/carrier fringe family only — frame back carrier,
          leaf louvre, capybara body shimmer — on both plate planes (preview
          only, no effect on the fab masks). Those periods track the Grating
          pitch above.
        </div>
        <div
          data-testid="pattern-scale-centerpiece-note"
          style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 6 }}
          title="plate.frag::uPatternScale — the centerpiece pitch is excluded from the scale multiplier on purpose."
        >
          The centerpiece switch/comb and its interlace &amp; ripple lanes are{' '}
          <b>not</b> scaled: the T/n parallax gap does not scale with the
          barrier pitch p, so magnifying p would magnify the swap angle
          atan((p/4)/(T/n)) by the same factor and the preview would answer
          &ldquo;does a hand tilt perform the swap?&rdquo; wrong by 2–8×. They
          stay at exact fab pitch, so at 1× they are sub-pixel and deliberately
          fade to flat gold — only <b>camera zoom</b> (scroll in) or a higher
          device pixel ratio resolves them.
        </div>
        {tilt ? (
          <div
            data-testid="switch-tilt-readout"
            title={
              `atan((p/4)/(T/n)) with p = ${tilt.periodUm} μm ` +
              `(recipe_data switch_interlace_period_um on the ${tilt.face} plate) and ` +
              `T/n = ${tilt.thicknessUm} μm / ${tilt.n} = ${tilt.gapUm.toFixed(1)} μm. ` +
              'Paraxial: the Snell-exact substrate walk (backend sim2d.tilt_for_shift_um) ' +
              'runs ~2% larger by 15°, so read this as ±0.1° near the default. It also ' +
              'describes the LAST COMPOSED plate — a spec edit still debouncing is not in ' +
              'it yet. Preview-only controls (Pattern scale, illumination) never change it.'
            }
            style={{ fontSize: 11, opacity: 0.75, lineHeight: 1.5, marginTop: 8 }}
          >
            {FACE_LABELS[tilt.face]} barrier switch: swaps at{' '}
            <b>≈ ±{tilt.deg.toFixed(1)}°</b> tilt
            <br />
            <span style={{ opacity: 0.75 }}>
              ({tilt.periodUm.toFixed(0)} μm comb ÷ 4 = {(tilt.periodUm / 4).toFixed(1)} μm
              back shift across the {tilt.gapUm.toFixed(0)} μm T/n gap)
            </span>
            {tilt.bakedDeg !== null && tilt.bakedPeriodUm !== null && (
              <div
                data-testid="switch-tilt-baked"
                style={{ color: KIT.error, marginTop: 4 }}
                title="recipe_data svg_bake_barrier_period_um — the barrier lattice ensure_plate_svg could actually resolve at this plate size. The design angle above is what the preview shows; this is what the exported mask would do."
              >
                ⚠ front.svg baked the comb at {tilt.bakedPeriodUm.toFixed(0)} μm →{' '}
                <b>≈ ±{tilt.bakedDeg.toFixed(1)}°</b> on the fabricated part
              </div>
            )}
          </div>
        ) : (
          boxManifest && (
            <div
              data-testid="switch-tilt-none"
              style={{ fontSize: 11, opacity: 0.5, lineHeight: 1.5, marginTop: 8 }}
            >
              No barrier-switch face on this box — assign one (colibrí ↔ globe,
              gear ↔ quill, wing flap) in the <b>Faces</b> rail to get its swap
              angle here.
            </div>
          )
        )}
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
          {boxSpec.bonded && artRim > ko
            ? `; art starts at the inner ply's window, ${mm1(boxSpec.glass.thickness_um)} mm ply + ${mm1(ov)} mm fold`
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
