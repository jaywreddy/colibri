/**
 * Client-side mirror of the Ring Box assembly contract (v2).
 *
 * Pure functions over a BoxSpec — no three.js, no React. The formulas here
 * are IDENTICAL to backend/app/assembly.py per the design contract; both
 * sides must produce the same numbers for the same spec. The 3D scene
 * (src/scene/BoxScene.tsx) and the cut-list UI (src/ui/BuildPanel.tsx) both
 * consume this module so geometry updates are instant and never wait on the
 * backend.
 *
 * All values are micrometers (um) unless suffixed _mm. Coordinates are
 * Y-up with the origin at the box center.
 */
import type { BoxSpec, FaceId, FoilFinish, PlateSpec } from './api';

/** Gap between hinge tube segments along the axis (um). */
export const SEGMENT_GAP_UM = 400;

/**
 * Minimum patternable aperture (um) a plate must keep after the keep-out rim
 * is subtracted from both sides — "a few mm" per the contract.
 */
export const MIN_APERTURE_UM = 3000;

/** Preview colors for the three solder/foil finish options. */
export const FOIL_COLORS: Record<FoilFinish, string> = {
  bright: '#c9ced6', // freshly flowed tin-lead / lead-free solder — cool silver
  copper: '#b87333', // bare copper foil left unsoldered — warm orange
  patina: '#34343a', // black sulfide patina treatment
  gold: '#e3b53b', // polished gold plating — rich yellow
  rose: '#c98a86', // rose-gold blush
  gunmetal: '#3a3f47', // dark blued gunmetal
};

/**
 * Micrometers -> millimeters rounded to 3 decimals, HALF AWAY FROM ZERO.
 *
 * Shared contract helper: `backend/app/assembly.py::round_mm3` mirrors this
 * formula term for term. Neither side may use its language's built-in
 * rounding — Python's `round()` is half-to-EVEN while JavaScript's
 * `Math.round()` is half-UP (and asymmetric for negatives), so the two
 * disagree on every dimension whose mm value lands exactly on a
 * half-thousandth: 24062.5 um is 24.062 mm to Python and 24.063 mm to JS,
 * and the manifest cut list then contradicts the BuildPanel readout.
 *
 * Because 1 um is exactly 0.001 mm we round the MICROMETER value to an
 * integer and divide once. That is deliberate: dividing first and
 * re-multiplying (`(um / 1000) * 1000`) reintroduces the quotient's
 * representation error and can floor a whole micrometer away (17.4 mm ->
 * 17399.999999999998 -> 17.399). One exact-integer division also lands on the
 * nearest double to n/1000 in both languages, so the fixture's
 * `toBe(width_mm)` equality holds bit for bit.
 *
 * Display formatting (toFixed) stays in the UI.
 */
export function roundMm3(valueUm: number): number {
  if (valueUm < 0) return -(Math.floor(-valueUm + 0.5) / 1000);
  return Math.floor(valueUm + 0.5) / 1000;
}

// -----------------------------------------------------------------------------
// Foil overlap + pattern keep-out
// -----------------------------------------------------------------------------

type FoilGlassSpec = Pick<BoxSpec, 'foil' | 'glass'> & { bonded?: boolean };

/** Effective wall thickness: one plate, or the bonded two-ply stack. */
export function wallUm(spec: Pick<BoxSpec, 'glass'> & { bonded?: boolean }): number {
  return spec.bonded ? 2 * spec.glass.thickness_um : spec.glass.thickness_um;
}

/**
 * Copper tape overlap onto each plate face: (tape_width - t) / 2, floored at 0.
 *
 * BONDED construction wraps a STEPPED edge — outer ply edge (p), the exposed
 * step ledge (p), inner ply edge (p) — so the tape consumes 3p before any
 * fold-over remains. Mirrors backend ``assembly.py::bonded_overlap_um``.
 */
export function overlapUm(spec: FoilGlassSpec): number {
  const consumed = spec.bonded ? 3 * spec.glass.thickness_um : spec.glass.thickness_um;
  return Math.max(0, (spec.foil.tape_width_um - consumed) / 2);
}

/** Pattern keep-out margin per edge: foil overlap + safety.
 * Bounds the FRONT artwork (foliage silhouette). Measured from the OUTER ply
 * edge in bonded construction. */
export function keepoutUm(spec: FoilGlassSpec): number {
  return overlapUm(spec) + spec.foil.safety_um;
}

/**
 * Back-carrier keep-out per edge: foil overlap ONLY (drops the safety margin).
 *
 * The back layer is a uniform moiré-carrier grating that should cover the whole
 * EXPOSED face — everything the folded foil doesn't physically hide — so the
 * shimmer reads edge-to-edge behind the front foliage. Only constraint: the
 * grating must not run under the foil. Mirrors backend
 * ``assembly.py::back_window_um``; always <= keepoutUm since safety >= 0.
 *
 * In BONDED construction this rim is measured from the INNER ply edge (the
 * interior fold starts there); note stampFaces converts it into the shared
 * outer frame by adding one ply (mirrors backend boxes.normalize_face_dims).
 */
export function backWindowUm(spec: FoilGlassSpec): number {
  return overlapUm(spec);
}

/**
 * FRONT-art rim of a BONDED face, from the OUTER ply edge: the larger of the
 * foil keep-out and the inner ply's carrier window seen in the outer frame
 * (one ply of ledge + the interior fold). Art between the two would sit over
 * copper tape with no back grating behind it. Mirrors backend
 * ``assembly.py::bonded_art_keepout_um``. Only meaningful when spec.bonded.
 */
export function bondedArtKeepoutUm(spec: FoilGlassSpec): number {
  return Math.max(keepoutUm(spec), spec.glass.thickness_um + backWindowUm(spec));
}

// -----------------------------------------------------------------------------
// Cut list — walls sit ON the bottom plate; lid rests on the wall rim.
// Local dims are width x height of each rectangular plate.
// -----------------------------------------------------------------------------

export type CutPlate = {
  face: FaceId;
  width_um: number;
  height_um: number;
  width_mm: number;
  height_mm: number;
};

export function cutList(spec: BoxSpec): CutPlate[] {
  // In BONDED construction these are the OUTER ply dims: the outer nested
  // shell is cut at the PLY thickness, so the same expressions apply verbatim
  // (mirrors backend face_cut_dims called with the ply). The inner plies come
  // from bondedCutList below.
  const t = spec.glass.thickness_um;
  const W = spec.width_um;
  const D = spec.depth_um;
  const H = spec.height_um;
  const mk = (face: FaceId, w: number, h: number): CutPlate => ({
    face,
    width_um: w,
    height_um: h,
    // Shared rounding rule — see roundMm3 / assembly.py::round_mm3.
    width_mm: roundMm3(w),
    height_mm: roundMm3(h),
  });
  return [
    mk('bottom', W, D),
    mk('top', W, D),
    mk('front', W, H - 2 * t),
    mk('back', W, H - 2 * t),
    mk('left', D - 2 * t, H - 2 * t),
    mk('right', D - 2 * t, H - 2 * t),
  ];
}

export type BondedCutPlate = CutPlate & {
  ply: 'outer' | 'inner';
  /** Inner ply only: per-edge inset relative to its outer ply (= one ply). */
  inset_um?: number;
};

/**
 * Full 12-plate bonded cut list — per face, outer ply then inner ply.
 *
 * The box is TWO NESTED shells each cut at the PLY thickness: outer at
 * (W, D, H), inner at (W - 2p, D - 2p, H - 2p). Every inner ply comes out
 * inset exactly one ply per edge, centered on its outer ply. Mirrors backend
 * ``assembly.py::bonded_cut_list``.
 */
export function bondedCutList(spec: BoxSpec): BondedCutPlate[] {
  const p = spec.glass.thickness_um;
  const outer = cutList(spec);
  const innerSpec: BoxSpec = {
    ...spec,
    width_um: spec.width_um - 2 * p,
    depth_um: spec.depth_um - 2 * p,
    height_um: spec.height_um - 2 * p,
  };
  const inner = new Map(cutList(innerSpec).map((c) => [c.face, c]));
  const out: BondedCutPlate[] = [];
  for (const o of outer) {
    out.push({ ...o, ply: 'outer' });
    const i = inner.get(o.face)!;
    out.push({ ...i, ply: 'inner', inset_um: p });
  }
  return out;
}

/**
 * Estimated copper-foil tape length: every plate edge is wrapped once, so
 * this is simply the sum of all 6 plate perimeters. Returned in centimeters
 * for the BuildPanel readout. Pure — no rounding beyond float math.
 */
export function copperTapeLengthCm(spec: BoxSpec): number {
  let totalUm = 0;
  for (const c of cutList(spec)) totalUm += 2 * (c.width_um + c.height_um);
  return totalUm / 10_000; // 1 cm = 10,000 um
}

/**
 * Return a copy of the spec with the box-level degrees of freedom stamped
 * into every face's PlateSpec: glass, cut-list width/height, and
 * weld_margin (= keep-out). Mirrors backend normalization so the spec the
 * frontend holds always agrees with what the backend will materialize.
 *
 * CONTRACT (matches backend ``boxes.py::normalize_face_dims``): a key that is
 * not one of the six cut-list faces is carried through UNMODIFIED, never
 * dropped. normalize_face_dims iterates FACE_IDS and leaves anything else in
 * ``self.faces`` untouched, and box_hash hashes the whole faces map — so
 * silently deleting a stray key here would make the same logical box hash and
 * serialize differently depending on which side normalized it last.
 */
export function stampFaces(spec: BoxSpec): BoxSpec {
  // Bonded: front art starts at the larger of the foil rim and the inner ply's
  // window (mirrors backend boxes.normalize_face_dims / bonded_art_keepout_um).
  const ko = spec.bonded ? bondedArtKeepoutUm(spec) : keepoutUm(spec);
  // Bonded: the back layer lives on the INNER ply, whose edge is already one
  // ply in — expressed in the shared outer-ply frame its window insets by
  // ply + interior fold (mirrors backend boxes.normalize_face_dims).
  const bw = spec.bonded
    ? spec.glass.thickness_um + backWindowUm(spec)
    : backWindowUm(spec);
  const cuts = new Map(cutList(spec).map((c) => [c.face, c]));
  // Keyed by string, not FaceId, because unknown keys survive the stamp.
  const faces: Record<string, PlateSpec> = {};
  for (const [fid, face] of Object.entries(spec.faces) as [FaceId, PlateSpec][]) {
    const cut = cuts.get(fid);
    if (!cut) {
      // Not a cut-list face: preserve it verbatim (backend leaves it unstamped
      // but present). Stamping it is impossible — it has no cut dims.
      faces[fid] = face;
      continue;
    }
    faces[fid] = {
      ...face,
      glass: { ...spec.glass },
      width_um: cut.width_um,
      height_um: cut.height_um,
      weld_margin_um: ko,
      // Back carrier grating covers the whole exposed face (foil overlap only).
      back_margin_um: bw,
      // Grating pitch is a box-level choice — stamp it onto every face (mirrors
      // backend boxes.normalize_face_dims). NOT an assembly formula.
      carrier_pitch_um: spec.carrier_pitch_um ?? 22.0,
    };
  }
  // Cast back to the six-face record: the widened key type exists only so the
  // unknown-key passthrough above is expressible.
  return { ...spec, faces: faces as BoxSpec['faces'] };
}

// -----------------------------------------------------------------------------
// Assembly positions (Y up, origin at box center)
// -----------------------------------------------------------------------------

export type PlatePlacement = {
  face: FaceId;
  /** Plate center in box coordinates (um). */
  center_um: [number, number, number];
  /** Plate-local width (along local +X) in um. */
  width_um: number;
  /** Plate-local height (along local +Y) in um. */
  height_um: number;
  thickness_um: number;
  /**
   * Euler rotation (radians, XYZ order) mapping plate-local axes
   * (+X = width, +Y = height, +Z = outward normal) into box coordinates.
   */
  rotation: [number, number, number];
  /** Outward normal in box coordinates. */
  outward: [number, number, number];
};

export function platePlacements(spec: BoxSpec): PlatePlacement[] {
  // Placement centers sit at the mid-plane of the WALL (the bonded stack when
  // spec.bonded); local plate dims stay the OUTER ply cut dims — the nested
  // inner ply is concentric, so the stack's footprint IS the outer ply's.
  const t = wallUm(spec);
  const hw = spec.width_um / 2;
  const hd = spec.depth_um / 2;
  const hh = spec.height_um / 2;
  const cuts = new Map(cutList(spec).map((c) => [c.face, c]));
  const mk = (
    face: FaceId,
    center: [number, number, number],
    rotation: [number, number, number],
    outward: [number, number, number]
  ): PlatePlacement => {
    const cut = cuts.get(face)!;
    return {
      face,
      center_um: center,
      width_um: cut.width_um,
      height_um: cut.height_um,
      thickness_um: t,
      rotation,
      outward,
    };
  };
  const HALF_PI = Math.PI / 2;
  return [
    mk('bottom', [0, -hh + t / 2, 0], [HALF_PI, 0, 0], [0, -1, 0]),
    mk('top', [0, hh - t / 2, 0], [-HALF_PI, 0, 0], [0, 1, 0]),
    mk('front', [0, 0, hd - t / 2], [0, 0, 0], [0, 0, 1]),
    mk('back', [0, 0, -hd + t / 2], [0, Math.PI, 0], [0, 0, -1]),
    mk('left', [-hw + t / 2, 0, 0], [0, -HALF_PI, 0], [-1, 0, 0]),
    mk('right', [hw - t / 2, 0, 0], [0, HALF_PI, 0], [1, 0, 0]),
  ];
}

// -----------------------------------------------------------------------------
// Soldered seams — 8 beads (4 bottom + 4 vertical corners)
// -----------------------------------------------------------------------------

export type SeamSegment = {
  id: string;
  axis: 'x' | 'y' | 'z';
  start_um: [number, number, number];
  end_um: [number, number, number];
};

export function seamSegments(spec: BoxSpec): SeamSegment[] {
  const t = spec.glass.thickness_um;
  const hw = spec.width_um / 2;
  const hd = spec.depth_um / 2;
  const hh = spec.height_um / 2;
  const yBottom = -hh + t;
  const yTop = hh - t;
  return [
    // 4 bottom seams at y = -hh + t
    { id: 'bottom-front', axis: 'x', start_um: [-hw, yBottom, hd], end_um: [hw, yBottom, hd] },
    { id: 'bottom-back', axis: 'x', start_um: [-hw, yBottom, -hd], end_um: [hw, yBottom, -hd] },
    // left/right walls sit BETWEEN front and back (D - 2t wide), so their bottom
    // seams stop t short of each end; the corners belong to the corner seams
    // (mirrors assembly.py::seam_list).
    { id: 'bottom-left', axis: 'z', start_um: [-hw, yBottom, -hd + t], end_um: [-hw, yBottom, hd - t] },
    { id: 'bottom-right', axis: 'z', start_um: [hw, yBottom, -hd + t], end_um: [hw, yBottom, hd - t] },
    // 4 vertical corner seams, y in [-hh + t, hh - t]
    { id: 'corner-front-left', axis: 'y', start_um: [-hw, yBottom, hd], end_um: [-hw, yTop, hd] },
    { id: 'corner-front-right', axis: 'y', start_um: [hw, yBottom, hd], end_um: [hw, yTop, hd] },
    { id: 'corner-back-left', axis: 'y', start_um: [-hw, yBottom, -hd], end_um: [-hw, yTop, -hd] },
    { id: 'corner-back-right', axis: 'y', start_um: [hw, yBottom, -hd], end_um: [hw, yTop, -hd] },
  ];
}

// -----------------------------------------------------------------------------
// Hinge — brass tube-and-rod along the back top edge
// -----------------------------------------------------------------------------

export type HingeSegment = {
  owner: 'body' | 'lid';
  center_x_um: number;
  length_um: number;
};

export type HingeLayout = {
  tube_r_um: number;
  rod_r_um: number;
  /** Hinge axis: parallel to X at (axis_y, axis_z). */
  axis_y_um: number;
  axis_z_um: number;
  run_length_um: number;
  segment_length_um: number;
  gap_um: number;
  rod_length_um: number;
  segments: HingeSegment[];
};

export function hingeLayout(spec: BoxSpec): HingeLayout {
  // The hinge axis rides the lid's mid-thickness — the bonded stack's when
  // spec.bonded. (Backend hinge_layout carries no axis; frontend-only.)
  const t = wallUm(spec);
  const r = spec.hinge.tube_od_um / 2;
  const n = spec.hinge.segments;
  const L = spec.hinge.coverage * spec.width_um;
  const segLen = (L - (n - 1) * SEGMENT_GAP_UM) / n;
  const segments: HingeSegment[] = [];
  for (let i = 0; i < n; i++) {
    segments.push({
      // Alternating body,lid,body,... — both ends body (n is odd).
      owner: i % 2 === 0 ? 'body' : 'lid',
      center_x_um: -L / 2 + segLen / 2 + i * (segLen + SEGMENT_GAP_UM),
      length_um: segLen,
    });
  }
  return {
    tube_r_um: r,
    rod_r_um: spec.hinge.rod_od_um / 2,
    axis_y_um: spec.height_um / 2 - t / 2,
    axis_z_um: -spec.depth_um / 2 - r,
    run_length_um: L,
    segment_length_um: segLen,
    gap_um: SEGMENT_GAP_UM,
    rod_length_um: L + 2 * spec.hinge.rod_od_um,
    segments,
  };
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/**
 * Validate that every plate keeps a positive patternable aperture after the
 * keep-out rim, and that the hinge layout is realizable. Returns a list of
 * human-readable errors; empty list = valid.
 *
 * MUST accept/reject exactly the same specs as the backend's
 * ``validate_assembly`` (app/assembly.py) — anything that passes here but
 * 400s on POST /boxes/generate is a contract bug.
 */
export function validateBox(spec: BoxSpec): string[] {
  if (spec.bonded) return validateBondedBox(spec);
  const errors: string[] = [];
  // Degenerate-input guards — mirror backend validate_assembly exactly:
  // non-positive glass/dims/tape and negative safety all reject there.
  const t = spec.glass.thickness_um;
  if (t <= 0) {
    errors.push(`Glass thickness must be positive (got ${t} um).`);
  }
  if (spec.width_um <= 0 || spec.depth_um <= 0 || spec.height_um <= 0) {
    errors.push(
      `Box dimensions must be positive (got W=${spec.width_um}, D=${spec.depth_um}, ` +
        `H=${spec.height_um} um).`
    );
  }
  // Walls are H - 2t tall and the left/right walls are D - 2t wide, so a box
  // shorter/shallower than two glass thicknesses has no walls at all. These
  // two predicates mirror backend validate_assembly verbatim. They are NOT
  // redundant with the non-positive-plate branch below: that one only agrees
  // by structural coincidence (H <= 2t happens to force a cut side <= 0), and
  // the coincidence breaks the moment either side's cut formula is edited.
  if (spec.height_um <= 2.0 * t) {
    errors.push(
      `Box height ${(spec.height_um / 1000).toFixed(1)} mm leaves no room for walls: ` +
        `walls are H - 2t = ${((spec.height_um - 2 * t) / 1000).toFixed(1)} mm tall with ` +
        `${(t / 1000).toFixed(1)} mm glass. Increase height or use thinner glass.`
    );
  }
  if (spec.depth_um <= 2.0 * t) {
    errors.push(
      `Box depth ${(spec.depth_um / 1000).toFixed(1)} mm leaves no room for the left/right ` +
        `walls between front and back (${((spec.depth_um - 2 * t) / 1000).toFixed(1)} mm). ` +
        `Increase depth or use thinner glass.`
    );
  }
  if (spec.foil.tape_width_um <= 0) {
    errors.push(`Foil tape width must be positive (got ${spec.foil.tape_width_um} um).`);
  }
  if (spec.foil.safety_um < 0) {
    errors.push(`Foil safety margin cannot be negative (got ${spec.foil.safety_um} um).`);
  }
  const ko = keepoutUm(spec);
  for (const cut of cutList(spec)) {
    const minSide = Math.min(cut.width_um, cut.height_um);
    if (minSide <= 0) {
      errors.push(`Plate '${cut.face}' has non-positive size — box too small for glass thickness.`);
      continue;
    }
    // Predicate arranged EXACTLY as backend validate_assembly writes it
    // (`min_side <= 2.0 * ko + MIN_APERTURE_UM`), never the algebraically
    // equal `minSide - 2 * ko <= MIN_APERTURE_UM`. With a keep-out that is not
    // binary-representable (any fractional tape width or glass thickness) the
    // two arrangements differ by an ULP right at the boundary — which is
    // exactly where a spec accepted here would 400 on POST /boxes/generate.
    // `aperture` below is display-only, never the decision.
    if (minSide <= 2.0 * ko + MIN_APERTURE_UM) {
      const aperture = minSide - 2.0 * ko;
      errors.push(
        `Plate '${cut.face}': patternable aperture ${(aperture / 1000).toFixed(1)} mm ` +
          `<= ${(MIN_APERTURE_UM / 1000).toFixed(1)} mm minimum ` +
          `(keep-out ${(ko / 1000).toFixed(1)} mm per edge).`
      );
    }
  }
  const { segments: n, coverage, tube_od_um, rod_od_um } = spec.hinge;
  if (n < 3 || n % 2 === 0) {
    errors.push(`Hinge segments must be odd and >= 3 (got ${n}).`);
  }
  if (!(coverage > 0 && coverage <= 1)) {
    errors.push(`Hinge coverage must be in (0, 1] of the box width (got ${coverage}).`);
  }
  if (rod_od_um >= tube_od_um) {
    errors.push(
      `Hinge rod OD (${(rod_od_um / 1000).toFixed(2)} mm) must be smaller than the tube OD ` +
        `(${(tube_od_um / 1000).toFixed(2)} mm) so the rod can pass through the tube.`
    );
  }
  if (n >= 3 && n % 2 === 1 && coverage > 0 && coverage <= 1) {
    const h = hingeLayout(spec);
    if (h.segment_length_um <= tube_od_um) {
      errors.push(
        `Hinge tube segments come out ${(h.segment_length_um / 1000).toFixed(2)} mm — ` +
          `shorter than the tube OD (${(tube_od_um / 1000).toFixed(2)} mm) and uncuttable. ` +
          'Reduce the segment count or increase hinge coverage.'
      );
    }
  }
  return errors;
}

/**
 * Bonded-construction validator — the two-ply sibling of validateBox.
 *
 * MUST accept/reject exactly the same specs as the backend's
 * ``validate_bonded_assembly`` (app/assembly.py): the inner nested shell must
 * exist (H > 4p, D > 4p), foil sanity matches the single-ply path, the
 * patternable aperture is checked on BOTH plies (front art on the outer ply
 * behind keepoutUm, back art on the inner ply behind backWindowUm), and the
 * hinge checks are shared. Predicates keep the backend's exact arrangement
 * (see the ULP note in validateBox).
 */
export function validateBondedBox(spec: BoxSpec): string[] {
  const errors: string[] = [];
  const p = spec.glass.thickness_um;
  if (p <= 0) {
    errors.push(`Ply thickness must be positive (got ${p} um).`);
  }
  if (spec.width_um <= 0 || spec.depth_um <= 0 || spec.height_um <= 0) {
    errors.push(
      `Box dimensions must be positive (got W=${spec.width_um}, D=${spec.depth_um}, ` +
        `H=${spec.height_um} um).`
    );
  }
  if (spec.height_um <= 4.0 * p) {
    errors.push(
      `Box height ${(spec.height_um / 1000).toFixed(1)} mm leaves no room for the bonded ` +
        `walls: the INNER wall ply is H - 4p = ${((spec.height_um - 4 * p) / 1000).toFixed(1)} mm ` +
        `tall with ${(p / 1000).toFixed(1)} mm plies. Increase height or use thinner stock.`
    );
  }
  if (spec.depth_um <= 4.0 * p) {
    errors.push(
      `Box depth ${(spec.depth_um / 1000).toFixed(1)} mm leaves no room for the bonded ` +
        `left/right walls: the INNER ply is D - 4p = ${((spec.depth_um - 4 * p) / 1000).toFixed(1)} mm ` +
        `wide with ${(p / 1000).toFixed(1)} mm plies. Increase depth or use thinner stock.`
    );
  }
  if (spec.foil.tape_width_um <= 0) {
    errors.push(`Foil tape width must be positive (got ${spec.foil.tape_width_um} um).`);
  }
  if (spec.foil.safety_um < 0) {
    errors.push(`Foil safety margin cannot be negative (got ${spec.foil.safety_um} um).`);
  }
  if (spec.foil.tape_width_um < 3.0 * p) {
    errors.push(
      `Foil tape ${(spec.foil.tape_width_um / 1000).toFixed(2)} mm is narrower than the ` +
        `${((3.0 * p) / 1000).toFixed(2)} mm stepped edge it must wrap (three plies of ` +
        `${(p / 1000).toFixed(2)} mm): nothing folds onto either face. Use wider tape ` +
        `(3/8″ for 2.25 mm plies) or thinner stock.`
    );
  }
  const ko = bondedArtKeepoutUm(spec); // front art: foil rim or inner-ply window, whichever is larger
  const bw = backWindowUm(spec); // from the INNER ply's own edge
  if (p > 0 && spec.height_um > 4 * p && spec.depth_um > 4 * p) {
    for (const cut of bondedCutList(spec)) {
      const rim = cut.ply === 'outer' ? ko : bw;
      const minSide = Math.min(cut.width_um, cut.height_um);
      // Same arrangement as the backend predicate — see validateBox's ULP note.
      if (minSide <= 2.0 * rim + MIN_APERTURE_UM) {
        errors.push(
          `Plate '${cut.face}' ${cut.ply} ply: patternable aperture ` +
            `${((minSide - 2.0 * rim) / 1000).toFixed(1)} mm <= ` +
            `${(MIN_APERTURE_UM / 1000).toFixed(1)} mm minimum ` +
            `(keep-out ${(rim / 1000).toFixed(1)} mm per edge).`
        );
      }
    }
  }
  const { segments: n, coverage, tube_od_um, rod_od_um } = spec.hinge;
  if (n < 3 || n % 2 === 0) {
    errors.push(`Hinge segments must be odd and >= 3 (got ${n}).`);
  }
  if (!(coverage > 0 && coverage <= 1)) {
    errors.push(`Hinge coverage must be in (0, 1] of the box width (got ${coverage}).`);
  }
  if (rod_od_um >= tube_od_um) {
    errors.push(
      `Hinge rod OD (${(rod_od_um / 1000).toFixed(2)} mm) must be smaller than the tube OD ` +
        `(${(tube_od_um / 1000).toFixed(2)} mm) so the rod can pass through the tube.`
    );
  }
  if (n >= 3 && n % 2 === 1 && coverage > 0 && coverage <= 1) {
    const h = hingeLayout(spec);
    if (h.segment_length_um <= tube_od_um) {
      errors.push(
        `Hinge tube segments come out ${(h.segment_length_um / 1000).toFixed(2)} mm — ` +
          `shorter than the tube OD (${(tube_od_um / 1000).toFixed(2)} mm) and uncuttable. ` +
          'Reduce the segment count or increase hinge coverage.'
      );
    }
  }
  return errors;
}
