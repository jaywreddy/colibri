import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as api from '../../src/api';
import * as P from '../../src/production';

function mockFetchOk(body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  });
}

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetchOk({}));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('api', () => {
  it('listPatterns GETs /patterns', async () => {
    const spy = mockFetchOk([{ slug: 'photo-halftone' }]);
    vi.stubGlobal('fetch', spy);
    const out = await api.listPatterns();
    expect(spy).toHaveBeenCalledWith('/patterns');
    expect(out).toEqual([{ slug: 'photo-halftone' }]);
  });

  it('throws on non-ok response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}),
        text: async () => 'boom',
      })
    );
    await expect(api.listPatterns()).rejects.toThrow(/500/);
  });

  it('generateBox POSTs the v2 spec envelope with box_id and force', async () => {
    const spy = mockFetchOk({ kind: 'box', id: 'x' });
    vi.stubGlobal('fetch', spy);
    await api.generateBox(api.defaultBoxSpec(), { boxId: 'my-box', force: true });
    const [url, opts] = spy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/boxes/generate');
    expect(opts.method).toBe('POST');
    const body = JSON.parse(opts.body as string);
    expect(body.box_id).toBe('my-box');
    expect(body.force).toBe(true);
    expect(body.width_um).toBe(32000);
    expect(body.glass.n).toBe(1.4585);
    expect(body.foil.tape_width_um).toBe(6350);
    expect(body.hinge.style).toBe('tube');
    expect(Object.keys(body.faces)).toHaveLength(6);
  });
});

describe('defaultBoxSpec (contract defaults)', () => {
  it('matches the PRODUCTION box exactly (= backend default_box_spec)', () => {
    const s = api.defaultBoxSpec();
    expect(s.width_um).toBe(32000);
    expect(s.depth_um).toBe(32000);
    expect(s.height_um).toBe(35000);
    // Six single plies (2026-09-16): no bond — the flag and its two-ply math
    // are gone — 1/4" tape, and the art rim pinned at the 3.64 mm the
    // 15 September plate was written with.
    expect('bonded' in s).toBe(false);
    expect(s.art_rim_um).toBe(P.ART_RIM_UM);
    expect(s.glass).toEqual({ thickness_um: 2250, material: 'fused quartz', n: 1.4585 });
    expect(s.foil).toEqual({
      tape_width_um: 6350,
      safety_um: 500,
      bead_um: 2000,
      finish: 'bright',
    });
    expect(s.hinge).toEqual({
      style: 'tube',
      tube_od_um: 2400,
      rod_od_um: 1600,
      segments: 5,
      coverage: 0.8,
    });
    // The eye-sized carrier (backend witness_geom.BOX_CARRIER_UM, 0.75' at
    // 300 mm) — not the 22 um design pitch, and not the 99 um the old 'gap'
    // scaling produced. It is stamped onto every face by normalize_face_dims,
    // so a drift here silently re-pitches all six gratings.
    expect(s.carrier_pitch_um).toBe(65.5);
    expect(P.CARRIER_UM).toBe(65.5);
    expect(s.metal).toBe('gold');
    expect(s.label).toBe('');
  });

  it('applies the production six-face plan with per-face frame seeds', () => {
    // Production plan: top = J+P monogram, front = the Atlantic globe,
    // left/right/back = three halftone photographs, bottom = solid gold. Every
    // wall is SINGLE-PLY. Per-face frame profiles stay seeded 100..105 in
    // profile order, at the production motif scale.
    const s = api.defaultBoxSpec();
    const expectedSeed: Record<string, number> = {
      front: 100, back: 101, top: 102, bottom: 103, left: 104, right: 105,
    };
    // The band-composition dials of backend boxes._FACE_FRAME_PROFILE. The seed
    // alone does not make a face distinct — these four do, and a face that
    // silently fell back to the FrameSpec defaults would still pass every other
    // assertion here while composing the wrong border.
    const expectedProfile: Record<string, [number, number, number, number]> = {
      front: [0.75, 0.9, 1.2, 1.1],
      back: [1.0, 0.6, 0.9, 0.8],
      top: [0.55, 1.05, 1.35, 1.25],
      bottom: [0.9, 0.75, 1.0, 0.9],
      left: [0.65, 1.0, 1.25, 1.15],
      right: [0.85, 0.8, 1.05, 0.95],
    };
    expect(s.faces.front!.pattern_slug).toBe('globe-atlantic');
    expect(s.faces.top!.pattern_slug).toBe('monogram-jp');
    expect(s.faces.back!.pattern_slug).toBe('photo-halftone');
    expect(s.faces.bottom!.pattern_slug).toBe('solid-gold');
    expect(s.faces.left!.pattern_slug).toBe('photo-halftone');
    expect(s.faces.right!.pattern_slug).toBe('photo-halftone');
    expect(s.faces.left!.pattern_params).toEqual({ image: 'beach', colour_mode: 'authored' });
    expect(s.faces.right!.pattern_params).toEqual({ image: 'sunset', colour_mode: 'authored' });
    expect(s.faces.back!.pattern_params).toEqual({ image: 'paris', colour_mode: 'authored' });
    api.FACE_IDS.forEach((fid) => {
      const f = s.faces[fid]!;
      expect(f.frame.seed).toBe(expectedSeed[fid]);
      expect(f.frame.motif_scale).toBe(0.68);
      expect(f.frame.band_um).toBe(2400);
      expect([
        f.frame.edge_gradient, f.frame.understory,
        f.frame.border_vine, f.frame.corner_fans,
      ]).toEqual(expectedProfile[fid]);
      expect(f.glass).toEqual(s.glass);
      // 'fixed' means carrier_pitch_um IS the fabricated pitch. Under 'gap'
      // the compositor multiplies it by the glass's parallax ratio (x4.5 on
      // 2.25 mm quartz) and the 65.5 um carrier lands at 295 um.
      expect(f.carrier_scale_mode).toBe('fixed');
      expect(f.carrier_pitch_um).toBe(65.5);
      // EVERY wall is one written ply over a bare inner ply (2026-09-15).
      expect(f.single_ply).toBe(true);
      // Stamped front-art rim (bondedArtKeepoutUm). The bonded 2.25 mm plies
      // make the 3/8" (9525 um) tape wrap a stepped edge that consumes 3p =
      // 6750 um, leaving a 1387.5 um fold; the foil rim is 1887.5 but the art
      // starts at the inner ply's window, 2250 + 1387.5 = 3637.5 from the edge,
      // so every front feature has the back carrier behind it.
      expect(f.weld_margin_um).toBe(3637.5);
      expect(f.back_margin_um).toBe(3637.5);
    });
  });
});

describe('PICKER_SLUGS (what the face picker may offer)', () => {
  it('is exactly the five constructions the production box is built from', () => {
    // The visualizer shows ONE box. A slug that is not one of its own
    // constructions must not be assignable to a wall, whatever a stale backend
    // still registers — so this is a whitelist, and it is pinned here.
    expect(api.PICKER_SLUGS).toEqual([
      'monogram-jp',
      'globe-atlantic',
      'photo-halftone',
      'solid-gold',
      'blank',
    ]);
  });

  it('covers every slug the default box actually uses', () => {
    const s = api.defaultBoxSpec();
    for (const fid of api.FACE_IDS) {
      expect(api.PICKER_SLUGS).toContain(s.faces[fid]!.pattern_slug);
    }
  });

  it('keeps the two-ply exemplars OUT of the picker', () => {
    // globe-duo-phase stays registered so the renderer's barrier-interlace
    // path (and the @effects suite) has a subject — but no production wall is
    // two-ply, so it is reachable only by setting the slug directly.
    for (const slug of api.DEV_EXEMPLAR_SLUGS) {
      expect(api.PICKER_SLUGS).not.toContain(slug);
    }
  });
});
