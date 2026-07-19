import * as THREE from 'three';

/**
 * Organic solder-bead geometry.
 *
 * A real hand-flowed solder seam is not a uniform cylinder: it is a convex
 * fillet with low-frequency radius undulation along its length, rounded blobby
 * ends where the iron lifted, and fatter accumulations where seams meet at a
 * corner. This builds a tube-of-revolution BufferGeometry whose per-ring radius
 * is modulated by a seeded noise function so it never flickers across rebuilds.
 *
 * Efficiency: radialSeg ~ 14, lengthSeg scales with length but is capped, so a
 * typical 8-seam box stays well under the ~60k-triangle budget.
 */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 1D smooth noise from a random control array, sampled with cubic smoothstep. */
function makeProfile(seed: number, n: number): (t: number) => number {
  const ctrl = new Float32Array(n);
  const rnd = mulberry32(seed);
  for (let i = 0; i < n; i++) ctrl[i] = rnd() * 2 - 1; // -1..1
  return (t: number): number => {
    const f = t * (n - 1);
    const i0 = Math.max(0, Math.min(n - 1, Math.floor(f)));
    const i1 = Math.min(n - 1, i0 + 1);
    const frac = f - i0;
    const s = frac * frac * (3 - 2 * frac);
    return ctrl[i0] + (ctrl[i1] - ctrl[i0]) * s;
  };
}

export type BeadParams = {
  /** Base bead radius (mm). */
  radius: number;
  /** Seam length (mm). */
  length: number;
  /** Deterministic seed (derive from seam id). */
  seed: number;
  /** Radial segments (default 14). */
  radialSeg?: number;
  /** Undulation amplitude as a fraction of radius (default 0.08). */
  undulation?: number;
};

/**
 * Build a bead centred at the origin, running along +Y (matches the existing
 * CylinderGeometry axis so the caller's rotations still apply). Includes
 * spherical end caps that bulge slightly past the seam ends (blobby lifts).
 *
 * The cross-section is a slightly asymmetric fillet: the bead is fatter on the
 * outward side (local +X) than the embedded side, approximating a fillet that
 * wicks up the two plate faces.
 */
export function makeBeadGeometry(p: BeadParams): THREE.BufferGeometry {
  const radius = p.radius;
  const length = p.length;
  const radialSeg = p.radialSeg ?? 14;
  const und = p.undulation ?? 0.08;

  // length segments: ~1 per 0.35mm, clamped 10..40
  const lengthSeg = Math.max(10, Math.min(40, Math.round(length / 0.35)));
  const capSeg = 4; // rings per hemispherical cap

  const profile = makeProfile(p.seed, 7);
  const blob = makeProfile(p.seed ^ 0x5bd1, 5); // extra low-freq lumps

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  // rings: [bottom cap][body][top cap]; each ring is a y + a radius scale.
  type Ring = { y: number; scale: number };
  const rings: Ring[] = [];

  // bottom hemispherical cap (y from -length/2 - radius up to -length/2)
  for (let i = 0; i <= capSeg; i++) {
    const a = (i / capSeg) * (Math.PI / 2); // 0..90deg
    const y = -length / 2 - radius * Math.cos(a);
    const scale = Math.sin(a);
    rings.push({ y, scale });
  }
  // body
  for (let i = 0; i <= lengthSeg; i++) {
    const t = i / lengthSeg;
    const y = -length / 2 + t * length;
    // undulation: two-octave profile so it wobbles a few percent
    const u = profile(t) * 0.6 + blob(t) * 0.4;
    const scale = 1 + u * und;
    rings.push({ y, scale });
  }
  // top hemispherical cap
  for (let i = 1; i <= capSeg; i++) {
    const a = (i / capSeg) * (Math.PI / 2);
    const y = length / 2 + radius * Math.sin(a);
    const scale = Math.cos(a);
    rings.push({ y, scale });
  }

  const ringCount = rings.length;
  for (let ri = 0; ri < ringCount; ri++) {
    const ring = rings[ri];
    for (let s = 0; s <= radialSeg; s++) {
      const theta = (s / radialSeg) * Math.PI * 2;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      // Asymmetric fillet: fatter toward outward local +X, thinner on the
      // embedded (-Y-ish, here local -X used as "into the joint") side.
      const asym = 1 + 0.18 * cos; // widen +X, narrow -X
      const r = radius * ring.scale * asym;
      const x = cos * r;
      const z = sin * r;
      positions.push(x, ring.y, z);
      // approximate normal (radial, cap rings get a y component)
      const ny =
        ri <= capSeg
          ? -Math.max(0, 1 - ring.scale) // bottom cap points down
          : ri >= ringCount - capSeg
            ? Math.max(0, 1 - ring.scale)
            : 0;
      const nrm = new THREE.Vector3(cos * asym, ny * 1.2, sin).normalize();
      normals.push(nrm.x, nrm.y, nrm.z);
      // UVs: u wraps around the bead a few times (blotches around the tube);
      // v tiles along the LENGTH proportional to physical distance so the
      // solder blotches stay roughly square instead of smearing into stripes.
      const circumference = 2 * Math.PI * radius;
      const uTiles = Math.max(1, Math.round(circumference / 1.2));
      const vTiles = Math.max(1, (ring.y + length / 2 + radius) / 1.2);
      uvs.push((s / radialSeg) * uTiles, vTiles);
    }
  }

  const stride = radialSeg + 1;
  for (let ri = 0; ri < ringCount - 1; ri++) {
    for (let s = 0; s < radialSeg; s++) {
      const a = ri * stride + s;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  return geo;
}

/**
 * Corner junction blob — a lumpy sphere placed where seams meet, so the 8 box
 * corners read as accumulated solder rather than clean rod ends.
 */
export function makeCornerBlob(radius: number, seed: number): THREE.BufferGeometry {
  const geo = new THREE.IcosahedronGeometry(radius, 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const rnd = mulberry32(seed);
  // Deterministic per-vertex lump: perturb along the normal by up to +/-18%.
  const offsets: number[] = [];
  const vcount = pos.count;
  for (let i = 0; i < vcount; i++) offsets.push(rnd() * 2 - 1);
  const v = new THREE.Vector3();
  for (let i = 0; i < vcount; i++) {
    v.fromBufferAttribute(pos, i);
    const lump = 1 + offsets[i] * 0.18;
    v.multiplyScalar(lump);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}
