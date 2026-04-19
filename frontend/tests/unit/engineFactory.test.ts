/**
 * Engines must each set `uUseFft` to the correct value on activate/deactivate.
 * Past regression: switching engines without calling deactivate left stale
 * atlas textures bound to the material.
 */
import { describe, it, expect, vi } from 'vitest';
import { StylizedEngine } from '../../src/engines/StylizedEngine';
import { FraunhoferEngine } from '../../src/engines/FraunhoferEngine';
import { WavePropEngine } from '../../src/engines/WavePropEngine';

// Minimal stand-in for THREE.ShaderMaterial's shape used by engines.
function makeCtx() {
  return {
    scene: {} as any,
    plateMesh: {} as any,
    material: {
      uniforms: {
        uFront: { value: null },
        uBack: { value: null },
        uFftAtlas: { value: null },
        uUseFft: { value: 0.0 },
      },
      needsUpdate: false,
    } as any,
    frontTex: {} as any,
    backTex: {} as any,
  };
}

const MANIFEST = {
  slug: 'wayuu-kanasu-moire',
  variant: 'abc123',
  name: '',
  description: '',
  tags: [],
  params: {},
  substrate: { thickness_um: 500, material: 'fused silica', n: 1.46 },
  extent_um: [2000, 2000] as [number, number],
  pixel_pitch_um: 0.5,
  min_feature_um: 2.0,
  extra: {},
  files: {
    front_png: '',
    back_png: '',
    front_svg: '',
    back_svg: '',
    thumbnail: '',
  },
};

describe('engines', () => {
  it('StylizedEngine forces uUseFft to 0 on activate', async () => {
    const ctx = makeCtx();
    ctx.material.uniforms.uUseFft.value = 1.0; // pretend a previous engine left it on
    await new StylizedEngine().activate(ctx, MANIFEST);
    expect(ctx.material.uniforms.uUseFft.value).toBe(0.0);
    expect(ctx.material.needsUpdate).toBe(true);
  });

  it('FraunhoferEngine.deactivate clears uUseFft', () => {
    const ctx = makeCtx();
    ctx.material.uniforms.uUseFft.value = 1.0;
    new FraunhoferEngine().deactivate(ctx);
    expect(ctx.material.uniforms.uUseFft.value).toBe(0.0);
  });

  it('WavePropEngine.deactivate clears uUseFft', () => {
    const ctx = makeCtx();
    ctx.material.uniforms.uUseFft.value = 1.0;
    new WavePropEngine().deactivate(ctx);
    expect(ctx.material.uniforms.uUseFft.value).toBe(0.0);
  });

  it('all three engines expose distinct ids matching EngineTier', () => {
    const ids = [new StylizedEngine().id, new FraunhoferEngine().id, new WavePropEngine().id];
    expect(new Set(ids).size).toBe(3);
    expect(ids).toEqual(expect.arrayContaining(['stylized', 'fraunhofer', 'waveprop']));
  });
});
