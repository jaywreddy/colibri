import * as THREE from 'three';
import type { Engine, EngineContext } from './Engine';
import { propagateSim, type PatternManifest } from '../api';

export class WavePropEngine implements Engine {
  readonly id = 'waveprop' as const;
  readonly name = 'Tier 3 — Angular spectrum';

  private atlasTex: THREE.Texture | null = null;

  async activate(ctx: EngineContext, m: PatternManifest): Promise<void> {
    const resp = await propagateSim(
      m.slug,
      m.variant,
      [0.65, 0.55, 0.45],
      [-15, 0, 15]
    );
    const loader = new THREE.TextureLoader();
    const tex = await loader.loadAsync(resp.atlas_png);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.atlasTex = tex;
    // Reuse the FFT atlas channel in the shader; the composite is intensity
    // so it slots cleanly into the same halo path in laser/backlight modes.
    ctx.material.uniforms.uFftAtlas.value = tex;
    ctx.material.uniforms.uUseFft.value = 1.0;
    ctx.material.needsUpdate = true;
  }

  deactivate(ctx: EngineContext): void {
    ctx.material.uniforms.uUseFft.value = 0.0;
    if (this.atlasTex) {
      this.atlasTex.dispose();
      this.atlasTex = null;
    }
  }
}
