import * as THREE from 'three';
import type { Engine, EngineContext } from './Engine';
import { fftSim, type PatternManifest } from '../api';

export class FraunhoferEngine implements Engine {
  readonly id = 'fraunhofer' as const;
  readonly name = 'Tier 2 — Fraunhofer FFT composite';

  private atlasTex: THREE.Texture | null = null;

  async activate(ctx: EngineContext, m: PatternManifest): Promise<void> {
    const resp = await fftSim(m.slug, m.variant, [0.65, 0.55, 0.45]);
    const loader = new THREE.TextureLoader();
    const tex = await loader.loadAsync(resp.atlas_png);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.atlasTex = tex;
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
