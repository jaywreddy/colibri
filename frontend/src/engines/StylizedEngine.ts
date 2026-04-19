import type { Engine, EngineContext } from './Engine';
import type { PatternManifest } from '../api';

export class StylizedEngine implements Engine {
  readonly id = 'stylized' as const;
  readonly name = 'Tier 1 — Stylized parallax';

  async activate(ctx: EngineContext, _m: PatternManifest): Promise<void> {
    ctx.material.uniforms.uUseFft.value = 0.0;
    ctx.material.needsUpdate = true;
  }

  deactivate(_ctx: EngineContext): void {}
}
