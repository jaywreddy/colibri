import * as THREE from 'three';
import type { PatternManifest } from '../api';

export type EngineContext = {
  scene: THREE.Scene;
  plateMesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  frontTex: THREE.Texture;
  backTex: THREE.Texture;
};

export interface Engine {
  readonly id: 'stylized' | 'fraunhofer' | 'waveprop';
  readonly name: string;
  activate(ctx: EngineContext, manifest: PatternManifest): Promise<void>;
  deactivate(ctx: EngineContext): void;
}
