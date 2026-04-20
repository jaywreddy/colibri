import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import vert from '../shaders/plate.vert';
import frag from '../shaders/plate.frag';
import { log } from '../logger';
import { useStore } from '../store';
import { StylizedEngine } from '../engines/StylizedEngine';
import { FraunhoferEngine } from '../engines/FraunhoferEngine';
import { WavePropEngine } from '../engines/WavePropEngine';
import type { Engine } from '../engines/Engine';

const ENGINES: Record<string, Engine> = {
  stylized: new StylizedEngine(),
  fraunhofer: new FraunhoferEngine(),
  waveprop: new WavePropEngine(),
};

export default function PlateScene() {
  const mountRef = useRef<HTMLDivElement>(null);
  const threeRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    mesh: THREE.Mesh;
    material: THREE.ShaderMaterial;
    light: THREE.Object3D;
    controls: OrbitControls;
    frontTex: THREE.Texture;
    backTex: THREE.Texture;
    currentEngine: Engine | null;
  } | null>(null);

  const manifest = useStore((s) => s.manifest);
  const illumination = useStore((s) => s.illumination);
  const laserColor = useStore((s) => s.laserColor);
  const engine = useStore((s) => s.engine);
  const lightAz = useStore((s) => s.lightAzimuthDeg);
  const lightEl = useStore((s) => s.lightElevationDeg);

  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.position.set(0, 0.4, 1.8);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    mount.appendChild(renderer.domElement);

    const geom = new THREE.PlaneGeometry(1.0, 1.0, 1, 1);
    const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    blank.needsUpdate = true;

    const material = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        uFront: { value: blank },
        uBack: { value: blank },
        uFftAtlas: { value: blank },
        uUseFft: { value: 0.0 },
        uWavelengthSlot: { value: 1 }, // green
        uExtentUm: { value: 2000.0 },
        uThicknessUm: { value: 500.0 },
        uN: { value: 1.46 },
        uIllumination: { value: 0 },
        uLaserColor: { value: new THREE.Color(0x33ff88) },
        uBacklightColor: { value: new THREE.Color(0xffffff) },
        uAmbientColor: { value: new THREE.Color(0xffffff) },
        uLightWorld: { value: new THREE.Vector3(2, 3, 3) },
      },
    });

    const mesh = new THREE.Mesh(geom, material);
    scene.add(mesh);

    const light = new THREE.Object3D();
    scene.add(light);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.6;
    controls.maxDistance = 4.0;

    // WebGL context loss is common in headless Chromium and on driver hiccups.
    // Logging both ends makes it possible to correlate a blank canvas in an
    // E2E failure with the actual cause.
    const canvas = renderer.domElement;
    const onContextLost = (e: Event) => {
      e.preventDefault();
      log('webgl_context_lost');
    };
    const onContextRestored = () => log('webgl_context_restored');
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);

    // Emit tilt_changed when the user orbits the camera. Throttled to 50ms so
    // a drag doesn't spam the ring buffer.
    let lastTiltLog = 0;
    const onControlsChange = () => {
      const now = performance.now();
      if (now - lastTiltLog < 50) return;
      lastTiltLog = now;
      const az = controls.getAzimuthalAngle();
      const pol = controls.getPolarAngle();
      log('tilt_changed', {
        az_deg: +((az * 180) / Math.PI).toFixed(2),
        polar_deg: +((pol * 180) / Math.PI).toFixed(2),
      });
    };
    controls.addEventListener('change', onControlsChange);

    threeRef.current = {
      scene,
      camera,
      renderer,
      mesh,
      material,
      light,
      controls,
      frontTex: blank,
      backTex: blank,
      currentEngine: null,
    };
    (window as unknown as { __three: unknown }).__three = threeRef.current;

    const onResize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    onResize();
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    let rafId = 0;
    const tick = () => {
      controls.update();
      // Update the tangent-space light direction uniform
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      controls.removeEventListener('change', onControlsChange);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // React to manifest changes — reload textures + re-activate engine.
  useEffect(() => {
    const t = threeRef.current;
    if (!t || !manifest) return;
    const loader = new THREE.TextureLoader();
    Promise.all([
      loader.loadAsync(manifest.files.front_png),
      loader.loadAsync(manifest.files.back_png),
    ]).then(([front, back]) => {
      for (const tex of [front, back]) {
        tex.colorSpace = THREE.LinearSRGBColorSpace;
        tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.magFilter = THREE.LinearFilter;
        tex.minFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        tex.anisotropy = 8;
        tex.needsUpdate = true;
      }
      if (t.frontTex !== front) t.frontTex.dispose();
      if (t.backTex !== back) t.backTex.dispose();
      t.frontTex = front;
      t.backTex = back;
      t.material.uniforms.uFront.value = front;
      t.material.uniforms.uBack.value = back;
      t.material.uniforms.uExtentUm.value = manifest.extent_um[0];
      t.material.uniforms.uThicknessUm.value = manifest.substrate.thickness_um;
      t.material.uniforms.uN.value = manifest.substrate.n;
      log('texture_bound', {
        slug: manifest.slug,
        variant: manifest.variant,
        front: { w: front.image?.width ?? 0, h: front.image?.height ?? 0 },
        back: { w: back.image?.width ?? 0, h: back.image?.height ?? 0 },
      });

      const eng = ENGINES[engine];
      const ctx = {
        scene: t.scene,
        plateMesh: t.mesh,
        material: t.material,
        frontTex: front,
        backTex: back,
      };
      if (t.currentEngine && t.currentEngine.id !== eng.id) {
        t.currentEngine.deactivate(ctx);
      }
      eng.activate(ctx, manifest).then(() => {
        t.currentEngine = eng;
      });
    });
  }, [manifest, engine]);

  // Illumination / laser color → uniforms
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    t.material.uniforms.uIllumination.value =
      illumination === 'ambient' ? 0 : illumination === 'laser' ? 1 : 2;
    const laserRgb: Record<string, number> = {
      red: 0xff3355,
      green: 0x44ff88,
      blue: 0x3388ff,
    };
    t.material.uniforms.uLaserColor.value.setHex(laserRgb[laserColor]);
    t.material.uniforms.uWavelengthSlot.value =
      laserColor === 'red' ? 0 : laserColor === 'green' ? 1 : 2;
  }, [illumination, laserColor]);

  // Light position
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    const az = (lightAz * Math.PI) / 180;
    const el = (lightEl * Math.PI) / 180;
    const r = 3.0;
    const x = r * Math.cos(el) * Math.sin(az);
    const y = r * Math.sin(el);
    const z = r * Math.cos(el) * Math.cos(az);
    t.material.uniforms.uLightWorld.value.set(x, y, z);
  }, [lightAz, lightEl]);

  return (
    <div
      ref={mountRef}
      style={{ width: '100%', height: '100%', minHeight: 0, minWidth: 0 }}
    />
  );
}
