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
import { RECIPE_IDS, fetchCarpet, type RenderRecipe } from '../api';

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
    viewATex: THREE.Texture;
    viewBTex: THREE.Texture;
    carpetTex: THREE.Texture;
    currentEngine: Engine | null;
  } | null>(null);

  const manifest = useStore((s) => s.manifest);
  const illumination = useStore((s) => s.illumination);
  const laserColor = useStore((s) => s.laserColor);
  const engine = useStore((s) => s.engine);
  const lightAz = useStore((s) => s.lightAzimuthDeg);
  const lightEl = useStore((s) => s.lightElevationDeg);
  const zSlice = useStore((s) => s.zSlice);
  const setCarpetAtlasUrl = useStore((s) => s.setCarpetAtlasUrl);

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
        // Default to the back-compat stylized path until a manifest with a
        // real recipe arrives. See RECIPE_IDS in src/api.ts.
        uRecipe: { value: RECIPE_IDS.stylized_amplitude },
        // --- iridescent_grating recipe (only read when uRecipe == 0) --------
        uGratingPeriodUm: { value: 4.0 },
        uGratingOrientation: { value: 0.0 },
        uLaserWavelengthUm: { value: 0.55 },
        // --- stereo_lenticular recipe (only read when uRecipe == 1) --------
        uViewA: { value: blank },
        uViewB: { value: blank },
        uSlitOrientation: { value: 0.0 },
        uSlitPeriodUm: { value: 40.0 },
        // --- near_field_carpet recipe (only read when uRecipe == 3) --------
        uCarpetAtlas: { value: blank },
        uZSlice: { value: 0.5 },
        uCarpetRows: { value: 0 },
        uHasCarpet: { value: false },
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
      viewATex: blank,
      viewBTex: blank,
      carpetTex: blank,
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
      // Legacy manifests on disk (from before Phase A) don't carry a recipe
      // — treat them as stylized_amplitude so every pattern keeps rendering.
      // An unknown recipe name (backend shipped a new recipe before the
      // frontend knew about it) would otherwise push `undefined` into an int
      // uniform; guard explicitly so the plate never hits that state.
      const rawRecipe = manifest.render_recipe;
      const recipe: RenderRecipe =
        rawRecipe && rawRecipe in RECIPE_IDS
          ? (rawRecipe as RenderRecipe)
          : 'stylized_amplitude';
      if (rawRecipe && recipe !== rawRecipe) {
        log('recipe_unknown_fallback', { slug: manifest.slug, requested: rawRecipe });
      }
      t.material.uniforms.uRecipe.value = RECIPE_IDS[recipe];

      // Per-recipe uniform hookup. Pull from recipe_data (generator-supplied)
      // and fall back to the pattern's nominal params when present.
      const rd = manifest.recipe_data ?? {};
      const params = manifest.params ?? {};
      if (recipe === 'iridescent_grating') {
        const period =
          Number(rd.period_um ?? params.period_um ?? 4.0) || 4.0;
        const orientRad =
          ((Number(rd.orientation_deg ?? 0) || 0) * Math.PI) / 180;
        t.material.uniforms.uGratingPeriodUm.value = period;
        t.material.uniforms.uGratingOrientation.value = orientRad;
      }

      // stereo_lenticular: load view_a / view_b textures + set slit axis.
      // We default-bind blank then overwrite asynchronously so a slow fetch
      // doesn't blank the canvas. The recipe_bound + texture_bound events
      // still fire above with the synchronous front/back state, so E2E
      // assertions that just want "binding happened" don't need to wait for
      // view textures to arrive.
      if (recipe === 'stereo_lenticular') {
        const viewAUrl = typeof rd.view_a_png === 'string' ? rd.view_a_png : null;
        const viewBUrl = typeof rd.view_b_png === 'string' ? rd.view_b_png : null;
        const slitDeg = Number(rd.slit_axis_deg ?? 0) || 0;
        const slitPeriod = Number(rd.slit_period_um ?? params.slit_period_um ?? params.period_um ?? 40.0) || 40.0;
        t.material.uniforms.uSlitOrientation.value = (slitDeg * Math.PI) / 180;
        t.material.uniforms.uSlitPeriodUm.value = slitPeriod;
        if (viewAUrl && viewBUrl) {
          const vLoader = new THREE.TextureLoader();
          Promise.all([vLoader.loadAsync(viewAUrl), vLoader.loadAsync(viewBUrl)]).then(
            ([va, vb]) => {
              for (const tex of [va, vb]) {
                tex.colorSpace = THREE.LinearSRGBColorSpace;
                tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
                tex.magFilter = THREE.LinearFilter;
                tex.minFilter = THREE.LinearFilter;
                tex.generateMipmaps = false;
                tex.anisotropy = 8;
                tex.needsUpdate = true;
              }
              if (t.viewATex !== va) t.viewATex.dispose();
              if (t.viewBTex !== vb) t.viewBTex.dispose();
              t.viewATex = va;
              t.viewBTex = vb;
              t.material.uniforms.uViewA.value = va;
              t.material.uniforms.uViewB.value = vb;
              log('stereo_views_bound', {
                slug: manifest.slug,
                view_a: { w: va.image?.width ?? 0, h: va.image?.height ?? 0 },
                view_b: { w: vb.image?.width ?? 0, h: vb.image?.height ?? 0 },
              });
            }
          );
        }
      } else {
        // Other recipes: clear any stale stereo textures so they don't bleed
        // into the shader if uRecipe == 1 ever runs transiently.
        t.material.uniforms.uViewA.value = t.frontTex; // harmless placeholder
        t.material.uniforms.uViewB.value = t.frontTex;
      }

      // near_field_carpet: fetch the z-sweep atlas from /sim/carpet using the
      // recipe_data z-range + design wavelength. Bound asynchronously so the
      // plate renders immediately (recipe 3 falls through to runStylized when
      // uHasCarpet=false, so the plate isn't blank while we wait).
      if (recipe === 'near_field_carpet') {
        t.material.uniforms.uHasCarpet.value = false;
        t.material.uniforms.uCarpetRows.value = 0;
        const wavelength_um = Number(rd.design_wavelength_um ?? 0.55) || 0.55;
        const z_min_um = Number(rd.z_min_um ?? 0.0) || 0.0;
        const z_max_um = Number(rd.z_max_um ?? 4000.0) || 4000.0;
        const n_slices = Math.max(
          2,
          Math.floor(Number(rd.n_slices ?? 48) || 48)
        );
        const slugAtFetch = manifest.slug;
        const variantAtFetch = manifest.variant;
        fetchCarpet(slugAtFetch, variantAtFetch, {
          wavelength_um,
          z_min_um,
          z_max_um,
          n_slices,
          downsample: 8,
          tile_size: 128,
        })
          .then((res) => {
            // Drop the result if the user has already selected a different
            // variant (selection race — mirrors the propagate cache guard).
            if (
              !threeRef.current ||
              threeRef.current !== t ||
              manifest.slug !== slugAtFetch ||
              manifest.variant !== variantAtFetch
            ) {
              return;
            }
            const loader2 = new THREE.TextureLoader();
            loader2.loadAsync(res.atlas_png).then((tex) => {
              tex.colorSpace = THREE.LinearSRGBColorSpace;
              tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
              tex.magFilter = THREE.LinearFilter;
              tex.minFilter = THREE.LinearFilter;
              tex.generateMipmaps = false;
              tex.anisotropy = 1;
              tex.needsUpdate = true;
              if (t.carpetTex !== tex) t.carpetTex.dispose();
              t.carpetTex = tex;
              t.material.uniforms.uCarpetAtlas.value = tex;
              t.material.uniforms.uCarpetRows.value = res.rows;
              t.material.uniforms.uHasCarpet.value = true;
              setCarpetAtlasUrl(res.atlas_png);
              log('carpet_fetched', {
                slug: slugAtFetch,
                variant: variantAtFetch,
                rows: res.rows,
                cached: res.cached,
              });
            });
          })
          .catch((e) => {
            log('carpet_fetch_error', {
              slug: slugAtFetch,
              variant: variantAtFetch,
              message: (e as Error).message,
            });
          });
      } else {
        // Non-carpet recipe: clear any stale carpet state so a transient
        // uRecipe == 3 frame doesn't show the previous pattern's z-sweep.
        t.material.uniforms.uHasCarpet.value = false;
        t.material.uniforms.uCarpetRows.value = 0;
        setCarpetAtlasUrl(null);
      }

      log('recipe_bound', { slug: manifest.slug, recipe });
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
    // Wavelength in μm for iridescent_grating recipe (laser-spot gating).
    const laserUm: Record<string, number> = { red: 0.65, green: 0.55, blue: 0.45 };
    t.material.uniforms.uLaserWavelengthUm.value = laserUm[laserColor] ?? 0.55;
  }, [illumination, laserColor]);

  // z-slice slider → uZSlice. Decoupled from the manifest effect so dragging
  // the slider is a zero-cost uniform push, not a full texture rebind.
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    t.material.uniforms.uZSlice.value = zSlice;
  }, [zSlice]);

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
