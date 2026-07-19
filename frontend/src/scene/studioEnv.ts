import * as THREE from 'three';

/**
 * A bright neutral studio "lightbox" scene used only as a source for PMREM.
 *
 * RoomEnvironment is too dim: metalness=1 surfaces reflect it and crush to near
 * black against our dark backdrop. Instead we build a small enclosing box of
 * emissive panels — a big soft top light, two side scrims, and a warm/cool pair
 * — so every metal has bright, directional-but-soft reflections to catch. This
 * scene is rendered once by PMREMGenerator.fromScene() and then thrown away;
 * it is never added to the visible scene.
 */
export function buildStudioEnvScene(): THREE.Scene {
  const scene = new THREE.Scene();
  // A fairly bright neutral fill so a flat metal facet reflecting toward an
  // "empty" direction still returns a mid-grey, never pure black.
  scene.background = new THREE.Color(0x6f757f);

  const geo = new THREE.BoxGeometry(1, 1, 1);
  geo.deleteAttribute('uv');
  geo.deleteAttribute('normal');

  // Enclosing room, emissive-ish (basic material = unlit, so it reads as a
  // uniform bright grey from every reflection angle). This is the FLOOR of the
  // reflected luminance: a metalness=1 facet whose mirror direction misses
  // every panel still reflects this, so it can never crush to black. Raised
  // (was 0x4c525c) so foil strips framing off-axis faces keep their tint in the
  // standard 3/4 view instead of reading as dead-black bands.
  const room = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ side: THREE.BackSide, color: 0x6a707b })
  );
  room.scale.set(16, 9, 16);
  scene.add(room);

  // Vertical hemisphere gradient painted onto the four side walls: brighter
  // toward the top, darker toward the floor. A metalness=1 facet that points
  // sideways (like a vertical foil strip on a turned face) reflects one of
  // these walls, so it now returns a graded mid-tone with tint rather than the
  // flat crush the old uniform room produced. Sky/ground bands are gentle so
  // the metal keeps direction without a hard horizon line.
  const wallGrad = (() => {
    const cv = document.createElement('canvas');
    cv.width = 4;
    cv.height = 256;
    const c2d = cv.getContext('2d')!;
    const g = c2d.createLinearGradient(0, 0, 0, 256);
    g.addColorStop(0.0, '#aeb6c2'); // sky band — bright
    g.addColorStop(0.5, '#7c828d'); // horizon — mid
    g.addColorStop(1.0, '#565b64'); // ground — dim but never black
    c2d.fillStyle = g;
    c2d.fillRect(0, 0, 4, 256);
    return new THREE.CanvasTexture(cv);
  })();
  const wallMat = new THREE.MeshBasicMaterial({
    map: wallGrad,
    side: THREE.BackSide,
  });
  const wallBox = new THREE.Mesh(geo, wallMat);
  wallBox.scale.set(15.6, 8.8, 15.6);
  scene.add(wallBox);

  // Gradient ceiling glow: a brighter band up high, darker toward the floor,
  // gives metals a natural top-lit falloff instead of flat grey.
  const ceil = new THREE.Mesh(
    new THREE.PlaneGeometry(15, 15),
    new THREE.MeshBasicMaterial({ color: 0x9aa2b0, side: THREE.DoubleSide })
  );
  ceil.position.set(0, 4.4, 0);
  ceil.rotation.x = Math.PI / 2;
  scene.add(ceil);

  const panel = (
    w: number,
    h: number,
    color: number,
    intensity: number,
    pos: [number, number, number],
    lookAt: [number, number, number]
  ) => {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide })
    );
    (m.material as THREE.MeshBasicMaterial).color.multiplyScalar(intensity);
    m.position.set(...pos);
    m.lookAt(...lookAt);
    scene.add(m);
    return m;
  };

  // Big soft key from above (slightly warm) — the dominant reflection.
  panel(9, 9, 0xfff4e6, 3.2, [1.5, 6, 2.5], [0, 0, 0]);
  // Cool fill from the left.
  panel(7, 7, 0xdce6ff, 2.6, [-6, 1.5, -2], [0, 0, 0]);
  // Warm rim from behind-right, low.
  panel(5, 5, 0xffe9c8, 2.4, [5, 0.5, -5], [0, 0, 0]);
  // Broad dim fill lobes wrapping the mid-height ring so a metalness=1 facet
  // that points to ANY horizontal direction (the off-key foil strips the
  // verifier flagged as flat-black) reflects a tinted mid-tone, not the dark
  // backdrop. Large + low intensity = soft, no hard highlight, just a lift off
  // the black floor that keeps each finish's tint readable off-axis.
  panel(11, 8, 0xe6ecf5, 1.5, [-8, 2.0, 3], [0, 1.5, 0]); // left-front
  panel(11, 8, 0xf2ede6, 1.5, [8, 2.0, 3], [0, 1.5, 0]); // right-front
  panel(11, 8, 0xe9edf4, 1.3, [-7, 2.0, -6], [0, 1.5, 0]); // left-back
  panel(11, 8, 0xf0ece6, 1.3, [7, 2.0, -6], [0, 1.5, 0]); // right-back
  // Large bright FRONT bounce (toward the viewer hemisphere) so the fronts of
  // flat wall foil — whose mirror reflection points back toward the camera —
  // reflect brightness instead of the dark backdrop. This is the panel that
  // rescues front-facing foil strips from crushing to black.
  panel(16, 12, 0xeef1f6, 2.6, [0, 1.5, 9], [0, 0, 0]);
  // Second front-quarter bounce, warmer, offset so turning the box slides a
  // highlight across the walls.
  panel(10, 9, 0xf6efe6, 2.0, [6, 2.5, 7], [0, 0, 0]);
  panel(10, 9, 0xe8eef6, 2.0, [-6, 2.5, 7], [0, 0, 0]);
  // Soft under-bounce (dim) so undersides of beads catch a little.
  panel(8, 8, 0xc8ccd4, 0.7, [0, -5, 1], [0, 0, 0]);

  // A couple of small bright specular "hero" panels to give crisp streak
  // highlights that travel across the brushed foil as the box turns.
  panel(0.8, 6, 0xffffff, 6.0, [3.5, 4, 4.5], [0, 0, 0]);
  panel(0.7, 5, 0xffffff, 5.0, [-3.5, 3.5, 3], [0, 0, 0]);

  return scene;
}
