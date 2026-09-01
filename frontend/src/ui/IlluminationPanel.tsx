import { log } from '../logger';
import { useStore } from '../store';
import { CheckRow, ChipRow, Disclosure, Section, SliderRow } from './kit';

/** Illumination mode, backdrop, presentation props, and light direction. */
export default function IlluminationPanel() {
  const illumination = useStore((s) => s.illumination);
  const setIllumination = useStore((s) => s.setIllumination);
  const laserColor = useStore((s) => s.laserColor);
  const setLaserColor = useStore((s) => s.setLaserColor);
  const backdrop = useStore((s) => s.backdrop);
  const setBackdrop = useStore((s) => s.setBackdrop);
  const showRing = useStore((s) => s.showRing);
  const setShowRing = useStore((s) => s.setShowRing);
  const az = useStore((s) => s.lightAzimuthDeg);
  const el = useStore((s) => s.lightElevationDeg);
  const setLight = useStore((s) => s.setLight);

  return (
    <>
      <Section title="Illumination" testId="section-illumination" persistId="illumination">
        {/* Primary modes: ambient (studio) and backlight (light table — the mask
            inspection view: bright field behind the box, gold as silhouette).
            Laser is demoted to Experimental below: its preview is a flat
            coverage tint with no diffraction physics yet. */}
        <ChipRow
          chips={(['ambient', 'backlight'] as const).map((m) => ({
            value: m,
            label: m === 'backlight' ? 'light table' : m,
            testId: `illum-${m}`,
          }))}
          value={illumination === 'laser' ? undefined : illumination}
          onSelect={(m) => {
            log('illumination_changed', { from: illumination, to: m });
            setIllumination(m);
          }}
        />
        <ChipRow
          label="Backdrop"
          chips={(['studio', 'velvet', 'daylight'] as const).map((b) => ({
            value: b,
            label: b,
            testId: `backdrop-${b}`,
          }))}
          value={backdrop}
          onSelect={(b) => {
            log('backdrop_changed', { from: backdrop, to: b });
            setBackdrop(b);
          }}
        />
        <CheckRow
          label="Show ring & cushion (display prop)"
          checked={showRing}
          onChange={(on) => {
            log('show_ring_toggled', { on });
            setShowRing(on);
          }}
          testId="show-ring"
        />
        <Disclosure label="Experimental" testId="illum-experimental">
          {/* Laser: a flat coverage tint today — kept for curiosity, demoted
              until it renders honest diffraction orders. */}
          <ChipRow
            label="Laser mode"
            chips={(['laser'] as const).map((m) => ({
              value: m,
              label: illumination === 'laser' ? 'laser (on)' : 'laser',
              testId: `illum-${m}`,
            }))}
            value={illumination === 'laser' ? 'laser' : undefined}
            onSelect={() => {
              const next = illumination === 'laser' ? 'ambient' : 'laser';
              log('illumination_changed', { from: illumination, to: next });
              setIllumination(next);
            }}
          />
          {illumination === 'laser' && (
            <ChipRow
              label="Laser"
              chips={(['red', 'green', 'blue'] as const).map((c) => ({
                value: c,
                label: c,
                testId: `laser-${c}`,
              }))}
              value={laserColor}
              onSelect={(c) => {
                log('laser_color_changed', { from: laserColor, to: c });
                setLaserColor(c);
              }}
            />
          )}
        </Disclosure>
      </Section>

      <Section
        title="Light direction"
        testId="section-light-direction"
        persistId="light-direction"
      >
        <SliderRow
          label="Azimuth"
        value={az}
        min={-180}
        max={180}
        step={1}
        unit="°"
        decimals={0}
        onChange={(nextAz) => {
          log('light_moved', { az: nextAz, el });
          setLight(nextAz, el);
        }}
        testId="light-az"
      />
      <SliderRow
        label="Elevation"
        value={el}
        min={5}
        max={89}
        step={1}
        unit="°"
        decimals={0}
        onChange={(nextEl) => {
          log('light_moved', { az, el: nextEl });
          setLight(az, nextEl);
        }}
        testId="light-el"
      />
      </Section>
    </>
  );
}
