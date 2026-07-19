import { log } from '../logger';
import { useStore } from '../store';
import { ChipRow, Section, SliderRow } from './kit';

/** Illumination mode, laser color, and light-direction controls. */
export default function IlluminationPanel() {
  const illumination = useStore((s) => s.illumination);
  const setIllumination = useStore((s) => s.setIllumination);
  const laserColor = useStore((s) => s.laserColor);
  const setLaserColor = useStore((s) => s.setLaserColor);
  const az = useStore((s) => s.lightAzimuthDeg);
  const el = useStore((s) => s.lightElevationDeg);
  const setLight = useStore((s) => s.setLight);

  return (
    <>
      <Section title="Illumination" testId="section-illumination" persistId="illumination">
        <ChipRow
          chips={(['ambient', 'laser', 'backlight'] as const).map((m) => ({
            value: m,
            label: m,
            testId: `illum-${m}`,
          }))}
          value={illumination}
          onSelect={(m) => {
            log('illumination_changed', { from: illumination, to: m });
            setIllumination(m);
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
