import { useEffect, useRef, useState } from 'react';
import { getDefault, listPatterns, type PatternDescriptor } from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import { parseSceneFromUrl } from '../debug';

const THEME_ORDER: PatternDescriptor['theme'][] = ['Colombia', 'Global Travel'];

const TIER_COLORS: Record<number, { bg: string; fg: string }> = {
  1: { bg: '#1f3a26', fg: '#8fe0a1' },  // green — ambient / parallax
  2: { bg: '#3a2f1f', fg: '#e6b866' },  // amber — diffractive / CGH
  3: { bg: '#2c1f3a', fg: '#c697e8' },  // violet — wave-propagated
};

export default function Gallery() {
  const [patterns, setPatterns] = useState<PatternDescriptor[]>([]);
  const activeSlug = useStore((s) => s.activeSlug);
  const selectPattern = useStore((s) => s.selectPattern);
  const beginSelect = useStore((s) => s.beginSelect);
  const selectPatternIfCurrent = useStore((s) => s.selectPatternIfCurrent);
  const setCatalog = useStore((s) => s.setCatalog);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    listPatterns().then((pts) => {
      setPatterns(pts);
      setCatalog(pts);
      log('catalog_loaded', { count: pts.length });
      if (pts.length && !activeSlug) {
        // URL deep-link: ?pattern=<slug>&illum=...&cam=az,el&light=az,el&...
        // Schema documented in tools/preview_inspect.md. Pattern slug is
        // honored only if it matches the catalog; everything else is
        // applied via window.__debug.applyScene after the initial select.
        const preset = parseSceneFromUrl();
        const fromUrl = preset.pattern && pts.some((p) => p.slug === preset.pattern);
        const initialSlug = fromUrl ? preset.pattern! : pts[0].slug;
        getDefault(initialSlug).then((m) => {
          selectPattern(initialSlug, m);
          log('pattern_selected', {
            slug: initialSlug,
            variant: m.variant,
            auto: true,
            from_url: !!fromUrl,
          });
          const hasMore =
            preset.illumination ||
            preset.laserColor ||
            preset.cameraAzEl ||
            preset.light ||
            preset.params;
          if (hasMore) {
            // applyScene awaits texture_bound + secondary fetch internally.
            // Drop the redundant `pattern` field so we don't re-select.
            const { pattern: _pattern, ...rest } = preset;
            window.__debug?.applyScene(rest).catch((e) => {
              log('url_preset_failed', { error: (e as Error).message });
            });
          }
        });
      }
    });
    return () => abortRef.current?.abort();
  }, []);

  /**
   * Race-safe selection. The last click wins: any previous in-flight fetch is
   * aborted and — as a belt-and-suspenders — if an abort slips through, the
   * `selectPatternIfCurrent` guard drops the stale response.
   */
  const onSelect = (slug: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    beginSelect(slug);
    log('pattern_select_requested', { slug });
    getDefault(slug, { signal: controller.signal })
      .then((m) => {
        const committed = selectPatternIfCurrent(slug, m);
        log('pattern_selected', { slug, variant: m.variant, committed });
      })
      .catch((err: Error) => {
        if (err.name === 'AbortError') {
          log('pattern_select_aborted', { slug });
          return;
        }
        log('pattern_load_failed', { slug, error: err.message });
        console.error('pattern load failed', slug, err);
      });
  };

  const grouped: Record<string, PatternDescriptor[]> = {};
  for (const p of patterns) {
    const key = p.theme ?? 'Colombia';
    (grouped[key] ||= []).push(p);
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        padding: 8,
        overflowY: 'auto',
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7, padding: '8px 4px' }}>
        PATTERN CATALOG
      </div>
      {THEME_ORDER.filter((t) => grouped[t]?.length).map((theme) => (
        <div key={theme} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <h3
            data-theme={theme}
            style={{
              margin: '12px 4px 4px',
              fontSize: 12,
              letterSpacing: 1.2,
              opacity: 0.85,
              color: '#cfd3d9',
              textTransform: 'uppercase',
              borderBottom: '1px solid #2a2f36',
              paddingBottom: 4,
            }}
          >
            {theme}
          </h3>
          {grouped[theme].map((p) => {
            const active = p.slug === activeSlug;
            const tierColor = TIER_COLORS[p.tier] ?? TIER_COLORS[1];
            return (
              <button
                key={p.slug}
                data-slug={p.slug}
                data-tier={p.tier}
                data-theme={p.theme}
                onClick={() => onSelect(p.slug)}
                style={{
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 6,
                  border: active ? '1px solid #8ab4ff' : '1px solid #2a2f36',
                  background: active ? '#1d2434' : '#141820',
                  color: '#e8eaed',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 6,
                  }}
                >
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{p.name}</div>
                  <span
                    data-testid="tier-badge"
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      letterSpacing: 0.6,
                      padding: '1px 6px',
                      borderRadius: 10,
                      background: tierColor.bg,
                      color: tierColor.fg,
                      flexShrink: 0,
                    }}
                  >
                    T{p.tier}
                  </span>
                </div>
                <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.3 }}>
                  {p.tags.map((t) => (
                    <span
                      key={t}
                      style={{
                        display: 'inline-block',
                        marginRight: 4,
                        padding: '1px 6px',
                        background: '#22262d',
                        borderRadius: 3,
                      }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}
