import { useEffect, useMemo, useRef } from 'react';
import { generatePattern } from '../api';
import { log } from '../logger';
import { useStore } from '../store';

function useDebounce<T extends (...args: never[]) => void>(fn: T, ms: number): T {
  const timer = useRef<number | null>(null);
  const latest = useRef(fn);
  latest.current = fn;
  return useMemo(
    () =>
      ((...args: Parameters<T>) => {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => latest.current(...args), ms);
      }) as T,
    [ms]
  );
}

export default function ParameterPanel() {
  const catalog = useStore((s) => s.catalog);
  const activeSlug = useStore((s) => s.activeSlug);
  const params = useStore((s) => s.params);
  const patchParams = useStore((s) => s.patchParams);
  const setManifest = useStore((s) => s.setManifest);

  const spec = useMemo(
    () => catalog.find((c) => c.slug === activeSlug) ?? null,
    [catalog, activeSlug]
  );

  const regen = useDebounce(async (slug: string, p: Record<string, unknown>) => {
    const t0 = performance.now();
    log('param_regen_start', { slug, params: p });
    try {
      const m = await generatePattern(slug, p);
      setManifest(m);
      log('param_regen_done', {
        slug,
        variant: m.variant,
        duration_ms: Math.round(performance.now() - t0),
      });
    } catch (e) {
      const err = e as Error;
      log('param_regen_failed', {
        slug,
        duration_ms: Math.round(performance.now() - t0),
        error: err.message,
      });
      console.error(e);
    }
  }, 250);

  useEffect(() => {
    if (activeSlug && spec && Object.keys(params).length > 0) {
      regen(activeSlug, params);
    }
  }, [params, activeSlug, spec, regen]);

  if (!spec) {
    return <div style={{ padding: 12, opacity: 0.6 }}>Pick a pattern.</div>;
  }

  return (
    <div
      data-testid="parameter-panel"
      style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}
    >
      <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7 }}>PARAMETERS</div>
      <div style={{ fontSize: 13, fontWeight: 600 }}>{spec.name}</div>
      <div style={{ fontSize: 11, opacity: 0.7, lineHeight: 1.4 }}>{spec.description}</div>
      {spec.params.map((p) => {
        const v = params[p.name] ?? (p.default as unknown);
        if (p.type === 'choice') {
          return (
            <label key={p.name} style={labelStyle}>
              <span>
                {p.label}
                {p.unit ? <em style={{ opacity: 0.5 }}> ({p.unit})</em> : null}
              </span>
              <select
                value={String(v)}
                onChange={(e) => {
                  log('param_changed', { slug: activeSlug, name: p.name, value: e.target.value });
                  patchParams({ [p.name]: e.target.value });
                }}
                style={inputStyle}
              >
                {p.choices?.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          );
        }
        if (p.type === 'bool') {
          return (
            <label key={p.name} style={labelStyle}>
              <span>{p.label}</span>
              <input
                type="checkbox"
                checked={Boolean(v)}
                onChange={(e) => {
                  log('param_changed', { slug: activeSlug, name: p.name, value: e.target.checked });
                  patchParams({ [p.name]: e.target.checked });
                }}
              />
            </label>
          );
        }
        return (
          <label key={p.name} style={labelStyle}>
            <span>
              {p.label}
              {p.unit ? <em style={{ opacity: 0.5 }}> ({p.unit})</em> : null}
              <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>
                {Number(v).toFixed(p.type === 'int' ? 0 : 2)}
              </span>
            </span>
            <input
              type="range"
              min={p.min}
              max={p.max}
              step={p.step ?? (p.type === 'int' ? 1 : 0.01)}
              value={Number(v)}
              onChange={(e) => {
                const raw = e.target.value;
                const value = p.type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
                log('param_changed', { slug: activeSlug, name: p.name, value });
                patchParams({ [p.name]: value });
              }}
            />
          </label>
        );
      })}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  fontSize: 12,
};

const inputStyle: React.CSSProperties = {
  background: '#141820',
  border: '1px solid #2a2f36',
  color: '#e8eaed',
  borderRadius: 4,
  padding: '4px 6px',
};
