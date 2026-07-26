/**
 * Ring Box Studio UI kit — the single place panel styling lives.
 *
 * Small, dependency-free building blocks shared by every panel: collapsible
 * Section, SliderRow / NumberRow / SelectRow / TextRow / CheckRow form rows,
 * ChipRow preset chips, Swatch color chips, Disclosure for advanced options,
 * Button, and the Shimmer / FailedTile thumbnail placeholders. One consistent
 * dark-studio style; no external UI deps.
 */
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';

export const KIT = {
  text: '#e8eaed',
  panel: '#0f1218',
  field: '#141820',
  raised: '#1d2434',
  border: '#2a2f36',
  divider: '#22262d',
  accent: '#8ab4ff',
  error: '#ff8888',
} as const;

export const INPUT_STYLE: CSSProperties = {
  background: KIT.field,
  border: `1px solid ${KIT.border}`,
  color: KIT.text,
  borderRadius: 4,
  padding: '4px 6px',
  fontSize: 12,
};

export const BUTTON_STYLE: CSSProperties = {
  padding: '4px 10px',
  background: KIT.raised,
  border: `1px solid ${KIT.border}`,
  borderRadius: 4,
  color: KIT.text,
  fontSize: 12,
  cursor: 'pointer',
};

const ROW: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  fontSize: 12,
  marginBottom: 8,
};

const VALUE: CSSProperties = { float: 'right', opacity: 0.7, fontSize: 10 };

// One-time keyframes for the thumbnail loading shimmer (no CSS file needed).
const KIT_CSS = `
@keyframes kit-shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
`;
if (typeof document !== 'undefined' && !document.getElementById('kit-css')) {
  const el = document.createElement('style');
  el.id = 'kit-css';
  el.textContent = KIT_CSS;
  document.head.appendChild(el);
}

/** Uppercase letterspaced mini-header used inside panel bodies. */
export function SubHeader({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7, marginBottom: 6 }}>
      {children}
    </div>
  );
}

export function Button({
  children,
  onClick,
  testId,
  title,
  active = false,
  disabled = false,
  style,
}: {
  children: ReactNode;
  onClick: () => void;
  testId?: string;
  title?: string;
  active?: boolean;
  disabled?: boolean;
  style?: CSSProperties;
}) {
  return (
    <button
      data-testid={testId}
      title={title}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      style={{
        ...BUTTON_STYLE,
        borderColor: active ? KIT.accent : KIT.border,
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'default' : 'pointer',
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// Remembers each Section's open/closed state across reloads. Keyed by
// `persistId` (falls back to the title), namespaced so it can't collide with
// other localStorage keys. Wrapped in try/catch — private-mode / disabled
// storage must never break the panels.
const SECTION_STORE_PREFIX = 'rbs.section.';
function readSectionOpen(key: string, fallback: boolean): boolean {
  try {
    const v = window.localStorage.getItem(SECTION_STORE_PREFIX + key);
    if (v === '0') return false;
    if (v === '1') return true;
  } catch {
    /* storage unavailable — use fallback */
  }
  return fallback;
}
function writeSectionOpen(key: string, open: boolean): void {
  try {
    window.localStorage.setItem(SECTION_STORE_PREFIX + key, open ? '1' : '0');
  } catch {
    /* storage unavailable — nothing to persist */
  }
}

/**
 * Collapsible panel section with an uppercase header + chevron. This is the
 * single collapsible primitive shared by BOTH the left (Build) and right
 * (Faces) rails, so their sections look and behave identically. Open/closed
 * state persists across reloads, keyed by `persistId` (or the title).
 */
export function Section({
  title,
  children,
  defaultOpen = true,
  testId,
  persistId,
  onOpen,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  testId?: string;
  /** Stable key for persisting open state; defaults to `title`. */
  persistId?: string;
  /**
   * Fires every time the section transitions closed -> open (incl. mount when
   * it starts open). Same contract as Disclosure's onOpen: callers that want
   * once-only behavior must dedupe — firing on EVERY open is what lets
   * consumers retry lazy loads that failed earlier.
   */
  onOpen?: () => void;
}) {
  const key = persistId ?? title;
  const [open, setOpen] = useState(() => readSectionOpen(key, defaultOpen));
  // Latest-callback ref so an inline arrow prop doesn't re-fire the effect on
  // every parent render; `wasOpen` makes this transition-edge triggered.
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open && !wasOpen.current) onOpenRef.current?.();
    wasOpen.current = open;
  }, [open]);
  const toggle = () => {
    setOpen((o) => {
      const next = !o;
      writeSectionOpen(key, next);
      return next;
    });
  };
  return (
    <div data-testid={testId} style={{ borderBottom: `1px solid ${KIT.divider}` }}>
      <button
        onClick={toggle}
        aria-expanded={open}
        style={{
          width: '100%',
          textAlign: 'left',
          background: 'transparent',
          border: 'none',
          color: KIT.text,
          padding: '10px 12px',
          fontSize: 11,
          letterSpacing: 1.5,
          opacity: 0.75,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span>{title.toUpperCase()}</span>
        <span
          aria-hidden
          style={{
            fontSize: 9,
            opacity: 0.6,
            display: 'inline-block',
            transform: open ? 'rotate(0deg)' : 'rotate(-90deg)',
            transition: 'transform 120ms ease',
          }}
        >
          ▾
        </span>
      </button>
      {open && <div style={{ padding: '0 12px 12px' }}>{children}</div>}
    </div>
  );
}

/** Lightweight inline disclosure for advanced/rarely-used options. */
export function Disclosure({
  label,
  children,
  defaultOpen = false,
  testId,
  onOpen,
}: {
  label: string;
  children: ReactNode;
  defaultOpen?: boolean;
  testId?: string;
  /**
   * Fires every time the disclosure transitions closed -> open (incl. mount
   * when defaultOpen). Callers that want once-only behavior must dedupe —
   * firing on every open is what lets consumers retry failed lazy loads.
   */
  onOpen?: () => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  // Latest-callback ref so an inline arrow prop doesn't re-fire the effect
  // on every parent render; `wasOpen` makes this transition-edge triggered.
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open && !wasOpen.current) onOpenRef.current?.();
    wasOpen.current = open;
  }, [open]);
  return (
    <div data-testid={testId} style={{ marginBottom: 8 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: 'transparent',
          border: 'none',
          color: KIT.text,
          opacity: 0.7,
          fontSize: 11,
          cursor: 'pointer',
          padding: 0,
          display: 'flex',
          gap: 6,
          alignItems: 'center',
        }}
      >
        <span style={{ fontSize: 9 }}>{open ? '▾' : '▸'}</span>
        <span>{label}</span>
      </button>
      {open && <div style={{ marginTop: 8 }}>{children}</div>}
    </div>
  );
}

/** Trailing-flush fallback (ms) for when rAF is starved (background tab). */
const SLIDER_FLUSH_MS = 64;

/**
 * Slider with a label + formatted value (+ optional numeric twin).
 *
 * Delivery of `onChange` is rAF-throttled on the LEADING edge: the first change
 * in an animation frame goes straight through, and any further changes in that
 * same frame are coalesced into one trailing delivery of the last value. A
 * pointer drag on a high-polling-rate mouse fires several input events per
 * frame, and each one used to reach the store — which for a box dimension
 * re-renders the panels and rebuilds the whole 3D scene graph.
 *
 * The leading edge is load-bearing, not an optimisation: a single programmatic
 * set (`locator.fill()`, a preset chip) must land in the store with no added
 * latency, because callers read the committed value straight afterwards. The
 * COMMITTED-VALUE CONTRACT IS UNCHANGED — every value the user lands on is
 * delivered, the last one always, and a parent that clamps or ignores a value
 * still wins the display.
 */
export function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit,
  decimals,
  testId,
  withNumber = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
  decimals?: number;
  testId?: string;
  withNumber?: boolean;
}) {
  const dec = decimals ?? (step >= 1 ? 0 : step >= 0.1 ? 1 : 2);
  // Local echo for coalesced (not-yet-delivered) values only. React restores a
  // controlled input's DOM value after any event whose handler did not move the
  // `value` prop, so without this the thumb would snap backwards for a frame
  // mid-drag. Null whenever nothing is in flight, so `value` is the single
  // source of truth at rest.
  const [draft, setDraft] = useState<number | null>(null);
  const shown = draft ?? value;
  const pendingRef = useRef<number | null>(null);
  const rafRef = useRef(0);
  const timerRef = useRef(0);
  // Latest-callback ref: the throttle outlives the render that scheduled it.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const clearScheduled = (): void => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timerRef.current) window.clearTimeout(timerRef.current);
    rafRef.current = 0;
    timerRef.current = 0;
  };
  const flush = (): void => {
    clearScheduled();
    const v = pendingRef.current;
    pendingRef.current = null;
    if (v !== null) onChangeRef.current(v);
  };
  const openWindow = (): void => {
    rafRef.current = requestAnimationFrame(flush);
    timerRef.current = window.setTimeout(flush, SLIDER_FLUSH_MS);
  };
  const push = (v: number): void => {
    if (rafRef.current || timerRef.current) {
      pendingRef.current = v;
      setDraft(v);
      return;
    }
    pendingRef.current = null;
    setDraft(null);
    onChangeRef.current(v);
    openWindow();
  };
  /** Discrete commit (numeric twin): no throttling, drop anything in flight. */
  const pushNow = (v: number): void => {
    clearScheduled();
    pendingRef.current = null;
    setDraft(null);
    onChangeRef.current(v);
  };
  // Stop echoing once the parent has caught up — and once it has had its say on
  // the final value, so a clamp or a refusal is what the user ends up seeing.
  useEffect(() => {
    if (pendingRef.current === null) setDraft(null);
  }, [value]);
  // Deliver, don't drop, a value still in flight when the row goes away.
  useEffect(() => () => flush(), []);
  const endGesture = (): void => {
    flush();
    setDraft(null);
  };
  return (
    <label style={ROW}>
      <span>
        {label}
        <span style={VALUE}>
          {shown.toFixed(dec)}
          {unit ? ` ${unit}` : ''}
        </span>
      </span>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <input
          data-testid={testId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={shown}
          onChange={(e) => push(parseFloat(e.target.value))}
          onPointerUp={endGesture}
          onKeyUp={endGesture}
          onBlur={endGesture}
          style={{ flex: 1 }}
        />
        {withNumber && (
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={Number(shown.toFixed(dec))}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (Number.isFinite(v)) pushNow(v);
            }}
            style={{ ...INPUT_STYLE, width: 58 }}
          />
        )}
      </div>
    </label>
  );
}

/**
 * Numeric field that commits on blur / Enter — never per keystroke — and
 * clamps the typed value to [min, max].
 *
 * Both behaviors are load-bearing, not polish: HTML number inputs do not
 * constrain typed text outside a form submit, and every keystroke used to
 * reach the store. Typing "2500" passes through 2, 25 and 250, so a foil-tape
 * edit patched three nonsense specs on the way, flashing validation errors and
 * (past the regen debounce) firing a multi-second /boxes/generate for a spec
 * the user never asked for. The draft therefore stays local until commit;
 * Escape abandons it.
 */
export function NumberRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit,
  testId,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  unit?: string;
  testId?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  // Abandon a pending draft when the committed value moves underneath us (our
  // own commit, or a preset chip writing the same field) so the box can never
  // show text that disagrees with the spec.
  useEffect(() => {
    setDraft(null);
  }, [value]);
  const commit = (raw: string) => {
    setDraft(null);
    const v = parseFloat(raw);
    if (!Number.isFinite(v)) return;
    const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v));
    if (clamped !== value) onChange(clamped);
  };
  return (
    <label style={ROW}>
      <span>
        {label}
        {unit ? <span style={VALUE}>{unit}</span> : null}
      </span>
      <input
        data-testid={testId}
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft ?? String(value)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit(e.currentTarget.value);
          } else if (e.key === 'Escape') {
            setDraft(null);
          }
        }}
        style={INPUT_STYLE}
      />
    </label>
  );
}

export function SelectRow({
  label,
  value,
  options,
  onChange,
  testId,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
  testId?: string;
}) {
  return (
    <label style={ROW}>
      <span>{label}</span>
      <select
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={INPUT_STYLE}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function TextRow({
  label,
  value,
  onChange,
  testId,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  testId?: string;
}) {
  return (
    <label style={ROW}>
      <span>{label}</span>
      <input
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={INPUT_STYLE}
      />
    </label>
  );
}

export function CheckRow({
  label,
  checked,
  onChange,
  testId,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  testId?: string;
}) {
  return (
    <label
      style={{
        ...ROW,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <span>{label}</span>
      <input
        data-testid={testId}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

export type Chip<T extends string | number> = {
  value: T;
  label: string;
  testId?: string;
};

/** Row of one-click preset chips; the chip matching `value` is highlighted. */
export function ChipRow<T extends string | number>({
  label,
  chips,
  value,
  onSelect,
}: {
  label?: string;
  chips: Chip<T>[];
  value?: T;
  onSelect: (v: T) => void;
}) {
  return (
    <div style={ROW}>
      {label && <span>{label}</span>}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {chips.map((c) => {
          const active = c.value === value;
          return (
            <button
              key={String(c.value)}
              data-testid={c.testId}
              onClick={() => onSelect(c.value)}
              aria-pressed={active}
              style={{
                ...BUTTON_STYLE,
                padding: '4px 10px',
                borderRadius: 12,
                background: active ? KIT.raised : KIT.field,
                borderColor: active ? KIT.accent : KIT.border,
              }}
            >
              {c.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Color chip with a dot — for finish pickers and the like. */
export function Swatch({
  color,
  label,
  active,
  onClick,
  testId,
}: {
  color: string;
  label: string;
  active: boolean;
  onClick: () => void;
  testId?: string;
}) {
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      aria-pressed={active}
      title={label}
      style={{
        ...BUTTON_STYLE,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        background: active ? KIT.raised : KIT.field,
        borderColor: active ? KIT.accent : KIT.border,
      }}
    >
      <span
        style={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          background: color,
          border: '1px solid rgba(255,255,255,0.25)',
          display: 'inline-block',
        }}
      />
      <span style={{ textTransform: 'capitalize' }}>{label}</span>
    </button>
  );
}

/**
 * Static placeholder for a thumbnail whose fetch FAILED — deliberately
 * unanimated so it reads differently from Shimmer. A shimmer that never
 * resolves is indistinguishable from a slow load, which is how failed pattern
 * previews used to sit spinning for a whole session.
 */
export function FailedTile({
  style,
  label = 'preview failed',
  title,
}: {
  style?: CSSProperties;
  label?: string;
  title?: string;
}) {
  return (
    <div
      title={title}
      style={{
        background: KIT.field,
        border: `1px dashed ${KIT.error}`,
        borderRadius: 4,
        color: KIT.error,
        opacity: 0.8,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        fontSize: 10,
        lineHeight: 1.2,
        textAlign: 'center',
        padding: 4,
        ...style,
      }}
    >
      <span aria-hidden style={{ fontSize: 14 }}>
        ↻
      </span>
      <span>{label}</span>
    </div>
  );
}

/** Animated placeholder block shown while a thumbnail loads. */
export function Shimmer({ style }: { style?: CSSProperties }) {
  return (
    <div
      style={{
        background: `linear-gradient(90deg, ${KIT.field} 25%, ${KIT.raised} 50%, ${KIT.field} 75%)`,
        backgroundSize: '200% 100%',
        animation: 'kit-shimmer 1.2s linear infinite',
        borderRadius: 4,
        ...style,
      }}
    />
  );
}
