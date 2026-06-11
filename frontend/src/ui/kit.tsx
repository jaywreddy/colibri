/**
 * Ring Box Studio UI kit — the single place panel styling lives.
 *
 * Small, dependency-free building blocks shared by every panel: collapsible
 * Section, SliderRow / NumberRow / SelectRow / TextRow / CheckRow form rows,
 * ChipRow preset chips, Swatch color chips, Disclosure for advanced options,
 * and Button. One consistent dark-studio style; no external UI deps.
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

/** Collapsible panel section with an uppercase header. */
export function Section({
  title,
  children,
  defaultOpen = true,
  testId,
  onFirstOpen,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  testId?: string;
  /** Fires once, the first time the section becomes open (incl. mount). */
  onFirstOpen?: () => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const fired = useRef(false);
  useEffect(() => {
    if (open && !fired.current) {
      fired.current = true;
      onFirstOpen?.();
    }
  }, [open, onFirstOpen]);
  return (
    <div data-testid={testId} style={{ borderBottom: `1px solid ${KIT.divider}` }}>
      <button
        onClick={() => setOpen(!open)}
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
          justifyContent: 'space-between',
        }}
      >
        <span>{title.toUpperCase()}</span>
        <span style={{ opacity: 0.6 }}>{open ? '−' : '+'}</span>
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

/** Slider with a label + formatted value (+ optional numeric twin). */
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
  return (
    <label style={ROW}>
      <span>
        {label}
        <span style={VALUE}>
          {value.toFixed(dec)}
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
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{ flex: 1 }}
        />
        {withNumber && (
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={Number(value.toFixed(dec))}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (Number.isFinite(v)) onChange(v);
            }}
            style={{ ...INPUT_STYLE, width: 58 }}
          />
        )}
      </div>
    </label>
  );
}

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
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
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
