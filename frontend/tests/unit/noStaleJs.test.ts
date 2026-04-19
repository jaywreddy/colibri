/**
 * The single most valuable test in this repo.
 *
 * Background: Vite's moduleResolution="bundler" resolves extensionless imports
 * to `.js` BEFORE `.tsx`/`.ts`. If stale compiled `.js` files sit next to their
 * TypeScript sources (e.g. from a past `tsc --emit` run), the `.js` wins and
 * every edit to the `.tsx` silently has zero effect. This class of bug cost
 * multiple hours of debugging before it was understood — this test ensures it
 * cannot happen again without a red suite.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, '..', '..');
const SRC = join(FRONTEND_ROOT, 'src');

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

describe('stale .js shadow guard', () => {
  const files = walk(SRC);

  it('no `.js` sibling exists for any `.tsx`/`.ts` source', () => {
    const tsFiles = files.filter(
      (f) => (f.endsWith('.ts') && !f.endsWith('.d.ts')) || f.endsWith('.tsx')
    );
    const offenders: string[] = [];
    for (const ts of tsFiles) {
      const sibling = ts.replace(/\.tsx?$/, '.js');
      if (files.includes(sibling)) offenders.push(sibling);
    }
    expect(offenders, `stale .js files found (delete them):\n${offenders.join('\n')}`)
      .toEqual([]);
  });

  it('tsconfig.json has noEmit=true so tsc cannot re-create the shadows', () => {
    const raw = readFileSync(join(FRONTEND_ROOT, 'tsconfig.json'), 'utf-8');
    // Strip line comments for safe JSON.parse
    const stripped = raw.replace(/^\s*\/\/.*$/gm, '');
    const cfg = JSON.parse(stripped);
    expect(cfg.compilerOptions?.noEmit).toBe(true);
  });
});
