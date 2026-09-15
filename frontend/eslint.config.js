// @ts-check
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

/**
 * Flat config, deliberately minimal.
 *
 * This repo had NO linter until now, so every rule turned on here is a rule
 * that fires on existing code. The set below is the one that catches real
 * defects rather than style: the recommended TS rules (no type-aware pass —
 * that wants a project service and a slower run, and this host runs one heavy
 * process at a time), plus react-hooks, whose exhaustive-deps warning is the
 * only automated check on the dependency arrays that BoxScene's rebuild and
 * rebind effects live or die by.
 *
 * Formatting is NOT linted: prettier owns it (`pnpm format`), and this PR
 * deliberately does not reformat the tree — running the formatter over 18k
 * lines would bury every real change in a whitespace diff.
 */
export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'test-results/**', 'playwright-report/**'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // `_`-prefixed args/vars are the documented "deliberately unused" form
      // (see App.tsx's destructuring drops), and tsc's noUnusedLocals already
      // fails the build on the accidental ones.
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrors: 'none',
        },
      ],
    },
  },
  {
    // The e2e specs reach into window.__studio — an untyped debug surface by
    // design — and Playwright's page.evaluate hands back `any` from the page
    // context. Downgrading here keeps the rule meaningful in src/.
    files: ['tests/e2e/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  {
    files: ['**/*.js', '**/*.mjs'],
    ...tseslint.configs.disableTypeChecked,
  },
  {
    // scripts/shot.mjs is a node CLI that also ships browser code as strings to
    // Playwright's page.evaluate, so half its identifiers (window, fetch) are
    // resolved in the PAGE and the other half (process, console) in node.
    // no-undef cannot tell those apart without a globals package, and it is the
    // rule TypeScript already subsumes everywhere else in this project.
    files: ['scripts/**/*.mjs'],
    rules: { 'no-undef': 'off' },
  },
);
