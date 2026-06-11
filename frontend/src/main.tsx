import { createRoot } from 'react-dom/client';
import App from './App';
import { useStore } from './store';

// Expose the zustand store for E2E tests so they can read/write state
// deterministically without going through the DOM (React-controlled
// <input type="range"> swallows programmatic .value writes; driving the
// store is the reliable path). Safe in production — it's just a hook.
// The 3D scene additionally exposes window.__studio (three handles + setLid).
(window as unknown as { __store: typeof useStore }).__store = useStore;

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
