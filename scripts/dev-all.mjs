// Combined dev supervisor: backend (uvicorn :8765) + frontend (vite :5173)
// in one process tree, so the Claude preview registers a single server whose
// port is the UI (5173) — the preview pane never lands on the raw API again.
// Used by .claude/launch.json ("app"). `just dev` remains the human-facing way.
//
// If either child exits, the other is torn down and we exit with its code.
import { spawn } from 'node:child_process';

const children = [];
let shuttingDown = false;

function launch(name, cmd, args) {
  // shell:true because pnpm/uv resolve to .cmd shims on Windows; args contain
  // no spaces so cmd.exe quoting is a non-issue.
  const child = spawn(cmd, args, { stdio: 'inherit', shell: true });
  children.push(child);
  child.on('exit', (code) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.error(`[dev-all] ${name} exited (code ${code}); stopping the rest.`);
    for (const c of children) if (c !== child && c.exitCode === null) c.kill();
    process.exit(code ?? 1);
  });
  return child;
}

launch('backend', 'uv', [
  'run', '--directory', 'backend',
  'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8765',
]);
launch('frontend', 'pnpm', ['--dir', 'frontend', 'run', 'dev']);

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    shuttingDown = true;
    for (const c of children) if (c.exitCode === null) c.kill();
    process.exit(0);
  });
}
