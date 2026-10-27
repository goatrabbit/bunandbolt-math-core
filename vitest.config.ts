import { defineConfig } from 'vitest/config';

export default defineConfig({
  // High-order numerical quadrature can exceed the default 5 seconds on Node 22 or shared CI CPUs.
  // This is a functional suite, not a performance benchmark; all numerical tolerances stay in the tests.
  test: { include: ['tests/**/*.test.ts', 'examples/**/*.test.ts'], testTimeout: 30_000 },
});
