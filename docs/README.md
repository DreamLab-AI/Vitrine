# Documentation — Vitrine / LichtFeld Studio

> **Rename note (2026-06-04):** The pipeline product is renamed from **Video-to-Gaussian** to
> **Vitrine** (a museum display case; pairs with the upstream LichtFeld Studio engine). CLI/package
> id: `vitrine`. The web setup tool is **Vitrine Onboarding**. This is a documentation-level rename
> only — code identifiers still say `video2splat` / `gaussian-toolkit`; a full code rename is a
> separate scheduled follow-up (ADR-015).

The documentation is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## v3 Architecture (Proposed)

The v3 design introduces a single-manifest, agent-orchestrated pipeline. Key new pages:

- [Vitrine Onboarding](onboarding.md) — the user-facing entry point: wizard, hardware selection, secret containment, and hand-off to the Claude Code overseer.
- [v3 Pipeline Design](architecture/v3-pipeline.md) — `v2g-net`, `exhibit.toml`, serial model lifecycle, and agent-controlled ComfyUI recovery.
- [Architecture overview](architecture.md) — existing v2 deployment plus a clearly-marked v3 section.

## Prerequisites
- [Node.js](https://nodejs.org/) (>=18.0)
- [pnpm](https://pnpm.io/installation) (Package manager)

## Local Development
All commands are run from the `docs` directory.

```bash
pnpm install
```

This command installs all dependencies.

```bash
pnpm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
pnpm build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.
