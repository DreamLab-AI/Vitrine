# Vitrine Onboarding

A small Rust/Axum + vanilla-JS web wizard that lets a curator describe an
exhibit and its objects, then round-trips everything into an ADR-013 / D-013.1
`exhibit.toml` manifest. Implements ADR-015 (D-015.2 schema-driven editor,
D-015.4 server-side secret containment).

## Build & run

```bash
cd onboarding
cargo run                      # builds and serves on 0.0.0.0:8088
# open http://localhost:8088
```

Environment variables:

| Var                  | Default     | Purpose                                   |
|----------------------|-------------|-------------------------------------------|
| `VITRINE_OUTPUT_DIR` | `./out`     | Where `exhibit.toml` and `.secrets.env` go |
| `VITRINE_STATIC_DIR` | `./static`  | Frontend asset directory                  |
| `RUST_LOG`           | `info`      | Tracing filter                            |

## Wizard flow

1. **Exhibit identity** — id, name, venue, date, curator, description.
2. **Drive source** — folder URL, rclone remote, recursive flag.
3. **Objects** — dynamic add/remove rows (id, name, SAM3 concept, description,
   priority `key|standard`, expected count).
4. **Secrets** — env-var **names** plus an optional token paste (warned it stays
   server-side).
5. **Pipeline & oversight** — `mesh_backend` (`tsdf|milo|come|gaussianwrapping|auto`),
   `matcher` (`exhaustive|aliked_lightglue`), `backend`
   (`claude_code|gemma_local`), `artifact_vlm` (`gemma_local|claude_code`).
6. **Review** — summary (token value never shown) then **Generate exhibit.toml**.

## API surface

| Method | Path            | Behaviour                                                        |
|--------|-----------------|-----------------------------------------------------------------|
| GET    | `/`             | Serves `static/index.html`                                      |
| GET    | `/api/health`   | `{"status":"ok"}`                                               |
| POST   | `/api/manifest` | Builds the manifest, writes `exhibit.toml`, returns `toml_path` |
| GET    | `/static/*`     | Static assets via `tower-http` `ServeDir`                       |

`POST /api/manifest` accepts:

```json
{
  "exhibit":  { "id": "...", "name": "...", "venue": "...", "date": "...", "curator": "...", "description": "..." },
  "drive":    { "url": "...", "rclone_remote": "gdrive", "recursive": true },
  "objects":  [ { "id": "obj-001", "name": "...", "sam3_concept": "...", "description": "...", "priority": "key", "expected_count": 1 } ],
  "secrets":  { "hf_token_env": "HF_TOKEN", "gcloud_credentials_env": "GOOGLE_APPLICATION_CREDENTIALS", "gcloud_project": "dreamlab-v2g", "hf_token_value": "(optional)" },
  "pipeline": { "mesh_backend": "come", "matcher": "aliked_lightglue" },
  "oversight":{ "backend": "claude_code", "artifact_vlm": "gemma_local" }
}
```

Response (no secret values):

```json
{ "ok": true, "toml_path": "./out/exhibit.toml", "secret_contained": true }
```

## Secret-containment guarantee

The emitted `[secrets]` block contains **only** `env:NAME` indirection
references (`hf_token = "env:HF_TOKEN"`), exactly as ADR-013 requires — the raw
token value is never written into the TOML and never returned in any HTTP
response.

If the client posts an optional `hf_token_value`, the server writes it to a
separate `<output>/.secrets.env` file with `0600` permissions
(`std::os::unix::fs::PermissionsExt`) and sets only a boolean `secret_contained`
flag in the response. The review pane in the browser also masks the value.

## Output paths

- `${VITRINE_OUTPUT_DIR:-./out}/exhibit.toml` — the manifest.
- `${VITRINE_OUTPUT_DIR:-./out}/.secrets.env` — `0600`, only if a raw token was
  pasted; holds `NAME=value` lines, never committed, never echoed.
