# Tach negative examples

Each child directory is a complete Tach project with exactly one `.tach` file. These projects intentionally demonstrate compiler failures or successful builds with warnings. They are test fixtures as well as small, copyable explanations of what Tach rejects and what it can prove is suspicious.

| Project | Result | Diagnostic codes |
| --- | --- | --- |
| `syntax-errors` | error | `parser` |
| `lexer-errors` | error | `lexer` |
| `manifest-error` | error | `manifest` |
| `name-and-type-errors` | error | `semantic` |
| `control-flow-errors` | error | `semantic` |
| `documentation-errors` | error | `semantic` |
| `divergent-barrier` | error | `semantic` |
| `missing-import` | error | `import` |
| `name-collision` | error | `name` |
| `dead-code` | warning | `discarded-value`, `unreachable-function`, `unused-binding` |
| `launch-and-control` | warning | `constant-condition`, `zero-dispatch` |
| `memory-access` | warning | `constant-write-index`, `strided-access` |
| `no-effect-kernel` | warning | `no-effect-kernel`, `unused-binding` |

Run `tach check` inside a project for the human report, or `tach check --json` for the stable machine-readable record.
