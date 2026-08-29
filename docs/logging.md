# Logging and verbosity

`repo2readme` writes progress to the console with Rich. Diagnostics go through
Python's `logging` and are routed through a Rich handler, so they no longer cut
across the progress bar.

## Verbosity flags

| Flag | Console level | Use it when |
|---|---|---|
| *(none)* | `WARNING` | Normal runs. |
| `-v` | `INFO` | You want to see what the run is doing. |
| `-vv` | `DEBUG` | Debugging, including third-party libraries. |
| `-q`, `--quiet` | `ERROR` | Scripted runs where only failures matter. |

```bash
repo2readme run --local . -v
repo2readme run --local . -vv
repo2readme run --local . --quiet
```

`--quiet` and `-v` together is a usage error rather than a silent
last-flag-wins, since it always means one of them was unintended.

## Third-party log noise

`langchain`, `httpx`, `httpcore`, `urllib3`, `openai`, `groq` and the Google
client libraries are capped at `WARNING`, so `-v` shows what `repo2readme` is
doing without drowning it in HTTP traces. `-vv` lifts the cap when you actually
need those traces.

## Writing a log file

```bash
repo2readme run --local . --log-file repo2readme.log
```

The file always records `DEBUG`, independently of the console level, which is
what makes it useful to attach to a bug report:

```
2026-08-13 11:02:41,880 WARNING  repo2readme.summarize.summary: Summary error for src/api.py: Error code: 429 ...
2026-08-13 11:02:41,881 DEBUG    repo2readme.cache: Cache hit for src/models.py
```

Combining `--quiet` with `--log-file` gives a silent console and a complete log
on disk.

If the path cannot be opened, the run prints a message and continues with
console logging rather than aborting.

## Markup in console output

Progress and results are printed through Rich, whose markup syntax uses square
brackets: `[green]...[/green]` renders in green. Style tags in those lines are
written by repo2readme itself.

Everything interpolated into them — a file path, an exception, a provider's
error text — is escaped first, so a bracket in a filename is shown rather than
parsed. That matters for dynamic-route files (`[slug].tsx`, `[id].vue`) and for
provider errors carrying JSON, either of which would otherwise be silently
truncated or, in the case of a stray closing tag, abort the line with a
`MarkupError`.

Diagnostics go through the logging handler, which is constructed with
`markup=False` and has never parsed markup, so `-v` output is unaffected either
way.

## Version

```bash
repo2readme --version
repo2readme, version 1.0.5
```

## See also

- [CLI reference](./cli-reference.md)
- [Troubleshooting](./troubleshooting.md)
