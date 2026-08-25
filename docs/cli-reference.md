# CLI Reference

## `repo2readme run`

Generates a `README.md` by analyzing a repository (git URL or local path).

```bash
repo2readme run [OPTIONS]
```

### Options

| Flag | Short | Description |
|---|---|---|
| `--url <URL>` | `-u` | Git repository URL to clone and process. Any form `git clone` accepts. |
| `--local <PATH>` | `-l` | Path to a local repository. |
| `--output <FILE_PATH>` | `-o` | File path to save the generated README. Printed to stdout when omitted. |
| `--force` | `-f` | Overwrite the output file and skip the token estimation confirmation prompt. |
| `--backup` | | Keep a copy of the file being replaced, alongside it with a `.bak` suffix. |
| `--create-dirs` / `--no-create-dirs` | | Create the output file's parent directory when missing. On by default. |
| `--strict` | | Exit with a non-zero status if any file fails to summarize. |
| `--respect-gitignore` | | Honor `.gitignore` and `.git/info/exclude` patterns during repository traversal. This is opt-in; default behavior is unchanged. |
| `--dry-run` | | Preview the analysis (repo tree, token estimate, files to be processed) without making any API calls or requiring API keys. |
| `--include <PATTERN>` | | Glob pattern for files to include, even if normally filtered out. Can be passed multiple times. |
| `--exclude <PATTERN>` | | Glob pattern for files to exclude. Can be passed multiple times. |
| `--max-content-chars <N>` | | Characters of a file to send in one request. Defaults to 40000. `0` sends files whole. |
| `--max-file-size-kb <N>` | | Skip files larger than N KB. |
| `--provider <NAME>` | | LLM provider to use. See `repo2readme providers`. |
| `--model <NAME>` | | Model name. Defaults to the selected provider's default model. |
| `--base-url <URL>` | | Base URL override for OpenAI-compatible providers. |

You must provide exactly one of `--url` or `--local`.

### `--url`

Any URL form `git clone` accepts works, not just `https://github.com/...`:

```bash
repo2readme run --url https://github.com/user/repo
repo2readme run --url https://github.com/user/repo.git
repo2readme run --url git@github.com:user/repo.git
repo2readme run --url ssh://git@github.com/user/repo.git
repo2readme run --url git://github.com/user/repo.git
repo2readme run --url https://gitlab.com/group/subgroup/repo
repo2readme run --url https://git.company.internal/team/service.git
```

Schemes are matched case-insensitively, and `git+ssh://` / `git+https://`
prefixes are accepted. A `file://` URL is treated as a local path. Anything
else (`s3://`, for example) is rejected up front with the list of supported
schemes, rather than being read as a directory name.

The repository is cloned shallowly (`--depth 1`) into a private temporary
directory that is removed when the run finishes, so two runs against the same
repository cannot interfere with each other.

### `--branch`

Branch to clone when using `--url`. Defaults to `main`.

```bash
repo2readme run --url https://github.com/user/repo --branch develop
```

### `--output`

Where to save the generated README. Without it, the README is printed to
stdout.

```bash
repo2readme run --local . --output README.md
repo2readme run --local . --output docs/generated/README.md   # directory created
```

The path is checked when the command starts, not when the file is written — a
path that cannot be written fails in a second, with exit code 2, instead of
after the whole repository has been summarized:

```
/path/to/repo is a directory. Give --output a file path, for example
/path/to/repo/README.md.
```

Missing parent directories are created. Pass `--no-create-dirs` to have a
missing directory be an error instead.

The write itself goes through a temporary file in the destination's own
directory and then an atomic rename, so an interrupted or failed write leaves
the previous README exactly as it was. `--backup` additionally keeps a copy of
what was replaced:

```bash
repo2readme run --local . --output README.md --backup
# Saved to /path/to/README.md
# Previous version kept at /path/to/README.md.bak
```

If the write fails anyway, the generated README is printed to stdout rather
than discarded, and the run exits 1.

### `--max-workers`

Number of worker threads used to read and process files. Defaults to 4, capped
at the number of files. This applies to both `--local` and `--url` runs.

```bash
repo2readme run --url https://github.com/user/repo --max-workers 8
```

### `--max-content-chars`

How much of a file to put in one request. Defaults to 40,000 characters —
roughly 13,000 tokens, which leaves room for the prompt and the answer inside a
16k context window.

A longer file is sent as its head and its tail with the middle replaced by a
marker:

```
line 0: def build_client():
...
... 7,891 line(s) omitted from the middle ...

    return Client(session)
```

Imports and module docstrings live at the top, the public entry points and the
`main` guard at the bottom, so the two ends carry most of what a summary needs.
The cut is always made on line boundaries, so the excerpt is never spliced
through the middle of a statement, and the summary records `"truncated": true`
along with how many lines were left out.

```bash
repo2readme run --local . --max-content-chars 80000   # a large context window
repo2readme run --local . --max-content-chars 0       # send files whole
```

Without a budget, a file just under `--max-file-size-kb` went to the model in
one piece — around 200 KB, or 60–70 thousand tokens. Most providers reject that
outright, and the rejection is classified as permanent (retrying an identical
oversized request cannot succeed), so the file was dropped from the run after
being paid for. Lowering `--max-file-size-kb` is not the same fix: it drops the
same file earlier instead of describing it.

The budget is part of the cache key, so changing it re-summarizes the files it
affects.

### `--respect-gitignore`

Honor `.gitignore` and `.git/info/exclude` patterns during repository traversal. This is opt-in, so the default behavior remains unchanged. When enabled, files and directories matching gitignore rules are skipped before language detection, parsing, summarization, and token estimation.

```bash
repo2readme run --local ./repo --respect-gitignore
repo2readme run --url https://github.com/user/repo --respect-gitignore
```

A `.gitignore` in a subdirectory applies to that subdirectory's subtree, the
way git applies it, so a monorepo that keeps its ignore rules next to each
package is honored:

```
.gitignore                  # *.log
frontend/.gitignore         # build/
frontend/build/bundle.js    # skipped
frontend/src/app.js         # analyzed
```

`.git/info/exclude` is repository-wide and is only read at the repository root.
`core.excludesFile` is deliberately not consulted: it is per-machine, and
honoring it would make the same repository produce a different README on a
different machine.

### `--strict`

Summarization is best-effort: a file that fails (rate limit, timeout, bad
response) is skipped and the README is generated from the rest. Every run
prints a report when that happens:

```
Summarization report

Succeeded          : 37/40
Failed             : 3/40

3 file(s): Error code: 429 - rate limit reached for model ...
    - src/api/client.py
    - src/api/routes.py
    - src/api/schema.py

The README was generated from the files that succeeded.
```

Failed files are never passed to the README prompt. If *every* file fails the
run stops with exit code 1 instead of asking the model to write a README from
nothing.

Use `--strict` in CI to turn any failure into a non-zero exit code. The README
is still written first, so you keep the partial result:

```bash
repo2readme run --local . --output README.md --force --strict
```

### `--dry-run`

Runs local analysis only — repo tree generation, file filtering, and token estimation — with no LLM calls and no API keys required. Useful for verifying your include/exclude filters before spending tokens.

```bash
repo2readme run --local ./path/to/your/repo --dry-run
```

Example output:

```
Repository Tree

project/
├── src/
├── tests/
└── README.md

Files to be processed

✓ src/main.py
✓ src/api.py
✓ tests/test_api.py
...

Repository Analysis

Files selected     : 45
Estimated tokens   : ~120,000
Request size       : ~420.5 KB

Dry run complete.
No API requests were made.
```

## `repo2readme providers`

Prints the supported LLM providers with their aliases, default models and API
key environment variables.

```bash
repo2readme providers
```

```text
        Supported providers
Provider    Aliases  Default model            API key env var
groq        -        openai/gpt-oss-120b      GROQ_API_KEY
google      gemini   gemini-2.5-flash         GOOGLE_API_KEY
...
ollama      -        llama3                   not required
```

## `repo2readme reset`

Deletes the locally stored API key configuration file (`~/.repo2readme_env.json`).

```bash
repo2readme reset
```

You'll be prompted to re-enter your `GROQ_API_KEY` and `GOOGLE_API_KEY` on the next `run`.

## See also

- [Configuration](./configuration.md) — API keys and env vars
- [Examples](./examples.md) — real command examples
