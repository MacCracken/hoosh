# ADR-015: The command line through cmdit

**Status:** Accepted (2.12.0)
**Date:** 2026-11-11

## Context

hoosh's command line was hand-rolled. 2.5.8 gave it flags: `--name value`, `--name=value` and `-s value`. 2.9.19
moved the scanners from `main.cyr` into `src/lib/cli.cyr`, so the suite ran them for real. Three functions read
argv one flag at a time: `cli_flag_val` (first-wins), `cli_flag_set` and `cli_positional`. Each verb asked them for
what it wanted, after `main()` had already read the config, detected the hardware and opened storage.

That shape had defects no single fix could remove:
- **Unknown or misplaced input was ignored.** `cli_positional` assumed every dash token but `--stream` takes a value.
  So an unknown flag ate the next positional, a dash-led prompt was skipped, and `--` counted as a flag.
  `serve -p` with no value, `serve -p abc`, `serve 8088 extra` and `infer a b c` all ran as if the bad part were
  absent.
- **The verb was always `argv(1)`.** `-c` was read wherever it stood, but `hoosh -c f.cyml serve` failed as
  `unknown command: -c`.
- **Nothing answered before setup.** Help and version ran after the config parse, hardware detection and storage
  open, and `hoosh serve --help` started the gateway.
- **Exit codes did not separate kinds of failure.** A usage error and a failed command both exited 1.

cmdit is the AGNOS CLI parser (agnosticos `planning/cmdit.md`, the 2026-06-25 ecosystem CLI review). It is
getopt-long with verb dispatch, and its API has been frozen since 1.0.0. 1.5.1 pins cyrius 6.7.18, hoosh's pin, and
its bundle has no git sub-deps and no name in common with hoosh's compile set.

## Decision

1. **hoosh's command line is a cmdit verb tool** (`[deps.cmdit]`, tag 1.5.1). Its handles:
   - **global:** `-c/--config`, and the auto `-h/--help` and `-V/--version`;
   - **verbs:** `serve`, `models`, `health`, `infer`, `info`, `help` and `version`, plus `bench`, hidden from help,
     the command list and completions (`cmdit_verb_hidden`);
   - **per verb:** each its own handle with its own flags and positional bounds.
2. **The whole command line is parsed first.** `cli_parse_argv(argc, argv, out)` fills a record of what to do and
   with what. It is pure (cmdit's `_argv` cores), so the suite drives it with synthetic argv. `main()` answers help,
   version and usage errors before anything else runs, and runs the verb from the record after setup.
3. **Usage errors exit 2, failed commands 1**, cmdit's `CMDIT_EXIT_*` (the AGNOS userland convention). A usage error
   prints cmdit's line, which names what was typed, then the verb's usage, or a pointer to `--help` before any verb.
4. **What stays as it was:**
   - `hoosh help` / `--help` keep the hand-written `cmd_help` on stdout, since cmdit has no renderer for its per-verb
     rows; `<verb> --help` is cmdit's generated help, also on stdout (`cmdit_help_fd(h, 1)`);
   - the version line stays `hoosh v<version>` on stdout;
   - no command still means `serve`;
   - `serve`'s bare positional port stays until 3.0.0 (the roadmap's backlog), and a given `-p` wins over it.
5. **hoosh checks the bare port and bench's count itself.** cmdit has no API to set an error, so those two carry a
   sentence in the record, and `cli_print_error` writes it with control bytes shown as `?`, as cmdit's lines do.

## Consequences

**Deliberate changes (CHANGELOG 2.12.0):**
- Unknown flags, a missing value, a bad or out-of-range port, and positionals past a verb's bound are exit-2
  errors. 2.11.x ignored them.
- Repeated flags are last-wins (cmdit's rule). They were first-wins.
- An unknown command exits 2, not 1, with cmdit's line and the command list on stderr.
- A dash-led prompt follows `--`.
- `-c` binds on either side of the verb, and `hoosh -c f.cyml` alone serves.
- Help, version and usage errors no longer touch the config, hardware or storage. `serve --help` prints serve's help.

**Costs:**
- The release build grows by about 41.7 KB (1.5%).
- A parse takes about 2.8 µs, once per process (`cli_parse_serve`).
- cmdit's one-pass caveats apply: a global value flag right before the verb can take the verb as its value
  (`hoosh --config serve ...`). Write `--config=serve` to be unambiguous.

**Dropped:** the three scanners, `cli_args`, `cli_infer_args` and `main.cyr`'s wrappers.

## Alternatives considered

- **Fix the scanners in place.** Teaching `cli_positional` every flag's arity, adding `--` and moving `-c` ahead of
  the verb is a parser rewrite, and one more hand-rolled parser in the ecosystem the CLI review set out to end.
- **The stdlib's `lib/flags.cyr`.** It has no verb dispatch, which is cmdit's reason to exist.

## References

- cmdit `README.md` ("Verb dispatch", the API reference), `docs/adr/0002-verb-dispatch.md`.
- agnosticos `docs/development/planning/cmdit.md`.
- CHANGELOG 2.5.8 (flags), 2.7.0 (`infer -m`), 2.9.19 (`cli.cyr`), 2.12.0.
