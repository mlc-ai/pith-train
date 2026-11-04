# pithtrain/kernels style guide

How code in `pithtrain/kernels/` is written. A change keeps the kernels' behaviour bit-identical
and their speed unchanged (section 9). Rules marked **(checked)** are enforced by
`tests/test_import_layers.py` or ruff; the rest are checked by review.

## 1. Interface and files

- **The package is a black box behind `__all__`.** `__init__.py` holds only one-line
  `from pithtrain.kernels.<module> import a, b` imports and `__all__ = []` built with grouped
  `__all__.append("name")` calls, one comment per feature area. Code outside the package imports
  `from pithtrain.kernels import <name>` only. **(checked)**
- **Every module has a literal `__all__ = [...]`**, and a module imports from a sibling only names
  in that sibling's `__all__`. **(checked)**
- **No star imports, no module `__getattr__`, no export dicts.**
- **Files are flat and named by feature, then architecture**: `<feature>.py` (the picker, or a
  kernel both architectures run), `<feature>_sm90.py` (CuTe-DSL), `<feature>_sm100.py` (Triton).
  Shared SM90-only helpers carry `_sm90`; arch-neutral shared leaves carry no suffix.
- **Lazy heavy imports.** A module the package imports eagerly never imports cutlass, QuACK or FA4
  at module level; they are imported inside the function that runs them. **(checked)**
- **Callers ask capability questions, not architecture questions**: `fp8_fused_experts_supported()`
  and similar, named for the feature. No architecture query is exported.
- **One file per feature per architecture**, forward and backward together, until a direction is
  its own pipeline or the file passes about 1.5k lines.

## 2. Architecture dispatch

- `arch.py` holds only constants probed once at import: `ARCH_MAJOR`, `SM_COUNT`, `NATIVE_E8M0`.
  No helper functions there.
- A picker dispatches with `match`, one explicit case per architecture, each importing its
  implementation inside the case, the default raising inline with this exact message:
  ```python
  match ARCH_MAJOR:
      case 9:
          from pithtrain.kernels.x_sm90 import f_sm90
          return f_sm90(...)
      case 10:
          from pithtrain.kernels.x_sm100 import f_sm100
          return f_sm100(...)
      case _:
          raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
  ```
- A single condition stays a comparison: `return ARCH_MAJOR == 9`, `if ARCH_MAJOR != 9: return None`.
- The picker exports the contract name (`fp8_quant_grad`); its implementations carry the file's
  suffix (`fp8_quant_grad_sm90`, `fp8_quant_grad_sm100`).

## 3. Names

A name says what the value is for. A reader should know a variable's role from its name alone,
without finding where it is assigned.

### 3.1 Never

- **No leading underscore anywhere**: module names, classes, methods, attributes, locals,
  parameters, import aliases. `__all__` says what is public. Allowed: `_` as a throwaway, dunders,
  and reads of a library's own `_` API (`mod._mint`, `torch._C`). **(checked)**
- **No unexplained one- or two-letter names, or abbreviations a reader has to decode** (`v`, `r`,
  `c`, `q`, `w`, `g`, `e`, `rv`, `sw`, `cl`, `qq`, `c4`, `rw`, `st`, `cnt`, `dpre`, `gw`).
- **No two spellings of one concept** (`row_ok` and `rok`, `scale_t` and `tscale`, `pipe_v` and
  `pipe_V`). Pick one and use it everywhere in the file.
- **No name that shadows a builtin** (`compile`, `max`, `min`, `sum`, `id`, `iter`, `len`).

### 3.2 Allowed short names

These are conventional and stay short:

| Name | Meaning |
|---|---|
| `M`, `N`, `K`, `G`, `H`, `T`, `B` | shape dimensions: rows, columns, reduction, groups (experts), hidden, tokens, batch |
| `BLOCK`, `BLOCK_M`, `TILE`, `THREADS`, `WARPS` | compile-time tile and launch sizes (upper case) |
| `i`, `j`, `k` | loop counters over a short constexpr range, only when the loop body is short |
| `tidx`, `lane`, `warp`, `bx`, `by`, `bz` | thread index, lane in warp, warp in block, block indices |
| `x`, `out`, `y` | a function's main input and output tensors |
| `fn`, `key` | a compiled kernel and its cache key |
| `amax`, `scale`, `rcp` | absolute maximum, FP8 scale, reciprocal of the scale |
| `_` | a value that is ignored |

### 3.3 Role vocabulary

Names are built from these parts, so the same kind of value always reads the same way:

| Pattern | Use for | Examples |
|---|---|---|
| `num_<things>` | a count | `num_groups`, `num_tiles`, `num_warps` |
| `<thing>_idx` | an index into a sequence | `expert_idx`, `row_idx`, `group_idx` |
| `<thing>_start`, `<thing>_end` | a half-open range | `row_start`, `group_end` |
| `is_<state>`, `has_<thing>`, `use_<option>` | a boolean, or a compile-time switch | `is_last`, `has_expand`, `use_smem` |
| `<thing>_ok` | an Int32 / mask predicate guarding a load or store | `row_ok`, `col_ok` |
| `acc_<what>` | an accumulator | `acc_amax`, `acc_sum` |
| `<tensor>_base` | the address of a tensor's first element | `x_base`, `scale_base` |
| `<thing>_addr` | a computed element address | `row_addr`, `scale_addr` |
| `s_<buffer>` | a shared-memory buffer (CuTe tensor or address) | `s_amax`, `s_tile` |
| `m<Name>` | a `cute.Tensor` argument of a CuTe `__call__` / kernel in global memory | `mX`, `mOut`, `mOffs` |
| `<dim>_offs` | Triton offset vectors | `row_offs`, `col_offs` |
| `<tensor>_ptrs`, `<what>_mask` | Triton pointer blocks and their masks | `out_ptrs`, `row_mask` |
| `<value>_bits` | an integer holding a float's bit pattern | `scale_bits`, `amax_bits` |
| `<value>_t`, `<value>_kmajor` | the transposed or K-major copy of a value (never a type: types are `<value>_type`) | `out_t`, `scale_t`, `tok_kmajor` |
| `<value>_lo`, `<value>_hi` | the two halves of a packed value | `words_lo`, `words_hi` |

Where a kernel implements a formula, use the formula's symbols for its matrices and their
gradients (`gate`, `up`, `grad_gate`, `dX`), the same in the code as in the docstring.

### 3.4 Functions and classes

- Functions are verbs or verb phrases (`load_rows`, `reduce_amax`, `store_tile`); queries are
  `<feature>_supported`. Kernel classes are nouns (`BlockwiseTranspose`).
- A name that would clash with a builtin or an import gets a more specific name
  (`compile_kernel`, `ptx_f32`, `struct_fields`), not an underscore.
- Generic names in the flat namespace get a feature prefix (`bf16_expert_gate_up_dgrad`, not
  `dgrad`).
- Exported names (anything in `__all__` of `__init__.py`) and their parameters are the public
  contract: renaming one is an interface change, made across every caller at once.

## 4. Structure inside a kernel

A kernel reads as a short sequence of named phases. The kernel function sets up its coordinates
and buffers, then calls one function per phase; each phase is small enough to read at once.

### 4.1 The kernel body

- **The kernel entry is an outline.** A `@cute.kernel` method or `@triton.jit` kernel holds its
  setup (thread / block coordinates, shared-memory allocation, base addresses) and then the calls
  that make up the algorithm, in order:
  ```python
  @cute.kernel
  def kernel(self, x_base: Int64, out_base: Int64, M: Int32, K: Int32):
      tidx, warp, lane, row0, col0 = thread_coords()
      s_amax = allocate_amax_buffer()
      rows = load_rows(x_base, row0, col0, M, K)
      amax_bits = reduce_block_amax(rows, warp, lane, s_amax)
      quant = quantize_rows(rows, amax_bits)
      store_rows(out_base, quant, row0, col0, M, K)
  ```
- **Split when a kernel passes about 80 lines**, or sooner when it holds two clear phases. Split
  along the algorithm's phases (load, reduce, quantize, transpose, store, reduce across warps),
  not at arbitrary line counts. Do not split out a piece shorter than about 10 lines unless it is
  reused or names a step the reader would otherwise have to work out.
- **Helpers are macros.** `@cute.jit` functions, `dsl_user_op` wrappers and `@triton.jit` helpers
  are inlined when the kernel compiles: calling one costs nothing at runtime, so use them the way C
  code uses macros, to give every step a name. Three levels, each built from the one below:
  - **instruction wrappers**: one PTX instruction or a short fixed sequence, as a `dsl_user_op`
    (`ld_v4`, `prmt`, `max_u32`, `cast4`), only when the PTX is used from two or more places
    (section 5); PTX used once is written inline where it is used;
  - **building blocks**: a few lines of arithmetic or data movement over registers, as a
    `@cute.jit` / `@triton.jit` function (`transpose4`, `swizzle`, `thread_coords`);
  - **phases**: one step of the algorithm (`load_rows`, `reduce_block_amax`, `store_tile`).
  The kernel entry calls phases; phases call building blocks and wrappers; nothing calls upward.
  Section 9 confirms the speed is unchanged.
- **A section banner is not a substitute for a function.** If a kernel needs a comment like
  `# --- store ---` to be readable, that section is a phase function.

### 4.2 Phase functions

- **Named for what they do**, a verb plus the object: `load_tile`, `reduce_row_amax`,
  `quantize_block`, `transpose_block`, `store_kmajor`, `find_group`.
- **Arguments say what the phase needs, no more.** When a phase needs more than about 8 values
  that always travel together (a tile's coordinates and bounds, a table row's fields), group them
  in a small class or a `@cute.jit` helper that computes them (`Tile(...)`, `thread_coords()`).
  Constants of the kernel instance stay on `self` (`self.N`, `self.stages`) instead of being passed.
- **Returns its results**; it does not mutate shared Python lists it was handed unless the caller
  builds them for that purpose.
- **A plain Python method or function is a fine phase.** Called from a `@cute.kernel`, it is traced
  inline like a `@cute.jit` helper. Keep the decorator the code already used: `@cute.jit` turns a
  Python `range` loop into a dynamic loop, which breaks loops that build Python lists of registers
  (those stay `range` / `range_constexpr` in an undecorated phase).
- **Unpack a coordinates object once.** A phase that reads several fields of a bundle (`Tile`,
  `SubBlock`) binds the ones it uses at its top under the same names
  (`lane, warp, row0 = sub_block.lane, sub_block.warp, sub_block.row0`) instead of repeating
  `sub_block.` in every expression; the spelling stays the same.
- **Compile-time variation stays compile-time.** Optional behaviour is a `const_expr` switch or a
  callable argument (`activation`, `mask_fn`) checked with `const_expr(fn is not None)`, so a
  disabled feature generates no code and the main path stays one function.
- **Order is part of the behaviour.** Do not reorder loads, stores, barriers, fences or
  side-effecting `asm` while extracting a phase; the extracted function issues them in the same
  order as before. In Triton the order of computations matters too: a helper that takes an
  operand computed earlier than before (an fp32 copy widened up front instead of after the amax)
  changed the schedule and cost 19% on `rowwise_transpose_sm100`. Extract a step only with its
  inputs computed where they were.

### 4.3 Warp-specialized kernels

When different warps of a CTA do different jobs (a producer warp loading tiles while consumer
warps compute, a scheduler warp handing out work, an epilogue warp storing), the kernel entry
dispatches on the warp's role and gives each role its own method:

```python
@cute.kernel
def kernel(self, ...):
    # setup shared by every role: coordinates, shared memory, barriers / pipelines
    if warp_idx in self.load_warps:
        self.load_loop(...)
    if warp_idx in self.compute_warps:
        self.compute_loop(...)
    if warp_idx == self.store_warp:
        self.store_loop(...)
```

- **One method per role**, named `<role>_loop` (or `<role>` when it runs once), holding that role's
  whole loop. Per-iteration work inside a role is a step method (`compute_step`, `load_tile`).
- **The warp ids of each role are attributes set in `__init__`** (`self.load_warps`,
  `self.store_warp`), never literals in the dispatch, so the role layout is stated in one place.
- **Roles talk only through the objects created in the shared setup** (barriers, pipelines,
  shared-memory buffers), passed in explicitly; a role never reaches into another role's state.
- A kernel where every warp does the same thing (most of ours today) has no roles: it is one
  outline of phases (4.1).

### 4.4 Host-side launchers

- The launcher (the plain Python function the picker calls) does, in order: input checks and
  layout fixes (`contiguous`), output allocation, the compile-cache lookup, the launch. Each step
  may be a helper (`store_mode`, `compile_blockwise`) when it is more than a few lines.
- Compile caches are one module-level dict per file, keyed by a tuple that starts with a string
  naming the kernel.

## 5. Shared helpers

Code that appears in more than one file lives once, in the shared module for its kind:

| Kind | Module |
|---|---|
| Inline-PTX `dsl_user_op`s used by two or more files: loads, stores, shuffles, conversions (`ld_v4`, `st_v4`, `prmt`, `max_u32`, ...) | `cute_ptx_sm90.py` |
| E8M0 scale math and FP8 quantization steps (`e8m0_scale`, `quant4_e4m3`, the 4-way amax) | `fp8_e8m0_scale_sm90.py` |
| Fake tensors and streams for `cute.compile` (`fake`, `fake_stream`, `TORCH2CUTE`) | `cute_compile_sm90.py` |
| Layout of FP8 scales | `fp8_scale_layout.py` |
| The Triton E8M0 scale | `fp8_scale.py` |

- **A PTX wrapper earns its place by reuse.** Where it lives follows how widely it is used:
  - **used from one place**: no wrapper. The `asm(...)` call is written inline in the building
    block or phase that uses it, with a short comment naming the instruction when the PTX does not
    make it obvious. A single-use function that builds several instructions over several
    statements is a building block, not a wrapper: it stays, named for what it computes.
  - **used from two or more places in one file**: a wrapper in that file.
  - **used by two or more files**: a wrapper in `cute_ptx_sm90.py`, defined there once and
    imported; no kernel file keeps its own copy. `cute_ptx_sm90.py` holds nothing else.

  Building blocks and phases are different: a phase called once is still justified, because it
  names a step of the algorithm (4.1).
- **Identical means identical.** Two helpers with the same name merge only when their PTX,
  constraints, predication and `side_effects` flag are the same. If they differ, they are
  different helpers: give the variant a name that says how (`ld_v4_pred`, `ld_v4_nc`) and keep
  both.
- **A phase repeated across features moves to a shared module too** (the amax over four words, a
  transposed-tile store), named by what it does, with the same signature for every caller.
- **A helper used by one file stays in that file**, below the code that uses it or above it, but
  not in a shared module "for later".
- Moving a helper changes no behaviour: the moved function is byte-for-byte the same code.

## 6. Layout and formatting

- **The formatter is off for kernels** (`ruff format` and its pre-commit hook skip
  `pithtrain/kernels/`; I001 is ignored there). The other lint rules apply. **(checked)**
- **No line-length limit for code.** Every statement without a comment sits on one line: calls,
  signatures, imports (`from x import a, b, c`, no parentheses), tables. Very long argument lists
  (over about 256 columns) may be packed onto a few full lines.
- **Multi-line PTX stays as written**: triple-quoted assembly, and assembly built one string per
  line, are never joined.
- **Blank lines mark steps, not decoration.** One blank line separates the steps of a function
  (the phases of a kernel entry; the load, reduce and store of a phase); a step has no blank line
  inside it, a block opener (`if`, `for`, `with`) is not followed by one, and a function-local
  import is not followed by one. Two blank lines separate top-level definitions, one separates
  methods, never more. A comment about one definition sits directly above it; a comment about a
  group of definitions stands alone, with two blank lines on each side.
- **Parallel assignment** for consecutive, independent, aligned assignments:
  ```python
  pid_row, pid_col = tl.program_id(0), tl.program_id(1)
  lane, warp = tidx % 32, tidx // 32
  ```
  - Independent: no right-hand side reads a target of the same statement.
  - Aligned: the same called function, the same operator, all plain names / constants, or one
    value split by different operators (a quotient and remainder).
  - Not merged: unrelated values that happen to sit together (a load next to an offset, two
    constants that only share a value), or an assignment that a comment above describes alone.
  - At most 4 targets and 160 columns; no blank line between two such groups.

## 7. Docstrings and comments

- **Docstring quotes on their own lines**, one-line docstrings included:
  ```python
  def load_rows(...):
      """
      Load the thread's 8x8 block; rows outside [row_lo, row_hi) read as zero.
      """
  ```
- **Plain text**, no reST, roles or backticks; wrapped at 100 columns; no paragraph ends on a line
  of fewer than 3 words.
- **Contract only.** An exported function states what it computes, input shapes / dtypes / layout
  requirements, outputs (and scale layout where it matters), in-place behaviour, and SM90-only with
  its gate. An implementation behind a picker gets one line, or only what differs. A phase function
  gets one line saying what it does when its name does not say it all.
- **No history**: no round numbers, branch names, measurement stories or paths.
- **CuTe tensors whose layout is not obvious get their modes in a trailing comment**, the one
  annotation that says what rather than why, because a tensor's mode order cannot be read off its
  name: `tOrP = thr_mma.partition_A(sP)  # (MMA, MMA_M, MMA_K)`.
- **Comments explain a non-obvious why**: a numerical trick, a hardware constraint, a race, why an
  order matters, why a constant has its value. Comments restating code, section banners and stale
  `# noqa` / `# fmt: skip` markers go.
- **Dead code goes**: unused locals, parameters, imports and helpers (search the whole repository,
  tests included, before removing a helper).

## 8. Tests

- Every exported name has a pure-PyTorch verifier in `tests/kernels/test_<feature>.py` that imports
  only `pithtrain.kernels` names. FP8 bytes and E8M0 scales are compared exactly; GEMMs with
  `calc_diff` or tight tolerances. **(checked: coverage)**
- Never verify against an `_sm100` sibling, and never add an architecture override to run
  `_sm100` code on Hopper.
- A style change adds no tests. Tests that only record a past refactor are not kept.

## 9. Verifying a change

1. **Static**: `ruff check pithtrain/kernels` and `pytest tests/test_import_layers.py` (CPU only)
   pass.
2. **Layout-only changes** (line joins, docstrings, comments, parallel assignment): the AST is
   unchanged once docstrings are set aside and parallel assignments are expanded back into
   separate statements. No GPU run is needed.
3. **Anything the compilers see** (renames, extracted phases, moved helpers, removed dead code):
   - the verifiers in `tests/kernels` of every feature that imports the changed module pass;
   - every kernel those features launch, run against the code before the change on the same GPU,
     gives bit-identical outputs and a median time within 5% (+3 us for very short kernels).
     Timings are taken with nothing else running on the node: kernels timed side by side with
     other jobs vary by more than that. A timing over the limit is retaken once before it counts;
     a real regression is fixed (usually an order change, 4.2, or a lost `const_expr`) or the
     change to that kernel is dropped.

## 10. Per-file checklist

- [ ] No `_name` definitions; `__all__` literal and complete.
- [ ] Every name follows section 3; no decoding needed; one spelling per concept.
- [ ] Kernel entries are an outline of phase calls; no kernel over about 80 lines without reason.
- [ ] Steps named as helpers at the right level (wrapper, building block, phase); nothing calls upward.
- [ ] A warp-specialized kernel dispatches on role, one method per role, role warp ids as attributes.
- [ ] PTX wrappers only for PTX used from 2+ places (in the file if one file uses it, in
      `cute_ptx_sm90.py` if several do); single-use PTX inline; nothing duplicated.
- [ ] Statements on one line; parallel assignments where aligned; PTX untouched.
- [ ] Docstrings: quotes on own lines, contract only, 100 columns, no orphans; comments say why.
- [ ] No dead code.
- [ ] Section 9 checks pass.
