/*
 * m68k_cycles.h — the MC68000 instruction timing model, one source for the
 * recompiler (the costs it stamps into generated code) and the runner's
 * interpreter (the costs it charges for code it runs, e.g. RAM-resident code
 * in exact mode). Both must agree or the interpreted and generated machines
 * drift apart on cycle-exact comparisons.
 */
#pragma once
#include "m68k_decoder.h"

/* PRM model (Appendix B), the static cost of one instruction in 68000 cycles. */
int m68k_cycles_prm(const M68KInstr *instr);
/* Exact mode: the cost before any data-dependent part, matching the reference
 * emulator (Musashi via Genesis Plus GX); the data-dependent deltas (branch not
 * taken, DBcc exit, register shift counts, MUL/DIV operands, Scc Dn true) are
 * charged at run time by whoever executes the instruction. */
int m68k_cycles_exact(const M68KInstr *instr);
/* Dn or An (the register-operand test the timing rules use). */
int m68k_ea_is_reg(int ea);
