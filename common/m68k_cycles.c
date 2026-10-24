/*
 * m68k_cycles.c — the MC68000 instruction timing model (see m68k_cycles.h).
 * Moved verbatim from profiles/genesis/code_generator.c.
 */
#include "m68k_cycles.h"
#include <stdint.h>

/* =========================================================================
 * Cycle-accurate VBlank: per-instruction 68K cycle cost.
 *
 * Values derived from the Motorola 68000 Programmer's Reference Manual
 * (Appendix B, "Instruction Timing"), cross-checked against the
 * interpreter's cost code in
 *   clownmdemu-core/libraries/clown68000/source/interpreter/clown68000.c
 * lines 399-463 (EA-mode costs) and per-instruction microcode.
 *
 * Data-dependent costs (MULx/DIVx result magnitude, Dn-register shift
 * counts, BTST on memory with dynamic bit index) use averages near the
 * mid-range of the PRM spec. Structural costs (size, addressing mode,
 * MOVEM register count, immediate shift count) are exact.
 *
 * Callers emit `g_cycle_accumulator += N;` once per non-terminator
 * instruction; the accumulator fires glue_check_vblank when it crosses
 * g_vblank_threshold (109,312 cycles = scanline 224).
 * ========================================================================= */


/* popcount of a 16-bit mask — used for MOVEM register count. */
static int popcount16(uint16_t v) {
    int c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}

/* ===========================================================================
 * MC68000 instruction timing model (Programmer's Reference Manual, Appendix
 * B / "Instruction Execution Times").
 *
 * DECODER CONTRACT — the thing that made the old table wrong. `dst_ea` is
 * populated by exactly ONE decoder path: the MOVE/MOVEA family
 * (m68k_decoder.c:239). Every other instruction leaves `dst_ea` at its
 * memset-0 value, which reads back as "Dn". Any cost rule that keyed off
 * `dst_ea` therefore silently skipped the operand EA cost for CLR, TST, NEG,
 * NOT, ADDQ/SUBQ, the immediate-arith family, the bit instructions, Scc and
 * the memory shifts — a systematic undercount on ~27% of all instructions.
 * The operand EA for every non-MOVE instruction lives in `src_ea`.
 *
 * STATIC-STAMP POLICY. These costs are stamped once per address at codegen
 * time and summed into g_audio_cycle_counter at runtime, so instructions
 * whose real cost depends on runtime state must be pinned to one documented
 * value:
 *   - Bcc / DBcc / BRA / BSR      -> the TAKEN cost. Branch-dominated loops
 *                                    are the common case, and taken is a real
 *                                    hardware value rather than an average.
 *   - Scc                         -> the TRUE cost (Dn form; 6 not 4).
 *   - Register-count shifts       -> average count of 4.
 *   - MULU / MULS                 -> 38 + 2*8, the average-operand form.
 *   - DIVU / DIVS                 -> PRM worst case (the spread is narrow).
 * ======================================================================== */

/* PRM Table 8-1 "Effective Address Calculation Times" — cycles to compute
 * and FETCH an operand through this EA, on top of the instruction base.
 * Size-dependent: a longword costs one extra memory round-trip. */
static int ea_calc_cost(int ea, M68KSize sz)
{
    int is_long = (sz == M68K_SIZE_L);
    int mode    = (ea >> 3) & 7;
    int reg     =  ea       & 7;
    if (ea < 0) return 0;                      /* -1 = immediate pseudo-operand */
    switch (mode) {
    case 0: /* Dn */           return 0;
    case 1: /* An */           return 0;
    case 2: /* (An)        */  return is_long ? 8  : 4;
    case 3: /* (An)+       */  return is_long ? 8  : 4;
    case 4: /* -(An)       */  return is_long ? 10 : 6;
    case 5: /* d16(An)     */  return is_long ? 12 : 8;
    case 6: /* d8(An,Xn)   */  return is_long ? 14 : 10;
    case 7:
        switch (reg) {
        case 0: /* (xxx).W  */ return is_long ? 12 : 8;
        case 1: /* (xxx).L  */ return is_long ? 16 : 12;
        case 2: /* d16(PC)  */ return is_long ? 12 : 8;
        case 3: /* d8(PC,Xn)*/ return is_long ? 14 : 10;
        case 4: /* #imm     */ return is_long ? 8  : 4;
        }
    }
    return 0;
}

/* MOVE DESTINATION cost. Distinct from ea_calc_cost: on a MOVE destination
 * the predecrement is overlapped with the write, so -(An) costs the same as
 * (An) (4/8) rather than the 6/10 an operand FETCH through -(An) costs.
 * PC-relative and immediate are not legal destinations. */
static int ea_move_dst_cost(int ea, M68KSize sz)
{
    int is_long = (sz == M68K_SIZE_L);
    int mode    = (ea >> 3) & 7;
    int reg     =  ea       & 7;
    if (ea < 0) return 0;
    switch (mode) {
    case 0: /* Dn */           return 0;
    case 1: /* An */           return 0;
    case 2: /* (An)        */  return is_long ? 8  : 4;
    case 3: /* (An)+       */  return is_long ? 8  : 4;
    case 4: /* -(An)       */  return is_long ? 8  : 4;   /* NOT 10/6 */
    case 5: /* d16(An)     */  return is_long ? 12 : 8;
    case 6: /* d8(An,Xn)   */  return is_long ? 14 : 10;
    case 7:
        switch (reg) {
        case 0: /* (xxx).W  */ return is_long ? 12 : 8;
        case 1: /* (xxx).L  */ return is_long ? 16 : 12;
        }
    }
    return 0;
}

int m68k_ea_is_reg(int ea)      /* Dn or An */
{
    if (ea < 0) return 0;
    return ((ea >> 3) & 7) <= 1;
}

/* Dn / An / #imm — the source forms that add 2 cycles to the LONG
 * "<ea>,Dn" arithmetic forms (PRM's "+2 if source is register or
 * immediate" footnote on ADD/SUB/AND/OR/CMP and ADDA/SUBA). */
static int ea_is_reg_or_imm(int ea)
{
    if (ea < 0) return 1;
    int mode = (ea >> 3) & 7;
    return mode <= 1 || (mode == 7 && (ea & 7) == 4);
}

/* JMP / JSR / LEA / PEA carry their own complete per-mode timings rather
 * than base+EA. Column order: JMP, JSR, LEA, PEA. */
static int ea_control_cost(int ea, int col)
{
    static const int T[8][4] = {
        /*                JMP  JSR  LEA  PEA */
        /* 0 (An)     */ {  8,  16,   4,  12 },
        /* 1 (d16,An) */ { 10,  18,   8,  16 },
        /* 2 (d8,An,X)*/ { 14,  22,  12,  20 },
        /* 3 (xxx).W  */ { 10,  18,   8,  16 },
        /* 4 (xxx).L  */ { 12,  20,  12,  20 },
        /* 5 (d16,PC) */ { 10,  18,   8,  16 },
        /* 6 (d8,PC,X)*/ { 14,  22,  12,  20 },
        /* 7 other    */ {  8,  16,   4,  12 },
    };
    int mode = (ea >> 3) & 7;
    int reg  =  ea       & 7;
    int row;
    switch (mode) {
    case 2:  row = 0; break;
    case 5:  row = 1; break;
    case 6:  row = 2; break;
    case 7:
        switch (reg) {
        case 0:  row = 3; break;
        case 1:  row = 4; break;
        case 2:  row = 5; break;
        case 3:  row = 6; break;
        default: row = 7; break;
        }
        break;
    default: row = 7; break;
    }
    return T[row][col];
}
#define EACTL_JMP 0
#define EACTL_JSR 1
#define EACTL_LEA 2
#define EACTL_PEA 3

/* MOVEM addressing-mode surcharge, on top of the direction base and the
 * per-register term. PRM MOVEM table decomposed as base + this + n-term. */
static int ea_movem_extra(int ea)
{
    int mode = (ea >> 3) & 7;
    int reg  =  ea       & 7;
    switch (mode) {
    case 2: case 3: case 4:  return 0;   /* (An), (An)+, -(An) */
    case 5:                  return 4;   /* (d16,An)  */
    case 6:                  return 6;   /* (d8,An,Xn)*/
    case 7:
        switch (reg) {
        case 0: return 4;                /* (xxx).W   */
        case 1: return 8;                /* (xxx).L   */
        case 2: return 4;                /* (d16,PC)  */
        case 3: return 6;                /* (d8,PC,Xn)*/
        }
    }
    return 0;
}

int m68k_cycles_prm(const M68KInstr *instr)
{
    M68KSize sz      = instr->size;
    int      is_long = (sz == M68K_SIZE_L);
    /* THE operand EA for every non-MOVE instruction (see decoder contract
     * above). `ea` is its fetch cost; `op_is_reg` says the operand is a
     * register rather than memory. */
    int      ea        = ea_calc_cost(instr->src_ea, sz);
    int      op_is_reg = m68k_ea_is_reg(instr->src_ea);
    uint16_t w0        = instr->words[0];

    switch (instr->mnemonic) {
    /* ---- Control flow ------------------------------------------------ */
    case MN_NOP:   return 4;
    case MN_RTS:   return 16;
    case MN_RTE:   return 20;
    case MN_RTR:   return 20;
    case MN_STOP:  return 4;
    case MN_JSR:   return ea_control_cost(instr->src_ea, EACTL_JSR);
    case MN_JMP:   return ea_control_cost(instr->src_ea, EACTL_JMP);
    case MN_BSR:   return 18;                 /* .S and .W alike */
    case MN_BRA:   return 10;
    case MN_Bcc:   return 10;                 /* TAKEN (policy) — untaken is 8 .S / 12 .W */
    case MN_DBcc:  return 10;                 /* TAKEN (policy) — cc true 12, expired 14 */

    /* ---- Moves -------------------------------------------------------- */
    case MN_MOVE:
        /* Only MOVE/MOVEA populate dst_ea, and the destination uses the
         * MOVE-specific write table. */
        return 4 + ea + ea_move_dst_cost(instr->dst_ea, sz);
    case MN_MOVEA: return 4 + ea;
    case MN_MOVEQ: return 4;
    case MN_LEA:   return ea_control_cost(instr->src_ea, EACTL_LEA);
    case MN_PEA:   return ea_control_cost(instr->src_ea, EACTL_PEA);

    /* ---- MOVEM: direction base + mode surcharge + per-register term ---- */
    case MN_MOVEM: {
        int nregs   = (instr->word_count >= 2) ? popcount16(instr->words[1]) : 4;
        int mem2reg = (w0 >> 10) & 1;         /* dr bit: 1 = <ea> -> registers */
        int base    = mem2reg ? 12 : 8;
        return base + ea_movem_extra(instr->src_ea)
                    + (is_long ? 8 : 4) * nregs;
    }

    /* ---- Binary arithmetic / logic ------------------------------------ */
    case MN_ADD: case MN_SUB: case MN_AND: case MN_OR: {
        /* Direction bit 8: 0 = <ea>,Dn ; 1 = Dn,<ea>. */
        int to_mem = (w0 >> 8) & 1;
        if (to_mem)
            return (is_long ? 12 : 8) + ea;
        /* <ea>,Dn — long form costs 2 more when the source is a register
         * or an immediate (PRM footnote). */
        if (is_long)
            return (ea_is_reg_or_imm(instr->src_ea) ? 8 : 6) + ea;
        return 4 + ea;
    }

    case MN_CMP:
        /* CMP is <ea>,Dn only. */
        if (is_long)
            return 6 + ea;
        return 4 + ea;

    case MN_EOR:
        /* EOR is Dn,<ea> only. */
        if (op_is_reg) return is_long ? 8 : 4;
        return (is_long ? 12 : 8) + ea;

    case MN_ADDA: case MN_SUBA:
        if (is_long)
            return (ea_is_reg_or_imm(instr->src_ea) ? 8 : 6) + ea;
        return 8 + ea;

    case MN_CMPA:
        return 6 + ea;

    case MN_ADDQ: case MN_SUBQ:
        if (op_is_reg) {
            /* An destination is 8 for both word and long (PRM Table 8-8).
             * DELIBERATE DEVIATION FROM clown68000, which reports 10 here:
             * Action_ADDQ runs Action_ADD's standard register timing and
             * then StandardInstructionExecutionTimeQuick adds another 2 for
             * the An mode (clown68000.c:739), double-charging the surcharge.
             * The PRM value is 8. */
            if (((instr->src_ea >> 3) & 7) == 1) return 8;
            return is_long ? 8 : 4;
        }
        return (is_long ? 12 : 8) + ea;

    case MN_ADDX: case MN_SUBX:
        if (instr->predec_mem_form) return is_long ? 30 : 18;
        return is_long ? 8 : 4;

    case MN_CMPM:
        return is_long ? 20 : 12;

    /* ---- Immediate arithmetic ----------------------------------------- */
    case MN_ADDI: case MN_SUBI: case MN_ANDI: case MN_ORI: case MN_EORI:
        if (op_is_reg) return is_long ? 16 : 8;
        return (is_long ? 20 : 12) + ea;

    case MN_CMPI:
        if (op_is_reg) return is_long ? 14 : 8;
        return (is_long ? 12 : 8) + ea;

    case MN_ORI_TO_CCR:  case MN_ANDI_TO_CCR: case MN_EORI_TO_CCR:
    case MN_ORI_TO_SR:   case MN_ANDI_TO_SR:  case MN_EORI_TO_SR:
        return 20;

    /* ---- Shifts and rotates ------------------------------------------- */
    case MN_LSL: case MN_LSR:
    case MN_ASL: case MN_ASR:
    case MN_ROL: case MN_ROR:
    case MN_ROXL: case MN_ROXR: {
        /* Memory form: always one bit, always a word operand. */
        if (instr->mem_shift) return 8 + ea;
        /* Register form: base + 2 per bit shifted. Bit 5 selects a
         * register count, which is not statically knowable — policy is an
         * average of 4 (see the header comment). */
        int reg_count_mode = (w0 >> 5) & 1;
        int n;
        if (reg_count_mode) {
            n = 4;
        } else {
            n = (w0 >> 9) & 7;
            if (n == 0) n = 8;
        }
        return (is_long ? 8 : 6) + 2 * n;
    }

    /* ---- Multiply / divide (data-dependent; documented policy) --------- */
    case MN_MULU: case MN_MULS: return 38 + 2 * 8 + ea;   /* average operand */
    case MN_DIVU:               return 140 + ea;
    case MN_DIVS:               return 158 + ea;

    /* ---- Unary --------------------------------------------------------- */
    case MN_TST:
        /* TST does not write back: 4 + EA, no RMW surcharge. */
        if (op_is_reg) return 4;
        return 4 + ea;

    case MN_CLR:
    case MN_NEG: case MN_NEGX: case MN_NOT:
        if (op_is_reg) return is_long ? 6 : 4;
        return (is_long ? 12 : 8) + ea;

    case MN_NBCD: return op_is_reg ? 6  : 8  + ea;
    case MN_TAS:  return op_is_reg ? 4  : 14 + ea;

    /* ---- Sign extension ------------------------------------------------ */
    case MN_EXT:  return 4;
    case MN_SWAP: return 4;

    /* ---- Bit instructions ----------------------------------------------
     * Static (#imm) forms cost more on a register than the dynamic (Dn)
     * forms. Bit 8 of the opcode distinguishes them: set = dynamic. */
    case MN_BTST: {
        int dynamic = (w0 >> 8) & 1;
        if (op_is_reg) return dynamic ? 6 : 10;
        return (dynamic ? 4 : 8) + ea;
    }
    case MN_BCHG: case MN_BSET: case MN_BCLR: {
        int dynamic = (w0 >> 8) & 1;
        int is_bclr = (instr->mnemonic == MN_BCLR);
        if (op_is_reg) {
            /* Dn destination = a 32-bit operand. The 68000's 16-bit ALU only
             * has to touch the low word when the bit number is < 16, which
             * costs 2 cycles less. For the STATIC form the bit number is an
             * immediate in the extension word, so this is exactly knowable
             * at codegen time; for the DYNAMIC form it is a runtime register
             * and we hold the PRM high-word value. */
            if (dynamic) return is_bclr ? 10 : 8;
            int low_bit = (instr->imm32 & 31u) < 16u;
            if (is_bclr) return low_bit ? 12 : 14;
            return low_bit ? 10 : 12;
        }
        /* Memory destination is a BYTE operand — no high/low-word split.
         * Static (#imm) forms cost 12+EA, not 8+EA: the extension word
         * carrying the bit number is an extra prefetch. */
        return (dynamic ? 8 : 12) + ea;
    }

    /* ---- Link / Unlink -------------------------------------------------- */
    case MN_LINK: return 16;
    case MN_UNLK: return 12;

    /* ---- Conditional set ------------------------------------------------ */
    case MN_Scc:
        if (op_is_reg) return 6;              /* TRUE (policy); false is 4 */
        return 8 + ea;

    /* ---- Traps / privileged --------------------------------------------- */
    case MN_TRAP:    return 34;
    case MN_TRAPV:   return 4;                /* untaken */
    case MN_CHK:     return 10 + ea;          /* no-trap path */
    case MN_RESET:   return 132;
    case MN_ILLEGAL: return 34;

    /* ---- BCD ------------------------------------------------------------ */
    case MN_ABCD: case MN_SBCD:
        return instr->predec_mem_form ? 18 : 6;

    /* ---- Misc ------------------------------------------------------------ */
    case MN_EXG:       return 6;
    case MN_MOVE_USP:  return 4;
    case MN_MOVEC:     return 12;             /* 68010+; not reachable on 68000 */
    case MN_MOVE_SR:
        /* MOVE SR,<ea> writes; MOVE <ea>,SR reads. */
        if (instr->dst_is_ea)
            return op_is_reg ? 6 : (8 + ea);
        return 12 + ea;
    case MN_MOVE_CCR:
        return 12 + ea;                       /* MOVE <ea>,CCR (68000 has no MOVE CCR,<ea>) */
    case MN_MOVEP:     return is_long ? 24 : 16;

    case MN_OTHER:
    default:
        /* Unmodelled opcode: base 4 plus whatever operand fetch the
         * decoder did identify. */
        return 4 + ea;
    }
}

/* Exact mode: the instruction's cost before any data-dependent part, matching
 * the reference emulator the exact runtime is verified against (Genesis Plus GX,
 * whose 68000 core uses Musashi's timing). Where that differs from the PRM model
 * above, Musashi's value wins, because frame-identical equivalence with the
 * reference is the contract:
 *   - Bcc 10 / DBcc 12 / Scc Dn 4 (ST 6): taken / cc-true / false base; the
 *     untaken, loop-exit and true-set deltas are emitted inline.
 *   - register-count shifts and rotates: the 6/8 base; 2 per bit is inline.
 *   - MULU/MULS/DIVU/DIVS: operand fetch only; the operand-dependent time is
 *     computed at run time (x68k_mul*_cycles / x68k_div*_cycles).
 *   - ADDQ.W #,An 4, ANDI.L #,Dn 14, BSET/BCHG/BCLR on Dn without the
 *     bit-number < 16 discount (static 12/12/14, dynamic 8/8/10).
 * Verified opcode by opcode with XMen2Recomp tools/cycle_census.py. */
int m68k_cycles_exact(const M68KInstr *instr)
{
    M68KSize sz  = instr->size;
    int is_long  = (sz == M68K_SIZE_L);
    int op_is_reg = m68k_ea_is_reg(instr->src_ea);
    uint16_t w0  = instr->words[0];
    switch (instr->mnemonic) {
    case MN_Bcc:  return 10;
    case MN_DBcc: return 12;
    case MN_Scc:
        if (op_is_reg) return ((w0 >> 8) & 0xF) == 0 ? 6 : 4;
        break;
    case MN_LSL: case MN_LSR: case MN_ASL: case MN_ASR:
    case MN_ROL: case MN_ROR: case MN_ROXL: case MN_ROXR:
        if (!instr->mem_shift && ((w0 >> 5) & 1))
            return is_long ? 8 : 6;
        break;
    case MN_MULU: case MN_MULS: case MN_DIVU: case MN_DIVS:
        return ea_calc_cost(instr->src_ea, M68K_SIZE_W);
    case MN_ADDQ:
        if (((instr->src_ea >> 3) & 7) == 1 && ((w0 >> 6) & 3) == 1) return 4;   /* .W #,An */
        break;
    case MN_ANDI:
        if (op_is_reg && is_long) return 14;
        break;
    case MN_BSET: case MN_BCHG: case MN_BCLR:
        if (op_is_reg) {
            int dynamic = (w0 >> 8) & 1;
            int is_bclr = (instr->mnemonic == MN_BCLR);
            if (dynamic) return is_bclr ? 10 : 8;
            return is_bclr ? 14 : 12;
        }
        break;
    default:
        break;
    }
    return m68k_cycles_prm(instr);
}
