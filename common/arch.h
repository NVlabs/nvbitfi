/*
 * Copyright 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * NVBitFI_for_ODC changes (2026):
 *  - Single opcode table (X-macro) instead of three hand-synchronised lists.
 *  - Opcode set extended to Ampere/Hopper/Blackwell (sm_80 ... sm_120), built
 *    from the "Blackwell Instruction Set" table of the CUDA Binary Utilities
 *    docs (12.9) plus the opcodes actually emitted for sm_120 in cuBLAS 12.9.
 *  - New instruction groups: TC (tensor cores), SFU, FP16, UNIFORM, UNKNOWN.
 *  - Fixed group assignments: SUST was both "ld" and "nodest" (now nodest),
 *    HSETP2 writes only predicates (now pr), DSET is FP64 (was fp32).
 */

#ifndef ARCH_H
#define ARCH_H

//////////////////////////////////////////////////////////////////////////////
// Instruction groups.
//
// Every opcode belongs to exactly ONE "exclusive" group (the hardware unit /
// datapath that produces its result). G_GPPR and G_GP are "aggregate" groups
// defined by the destination operands, not by the opcode; they MUST stay the
// last two entries (counters[NUM_COUNTERS-2] and counters[NUM_COUNTERS-1]).
//
// Injectable destinations per group (see common/dests.h):
//  - exclusive groups      : every destination register of the instruction
//                            (GPRs + predicates, or URs + uniform predicates
//                            for G_UNIFORM)
//  - G_PR                  : predicate destinations
//  - G_GPPR                : GPR or predicate destinations, any non-uniform,
//                            known opcode
//  - G_GP                  : GPR destinations only (original NVBitFI default)
//  - G_NODEST              : nothing (never selected for injection)
//////////////////////////////////////////////////////////////////////////////
enum GroupType {
	G_FP64 = 0,  // FP64 arithmetic (DADD, DFMA, DMUL, ...)
	G_FP32,      // FP32 arithmetic (FADD, FFMA, FMUL, ...)
	G_LD,        // loads into the register file (LDG, LDS, LDSM, LDC, ...)
	G_PR,        // instructions that write predicate registers only (ISETP, FSETP, PLOP3, ...)
	G_NODEST,    // no register destination (stores, branches, barriers, async copies, ...)
	G_OTHERS,    // integer ALU, conversions, moves, shuffles, atomics, texture, misc
	G_TC,        // tensor core MMA writing registers (HMMA, IMMA, DMMA, QMMA, OMMA, *GMMA)
	G_SFU,       // special function unit (MUFU: sin, cos, ex2, lg2, rsq, rcp, ...)
	G_FP16,      // packed FP16/BF16/FP8 arithmetic (HADD2, HFMA2, QFMA4, ...)
	G_UNIFORM,   // uniform datapath: writes uniform registers/predicates (UIADD3, LDCU, S2UR, R2UR, ...)
	G_UNKNOWN,   // opcode not in the table below: counted, never injected (add it to the table!)
	G_GPPR,      // aggregate: writes a GPR or a predicate register
	G_GP,        // aggregate: writes a GPR
	NUM_INST_GROUPS
};

//////////////////////////////////////////////////////////////////////////////
// Opcode table: OPCODE(name, exclusive group)
//////////////////////////////////////////////////////////////////////////////
#define NVBITFI_OPCODE_TABLE(OPCODE) \
	/* ---- FP32 ---- */ \
	OPCODE(FADD, G_FP32) OPCODE(FADD2, G_FP32) OPCODE(FADD32I, G_FP32) \
	OPCODE(FCHK, G_PR) OPCODE(FCMP, G_FP32) OPCODE(FFMA, G_FP32) OPCODE(FFMA2, G_FP32) \
	OPCODE(FFMA32I, G_FP32) OPCODE(FHADD, G_FP32) OPCODE(FHFMA, G_FP32) OPCODE(FMNMX, G_FP32) \
	OPCODE(FMNMX3, G_FP32) OPCODE(FMUL, G_FP32) OPCODE(FMUL2, G_FP32) OPCODE(FMUL32I, G_FP32) \
	OPCODE(FSEL, G_FP32) OPCODE(FSET, G_FP32) OPCODE(FSETP, G_PR) OPCODE(FSWZADD, G_FP32) \
	OPCODE(IPA, G_FP32) \
	/* ---- SFU ---- */ \
	OPCODE(MUFU, G_SFU) OPCODE(RRO, G_SFU) \
	/* ---- FP64 ---- */ \
	OPCODE(DADD, G_FP64) OPCODE(DFMA, G_FP64) OPCODE(DMNMX, G_FP64) OPCODE(DMUL, G_FP64) \
	OPCODE(DSET, G_FP64) OPCODE(DSETP, G_PR) \
	/* ---- FP16 / BF16 / FP8 packed ---- */ \
	OPCODE(HADD2, G_FP16) OPCODE(HADD2_32I, G_FP16) OPCODE(HFMA2, G_FP16) OPCODE(HFMA2_32I, G_FP16) \
	OPCODE(HMNMX2, G_FP16) OPCODE(HMUL2, G_FP16) OPCODE(HMUL2_32I, G_FP16) OPCODE(HSET2, G_FP16) \
	OPCODE(HSETP2, G_PR) OPCODE(VHMNMX, G_FP16) OPCODE(QADD4, G_FP16) OPCODE(QFMA4, G_FP16) \
	OPCODE(QMUL4, G_FP16) \
	/* ---- Tensor cores (results in registers) ---- */ \
	OPCODE(HMMA, G_TC) OPCODE(IMMA, G_TC) OPCODE(BMMA, G_TC) OPCODE(DMMA, G_TC) \
	OPCODE(QMMA, G_TC) OPCODE(OMMA, G_TC) OPCODE(HGMMA, G_TC) OPCODE(IGMMA, G_TC) \
	OPCODE(QGMMA, G_TC) OPCODE(BGMMA, G_TC) \
	/* ---- Integer ---- */ \
	OPCODE(IDP, G_OTHERS) OPCODE(IDP4A, G_OTHERS) OPCODE(BFE, G_OTHERS) OPCODE(BFI, G_OTHERS) \
	OPCODE(BMSK, G_OTHERS) OPCODE(BREV, G_OTHERS) OPCODE(FLO, G_OTHERS) OPCODE(IABS, G_OTHERS) \
	OPCODE(IADD, G_OTHERS) OPCODE(IADD3, G_OTHERS) OPCODE(IADD32I, G_OTHERS) OPCODE(ICMP, G_OTHERS) \
	OPCODE(IMAD, G_OTHERS) OPCODE(IMAD32I, G_OTHERS) OPCODE(IMADSP, G_OTHERS) OPCODE(IMNMX, G_OTHERS) \
	OPCODE(IMUL, G_OTHERS) OPCODE(IMUL32I, G_OTHERS) OPCODE(ISCADD, G_OTHERS) OPCODE(ISCADD32I, G_OTHERS) \
	OPCODE(ISET, G_OTHERS) OPCODE(ISETP, G_PR) OPCODE(LEA, G_OTHERS) OPCODE(LOP, G_OTHERS) \
	OPCODE(LOP3, G_OTHERS) OPCODE(LOP32I, G_OTHERS) OPCODE(PLOP3, G_PR) OPCODE(POPC, G_OTHERS) \
	OPCODE(SHF, G_OTHERS) OPCODE(SHL, G_OTHERS) OPCODE(SHR, G_OTHERS) OPCODE(XMAD, G_OTHERS) \
	OPCODE(VABSDIFF, G_OTHERS) OPCODE(VABSDIFF4, G_OTHERS) OPCODE(VADD, G_OTHERS) OPCODE(VMAD, G_OTHERS) \
	OPCODE(VMNMX, G_OTHERS) OPCODE(VSET, G_OTHERS) OPCODE(VSETP, G_PR) OPCODE(VSHL, G_OTHERS) \
	OPCODE(VSHR, G_OTHERS) OPCODE(VIADD, G_OTHERS) OPCODE(VIADDMNMX, G_OTHERS) OPCODE(VIMNMX, G_OTHERS) \
	OPCODE(VIMNMX3, G_OTHERS) \
	/* ---- Conversion ---- */ \
	OPCODE(F2F, G_OTHERS) OPCODE(F2I, G_OTHERS) OPCODE(I2F, G_OTHERS) OPCODE(I2I, G_OTHERS) \
	OPCODE(I2IP, G_OTHERS) OPCODE(I2FP, G_OTHERS) OPCODE(F2IP, G_OTHERS) OPCODE(F2FP, G_OTHERS) \
	OPCODE(FRND, G_OTHERS) \
	/* ---- Movement ---- */ \
	OPCODE(MOV, G_OTHERS) OPCODE(MOV32I, G_OTHERS) OPCODE(MOVM, G_OTHERS) OPCODE(PRMT, G_OTHERS) \
	OPCODE(SEL, G_OTHERS) OPCODE(SGXT, G_OTHERS) OPCODE(SHFL, G_OTHERS) \
	/* ---- Predicate ---- */ \
	OPCODE(CSET, G_OTHERS) OPCODE(CSETP, G_PR) OPCODE(PSET, G_OTHERS) OPCODE(PSETP, G_PR) \
	OPCODE(P2R, G_OTHERS) OPCODE(R2P, G_PR) \
	/* ---- Texture / surface ---- */ \
	OPCODE(TEX, G_OTHERS) OPCODE(TLD, G_LD) OPCODE(TLD4, G_LD) OPCODE(TMML, G_OTHERS) \
	OPCODE(TXA, G_OTHERS) OPCODE(TXD, G_OTHERS) OPCODE(TXQ, G_OTHERS) OPCODE(TEXS, G_OTHERS) \
	OPCODE(TLD4S, G_LD) OPCODE(TLDS, G_LD) OPCODE(STP, G_OTHERS) \
	OPCODE(SUATOM, G_OTHERS) OPCODE(SULD, G_LD) OPCODE(SURED, G_NODEST) OPCODE(SUST, G_NODEST) \
	/* ---- Load ---- */ \
	OPCODE(LD, G_LD) OPCODE(LDC, G_LD) OPCODE(LDG, G_LD) OPCODE(LDL, G_LD) OPCODE(LDS, G_LD) \
	OPCODE(LDSM, G_LD) OPCODE(LDGMC, G_LD) OPCODE(LDT, G_LD) OPCODE(LDTM, G_LD) \
	/* ---- Store / async copy / reduction without return ---- */ \
	OPCODE(ST, G_NODEST) OPCODE(STG, G_NODEST) OPCODE(STL, G_NODEST) OPCODE(STS, G_NODEST) \
	OPCODE(STSM, G_NODEST) OPCODE(STAS, G_NODEST) OPCODE(STT, G_NODEST) OPCODE(STTM, G_NODEST) \
	OPCODE(LDGSTS, G_NODEST) OPCODE(LDGDEPBAR, G_NODEST) OPCODE(RED, G_NODEST) OPCODE(REDG, G_NODEST) \
	OPCODE(REDAS, G_NODEST) OPCODE(SYNCS, G_NODEST) OPCODE(FENCE, G_NODEST) \
	/* ---- Atomics / misc memory (return a value) ---- */ \
	OPCODE(MATCH, G_OTHERS) OPCODE(QSPC, G_OTHERS) OPCODE(ATOM, G_OTHERS) OPCODE(ATOMS, G_OTHERS) \
	OPCODE(ATOMG, G_OTHERS) \
	/* ---- Cache control / memory ordering ---- */ \
	OPCODE(CCTL, G_NODEST) OPCODE(CCTLL, G_NODEST) OPCODE(CCTLT, G_NODEST) OPCODE(ERRBAR, G_NODEST) \
	OPCODE(CGAERRBAR, G_NODEST) OPCODE(MEMBAR, G_NODEST) \
	/* ---- Control ---- */ \
	OPCODE(BRA, G_NODEST) OPCODE(BRX, G_NODEST) OPCODE(BRXU, G_NODEST) OPCODE(JMP, G_NODEST) \
	OPCODE(JMX, G_NODEST) OPCODE(JMXU, G_NODEST) OPCODE(SSY, G_NODEST) OPCODE(SYNC, G_NODEST) \
	OPCODE(CAL, G_NODEST) OPCODE(CALL, G_NODEST) OPCODE(JCAL, G_NODEST) OPCODE(PRET, G_NODEST) \
	OPCODE(RET, G_NODEST) OPCODE(BRK, G_NODEST) OPCODE(PBK, G_NODEST) OPCODE(CONT, G_NODEST) \
	OPCODE(PCNT, G_NODEST) OPCODE(EXIT, G_NODEST) OPCODE(PEXIT, G_NODEST) OPCODE(PREEXIT, G_NODEST) \
	OPCODE(LONGJMP, G_NODEST) OPCODE(PLONGJMP, G_NODEST) OPCODE(KIL, G_NODEST) OPCODE(KILL, G_NODEST) \
	OPCODE(BSSY, G_NODEST) OPCODE(BSYNC, G_NODEST) OPCODE(BREAK, G_NODEST) OPCODE(BMOV, G_OTHERS) \
	OPCODE(BPT, G_NODEST) OPCODE(IDE, G_NODEST) OPCODE(RAM, G_NODEST) OPCODE(RTT, G_NODEST) \
	OPCODE(SAM, G_NODEST) OPCODE(RPCMOV, G_OTHERS) OPCODE(WARPSYNC, G_NODEST) OPCODE(YIELD, G_NODEST) \
	OPCODE(NANOSLEEP, G_NODEST) OPCODE(ACQBULK, G_NODEST) OPCODE(ACQSHMINIT, G_NODEST) \
	OPCODE(ENDCOLLECTIVE, G_NODEST) OPCODE(ELECT, G_UNIFORM) \
	/* ---- Miscellaneous ---- */ \
	OPCODE(NOP, G_NODEST) OPCODE(CS2R, G_OTHERS) OPCODE(S2R, G_OTHERS) OPCODE(LEPC, G_OTHERS) \
	OPCODE(B2R, G_OTHERS) OPCODE(BAR, G_NODEST) OPCODE(R2B, G_NODEST) OPCODE(VOTE, G_OTHERS) \
	OPCODE(DEPBAR, G_NODEST) OPCODE(GETCRSPTR, G_OTHERS) OPCODE(GETLMEMBASE, G_OTHERS) \
	OPCODE(SETCRSPTR, G_NODEST) OPCODE(SETLMEMBASE, G_NODEST) OPCODE(PMTRIG, G_NODEST) \
	OPCODE(SETCTAID, G_NODEST) \
	/* ---- Uniform datapath (writes UR / UP) ---- */ \
	OPCODE(CREDUX, G_UNIFORM) OPCODE(CS2UR, G_UNIFORM) OPCODE(LDCU, G_UNIFORM) OPCODE(R2UR, G_UNIFORM) \
	OPCODE(REDUX, G_UNIFORM) OPCODE(S2UR, G_UNIFORM) OPCODE(UBMSK, G_UNIFORM) OPCODE(UBREV, G_UNIFORM) \
	OPCODE(UCLEA, G_UNIFORM) OPCODE(UFADD, G_UNIFORM) OPCODE(UF2F, G_UNIFORM) OPCODE(UF2FP, G_UNIFORM) \
	OPCODE(UF2I, G_UNIFORM) OPCODE(UF2IP, G_UNIFORM) OPCODE(UFFMA, G_UNIFORM) OPCODE(UFLO, G_UNIFORM) \
	OPCODE(UFMNMX, G_UNIFORM) OPCODE(UFMUL, G_UNIFORM) OPCODE(UFRND, G_UNIFORM) OPCODE(UFSEL, G_UNIFORM) \
	OPCODE(UFSET, G_UNIFORM) OPCODE(UFSETP, G_UNIFORM) OPCODE(UI2F, G_UNIFORM) OPCODE(UI2FP, G_UNIFORM) \
	OPCODE(UI2I, G_UNIFORM) OPCODE(UI2IP, G_UNIFORM) OPCODE(UIABS, G_UNIFORM) OPCODE(UIMNMX, G_UNIFORM) \
	OPCODE(UIADD3, G_UNIFORM) OPCODE(UIMAD, G_UNIFORM) OPCODE(UISETP, G_UNIFORM) OPCODE(ULEA, G_UNIFORM) \
	OPCODE(ULEPC, G_UNIFORM) OPCODE(ULOP, G_UNIFORM) OPCODE(ULOP3, G_UNIFORM) OPCODE(ULOP32I, G_UNIFORM) \
	OPCODE(UMOV, G_UNIFORM) OPCODE(UP2UR, G_UNIFORM) OPCODE(UPLOP3, G_UNIFORM) OPCODE(UPOPC, G_UNIFORM) \
	OPCODE(UPRMT, G_UNIFORM) OPCODE(UPSETP, G_UNIFORM) OPCODE(UR2UP, G_UNIFORM) OPCODE(USEL, G_UNIFORM) \
	OPCODE(USGXT, G_UNIFORM) OPCODE(USHF, G_UNIFORM) OPCODE(USHL, G_UNIFORM) OPCODE(USHR, G_UNIFORM) \
	OPCODE(UGETNEXTWORKID, G_UNIFORM) OPCODE(UVIADD, G_UNIFORM) OPCODE(UVIMNMX, G_UNIFORM) \
	OPCODE(UVIRTCOUNT, G_UNIFORM) OPCODE(VOTEU, G_UNIFORM) \
	/* ---- Uniform / tensor-memory / TMA ops without register result ---- */ \
	OPCODE(UCGABAR_ARV, G_NODEST) OPCODE(UCGABAR_WAIT, G_NODEST) OPCODE(USETMAXREG, G_NODEST) \
	OPCODE(UMEMSETS, G_NODEST) OPCODE(UREDGR, G_NODEST) OPCODE(USTGR, G_NODEST) \
	OPCODE(UBLKCP, G_NODEST) OPCODE(UBLKPF, G_NODEST) OPCODE(UBLKRED, G_NODEST) \
	OPCODE(UTMACCTL, G_NODEST) OPCODE(UTMACMDFLUSH, G_NODEST) OPCODE(UTMALDG, G_NODEST) \
	OPCODE(UTMAPF, G_NODEST) OPCODE(UTMAREDG, G_NODEST) OPCODE(UTMASTG, G_NODEST) \
	OPCODE(UTCATOMSWS, G_NODEST) OPCODE(UTCBAR, G_NODEST) OPCODE(UTCCP, G_NODEST) \
	OPCODE(UTCHMMA, G_NODEST) OPCODE(UTCIMMA, G_NODEST) OPCODE(UTCOMMA, G_NODEST) \
	OPCODE(UTCQMMA, G_NODEST) OPCODE(UTCSHIFT, G_NODEST)

#define NVBITFI_OPCODE_ENUM(name, grp) name,
enum InstructionType {
	NVBITFI_OPCODE_TABLE(NVBITFI_OPCODE_ENUM)
	NUM_ISA_INSTRUCTIONS,         // number of known opcodes
	UNKNOWN_OPCODE = NUM_ISA_INSTRUCTIONS  // index used for opcodes not in the table
};
#undef NVBITFI_OPCODE_ENUM

// Per-opcode counters include one extra slot for UNKNOWN_OPCODE.
#define NUM_OPCODE_COUNTERS (NUM_ISA_INSTRUCTIONS+1)
#define NUM_COUNTERS (NUM_OPCODE_COUNTERS+NUM_INST_GROUPS)

// Register class of the corrupted destination (logged with every injection)
enum RegType {
	REG_GPR = 0, // general purpose register (per thread, 32 bit)
	REG_PR,      // predicate register (per thread, 1 bit, P0-P6)
	REG_UR,      // uniform register (per warp, 32 bit)
	REG_UP,      // uniform predicate (per warp, 1 bit, UP0-UP6)
	NUM_REG_TYPES
};

// Per-instruction flags computed on the host (eligibility_flags in globals.h)
// and passed to the device functions.
#define FLAG_ELIG_OWN   0x1  // counted in the counter of its own (exclusive) group
#define FLAG_ELIG_GPPR  0x2  // counted in G_GPPR (writes a GPR or a PR)
#define FLAG_ELIG_GP    0x4  // counted in G_GP (writes a GPR)
#define FLAG_UNIFORM    0x8  // uniform-datapath instruction: executes once per warp

// List of the Bit Flip Models
// For 1-bit destinations (predicates) FLIP_SINGLE_BIT and FLIP_TWO_BITS both
// invert the predicate, RANDOM_VALUE writes a random 0/1, ZERO_VALUE writes 0.
enum BitFlipModel {
	FLIP_SINGLE_BIT = 0,  // flip a single bit
	FLIP_TWO_BITS, // flip two adjacent bits
	RANDOM_VALUE,  // write a random value.
	ZERO_VALUE, // write value 0
	NUM_BFM_TYPES
};

#endif
