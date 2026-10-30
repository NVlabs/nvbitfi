/*
 * NVBitFI_for_ODC: destination-register analysis of a SASS instruction.
 *
 * The original NVBitFI tokenised the SASS text and searched for the letters
 * 'R' and 'P' in the first operands. That mistakes uniform registers (UR4) for
 * GPRs (R4), treats RZ/PT (discard targets) as real destinations, gets the
 * width of 64/128-bit and tensor-core results wrong, and never looks at
 * uniform predicates. This header decides, for one static instruction, which
 * registers it writes, split by register class:
 *
 *   GPR : general purpose registers (per thread, 32 bit)       -> gpr_base..+num_gpr
 *   PR  : predicate registers P0..P6 (per thread, 1 bit)       -> pr_mask
 *   UR  : uniform registers (one per warp, 32 bit)             -> ur_base..+num_ur
 *   UP  : uniform predicates UP0..UP6 (one per warp, 1 bit)    -> up_mask
 *
 * It works on an abstract operand list so the same code is used by the NVBit
 * tools (operands from Instr::getOperand) and by the offline test that runs it
 * over real sm_120 SASS (tests/dests_test.cpp).
 *
 * Host-only code. Destination rules were derived from the operand shapes of
 * every opcode in the sm_120 SASS of cuBLAS/cuBLASLt 12.9.
 */
#ifndef DESTS_H
#define DESTS_H

#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include "arch.h"

//////////////////////////////////////////////////////////////////////////////
// Opcode name / group tables generated from NVBITFI_OPCODE_TABLE (arch.h)
//////////////////////////////////////////////////////////////////////////////
#define NVBITFI_OPCODE_NAME(name, grp) #name,
static const char *instTypeNames[NUM_OPCODE_COUNTERS] = {
	NVBITFI_OPCODE_TABLE(NVBITFI_OPCODE_NAME)
	"UNKNOWN"
};
#undef NVBITFI_OPCODE_NAME

#define NVBITFI_OPCODE_GROUP(name, grp) grp,
[[maybe_unused]] static const int instTypeGroups[NUM_OPCODE_COUNTERS] = {
	NVBITFI_OPCODE_TABLE(NVBITFI_OPCODE_GROUP)
	G_UNKNOWN
};
#undef NVBITFI_OPCODE_GROUP

static const char *instGrouptNames[NUM_INST_GROUPS] = {
	"fp64", "fp32", "ld", "pr", "nodest", "others", "tc", "sfu", "fp16", "uniform", "unknown", "gppr", "gp"
};

[[maybe_unused]] static const char *regTypeNames[NUM_REG_TYPES] = { "GPR", "PR", "UR", "UP" };

// Opcode index from the short opcode ("IMAD" for "IMAD.WIDE.U32").
// Returns UNKNOWN_OPCODE if the opcode is not in the table.
inline int opcode_index(const std::string &shortOpcode) {
	static std::map<std::string, int> m;
	if (m.empty()) {
		for (int i = 0; i < NUM_ISA_INSTRUCTIONS; i++) m[instTypeNames[i]] = i;
	}
	auto it = m.find(shortOpcode);
	return (it == m.end()) ? UNKNOWN_OPCODE : it->second;
}

inline int opcode_group(int opIdx) {
	return (opIdx >= 0 && opIdx < NUM_OPCODE_COUNTERS) ? instTypeGroups[opIdx] : G_UNKNOWN;
}

// Short opcode of a full opcode string ("IMAD.WIDE.U32" -> "IMAD")
inline std::string short_opcode(const std::string &fullOpcode) {
	return fullOpcode.substr(0, fullOpcode.find('.'));
}

//////////////////////////////////////////////////////////////////////////////
// Abstract operand list
//////////////////////////////////////////////////////////////////////////////
enum OpKind {
	OK_REG,       // Rn / RZ
	OK_PRED,      // Pn / PT
	OK_UREG,      // URn / URZ
	OK_UPRED,     // UPn / UPT
	OK_PR_ALL,    // "PR"  : the whole predicate register (R2P, P2R)
	OK_UPR_ALL,   // "UPR" : the whole uniform predicate register (UR2UP, UP2UR)
	OK_IMM,       // immediate
	OK_OTHER      // memory reference, constant bank, special register, ...
};

struct OpDesc {
	OpKind kind;
	int64_t num;  // register number (DISCARD_REG for RZ/URZ/PT/UPT) or immediate value
};

// Discard targets (RZ, URZ, PT, UPT) are reported by the front-end as
// num = DISCARD_REG, recognised by NAME, not by number: on Blackwell the
// uniform register file goes beyond UR63 (UR64..UR67 appear in cuBLAS sm_120
// SASS), so the Turing/Ampere convention "URZ = UR63" no longer holds.
static const int DISCARD_REG = -1;
static const int RZ_NUM = 255;   // GPR zero register
static const int PT_NUM = 7;     // always-true predicate
static const int MAX_UR = 256;   // upper bound for uniform register numbers

struct DestInfo {
	int gpr_base = -1;      // first destination GPR, -1 if none
	int num_gpr = 0;        // number of consecutive destination GPRs
	uint32_t pr_mask = 0;   // bit k set: Pk is a destination (k = 0..6)
	int ur_base = -1;       // first destination UR, -1 if none
	int num_ur = 0;         // number of consecutive destination URs
	uint32_t up_mask = 0;   // bit k set: UPk is a destination (k = 0..6)

	int num_pr() const { return __builtin_popcount(pr_mask); }
	int num_up() const { return __builtin_popcount(up_mask); }
	bool any() const { return num_gpr || pr_mask || num_ur || up_mask; }
	std::string str() const {
		std::stringstream ss;
		if (num_gpr) ss << "R" << gpr_base << "x" << num_gpr << " ";
		for (int k = 0; k < 7; k++) if (pr_mask >> k & 1) ss << "P" << k << " ";
		if (num_ur) ss << "UR" << ur_base << "x" << num_ur << " ";
		for (int k = 0; k < 7; k++) if (up_mask >> k & 1) ss << "UP" << k << " ";
		std::string s = ss.str();
		return s.empty() ? "-" : s.substr(0, s.size() - 1);
	}
};

inline std::vector<std::string> split_opcode(const std::string &fullOpcode) {
	std::vector<std::string> t;
	std::stringstream ss(fullOpcode);
	std::string tok;
	while (std::getline(ss, tok, '.')) t.push_back(tok);
	return t;
}

inline bool has_token(const std::vector<std::string> &t, const char *s) {
	return std::find(t.begin() + 1, t.end(), s) != t.end();
}

// Registers written per thread by a tensor core MMA, from the shape and the
// accumulator type, e.g. HMMA.16816.F32 -> 16*8 fp32 / 32 threads = 4 regs.
inline int mma_dest_regs(const std::vector<std::string> &t, int opIdx) {
	int M = 0, N = 0;
	size_t shapePos = 0;
	for (size_t k = 1; k < t.size() && !M; k++) {
		const std::string &s = t[k];
		if (s.empty() || !isdigit(s[0])) continue;
		if (s.find('x') != std::string::npos) {           // "8x8x4", "64x128x16"
			M = atoi(s.c_str());
			N = atoi(s.c_str() + s.find('x') + 1);
		} else if (s.size() >= 3) {                        // "16816", "1684", "884", "16832"
			if (s.compare(0, 2, "16") == 0) { M = 16; N = s[2] - '0'; }
			else { M = s[0] - '0'; N = s[1] - '0'; }
		}
		shapePos = k;
	}
	if (!M || !N) return 1;
	int bytes = 4;                                       // FP32 / S32 accumulators
	if (opIdx == DMMA) bytes = 8;
	else if (opIdx != IMMA && opIdx != IGMMA && opIdx != BMMA && opIdx != BGMMA) {
		for (size_t k = shapePos + 1; k < t.size(); k++) { // first type after the shape = accumulator
			if (t[k] == "F32") { bytes = 4; break; }
			if (t[k] == "F16" || t[k] == "BF16") { bytes = 2; break; }
		}
	}
	bool warpgroup = (opIdx == HGMMA || opIdx == IGMMA || opIdx == QGMMA || opIdx == BGMMA);
	int threads = warpgroup ? 128 : 32;
	int regs = (M * N / threads) * bytes / 4;
	return regs > 0 ? regs : 1;
}

// Number of consecutive registers (GPR or UR) written by the first destination
inline int dest_width(const std::string &fullOpcode, int opIdx) {
	std::vector<std::string> t = split_opcode(fullOpcode);
	int grp = opcode_group(opIdx);
	if (grp == G_TC) return mma_dest_regs(t, opIdx);
	switch (opIdx) {
		case DADD: case DFMA: case DMUL: case DMNMX:
			return 2;
		case LDSM:   // LDSM.16.M88.4 -> 4 regs, .2 -> 2, otherwise 1
			if (t.back() == "4") return 4;
			if (t.back() == "2") return 2;
			return 1;
		case CS2R:   // CS2R is 64-bit unless .32
			return has_token(t, "32") ? 1 : 2;
		case SHF: case USHF:  // funnel shift reads a 64-bit pair, writes 32 bits
			return 1;
		case F2F: case FRND: case UF2F: case UFRND: // first float type is the destination type
			for (size_t k = 1; k < t.size(); k++) {
				if (t[k] == "F64") return 2;
				if (t[k] == "F32" || t[k] == "F16" || t[k] == "BF16") return 1;
			}
			return 1;
		case I2F: case UI2F: // destination is a float: 64-bit only if F64
			return has_token(t, "F64") ? 2 : 1;
		case F2I: case UF2I: // destination is an integer
			return (has_token(t, "U64") || has_token(t, "S64")) ? 2 : 1;
		default:
			break;
	}
	if (has_token(t, "WIDE")) return 2;
	if (has_token(t, "256")) return 8;
	if (has_token(t, "128")) return 4;
	if (has_token(t, "64") || has_token(t, "U64") || has_token(t, "S64") || has_token(t, "F64")) return 2;
	return 1;
}

// Opcodes whose leading predicate destination(s) are followed by a GPR/UR
// destination (e.g. "SHFL.IDX PT, R3, ...", "ATOMG.E.ADD PT, R6, [...]",
// "LOP3.LUT P0, RZ, ...", "IMNMX.S64 PT, PT, R20, ...", "ELECT P0, UR4, PT").
inline bool pred_then_reg_dest(int opIdx) {
	switch (opIdx) {
		case SHFL: case ATOM: case ATOMS: case ATOMG: case SUATOM: case LOP3: case IMNMX:
		case ULOP3: case UIMNMX: case ELECT:
			return true;
		default:
			return false;
	}
}

// Maximum number of predicate destinations that directly follow (or lead) the
// register destination. VOTE.ANY R0, PT, P1 writes R0 and PT, P1 is a source.
inline int max_pred_dests(int opIdx) {
	switch (opIdx) {
		case VOTE: case VOTEU: case ELECT: case B2R:
			return 1;
		default:
			return 2;
	}
}

inline uint32_t imm_mask_after(const std::vector<OpDesc> &ops, size_t from) {
	for (size_t k = from; k < ops.size(); k++)
		if (ops[k].kind == OK_IMM) return (uint32_t)ops[k].num & 0x7f;
	return 0x7f;
}

// Main entry point: destinations of one static instruction.
// ops must NOT include the guard predicate (@P0 / @!UP1).
inline DestInfo analyze_dests(const std::string &fullOpcode, int opIdx, const std::vector<OpDesc> &ops) {
	DestInfo d;
	int grp = opcode_group(opIdx);
	if (grp == G_NODEST || ops.empty()) return d;

	auto addPR = [&](int64_t n) { if (n >= 0 && n < PT_NUM) d.pr_mask |= 1u << n; };
	auto addUP = [&](int64_t n) { if (n >= 0 && n < PT_NUM) d.up_mask |= 1u << n; };
	auto addGPR = [&](int64_t n) {
		if (n >= 0 && n < RZ_NUM) {
			d.gpr_base = (int)n;
			d.num_gpr = std::min(dest_width(fullOpcode, opIdx), RZ_NUM - (int)n);
		}
	};
	auto addUR = [&](int64_t n) {
		if (n >= 0 && n < MAX_UR) {
			d.ur_base = (int)n;
			d.num_ur = std::min(dest_width(fullOpcode, opIdx), MAX_UR - (int)n);
		}
	};

	size_t k = 0;
	const OpDesc &op0 = ops[0];
	switch (op0.kind) {
		case OK_REG: {          // Rd [, Pd [, Pd]] , sources...
			addGPR(op0.num);
			int maxp = max_pred_dests(opIdx);
			for (k = 1; k < ops.size() && maxp > 0 && ops[k].kind == OK_PRED; k++, maxp--) addPR(ops[k].num);
			break;
		}
		case OK_UREG: {         // URd [, UPd [, UPd]] , sources...
			addUR(op0.num);
			int maxp = max_pred_dests(opIdx);
			for (k = 1; k < ops.size() && maxp > 0 && ops[k].kind == OK_UPRED; k++, maxp--) addUP(ops[k].num);
			break;
		}
		case OK_PRED: {         // Pd [, Pd] [, Rd if pred_then_reg_dest] , sources...
			int maxp = max_pred_dests(opIdx);
			int npred = 0;
			for (k = 0; k < ops.size() && maxp > 0 && ops[k].kind == OK_PRED; k++, maxp--, npred++) addPR(ops[k].num);
			if (k < ops.size() && pred_then_reg_dest(opIdx)) {
				if (ops[k].kind == OK_REG) addGPR(ops[k].num);
				else if (ops[k].kind == OK_UREG) addUR(ops[k].num);
			}
			break;
		}
		case OK_UPRED: {        // UPd [, UPd] [, URd if pred_then_reg_dest] , sources...
			int maxp = max_pred_dests(opIdx);
			for (k = 0; k < ops.size() && maxp > 0 && ops[k].kind == OK_UPRED; k++, maxp--) addUP(ops[k].num);
			if (k < ops.size() && pred_then_reg_dest(opIdx) && ops[k].kind == OK_UREG) addUR(ops[k].num);
			break;
		}
		case OK_PR_ALL:         // R2P PR, Rs, mask  -> predicates selected by the mask
			d.pr_mask = imm_mask_after(ops, 1);
			break;
		case OK_UPR_ALL:        // UR2UP UPR, URs, mask
			d.up_mask = imm_mask_after(ops, 1);
			break;
		default:
			break;
	}

	// Sanity: register classes must match the datapath. Uniform-datapath
	// opcodes only write UR/UP (R2UR/REDUX read GPRs but write URs), other
	// opcodes never write UR/UP.
	if (grp == G_UNIFORM) {
		d.gpr_base = -1; d.num_gpr = 0;
		// predicates written by uniform opcodes are uniform predicates
	} else {
		d.ur_base = -1; d.num_ur = 0; d.up_mask = 0;
	}
	if (grp == G_PR) { d.gpr_base = -1; d.num_gpr = 0; }
	return d;
}

#endif
