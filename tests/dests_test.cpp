// Offline test of common/dests.h on real SASS (no GPU / NVBit needed).
//
// Build:  g++ -std=c++17 -O2 -I../common dests_test.cpp -o dests_test
// Input:  SASS lines, one instruction per line, as printed by
//         cuobjdump -sass (address and encoding columns stripped), e.g.
//           @!P3 LEA R32, P4, R13, R6, 0x2
// Usage:  ./dests_test --check            built-in expected cases
//         ./dests_test < file.sass        per-opcode summary + unknown opcodes
#include <cstdio>
#include <iostream>
#include <regex>
#include "dests.h"

// Text front-end: mimics what NVBit's Instr::getOperand() reports.
static OpDesc classify(std::string t) {
	while (!t.empty() && (t[0] == '-' || t[0] == '|' || t[0] == '!' || t[0] == '~' || t[0] == ' ')) t.erase(0, 1);
	static const std::regex reg("^R(\\d+|Z)\\b.*"), pred("^P(\\d|T)$"), ureg("^UR(\\d+|Z)\\b.*"),
		upred("^UP(\\d|T)$"), imm("^(0x[0-9a-fA-F]+|[0-9.e+-]+|\\+INF|-INF|QNAN)$");
	std::smatch m;
	std::string base = t.substr(0, t.find_first_of(".|"));
	if (std::regex_match(t, m, ureg)) return {OK_UREG, m[1] == "Z" ? DISCARD_REG : std::stoi(m[1])};
	if (std::regex_match(t, m, reg)) return {OK_REG, m[1] == "Z" ? DISCARD_REG : std::stoi(m[1])};
	if (std::regex_match(base, m, upred)) return {OK_UPRED, m[1] == "T" ? DISCARD_REG : std::stoi(m[1])};
	if (std::regex_match(base, m, pred)) return {OK_PRED, m[1] == "T" ? DISCARD_REG : std::stoi(m[1])};
	if (base == "PR") return {OK_PR_ALL, 8};
	if (base == "UPR") return {OK_UPR_ALL, 8};
	if (std::regex_match(t, imm)) return {OK_IMM, (int64_t)strtoll(t.c_str(), nullptr, 0)};
	return {OK_OTHER, 0};
}

static bool parse(const std::string &line, std::string &opcode, std::vector<OpDesc> &ops) {
	std::stringstream ss(line);
	std::string w;
	if (!(ss >> w)) return false;
	if (w[0] == '@') { if (!(ss >> w)) return false; } // guard predicate is not an operand
	opcode = w;
	std::string rest;
	std::getline(ss, rest);
	ops.clear();
	std::stringstream rs(rest);
	std::string tok;
	while (std::getline(rs, tok, ',')) {
		size_t a = tok.find_first_not_of(' ');
		if (a == std::string::npos) continue;
		ops.push_back(classify(tok.substr(a, tok.find_last_not_of(' ') - a + 1)));
	}
	return true;
}

static DestInfo dests_of(const std::string &line) {
	std::string opc; std::vector<OpDesc> ops;
	parse(line, opc, ops);
	return analyze_dests(opc, opcode_index(short_opcode(opc)), ops);
}

static int check() {
	struct { const char *sass, *expect; } cases[] = {
		{"FFMA R7, R0, R5, 1", "R7x1"},
		{"IMAD.WIDE R4, R7, 0x4, R4", "R4x2"},
		{"IMAD.WIDE.U32 R2, R5.reuse, 0x4, R2", "R2x2"},
		{"IADD3 R4, P0, P1, R2, R3, RZ", "R4x1 P0 P1"},
		{"IADD3 R4, PT, PT, R2, R3, RZ", "R4x1"},
		{"@!P3 LEA R32, P4, R13, R6, 0x2", "R32x1 P4"},
		{"@!P3 LEA.HI.X R33, R13, R7, R30, 0x2, P4", "R33x1"},
		{"ISETP.GE.AND P0, PT, R7, UR5, PT", "P0"},
		{"FSETP.GT.AND P0, P1, R0, R1, PT", "P0 P1"},
		{"PLOP3.LUT P0, P1, PT, PT, PT, 0x80, 0x0", "P0 P1"},
		{"HSETP2.GT.AND P0, PT, R1.H0_H0, R2, PT", "P0"},
		{"SHFL.IDX PT, R3, R2, R4, R5", "R3x1"},
		{"ATOMG.E.ADD.STRONG.GPU PT, R6, desc[UR12][R6.64], R5", "R6x1"},
		{"ATOMG.E.CAS.STRONG.GPU PT, RZ, [R22+0x4], R16, R17", "-"},
		{"LOP3.LUT P0, RZ, R0, 0xff, RZ, 0xc0, !PT", "P0"},
		{"LOP3.LUT P0, R5, R0, 0xff, RZ, 0xc0, !PT", "R5x1 P0"},
		{"IMNMX.S64 P1, PT, R20, R10, UR6, !PT, !PT", "R20x2 P1"},
		{"VOTE.ANY R0, PT, P1", "R0x1"},
		{"VOTE.ANY P0, P0", "P0"},
		{"FMNMX R3, R4, UR5, !PT", "R3x1"},
		{"SEL R0, R1, R2, P0", "R0x1"},
		{"R2P PR, R10, 0x6", "P1 P2"},
		{"P2R R4, PR, RZ, 0x2", "R4x1"},
		{"CS2R R32, SRZ", "R32x2"},
		{"CS2R.32 R5, SR_CLOCKLO", "R5x1"},
		{"LDG.E.128 R8, desc[UR4][R2.64]", "R8x4"},
		{"LDG.E.LTC128B.64 R8, desc[UR4][R2.64]", "R8x2"},
		{"LDS.U16 R3, [R2]", "R3x1"},
		{"LDSM.16.M88.4 R136, [R0+UR11]", "R136x4"},
		{"LDSM.16.M88.2 R68, [R68]", "R68x2"},
		{"SHF.L.U64.HI R5, R4, 0x2, R5", "R5x1"},
		{"IADD.64 R2, R2, 0x40", "R2x2"},
		{"MOV.64 R2, UR4", "R2x2"},
		{"F2F.F64.F32 R4, R2", "R4x2"},
		{"F2F.F32.F64 R4, R2", "R4x1"},
		{"I2F.U64.RP R4, UR4", "R4x1"},
		{"I2F.F64.S32 R4, R2", "R4x2"},
		{"F2I.U64.TRUNC R10, R10", "R10x2"},
		{"DFMA R4, R6, R8, R4", "R4x2"},
		{"DSETP.GT.AND P0, PT, R2, R4, PT", "P0"},
		{"HMMA.16816.F32 R12, R8, R14, RZ", "R12x4"},
		{"HMMA.16816.F16 R12, R8, R12, RZ", "R12x2"},
		{"HMMA.16816.F32.BF16 R12, R8, R14, RZ", "R12x4"},
		{"HMMA.1684.F32.TF32 R28, R18, R24, R28", "R28x4"},
		{"IMMA.16816.S8.S8 R16, R10.ROW, R0.COL, RZ", "R16x4"},
		{"IMMA.16832.S8.S8 R16, R10.ROW, R0.COL, RZ", "R16x4"},
		{"DMMA.8x8x4 R4, R2, R4, RZ", "R4x4"},
		{"QMMA.16832.F32.E4M3.E4M3 R4, R24, R22, R4, UP0", "R4x4"},
		{"QMMA.16832.F16.E4M3.E4M3 R4, R24, R22, R4", "R4x2"},
		{"MUFU.EX2 R3, R2", "R3x1"},
		{"STG.E.128 desc[UR4][R2.64], R8", "-"},
		{"LDGSTS.E.BYPASS.128 [R3], desc[UR4][R4.64]", "-"},
		{"SUST.D.BA.2D.STRONG.SM.TRAP [R2], R4, UR4", "-"},
		{"BRA 0x120", "-"},
		{"@P0 EXIT", "-"},
		{"S2UR UR4, SR_CTAID.X", "UR4x1"},
		{"LDCU UR5, c[0x0][0x390]", "UR5x1"},
		{"LDCU.64 UR4, c[0x0][0x358]", "UR4x2"},
		{"LDCU.128 UR12, c[0x0][0x3d0]", "UR12x4"},
		{"UIADD3 UR5, UPT, UPT, UR4, UR6, URZ", "UR5x1"},
		{"UIADD3 UR5, UP0, UP1, UR4, UR6, URZ", "UR5x1 UP0 UP1"},
		{"UIADD3.64 UR4, UPT, UPT, UR4, 0x20, URZ", "UR4x2"},
		{"UISETP.NE.AND UP0, UPT, UR4, URZ, UPT", "UP0"},
		{"ULOP3.LUT UP0, UR3, UR10, URZ, 0x33, !UPT", "UR3x1 UP0"},
		{"USHF.L.U64.HI UR5, UR4, 0x2, UR5", "UR5x1"},
		{"R2UR UR4, R2", "UR4x1"},
		{"REDUX.SUM.S32 UR4, R9", "UR4x1"},
		{"VOTEU.ANY UR7, UPT, PT", "UR7x1"},
		{"VOTEU.ALL UP0, P3", "UP0"},
		{"UMOV URZ, UR4", "-"},
		{"MOV RZ, R4", "-"},
		{"LDCU UR67, c[0x0][0x3d8]", "UR67x1"},
		{"UIMAD UR64, UR57, 0x6, URZ", "UR64x1"},
	};
	int fails = 0;
	for (auto &c : cases) {
		std::string got = dests_of(c.sass).str();
		bool ok = (got == c.expect);
		if (!ok) fails++;
		printf("%s %-55s -> %-16s%s\n", ok ? "ok  " : "FAIL", c.sass, got.c_str(), ok ? "" : (std::string(" (expected ") + c.expect + ")").c_str());
	}
	printf("\n%d/%zu cases passed\n", (int)(sizeof(cases) / sizeof(cases[0])) - fails, sizeof(cases) / sizeof(cases[0]));
	return fails ? 1 : 0;
}

int main(int argc, char **argv) {
	if (argc > 1 && std::string(argv[1]) == "--check") return check();

	// Summary over a SASS stream: one line per opcode with the destination
	// pattern of every distinct operand shape (first occurrence as example).
	std::map<std::string, std::map<std::string, std::pair<long, std::string>>> summary;
	std::map<std::string, long> unknown;
	std::string line, opc;
	std::vector<OpDesc> ops;
	long total = 0, withDest = 0;
	while (std::getline(std::cin, line)) {
		if (!parse(line, opc, ops)) continue;
		total++;
		std::string s = short_opcode(opc);
		int idx = opcode_index(s);
		if (idx == UNKNOWN_OPCODE) { unknown[s]++; continue; }
		DestInfo d = analyze_dests(opc, idx, ops);
		if (d.any()) withDest++;
		// pattern = destination classes and widths, without register numbers
		std::string pat;
		if (d.num_gpr) pat += "GPRx" + std::to_string(d.num_gpr) + " ";
		if (d.pr_mask) pat += "PRx" + std::to_string(d.num_pr()) + " ";
		if (d.num_ur) pat += "URx" + std::to_string(d.num_ur) + " ";
		if (d.up_mask) pat += "UPx" + std::to_string(d.num_up()) + " ";
		if (pat.empty()) pat = "none";
		auto &e = summary[s][pat];
		if (e.first++ == 0) e.second = line + "   =>  " + d.str();
	}
	for (auto &o : summary) {
		printf("%-10s [%s]\n", o.first.c_str(), instGrouptNames[opcode_group(opcode_index(o.first))]);
		for (auto &p : o.second) printf("    %-22s %9ld   e.g. %s\n", p.first.c_str(), p.second.first, p.second.second.c_str());
	}
	printf("\ninstructions: %ld, with an injectable destination: %ld\n", total, withDest);
	printf("unknown opcodes: %zu\n", unknown.size());
	for (auto &u : unknown) printf("    %s %ld\n", u.first.c_str(), u.second);
	return 0;
}
