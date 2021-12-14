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
 */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <assert.h>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <iterator>

#include "arch.h"

// Output log file 
std::string injOutputFilename = "nvbitfi-injection-log-temp.txt";
std::ofstream  fout;

// kernel_id is a counter we used to keep track of which kernel needs
// injection. We do not maintain 2 counters for kernel launches.  Ideally we
// want to maintain one for kernel launch and one for kernel exit because
// kernels can be launched currently and end in the order that's different than
// the order in which they were launched. This is a TODO for later time.
int kernel_id = 0;

void open_output_file(std::string filename) {
	static bool open_flag = false; // file will be parsed only once - performance enhancement
	if (!open_flag) {
 		fout.open(filename.c_str(), std::ifstream::out);
		open_flag = true;
	}
}


__managed__ uint64_t counters[NUM_COUNTERS]; // instruction counters

uint64_t get_inst_count(bool profiler=false) {
	uint64_t sum2 = 0;
	for (int i=NUM_ISA_INSTRUCTIONS; i<NUM_COUNTERS-2; i++) { // TODO: 2 here is for G_GPPR and G_GP 
		sum2 += counters[i];
	}
	if (profiler) {
		uint64_t sum1 = 0;
		for (int i=0; i<NUM_ISA_INSTRUCTIONS; i++) {
			sum1 += counters[i];
		}
		if (sum1!=sum2){
			std::cout<< "sum1=" << sum1 << " sum2=" << sum2 << "\n";
		}
		assert (sum1 == sum2);
	}

	return sum2;
}
void init_counters() {
	cudaDeviceSynchronize();
	for (int i=0; i<NUM_COUNTERS; i++) {
		counters[i] = 0;
	}
}

// remove spaces from a string - used mainly for kernel names
std::string removeSpaces(const char * str) {
	std::string s = str;
	s.erase(remove_if(s.begin(), s.end(), isspace), s.end());
	return s;
}

//////////////////////////////////////////////////////////////////////////////////
// Architecture-related global variables and functions
//////////////////////////////////////////////////////////////////////////////////


const char * instTypeNames[NUM_ISA_INSTRUCTIONS] = {
 "FADD", "FADD32I", "FCHK", "FCMP", "FFMA", "FFMA32I", "FMNMX", "FMUL",
 "FMUL32I", "FSEL", "FSET", "FSETP", "FSWZADD", "IPA", "MUFU", "RRO", "DADD", "DFMA",
 "DMNMX", "DMUL", "DSET", "DSETP", "HADD2", "HADD2_32I", "HFMA2", "HFMA2_32I",
 "HMUL2", "HMUL2_32I", "HSET2", "HSETP2", "IDP", "IDP4A", "BFE", "BFI", "BMSK", "BREV", "FLO", "IADD",
 "IADD3", "IADD32I", "ICMP", "IMAD", "IMAD32I", "IMADSP", "IMNMX", "IMUL",
 "IMUL32I", "ISCADD", "ISCADD32I", "ISET", "ISETP", "LEA", "LOP", "LOP3",
 "LOP32I", "PLOP3", "POPC", "SHF", "SHL", "SHR", "XMAD", "IMMA", "HMMA", "VABSDIFF", "VADD", "VMAD",
 "VMNMX", "VSET", "VSETP", "VSHL", "VSHR", "VABSDIFF4", "F2F", "F2I", "I2F",
 "I2I", "I2IP", "FRND", "MOV", "MOV32I", "PRMT", "SEL", "SGXT", "SHFL", "CSET", "CSETP", "PSET",
 "PSETP", "P2R", "R2P", "TEX", "TLD", "TLD4", "TMML", "TXA", "TXD", "TXQ",
 "TEXS", "TLD4S", "TLDS", "STP", "LD", "LDC", "LDG", "LDL", "LDS", "ST", "STG",
 "STL", "STS", "MATCH", "QSPC", "ATOM", "ATOMS", "RED", "CCTL", "CCTLL", "ERRBAR", "MEMBAR", "CCTLT",
 "SUATOM", "SULD", "SURED", "SUST", "BRA", "BRX", "JMP", "JMX", "SSY", "SYNC",
 "CAL", "JCAL", "PRET", "RET", "BRK", "PBK", "CONT", "PCNT", "EXIT", "PEXIT",
 "LONGJMP", "PLONGJMP", "KIL", "BSSY", "BSYNC", "BREAK", "BMOV", "BPT", "IDE", "RAM", "RTT", "SAM",
 "RPCMOV", "WARPSYNC", "YIELD", "NANOSLEEP",
 "NOP",
 "CS2R", "S2R", "LEPC", "B2R", "BAR", "R2B", "VOTE", "DEPBAR", "GETCRSPTR",
 "GETLMEMBASE", "SETCRSPTR", "SETLMEMBASE" , "PMTRIG", "SETCTAID"
 };

const char * instGrouptNames[NUM_INST_GROUPS] = {
	"fp64", "fp32", "ld", "pr", "nodest", "others", "gppr", "gp"
	};

int fp64Inst[] = {
 DADD, DFMA, DMNMX, DMUL
 };

int fp32Inst[] = {
 FADD, FADD32I, FCMP, FFMA, FFMA32I, FMNMX, FMUL, FMUL32I, FSEL, FSET, FSWZADD, IPA, DSET
 };

int ldInst[] = {
 LD, LDC, LDG, LDL, LDS, SULD, SUST, TLD, TLD4, TLD4S, TLDS
 };

int prOnlyInst[] = {
 FCHK, FSETP, DSETP, ISETP,  CSETP, PSETP, R2P, VSETP, PLOP3
 };


int noDestInst[] = {
 ST, STG, STL, STS, RED, CCTL, CCTLL, ERRBAR, MEMBAR, CCTLT, SURED, SUST,
 // Control Instructions
 BRA, BRX, JMP, JMX, SSY, SYNC, CAL, JCAL, PRET, RET, BRK, PBK, CONT, PCNT, EXIT, PEXIT, LONGJMP, PLONGJMP, KIL, BSSY, BSYNC, BREAK, BPT, IDE, RAM, RTT, SAM,
 // Miscellaneous Instructions
 WARPSYNC, YIELD, NANOSLEEP, 
 NOP, BAR, R2B, DEPBAR, SETCRSPTR, SETLMEMBASE, PMTRIG, SETCTAID
 };

int otherInst[] = {
 // Floating-point Instructions
 MUFU, RRO, HADD2, HADD2_32I, HFMA2, HFMA2_32I, HMUL2, HMUL2_32I, HSET2, HSETP2,
 // Integer Instructions
 IDP, IDP4A, BFE, BFI, BMSK, BREV, FLO, IADD, IADD3, IADD32I, ICMP, IMAD, IMAD32I, IMADSP, IMNMX, IMUL, IMUL32I, ISCADD, ISCADD32I, ISET, LEA, LOP, LOP3, LOP32I, POPC, SHF, SHL, SHR, XMAD,
 // MMA instructions
 IMMA, HMMA,
 // Video Instructions
 VABSDIFF, VADD, VMAD, VMNMX, VSET, VSHL, VSHR, VABSDIFF4,
 // Conversion Instructions
 F2F, F2I, I2F, I2I,I2IP, FRND,
 // Move Instructions
 MOV, MOV32I, PRMT, SEL, SGXT, SHFL,
 // Predicate/CC Instructions
 CSET,  PSET, P2R,
 // Texture Instructions
 TEX, TMML, TXA, TXD, TXQ, TEXS, STP,
 // Graphics Load/Store Instructions
 // Compute Load/Store Instructions
 MATCH, QSPC, ATOM, ATOMS, SUATOM,
 // Miscellaneous Instructions
 RPCMOV,
 BMOV, CS2R, S2R, LEPC, B2R, VOTE, GETCRSPTR, GETLMEMBASE
 }; 

// int bothGPRPR[] = {
// 	LEA, LOP, LOP3, LOP32I, SHFL, TEX, TLD, TLD4, TXD, 
// }

std::map<std::string, int> instTypeNameMap;
void initInstTypeNameMap() {
	for (int i=0; i<NUM_ISA_INSTRUCTIONS; i++) {
		instTypeNameMap[instTypeNames[i]] = i;
	}
}

// instruction: ISETP.GE.AND P0, PT, R0, c[0x0][0x158], PT
// instruction: NOP
// instruction: @P0 EXIT
// if first word doesn't start with '@', it's the opcode. If it starts with '@' then the 2nd word is the opcode. 
std::string extractOpcode(std::string instr) {
	// split string into string of words separated by space
	std::istringstream iss(instr);
	std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
	                                 std::istream_iterator<std::string>());
	if (words.size() > 0) {
		// std::cout << "Number of words: " << words.size() << " words[0] = " << words[0] << " words[0][0] = " << words[0][0] << std::endl;
		if (words[0][0] == '@') { // this is a predicated instruction, 2nd word will be the opcode 
			return words[1];
		} else {
			return words[0];
		}
	} else {
		return "";
	}
}

// input: ISETP.GE.AND 
// input: NOP
// return the instruction type (just ISETP in the above example)
std::string extractInstType(std::string opcode) {
	std::string:: size_type pos = opcode.find('.');
	if (pos != std::string::npos) {
		return opcode.substr(0,pos);
	} else {
		return opcode;
	}
}

// input: LD.U.128
// return 128
// input: HMMA.848.F16.F16
// return 64
// input: NOP
// return 0
int extractSize(std::string opcode) {
	if (opcode.find("HMMA") != std::string::npos)
		return 64;

	if (opcode.find("IMMA.8816.S8.") != std::string::npos)
		return 64;
	if (opcode.find("IMMA.8816.U8.") != std::string::npos) 
		return 64;

	// For LD, LDC, LDG instructions
	if (opcode.find("LD") == 0) {
		if (opcode.find("128") != std::string::npos) 
			return 128;
		else if (opcode.find("64") != std::string::npos) 
			return 64;
		else 
			return 32;
	}
	return 0;
}


bool checkOpType(int opcode, int *opTypeArr, int size) {
	for (int i=0; i<size; i++) 
		if (opcode == opTypeArr[i]) 
			return true;
	return false;
}

int getOpGroupNum(int opcode) {
	if (checkOpType(opcode, fp64Inst, sizeof(fp64Inst)/sizeof(int))) 
		return G_FP64;
	if (checkOpType(opcode, fp32Inst, sizeof(fp32Inst)/sizeof(int))) 
		return G_FP32;
	if (checkOpType(opcode, ldInst, sizeof(ldInst)/sizeof(int))) 
		return G_LD;
	if (checkOpType(opcode, prOnlyInst, sizeof(prOnlyInst)/sizeof(int))) 
		return G_PR;
	if (checkOpType(opcode, noDestInst, sizeof(noDestInst)/sizeof(int))) 
		return G_NODEST;
	if (checkOpType(opcode, otherInst, sizeof(otherInst)/sizeof(int))) 
		return G_OTHERS;
	return NUM_INST_GROUPS+1; // not a valid group number
}

bool isGPPRInst(int grp) {
	return (grp != G_NODEST);
}
bool isGPInst(int grp) {
	return ((grp != G_NODEST) && (grp != G_PR));
}

int extractRegNo(std::string token, int &num) {
	// std::string alpha = "abcdefghijklmnopqrstuvwxyz";
	std::string numeric = "1234567890";
	char regchar = 'X';
	int retval = 0;
	if (token.find('R') != std::string::npos) { // found a GPR
		regchar = 'R';
		retval = 0;
	} else if (token.find('P') != std::string::npos) { // found a PR
		regchar = 'P';
		retval = 1;
	} else {
		num = -1;
		return -1;
	}
	// extract the register number
	int pos1 = token.find(regchar)+1;
	int pos2 = token.find_first_not_of(numeric, pos1);
	std::string numStr = token.substr(pos1, pos2).c_str();
	if (numStr.compare("Z") == 0) { // found RZ register
		num = 255; // RZ = R255
	} else if (numStr.compare("T") == 0) { // found RZ register
		num = -1;
	} else {
		num = atoi(numStr.c_str());
	}
	return retval;
}

#endif
