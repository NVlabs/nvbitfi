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


#ifndef ARCH_H
#define ARCH_H 

#define NUM_COUNTERS NUM_INST_GROUPS+NUM_ISA_INSTRUCTIONS

// new inst: FSEL

enum InstructionType { 
 // Floating point instructions
 FADD = 0,
 FADD32I,
 FCHK,
 FCMP,
 FFMA,
 FFMA32I,
 FMNMX,
 FMUL,
 FMUL32I,
 FSEL,
 FSET,
 FSETP,
 FSWZADD,
 IPA,
 MUFU,
 RRO,
 DADD,
 DFMA,
 DMNMX,
 DMUL,
 DSET,
 DSETP,
 HADD2,
 HADD2_32I,
 HFMA2,
 HFMA2_32I,
 HMUL2,
 HMUL2_32I,
 HSET2,
 HSETP2,
 // Integer Instructions
 IDP,
 IDP4A,
 BFE,
 BFI,
 BMSK,
 BREV,
 FLO,
 IADD,
 IADD3,
 IADD32I,
 ICMP,
 IMAD,
 IMAD32I,
 IMADSP,
 IMNMX,
 IMUL,
 IMUL32I,
 ISCADD,
 ISCADD32I,
 ISET,
 ISETP,
 LEA,
 LOP,
 LOP3,
 LOP32I,
 PLOP3,
 POPC,
 SHF,
 SHL,
 SHR,
 XMAD,
 //MMA Instructions
 IMMA,
 HMMA,
 // Video Instructions
 VABSDIFF,
 VADD,
 VMAD,
 VMNMX, 
 VSET, 
 VSETP,
 VSHL, 
 VSHR, 
 VABSDIFF4,
 // Conversion Instructions
 F2F,
 F2I,
 I2F,
 I2I,
 I2IP,
 FRND,
 // Move Instructions
 MOV,
 MOV32I,
 PRMT,
 SEL,
 SGXT,
 SHFL,
 // Predicate/CC Instructions
 CSET,
 CSETP,
 PSET,
 PSETP,
 P2R,
 R2P,
 // Texture Instructions
 TEX,
 TLD,
 TLD4,
 TMML,
 TXA,
 TXD,
 TXQ,
 TEXS,
 TLD4S,
 TLDS,
 STP,
 // Load/Store Instructions
 LD,
 LDC,
 LDG,
 LDL,
 LDS,
 ST,
 STG,
 STL,
 STS,
 MATCH,
 QSPC,
 ATOM,
 ATOMS,
 RED,
 CCTL,
 CCTLL,
 ERRBAR,
 MEMBAR,
 CCTLT,
 SUATOM,
 SULD,
 SURED,
 SUST,
 // Control Instructions
 BRA,
 BRX,
 JMP,
 JMX,
 SSY,
 SYNC,
 CAL,
 JCAL,
 PRET,
 RET,
 BRK,
 PBK,
 CONT,
 PCNT,
 EXIT,
 PEXIT,
 LONGJMP,
 PLONGJMP,
 KIL,
 BSSY,
 BSYNC,
 BREAK,
 BMOV,
 BPT,
 IDE,
 RAM,
 RTT,
 SAM,
 RPCMOV,
 WARPSYNC,
 YIELD,
 NANOSLEEP,
 // Miscellaneous Instructions
 NOP,
 CS2R,
 S2R,
 LEPC,
 B2R,
 BAR,
 R2B,
 VOTE,
 DEPBAR,
 GETCRSPTR,
 GETLMEMBASE,
 SETCRSPTR,
 SETLMEMBASE,
 PMTRIG,
 SETCTAID,
 NUM_ISA_INSTRUCTIONS
 };

// List of instruction groups
enum GroupType { 
	G_FP64 = 0, // FP64 arithmetic instructions
	G_FP32, // FP32 arithmetic instructions 
	G_LD, // instructions that read from emory 
	G_PR, // instructions that write to PR registers only
	G_NODEST, // instructions with no destination register 
	G_OTHERS, 
	G_GPPR, // instructions that write to general purpose and predicate registers
		//  #GPPR registers = all instructions - G_NODEST
	G_GP, // instructions that write to general purpose registers 
		// #GP registers = all instructions - G_NODEST - G_PR
	NUM_INST_GROUPS
	};

// List of the Bit Flip Models
enum BitFlipModel {
	FLIP_SINGLE_BIT = 0,  // flip a single bit
	FLIP_TWO_BITS, // flip two adjacent bits
	RANDOM_VALUE,  // write a random value.
	ZERO_VALUE, // write value 0
	NUM_BFM_TYPES 
};



#endif
