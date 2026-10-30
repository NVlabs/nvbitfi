# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import sys, re, string, os, operator, math, datetime, random
import params as p

# Parse the profile written by profiler.so (one line per dynamic kernel):
#   NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, ..., UNKNOWN: 0, fp64: 0, ..., gp: 337920,
# Returns a list of [kname, kcount, instrs, counts] where counts is a dict
# {opcode or group name: count}. Counts are read by NAME, so adding opcodes or
# groups to common/arch.h needs no change here.
# Group counts are the INJECTABLE dynamic instances of the group; per-opcode
# counts are all executed dynamic instances (uniform instructions once per warp).
def read_inst_counts(d, app):
	countList = []
	fName = d + "/" + p.nvbit_profile_log 
	if not os.path.exists(fName):
		print ("%s file not found!" %fName )
		return countList

	f = open(fName, "r")
	for line in f:
		line = line.rstrip()
		if not line.startswith("NVBit-igprofile"):
			continue
		fields = line.split(';')
		kcount = fields[1].split(':')[1].strip()
		kname = fields[2].split('kernel_name:')[1].strip()
		icount = fields[4].split(':')[1].strip()
		counts = {}
		for e in fields[5].split(','):
			e = e.strip()
			if e != "":
				name, val = e.split(':')
				counts[name.strip()] = int(val)
		countList.append([kname, int(kcount), int(icount), counts])
	f.close()

	return countList 

# names of the per-opcode counters, in the order of the profile
def get_opcode_names(countList):
	if len(countList) == 0:
		return []
	return [k for k in countList[0][3] if k not in p.IGID_STR]

# return the total number of injectable instances of group igid across all kernels
def get_group_count(countList, igid):
	name = p.IGID_STR[igid]
	return sum(item[3].get(name, 0) for item in countList)

# return {name: total count} for every opcode and group, across all the kernels
def get_total_counts(countList):
	total = {}
	for item in countList:
		for k, v in item[3].items():
			total[k] = total.get(k, 0) + v
	return total

# return total number of executed instructions in the countList 
def get_total_insts(countList, with_will_not_execute):
	total = 0
	for l in countList:
		# kname:kcount:instrns 
		total += int(l[2])
	return total

# convert a global injection index (0 .. group total - 1) into
# [kernel name, kernel count, index of the instance inside that kernel]
def get_injection_site_info(countList, inj_num, igid):
	start = 0
	name = p.IGID_STR[igid]
	for item in countList:
		n = item[3].get(name, 0)
		if start <= inj_num < start + n:
			return [item[0], item[1], inj_num-start] # return [kname, kcount, inj_num in this kernel]
		start += n
	return ["", -1, -1]

def set_env(app, is_profiler):
	# Make sure that you use the same ENV variables in the run scripts
	os.environ['BIN_DIR'] = p.bin_dir[app]
	os.environ['BIN_PATH'] = p.bin_dir[app]
	os.environ['APP_DIR'] = p.app_dir[app]
	os.environ['DATASET_DIR'] = p.app_data_dir[app]
	if is_profiler: 
		os.environ['PRELOAD_FLAG'] = "LD_PRELOAD=" + p.PROFILER_LIB
	else:
		os.environ['PRELOAD_FLAG'] = "LD_PRELOAD=" + p.INJECTOR_LIB 
	if p.verbose: print ("BIN_DIR=%s" %(os.environ['BIN_DIR']))
	if p.verbose: print ("PRELOAD_FLAG=%s" %(os.environ['PRELOAD_FLAG']))
	if p.verbose: print ("RODINIA=%s" %(os.environ['RODINIA']))
	if p.verbose: print ("APP_DIR=%s" %(os.environ['APP_DIR']))

