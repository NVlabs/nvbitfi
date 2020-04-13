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


import os, sys, re, string, math, datetime, time, pkgutil
import params as p
import common_functions as cf 

output_format = []
for cat in range(p.NUM_CATS-1):
	output_format.append(p.CAT_STR[cat])

fname_prefix = ""

results_app_table = {} # app, igid, bfm, outcome, 
num_injections_app_table = {} # app, igid, bfm, num_injections
runtime_app_table = {} # app, igid, bfm, runtime
runtime_app_nt_table = {} # app, igid, bfm, runtime without Timeouts
results_kname_table = {} # app, kname, igid, bfm, outcome, 
results_kiid_table = {} # app, kname, kid, igid, bfm, outcome, 

def check_and_create_nested_dict(dict_name, k1, k2, k3, k4="", k5="", k6=""):
	if k1 not in dict_name:
		dict_name[k1] = {}
	if k2 not in dict_name[k1]:
		dict_name[k1][k2] = {}
	if k3 not in dict_name[k1][k2]:
		dict_name[k1][k2][k3] = 0 if k4 == "" else {}
	if k4 == "":
		return
	if k4 not in dict_name[k1][k2][k3]:
		dict_name[k1][k2][k3][k4] = 0 if k5 == "" else {}
	if k5 == "":
		return
	if k5 not in dict_name[k1][k2][k3][k4]:
		dict_name[k1][k2][k3][k4][k5] = 0 if k6 == "" else {}
	if k6 == "":
		return
	if k6 not in dict_name[k1][k2][k3][k4][k5]:
		dict_name[k1][k2][k3][k4][k5][k6] = 0

###############################################################################
# Add the injection result to the results*table dictionary 
###############################################################################
def add(app, kname, kiid, igid, bfm, outcome, runtime):
	check_and_create_nested_dict(results_app_table, app, igid, bfm, outcome)
	results_app_table[app][igid][bfm][outcome] += 1

	check_and_create_nested_dict(num_injections_app_table, app, igid, bfm)
	num_injections_app_table[app][igid][bfm] += 1

	check_and_create_nested_dict(runtime_app_table, app, igid, bfm)
	runtime_app_table[app][igid][bfm] += runtime

	if outcome != p.TIMEOUT: 
		check_and_create_nested_dict(runtime_app_nt_table, app, igid, bfm)
		runtime_app_nt_table[app][igid][bfm] += runtime

	check_and_create_nested_dict(results_kname_table, app, kname, igid, bfm, outcome)
	results_kname_table[app][kname][igid][bfm][outcome] += 1

	check_and_create_nested_dict(results_kiid_table, app, kname, kiid, igid, bfm, outcome)
	results_kiid_table[app][kname][kiid][igid][bfm][outcome] += 1


###############################################################################
# inst_fraction contains the fraction of IADD, FADD, IMAD, FFMA, ISETP, etc. 
# instructions per application
###############################################################################
inst_fraction = {}
inst_count = {}
def populate_inst_fraction():
	global inst_fraction
	for app in results_app_table:
		inst_counts = cf.get_total_counts(cf.read_inst_counts(p.app_log_dir[app], app))
		total = cf.get_total_insts(cf.read_inst_counts(p.app_log_dir[app], app), False)
		inst_fraction[app] = [total] + [1.0*i/total for i in inst_counts]
		inst_count[app] = inst_counts 

###############################################################################
# Print instruction distribution to a txt file
###############################################################################
def print_inst_fractions_tsv():
	f = open(fname_prefix + "instruction-fractions.tsv", "w")
	f.write("\t".join(["App", "Total"] + cf.get_inst_count_format().split(':')[2:]) + "\n")
	for app in inst_fraction: 
		f.write("\t".join([app] + map(str, inst_fraction[app])) + "\n")
	f.close()


def parse_results_file(app, inj_mode, igid, bfm):
	try:
		rf = open(p.app_log_dir[app] + "results-mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt", "r")
	except IOError: 
		print ("Error opening file: " + p.app_log_dir[app] + "results-mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt")
		print ("It is possible that no injections were performed for app=%s, inj_mode=%s, igid=%s, bfm=%s " %(app, inj_mode, str(igid), str(bfm)))
		return 

	num_lines = 0
	for line in rf: # for each injection site 
		# print(line)
		#Example line: 1;_Z22bpnn_layerforward_CUDAPfS_S_S_ii;0;26605491;0.506809798834;0.560204950825:..:MOV:773546:17:0.759537:3:dmesg:value_before_value_after, 
		#inj_count;kname;kcount;iid;opid;bid:pc:opcode:tid:injBID:runtime_sec:outcome_category:dmesg:value_before:value_after
		words1 = line.split(";")
		words2 = words1[5].split(":")
		[kname, invocation_index, opcode, injBID, runtime, outcome] = [words1[1], int(words1[2]), words2[2], words2[4], float(words2[5]), int(words2[6])]
		add(app, kname, invocation_index, igid, bfm, outcome, runtime)
		num_lines += 1
	rf.close()

	if num_lines == 0 and app in results_app_table and os.stat(p.app_log_dir[app] + "injection-list/mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt").st_size != 0: 
		print ("%s, inj_mode=%s, igid=%d, bfm=%d not done" %(app, inj_mode, igid, bfm))

###################################################################################
# Parse results files and populate summary to results table 
###################################################################################
def parse_results_apps(typ): 
	for app in p.parse_apps:
		if typ == p.INST_VALUE_MODE:
			for igid in p.parse_inst_value_igid_bfm_map:
				for bfm in p.parse_inst_value_igid_bfm_map[igid]:
					parse_results_file(app, typ, igid, bfm)
		elif typ == p.INST_ADDRESS_MODE:
			for igid in p.parse_inst_address_igid_bfm_map:
				for bfm in p.parse_inst_address_igid_bfm_map[igid]:
					parse_results_file(app, typ, igid, bfm)
		else:
			for bfm in p.parse_rf_bfm_list:
				parse_results_file(app, typ, "rf", bfm)

###############################################################################
# Convert a dictionary to list
# input: d (dictionary), s (size)
###############################################################################
def to_list(d, s):  
	# if a specific category is not found then make it zero
	l = []
	for i in range(1,s-1):
		if i not in d:
			d[i] = 0
		l.append(d[i])
	return l


###############################################################################
# Helper function
###############################################################################
def get_igid_list(inj_mode):
	if inj_mode == p.INST_VALUE_MODE:
		return p.parse_inst_value_igid_bfm_map 
	elif inj_mode == p.INST_ADDRESS_MODE:
		return p.parse_inst_address_igid_bfm_map 
	else: # if inj_mode == p.RF_MODE:
		return ["rf"]

def get_bfm_list(inj_mode, igid):
	if inj_mode == p.INST_VALUE_MODE:
		return p.parse_inst_value_igid_bfm_map[igid] 
	elif inj_mode == p.INST_ADDRESS_MODE:
		return p.parse_inst_address_igid_bfm_map[igid] 
	else: # if inj_mode == p.RF_MODE:
		return p.parse_rf_bfm_list

def get_igid_str(inj_mode, igid):
	if inj_mode == p.INST_VALUE_MODE or inj_mode == p.INST_ADDRESS_MODE:
		return p.IGID_STR[igid]
	else: # if inj_mode == p.RF_MODE:
		return "rf"

###############################################################################
# Print Stats to a txt file
###############################################################################
def print_stats_tsv(typ):
	f = open(fname_prefix + "stats.tsv", "w")
	f.write("\t".join(["App", "IGID", "Injection Model", "Num Jobs", "Total Runtime", "Total Runtime without Timeouts"]) + "\n")

	for app in num_injections_app_table: 
		f.write(app + "\t") 

		igid_list = get_igid_list(typ)
		for igid in igid_list: 
			f.write(get_igid_str(typ, igid) + "\t")

			bfm_list = get_bfm_list(typ, igid)
			for bfm in bfm_list: 
				if igid in num_injections_app_table[app]:
					if bfm in num_injections_app_table[app][igid]:
						f.write("\t".join([p.EM_STR[bfm], str(num_injections_app_table[app][igid][bfm]), str(runtime_app_table[app][igid][bfm]),  str(runtime_app_nt_table[app][igid][bfm])]) + "\n")
					else:
						f.write("\t".join([p.EM_STR[bfm], "0", "0"] + "\n"))
	f.close()


###############################################################################
# Print detailed NVBitFI results to a tsv file for analysis in a worksheet
###############################################################################
def print_detailed_results_tsv(typ):
	f = open(fname_prefix + "NVBitFI_details.tsv", "w")
	f.write("\t".join(["App", "IGID", "Injection Model"] + output_format) + "\n")

	for app in results_app_table: 
		f.write(app + "\t") # write app name

		igid_list = get_igid_list(typ)
		for igid in igid_list: 
			f.write(get_igid_str(typ, igid) + "\t")

			bfm_list = get_bfm_list(typ, igid)
			for bfm in bfm_list: 
				written = False
				if igid in results_app_table[app]:
					if bfm in results_app_table[app][igid]:
						f.write("\t".join([p.EM_STR[bfm]] + map(str,to_list(results_app_table[app][igid][bfm], p.NUM_CATS))) + "\n")
						written = True
				if not written:
					f.write("\t".join([p.EM_STR[bfm]] + map(str,to_list({}, p.NUM_CATS))))
	f.close()

###############################################################################
# Main function that processes files, analyzes results and prints them to an
# xlsx file
###############################################################################
def main():
	inj_type = "inst_value"
			
	parse_results_apps(inj_type) # parse sassifi results into local data structures
	# populate instruction fractions
	populate_inst_fraction()

	global fname_prefix 
	fname_prefix = p.NVBITFI_HOME + "/logs/" + "results/results_" + inj_type + "_" + str(p.NUM_INJECTIONS) + "_"

	print_inst_fractions_tsv()
	print_detailed_results_tsv(inj_type)
	print_stats_tsv(inj_type)

	print ("Results: %s" %(p.NVBITFI_HOME + "/logs/" + "results/"))


if __name__ == "__main__":
    main()
