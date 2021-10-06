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



import os, sys, re, string, operator, math, datetime, subprocess, time, multiprocessing, pkgutil
import params as p
import common_functions as cf

###############################################################################
# Basic functions and parameters
###############################################################################
before = -1

def print_usage():
	print ("Usage: \n run_injections.py standalone <clean>")
	print ("Example1: \"run_injections.py standalone\" to run jobs on the current system")
	print ("Example2: \"run_injections.py standalone clean\" to launch jobs on the current system and clean all previous logs/results")


############################################################################
# Print progress every 10 minutes for jobs submitted to the cluster
############################################################################
def print_heart_beat(nj):
	global before
	if before == -1:
		before = datetime.datetime.now()
	if (datetime.datetime.now()-before).seconds >= 10*60:
		print ("Jobs so far: %d" %nj)
		before = datetime.datetime.now()

def get_log_name(app, inj_mode, igid, bfm):
	return p.app_log_dir[app] + "results-mode" + str(inj_mode) + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt"

############################################################################
# Clear log conent. Default is to append, but if the user requests to clear
# old logs, use this function.
############################################################################
def clear_results_file(app):
	for bfm in p.rf_bfm_list: 
		open(get_log_name(app, p.RF_MODE, "rf", bfm)).close()
	for igid in p.inst_value_igid_bfm_map:
		for bfm in p.inst_value_igid_bfm_map[igid]:
			open(get_log_name(app, p.INST_VALUE_MODE, igid, bfm)).close()
	for igid in p.inst_address_igid_bfm_map:
		for bfm in p.inst_address_igid_bfm_map[igid]:
			open(get_log_name(app, p.INST_ADDRESS_MODE, igid, bfm)).close()

############################################################################
# count how many jobs are done
############################################################################
def count_done(fname):
	return sum(1 for line in open(fname)) # count line in fname 


############################################################################
# check queue and launch multiple jobs on a cluster 
# This feature is not implemented.
############################################################################
def check_and_submit_cluster(cmd):
    print ("This feature is not implement. Please write code here to submit jobs to your cluster.\n")
    sys.exit(-1)

############################################################################
# check queue and launch multiple jobs on the multigpu system 
############################################################################
jobs_list = []
pool = multiprocessing.Pool(p.NUM_GPUS) # create a pool

def check_and_submit_multigpu(cmd):
	jobs_list.append("CUDA_VISIBLE_DEVICES=" + str(len(jobs_list)) + " " + cmd)
	if len(jobs_list) == p.NUM_GPUS:
		pool.map(os.system, jobs_list) # launch jobs in parallel
		del jobs_list[:] # clear the list


###############################################################################
# Run Multiple injection experiments
###############################################################################
def run_multiple_injections_igid(app, inj_mode, igid, where_to_run):
	bfm_list = [] 
	if inj_mode == p.RF_MODE: 
		bfm_list = p.rf_bfm_list 
	if inj_mode == p.INST_VALUE_MODE:
		bfm_list = p.inst_value_igid_bfm_map[igid]
	if inj_mode == p.INST_ADDRESS_MODE:
		bfm_list = p.inst_address_igid_bfm_map[igid]
		
	for bfm in bfm_list:
		#print "App: %s, IGID: %s, EM: %s" %(app, p.IGID_STR[igid], p.EM_STR[bfm])
		total_jobs = 0
		inj_list_filenmae = p.app_log_dir[app] + "/injection-list/mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt"
		inf = open(inj_list_filenmae, "r")
		for line in inf: # for each injection site 
			total_jobs += 1
			if total_jobs > p.THRESHOLD_JOBS: 
				break; # no need to run more jobs
			l = line.strip().split() #Example: _Z24bpnn_adjust_weights_cudaPfiS_iS_S_ 0 1297034 0.877316323856 0.214340876321
			if len(l) >= 5: 
				# print (line) 
				[kcount, iid, opid, bid] = l[-4:] # obtains params for this injection
				kname = "".join(l[:-4]) # kernel name should contain all the arguments, with no spaces
				if p.verbose: print ("\n%d: app=%s, Kernel=%s, kcount=%s, igid=%d, bfm=%d, instID=%s, opID=%s, bitLocation=%s" %(total_jobs, app, kname, kcount, igid, bfm, iid, opid, bid))
				cmd = "%s %s/scripts/run_one_injection.py %s %s %s %s %s %s %s %s %s %s" %(p.PYTHON_P, p.NVBITFI_HOME, inj_mode, str(igid), str(bfm), app, "\""+kname+"\"", kcount, iid, opid, bid, total_jobs)
				if p.verbose: print (cmd)
				if where_to_run == "cluster":
					check_and_submit_cluster(cmd)
				elif where_to_run == "multigpu":
					check_and_submit_multigpu(cmd)
				else:
					os.system(cmd)
				if p.verbose: print ("done injection run ")
			else:
				print ("Line doesn't have enough params:%s" %line)
			print_heart_beat(total_jobs)


###############################################################################
# wrapper function to call either RF injections or instruction level injections
###############################################################################
def run_multiple_injections(app, inj_mode, where_to_run):
    if inj_mode == p.RF_MODE:
        run_multiple_injections_igid(app, inj_mode, "rf", where_to_run)
    else:
        igid_list = []
        if inj_mode == p.INST_VALUE_MODE:
            igid_list = p.inst_value_igid_bfm_map
        if inj_mode == p.INST_ADDRESS_MODE:
            igid_list = p.inst_address_igid_bfm_map
        for igid in igid_list: 
            run_multiple_injections_igid(app, inj_mode, igid, where_to_run)

###############################################################################
# Starting point of the execution
###############################################################################
def main(): 
	if len(sys.argv) >= 2: 
		where_to_run = sys.argv[1]
	
		if where_to_run != "standalone":
			if pkgutil.find_loader('lockfile') is None:
				print ("lockfile module not found. This python module is needed to run injection experiments in parallel." )
				sys.exit(-1)
	
		sorted_apps = [app for app, value in sorted(p.apps.items(), key=lambda e: e[1][3])] # sort apps according to expected runtimes
		for app in sorted_apps: 
			print ("\n" + app)
			if not os.path.isdir(p.app_log_dir[app]): os.system("mkdir -p " + p.app_log_dir[app]) # create directory to store summary
			if len(sys.argv) == 3: 
				if sys.argv[2] == "clean":
					clear_results_file(app) # clean log files only if asked for
	
			run_multiple_injections(app, "inst_value", where_to_run)
	
	else:
		print_usage()

if __name__ == "__main__":
    main()
