#!/bin/bash

## Try to keep this prefix at least for your job name
## so that SoC can keep track of our jobs

#SBATCH --job-name=cs3210_a2

## Change based on length of job and `sinfo` partitions available
## You shouldn't need to change this unless there are very limited GPUs on the standard partition for some reason

#SBATCH --partition=standard

## Specifying what kind of GPU you want. We will grade on A100 MIG instances, but feel free to develop on other machines.
## gpu:1 ==> any gpu. gpu:a100mig:1 gets you one of the A100 GPU shared instances, which are the most plentiful.

#SBATCH --gres=gpu:a100mig:1

## Time limit for the job, set to 30 sec for initial testing, but change this as necessary.

#SBATCH --time=00:00:30

## Just useful logfile names

#SBATCH --output=logs/cs3210_a2_%j.slurmlog
#SBATCH --error=logs/cs3210_a2_%j.slurmlog

## You shouldn't have to change these settings

#SBATCH --ntasks=1
#SBATCH --nodes=1

# Uncomment these two lines if we ask you for debugging information
# echo "DEBUG: All Slurm environment variables for this job"
# printenv | grep SLURM

echo "Job is running on $(hostname), started at $(date)"

# Get some output about GPU status before starting the job
nvidia-smi 

# Compile the code with make - this should produce the 'scanner' executable
echo "Running make to compile your code..."
srun make all

# Run the executable - you can replace the default arguments here with "$@" to pass in the arguments to this job script instead
echo "Running!"
# ./check.py signatures/sigs-both.txt tests/benign-0001.in tests/virus-0008-Win.Trojan.Agent-35503+Win.Trojan.Mybot-6324+Win.Trojan.Delf-802.in tests/benign-0002.in tests/virus-0009-Win.Worm.Gaobot-688.in tests/benign-0003.in tests/virus-0010-Win.Trojan.Corp-3.in tests/virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in tests/virus-0011-Win.Trojan.Sdbot-52.in tests/virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in tests/virus-0012-Win.Trojan.Bancos-1977+Html.Phishing.Auction-29.in tests/virus-0003-Win.Trojan.Anti-5+Win.Trojan.Mybot-8053.in tests/virus-0013-Win.Trojan.Pakes-518.in tests/virus-0004-Win.Worm.Gaobot-857+Win.Worm.Delf-1443+Win.Trojan.Mybot-6497.in tests/virus-0014-Win.Trojan.HIV-5+Html.Phishing.Auction-29.in tests/virus-0005-Win.Trojan.Krap-1+Win.Spyware.Banker-4712.in tests/virus-0015-Win.Spyware.Goldun-31.in tests/virus-0006-Win.Spyware.Banker-483.in tests/virus-0016-Win.Trojan.Pakes-480.in tests/virus-0007-Win.Trojan.Matrix-8.in tests/virus-0017-Win.Trojan.Mybot-8097+Win.Trojan.JetHome-2.in
### RECIPES: Examples of what you can run here instead of the line above
## Just runs the scanner executable directly
 srun nsys nvprof ./scanner  signatures/sigs-both.txt tests/benign-0001.in tests/virus-0008-Win.Trojan.Agent-35503+Win.Trojan.Mybot-6324+Win.Trojan.Delf-802.in tests/benign-0002.in tests/virus-0009-Win.Worm.Gaobot-688.in tests/benign-0003.in tests/virus-0010-Win.Trojan.Corp-3.in tests/virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in tests/virus-0011-Win.Trojan.Sdbot-52.in tests/virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in tests/virus-0012-Win.Trojan.Bancos-1977+Html.Phishing.Auction-29.in tests/virus-0003-Win.Trojan.Anti-5+Win.Trojan.Mybot-8053.in tests/virus-0013-Win.Trojan.Pakes-518.in tests/virus-0004-Win.Worm.Gaobot-857+Win.Worm.Delf-1443+Win.Trojan.Mybot-6497.in tests/virus-0014-Win.Trojan.HIV-5+Html.Phishing.Auction-29.in tests/virus-0005-Win.Trojan.Krap-1+Win.Spyware.Banker-4712.in tests/virus-0015-Win.Spyware.Goldun-31.in tests/virus-0006-Win.Spyware.Banker-483.in tests/virus-0016-Win.Trojan.Pakes-480.in tests/virus-0007-Win.Trojan.Matrix-8.in tests/virus-0017-Win.Trojan.Mybot-8097+Win.Trojan.JetHome-2.in 
## Works as above if you call this script as sbatch a2_slurm_job.sh signatures/sigs-exact.txt tests/benign-0001.in tests/virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in
# ./scanner $@
## Runs the debug version of the executable with NVIDIA's Nsight Systems profile in nvprof mode
 /usr/local/cuda/bin/nsys nvprof ./scanner-debug signatures/sigs-exact.txt tests/benign-0001.in tests/virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in

echo -e "\n====> Finished running.\n"

echo -e "\nJob completed at $(date)"
