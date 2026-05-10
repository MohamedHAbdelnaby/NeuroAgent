#!/bin/bash
#SBATCH --job-name=fmriprep_diag
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${PROJECT}/preprocessing/logs/fmriprep_diag_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fmriprep_diag_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${FS_LICENSE:?FS_LICENSE must be set (path to your FreeSurfer license.txt)}"

module load apptainer/1.4.0

SIF=/apps/common/software/fMRIPrep/25.1.3/fmriprep-25.1.3.sif
BIDS_DIR=${PROJECT}/Datasets/ADHD200/Brown
OUTPUT_DIR=${PROJECT}/preprocessing/fmriprep_output/Brown
WORK_DIR=$TMPDIR/fmriprep_diag_${SLURM_JOB_ID}
LICENSE=${FS_LICENSE}

mkdir -p $OUTPUT_DIR

echo "fMRIPrep Diagnostic"
echo "Node       : $SLURMD_NODENAME"
echo "Subject    : Brown / sub-0026001"
echo "Start time : $(date)"

apptainer run --nv \
    --bind /projects \
    --bind $TMPDIR \
    $SIF \
    $BIDS_DIR \
    $OUTPUT_DIR \
    participant \
    --participant-label 0026001 \
    --fs-license-file $LICENSE \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --task-id rest \
    --fs-no-reconall \
    --skip-bids-validation \
    --nthreads 8 \
    --mem-mb 22000 \
    -w $WORK_DIR \
    --notrack

echo "Exit code  : $?"
echo "Done       : $(date)"
