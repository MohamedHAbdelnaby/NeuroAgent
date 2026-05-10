#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-944%32
#SBATCH --output=${PROJECT}/preprocessing/logs/fmriprep_%A_%a.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fmriprep_%A_%a.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${FS_LICENSE:?FS_LICENSE must be set (path to your FreeSurfer license.txt)}"

module load apptainer/1.4.0
SIF=/apps/common/software/fMRIPrep/25.1.3/fmriprep-25.1.3.sif

MANIFEST=${PROJECT}/preprocessing/manifests/fmriprep_subjects.csv
LINE=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n' $MANIFEST)
SITE=$(echo $LINE | cut -d',' -f1)
SUBJECT=$(echo $LINE | cut -d',' -f2)
SUBJECT_LABEL=${SUBJECT#sub-}

BIDS_DIR=${PROJECT}/Datasets/ADHD200/$SITE
OUTPUT_DIR=${PROJECT}/preprocessing/fmriprep_output/$SITE
WORK_DIR=$TMPDIR/fmriprep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
LICENSE=${FS_LICENSE}

mkdir -p $OUTPUT_DIR

EXPECTED=$OUTPUT_DIR/sub-${SUBJECT_LABEL}/ses-1/func/sub-${SUBJECT_LABEL}_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
if [ -f "$EXPECTED" ]; then
    echo "Subject $SUBJECT ($SITE) already processed, skipping."
    exit 0
fi

echo "fMRIPrep Array Job"
echo "Array job  : $SLURM_ARRAY_JOB_ID"
echo "Task ID    : $SLURM_ARRAY_TASK_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Site       : $SITE"
echo "Subject    : $SUBJECT"
echo "Start time : $(date)"

apptainer run --nv \
    --bind /projects \
    --bind $TMPDIR \
    $SIF \
    $BIDS_DIR \
    $OUTPUT_DIR \
    participant \
    --participant-label $SUBJECT_LABEL \
    --fs-license-file $LICENSE \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --task-id rest \
    --fs-no-reconall \
    --skip-bids-validation \
    --nthreads 8 \
    --mem-mb 22000 \
    -w $WORK_DIR \
    --notrack

EXIT_CODE=$?

echo "Exit code  : $EXIT_CODE"
echo "Done       : $(date)"

exit $EXIT_CODE
