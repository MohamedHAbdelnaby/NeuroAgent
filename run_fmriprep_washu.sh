#!/bin/bash
#SBATCH --job-name=fmriprep_washu
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=895-944%32
#SBATCH --output=${PROJECT}/preprocessing/logs/fmriprep_%A_%a.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fmriprep_%A_%a.err

# WashU BOLD files use task-reststudy1/2/3 (not task-rest), so we omit --task-id and let fMRIPrep discover all BOLD files automatically.


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

DONE=$(find $OUTPUT_DIR/sub-${SUBJECT_LABEL} -name "*desc-preproc_bold.nii.gz" 2>/dev/null | head -1)
if [ -n "$DONE" ]; then
    echo "Subject $SUBJECT ($SITE) already processed, skipping."
    exit 0
fi

echo "fMRIPrep Array Job (WashU)"
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
