# Failure-mode analysis: study_postfix


## Llama-3.1-8B-Instruct


### ADHD failure breakdown (584 total: 204 FP, 79 FN)

- FP sites: NYU=59, Peking_1=52, Pittsburgh=31, OHSU=27, KKI=21, NeuroIMAGE=14
- FN sites: NYU=39, OHSU=16, Peking_1=10, NeuroIMAGE=9, KKI=5
- FP gender: female=143, male=61
- FN gender: female=12, male=67
- FP age:    mean=12.0  range=7.2-21.7
- FN age:    mean=11.2  range=7.4-20.1

**Top-5 high-confidence FPs:**
- `KKI_sub-1266183` (conf=1.0, site=KKI, age=9.67, sex=F)  reason: direct_v2
- `KKI_sub-1594156` (conf=1.0, site=KKI, age=12.87, sex=M)  reason: direct_v2
- `KKI_sub-1638334` (conf=1.0, site=KKI, age=10.44, sex=F)  reason: direct_v2
- `KKI_sub-1652369` (conf=1.0, site=KKI, age=10.6, sex=F)  reason: direct_v2
- `KKI_sub-1735881` (conf=1.0, site=KKI, age=9.39, sex=F)  reason: direct_v2

**Top-5 high-confidence FNs:**
- `KKI_sub-1873761` (conf=1.0, site=KKI, age=10.36, sex=F)  reason: direct_v2
- `KKI_sub-2081148` (conf=1.0, site=KKI, age=9.43, sex=F)  reason: direct_v2
- `NeuroIMAGE_sub-1438162` (conf=1.0, site=NeuroIMAGE, age=11.85, sex=F)  reason: direct_v2
- `NeuroIMAGE_sub-2029723` (conf=1.0, site=NeuroIMAGE, age=16.5, sex=M)  reason: direct_v2
- `NeuroIMAGE_sub-2961243` (conf=1.0, site=NeuroIMAGE, age=16.51, sex=M)  reason: direct_v2

### stroke_detection (1013 total: 430 FP, 0 FN, 0 invalid)


**Top-5 FPs:**
- `NeuroIMAGE_sub-2671604` (conf=0.9) reason: direct_v2
- `NeuroIMAGE_sub-8991934` (conf=0.9) reason: direct_v2
- `NeuroIMAGE_sub-4134561` (conf=0.9) reason: direct_v2
- `NeuroIMAGE_sub-1588809` (conf=0.9) reason: direct_v2
- `NeuroIMAGE_sub-3959823` (conf=0.9) reason: direct_v2

**Top-5 FNs:**

### stroke_lat (583 total: 24 FP, 131 FN, 0 invalid)


**Top-5 FPs:**
- `R010_sub-r010s027` (conf=1.0) reason: direct_v2
- `R011_sub-r011s034` (conf=1.0) reason: direct_v2
- `R049_sub-r049s010` (conf=1.0) reason: direct_v2
- `R010_sub-r010s029` (conf=0.99) reason: direct_v2
- `R011_sub-r011s003` (conf=0.99) reason: direct_v2

**Top-5 FNs:**
- `R009_sub-r009s024` (conf=1.0) reason: direct_v2
- `R009_sub-r009s073` (conf=1.0) reason: direct_v2
- `R009_sub-r009s093` (conf=1.0) reason: direct_v2
- `R010_sub-r010s014` (conf=1.0) reason: direct_v2
- `R031_sub-r031s007` (conf=1.0) reason: direct_v2

### tumor_lat (1563 total: 52 FP, 781 FN, 0 invalid)


**Top-5 FPs:**
- `BraTS-GLI-02194-106` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02228-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02412-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02060-100` (conf=0.99) reason: direct_v2
- `BraTS-GLI-02078-100` (conf=0.99) reason: direct_v2

**Top-5 FNs:**
- `BraTS-GLI-00008-102` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00046-101` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00063-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00078-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00078-101` (conf=1.0) reason: direct_v2

## Llama-3.1-8B-Instruct__neuroagent


### ADHD failure breakdown (584 total: 75 FP, 116 FN)

- FP sites: NYU=22, Pittsburgh=19, OHSU=11, KKI=9, Peking_1=8, NeuroIMAGE=6
- FN sites: NYU=64, OHSU=16, Peking_1=16, KKI=13, NeuroIMAGE=7
- FP gender: male=38, female=37
- FN gender: female=24, male=92
- FP age:    mean=12.4  range=7.2-20.4
- FN age:    mean=10.7  range=7.2-20.1

**Top-5 high-confidence FPs:**
- `KKI_sub-1535233` (conf=1.0, site=KKI, age=9.64, sex=M)  reason: rule (conf=1.00): Convergent ADHD evidence: 7 subcortical |z|>=1.5 + 2 circuit |z|>=1.5 -> ADHD -> 1
- `KKI_sub-1686265` (conf=1.0, site=KKI, age=8.02, sex=M)  reason: rule (conf=1.00): Convergent ADHD evidence: 4 subcortical |z|>=1.5 + 2 circuit |z|>=1.5 -> ADHD -> 1
- `KKI_sub-1735881` (conf=1.0, site=KKI, age=9.39, sex=F)  reason: rule (conf=1.00): Convergent ADHD evidence: 2 subcortical |z|>=1.5 + 1 circuit |z|>=1.5 -> ADHD -> 1
- `KKI_sub-2371032` (conf=1.0, site=KKI, age=10.73, sex=F)  reason: rule (conf=1.00): Convergent ADHD evidence: 1 subcortical |z|>=1.5 + 3 circuit |z|>=1.5 -> ADHD -> 1
- `KKI_sub-2768273` (conf=1.0, site=KKI, age=9.24, sex=F)  reason: rule (conf=1.00): Convergent ADHD evidence: 5 subcortical |z|>=1.5 + 1 circuit |z|>=1.5 -> ADHD -> 1

**Top-5 high-confidence FNs:**
- `KKI_sub-1541812` (conf=1.0, site=KKI, age=8.45, sex=F)  reason: rule (conf=1.00): Insufficient evidence: 0 subcortical |z|>=1.5 + 0 circuit |z|>=1.5; needs convergent (>=2 struct + >=1 fmri, or >=1 struct + >=2 fmri, or >=3 
- `KKI_sub-1577042` (conf=1.0, site=KKI, age=9.06, sex=M)  reason: rule (conf=1.00): Insufficient evidence: 0 subcortical |z|>=1.5 + 1 circuit |z|>=1.5; needs convergent (>=2 struct + >=1 fmri, or >=1 struct + >=2 fmri, or >=3 
- `KKI_sub-1873761` (conf=1.0, site=KKI, age=10.36, sex=F)  reason: rule (conf=1.00): Insufficient evidence: 0 subcortical |z|>=1.5 + 1 circuit |z|>=1.5; needs convergent (>=2 struct + >=1 fmri, or >=1 struct + >=2 fmri, or >=3 
- `KKI_sub-2014113` (conf=1.0, site=KKI, age=10.35, sex=M)  reason: rule (conf=1.00): Insufficient evidence: 0 subcortical |z|>=1.5 + 0 circuit |z|>=1.5; needs convergent (>=2 struct + >=1 fmri, or >=1 struct + >=2 fmri, or >=3 
- `KKI_sub-2026113` (conf=1.0, site=KKI, age=12.99, sex=F)  reason: rule (conf=1.00): Insufficient evidence: 0 subcortical |z|>=1.5 + 1 circuit |z|>=1.5; needs convergent (>=2 struct + >=1 fmri, or >=1 struct + >=2 fmri, or >=3 

### stroke_detection (1013 total: 161 FP, 174 FN, 0 invalid)


**Top-5 FPs:**
- `NeuroIMAGE_sub-2671604` (conf=1.0) reason: rule (conf=1.00): |TOTAL_LI|=0.0992 (>=0.012) or max |LI_*|=0.072 (>=0.10) -> stroke present -> 1
- `Pittsburgh_sub-0016035` (conf=1.0) reason: rule (conf=1.00): |TOTAL_LI|=0.0145 (>=0.012) or max |LI_*|=0.028 (>=0.10) -> stroke present -> 1
- `Pittsburgh_sub-0016036` (conf=1.0) reason: rule (conf=1.00): |TOTAL_LI|=0.0136 (>=0.012) or max |LI_*|=0.031 (>=0.10) -> stroke present -> 1
- `NeuroIMAGE_sub-8991934` (conf=1.0) reason: rule (conf=1.00): |TOTAL_LI|=0.1043 (>=0.012) or max |LI_*|=0.059 (>=0.10) -> stroke present -> 1
- `NYU_sub-0010055` (conf=1.0) reason: rule (conf=1.00): |TOTAL_LI|=0.0824 (>=0.012) or max |LI_*|=0.200 (>=0.10) -> stroke present -> 1

**Top-5 FNs:**
- `R040_sub-r040s004` (conf=1.0) reason: Brain morphology is roughly symmetric, indicating no stroke.
- `R009_sub-r009s013` (conf=1.0) reason: Brain morphology is roughly symmetric, indicating no stroke.
- `R040_sub-r040s024` (conf=0.998) reason: rule (conf=0.99): |TOTAL_LI|=0.0073 (<0.012) and max |LI_*|=0.100 (<0.10) -> brain symmetric -> control -> 0
- `R031_sub-r031s014` (conf=0.996) reason: rule (conf=0.99): |TOTAL_LI|=0.0119 (<0.012) and max |LI_*|=0.061 (<0.10) -> brain symmetric -> control -> 0
- `R009_sub-r009s012` (conf=0.996) reason: rule (conf=0.99): |TOTAL_LI|=0.0119 (<0.012) and max |LI_*|=0.012 (<0.10) -> brain symmetric -> control -> 0

### stroke_lat (583 total: 48 FP, 61 FN, 0 invalid)


**Top-5 FPs:**
- `R001_sub-r001s030` (conf=1.0) reason: Total LI is positive, indicating the right hemisphere is smaller, suggesting a right hemisphere stroke.
- `R009_sub-r009s049` (conf=1.0) reason: Total LI is positive, indicating the right hemisphere is smaller, suggesting a right hemisphere stroke.
- `R009_sub-r009s058` (conf=1.0) reason: Total LI is positive, indicating the right hemisphere is smaller, suggesting a right hemisphere stroke.
- `R009_sub-r009s113` (conf=1.0) reason: Total LI is positive, indicating the right hemisphere is smaller, suggesting a right hemisphere stroke.
- `R010_sub-r010s021` (conf=1.0) reason: Total LI is positive, indicating the right hemisphere is smaller, suggesting a right hemisphere stroke.

**Top-5 FNs:**
- `R009_sub-r009s007` (conf=1.0) reason: The stroke is lateralized to the left hemisphere based on the positive TOTAL LI value.
- `R009_sub-r009s018` (conf=1.0) reason: Total LI is negative, indicating left hemisphere stroke.
- `R009_sub-r009s032` (conf=1.0) reason: Total LI is negative, indicating left hemisphere stroke.
- `R009_sub-r009s052` (conf=1.0) reason: Total LI is negative, indicating left hemisphere stroke.
- `R009_sub-r009s057` (conf=1.0) reason: Total LI is negative, indicating left hemisphere stroke.

### tumor_lat (1563 total: 46 FP, 781 FN, 0 invalid)


**Top-5 FPs:**
- `BraTS-GLI-02060-100` (conf=1.0) reason: Tumor lateralization is determined by the sign of TOTAL_LI, which is negative (-0.0081), indicating the right hemisphere is larger.
- `BraTS-GLI-02078-103` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=-0.0711; FastSurfer counts tumor as tissue so tumor side is LARGER; sign <=0 -> RIGHT larger -> RIGHT tumor -> 1
- `BraTS-GLI-02085-105` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=-0.0632; FastSurfer counts tumor as tissue so tumor side is LARGER; sign <=0 -> RIGHT larger -> RIGHT tumor -> 1
- `BraTS-GLI-02169-101` (conf=1.0) reason: Tumor side is larger based on TOTAL_LI=-0.0042.
- `BraTS-GLI-02186-103` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=-0.1070; FastSurfer counts tumor as tissue so tumor side is LARGER; sign <=0 -> RIGHT larger -> RIGHT tumor -> 1

**Top-5 FNs:**
- `BraTS-GLI-00008-100` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1085; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0
- `BraTS-GLI-00008-102` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1137; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0
- `BraTS-GLI-00008-101` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1113; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0
- `BraTS-GLI-00008-103` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1119; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0
- `BraTS-GLI-00020-100` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1527; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0

## Mistral-7B-v0.3


### ADHD failure breakdown (584 total: 107 FP, 114 FN)

- FP sites: NYU=35, OHSU=26, Peking_1=16, Pittsburgh=16, NeuroIMAGE=8, KKI=6
- FN sites: NYU=51, OHSU=20, NeuroIMAGE=16, Peking_1=15, KKI=12
- FP gender: male=41, female=66
- FN gender: male=96, female=17, ?=1
- FP age:    mean=12.1  range=7.2-20.1
- FN age:    mean=11.2  range=7.2-20.5

**Top-5 high-confidence FPs:**
- `OHSU_sub-1421489` (conf=1.0, site=OHSU, age=8.75, sex=M)  reason: direct_v2
- `KKI_sub-1779922` (conf=0.99, site=KKI, age=10.84, sex=M)  reason: direct_v2
- `KKI_sub-3434578` (conf=0.99, site=KKI, age=8.12, sex=F)  reason: direct_v2
- `NeuroIMAGE_sub-6115230` (conf=0.99, site=NeuroIMAGE, age=19.39, sex=M)  reason: direct_v2
- `NYU_sub-0010076` (conf=0.99, site=NYU, age=15.95, sex=F)  reason: direct_v2

**Top-5 high-confidence FNs:**
- `NeuroIMAGE_sub-2029723` (conf=1.0, site=NeuroIMAGE, age=16.5, sex=M)  reason: direct_v2
- `NYU_sub-1780174` (conf=1.0, site=NYU, age=11.18, sex=M)  reason: direct_v2
- `NYU_sub-2107638` (conf=1.0, site=NYU, age=10.41, sex=M)  reason: direct_v2
- `OHSU_sub-0023000` (conf=1.0, site=OHSU, age=10.42, sex=F)  reason: direct_v2
- `OHSU_sub-2571197` (conf=1.0, site=OHSU, age=7.67, sex=M)  reason: direct_v2

### stroke_detection (1013 total: 15 FP, 480 FN, 0 invalid)


**Top-5 FPs:**
- `OHSU_sub-4103874` (conf=0.99) reason: direct_v2
- `NeuroIMAGE_sub-1588809` (conf=0.95) reason: direct_v2
- `NeuroIMAGE_sub-2352986` (conf=0.95) reason: direct_v2
- `NeuroIMAGE_sub-3007585` (conf=0.95) reason: direct_v2
- `NYU_sub-0010114` (conf=0.95) reason: direct_v2

**Top-5 FNs:**
- `R009_sub-r009s044` (conf=0.95) reason: direct_v2
- `R046_sub-r046s008` (conf=0.95) reason: direct_v2
- `R011_sub-r011s030` (conf=0.95) reason: direct_v2
- `R048_sub-r048s029` (conf=0.95) reason: direct_v2
- `R004_sub-r004s005` (conf=0.95) reason: direct_v2

### stroke_lat (583 total: 75 FP, 81 FN, 0 invalid)


**Top-5 FPs:**
- `R001_sub-r001s017` (conf=1.0) reason: direct_v2
- `R001_sub-r001s018` (conf=1.0) reason: direct_v2
- `R002_sub-r002s003` (conf=1.0) reason: direct_v2
- `R009_sub-r009s002` (conf=1.0) reason: direct_v2
- `R009_sub-r009s005` (conf=1.0) reason: direct_v2

**Top-5 FNs:**
- `R004_sub-r004s023` (conf=1.0) reason: direct_v2
- `R005_sub-r005s058` (conf=1.0) reason: direct_v2
- `R005_sub-r005s069` (conf=1.0) reason: direct_v2
- `R009_sub-r009s024` (conf=1.0) reason: direct_v2
- `R009_sub-r009s032` (conf=1.0) reason: direct_v2

### tumor_lat (1563 total: 52 FP, 772 FN, 0 invalid)


**Top-5 FPs:**
- `BraTS-GLI-02060-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02078-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02078-101` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02078-102` (conf=1.0) reason: direct_v2
- `BraTS-GLI-02078-103` (conf=1.0) reason: direct_v2

**Top-5 FNs:**
- `BraTS-GLI-00008-100` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00008-102` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00008-103` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00020-101` (conf=1.0) reason: direct_v2
- `BraTS-GLI-00046-100` (conf=1.0) reason: direct_v2

## Mistral-7B-v0.3__neuroagent


### ADHD failure breakdown (584 total: 0 FP, 154 FN)

- FP sites: 
- FN sites: NYU=77, OHSU=27, NeuroIMAGE=18, KKI=16, Peking_1=16
- FP gender: 
- FN gender: female=31, male=122, ?=1
- FP age:    n/a
- FN age:    mean=11.0  range=7.2-20.5

**Top-5 high-confidence FPs:**

**Top-5 high-confidence FNs:**
- `KKI_sub-1541812` (conf=1.0, site=KKI, age=8.45, sex=F)  reason: rule (conf=1.00): No fMRI; only 0 ADHD-relevant subcortical |z|>=1.5 (<4 threshold; ADHD morphometry effect d=-0.11 to -0.19 too small for individual prediction
- `KKI_sub-1623716` (conf=1.0, site=KKI, age=12.65, sex=F)  reason: rule (conf=1.00): No fMRI; only 0 ADHD-relevant subcortical |z|>=1.5 (<4 threshold; ADHD morphometry effect d=-0.11 to -0.19 too small for individual prediction
- `KKI_sub-1577042` (conf=1.0, site=KKI, age=9.06, sex=M)  reason: rule (conf=1.00): No fMRI; only 0 ADHD-relevant subcortical |z|>=1.5 (<4 threshold; ADHD morphometry effect d=-0.11 to -0.19 too small for individual prediction
- `KKI_sub-1873761` (conf=1.0, site=KKI, age=10.36, sex=F)  reason: rule (conf=1.00): No fMRI; only 0 ADHD-relevant subcortical |z|>=1.5 (<4 threshold; ADHD morphometry effect d=-0.11 to -0.19 too small for individual prediction
- `KKI_sub-2026113` (conf=1.0, site=KKI, age=12.99, sex=F)  reason: rule (conf=1.00): No fMRI; only 0 ADHD-relevant subcortical |z|>=1.5 (<4 threshold; ADHD morphometry effect d=-0.11 to -0.19 too small for individual prediction

### stroke_detection (978 total: 16 FP, 546 FN, 0 invalid)


**Top-5 FPs:**
- `Peking_1_sub-3554582` (conf=0.95) reason: Significant volume asymmetry was found between the left and right caudate regions (p = 0.001), likely reflecting a stroke lesion localized in the basal ganglia 
- `Pittsburgh_sub-0016013` (conf=0.95) reason: The subject's TOTAL LI is 0.08, which is above the threshold of 0.012 for stroke detection.
- `NYU_sub-0010055` (conf=0.9) reason: Significant hemispheric asymmetry detected with Left Caudate (z-score: -2.05, > 95% percentile by normative reference) and Right Putamen (z-score: -3.38, > 99.8
- `KKI_sub-2104012` (conf=0.9) reason: The subject KKI_sub-2104012 has a |TOTAL LI| of 0.08, which is above the threshold of 0.012 for stroke detection. Additionally, the left Caudate volume is 3.6 s
- `KKI_sub-3699991` (conf=0.9) reason: The subject KKI_sub-3699991 has a |TOTAL LI| of 0.08, which is above the threshold of 0.012 for stroke detection. Additionally, the left caudate nucleus has a l

**Top-5 FNs:**
- `R034_sub-r034s015` (conf=0.99) reason: The subject's |TOTAL LI| is 0.007, which is within the range of healthy controls (0.007 median), and significantly below the stroke subject range (0.016+). No s
- `R019_sub-r019s001` (conf=0.99) reason: The subject's |TOTAL LI| is below the 0.08 threshold, indicating hemispheric volumes within the symmetric range. No significant abnormalities were detected in t
- `R001_sub-r001s034` (conf=0.99) reason: |TOTAL LI| is below the 0.08 threshold, indicating hemispheric volumes within the symmetric range.
- `R015_sub-r015s009` (conf=0.99) reason: The |TOTAL LI| value is below the detection threshold of 0.012, indicating no significant hemispheric asymmetry or focal cortical loss.
- `R047_sub-r047s031` (conf=0.99) reason: The subject's |TOTAL LI| is below the threshold of 0.012 for stroke detection, indicating roughly symmetric brain morphology.

### stroke_lat (562 total: 4 FP, 271 FN, 0 invalid)


**Top-5 FPs:**
- `R028_sub-r028s005` (conf=0.99) reason: The Right Cerebral Hemisphere (RCH) shows a noticeable volume reduction (6.02%) compared to the Left Cerebral Hemisphere (LCH), which exceeds the age-corrected 
- `R024_sub-r024s011` (conf=0.95) reason: The TOTAL LI value is positive, indicating that the right hemisphere is smaller in volume compared to the left hemisphere. This suggests that the stroke is on t
- `R044_sub-r044s002` (conf=0.95) reason: The TOTAL LI value is positive (right > left), indicating that the right hemisphere is smaller in volume. This suggests a right hemisphere stroke.
- `R009_sub-r009s029` (conf=0.9) reason: The TOTAL LI value is positive, indicating that the right hemisphere is smaller in volume. This suggests that the stroke is on the right side, hence the lateral

**Top-5 FNs:**
- `R009_sub-r009s032` (conf=1.0) reason: TOTAL LI is -0.06, indicating that the left hemisphere is smaller, suggesting a left hemisphere stroke.
- `R009_sub-r009s057` (conf=1.0) reason: The TOTAL LI value is negative (-0.08), indicating that the Left hemisphere is smaller in volume compared to the Right hemisphere. This suggests that the stroke
- `R009_sub-r009s065` (conf=1.0) reason: The TOTAL LI value is negative, indicating that the left hemisphere is smaller. This suggests that the stroke is on the left hemisphere.
- `R010_sub-r010s013` (conf=1.0) reason: The TOTAL LI value is negative, indicating that the left hemisphere is smaller in volume compared to the right hemisphere, suggesting a left-sided stroke.
- `R017_sub-r017s118` (conf=1.0) reason: The TOTAL LI value is negative (-8.973), indicating a smaller Left hemisphere compared to the Right hemisphere. This suggests a Left hemisphere stroke.

### tumor_lat (1485 total: 398 FP, 373 FN, 0 invalid)


**Top-5 FPs:**
- `BraTS-GLI-00006-101` (conf=0.99) reason: The analysis reveals no significant structural abnormalities and the total left-right (LI) volume is negative, indicating a larger right hemisphere. This sugges
- `BraTS-GLI-00009-101` (conf=0.99) reason: The analysis reveals no significant structural abnormalities and the total left-right hemisphere volume (LI) is negative, indicating a larger right hemisphere. 
- `BraTS-GLI-00085-100` (conf=0.99) reason: The total intracranial volume (ICV) is 0 mm³, indicating no significant findings. Since the ICV is not significantly different between the hemispheres, the late
- `BraTS-GLI-00533-100` (conf=0.99) reason: The tumor is located in the right hemisphere as the total left-right (LI) volume is negative, indicating a larger right hemisphere.
- `BraTS-GLI-00533-101` (conf=0.99) reason: The total intracranial volume (ICV) is not provided, but based on the key findings, no significant abnormalities were detected. Since the analysis did not revea

**Top-5 FNs:**
- `BraTS-GLI-02789-101` (conf=1.0) reason: rule (conf=1.00): TOTAL_LI=+0.1671; FastSurfer counts tumor as tissue so tumor side is LARGER; sign >0 -> LEFT larger -> LEFT tumor -> 0
- `BraTS-GLI-00046-101` (conf=0.99) reason: The total intracranial volume (ICV) is 0 mm³, indicating no significant findings. Since the ICV is not different between the hemispheres, the lateralization is 
- `BraTS-GLI-00517-100` (conf=0.99) reason: The analysis reveals no significant structural abnormalities, and the total intracranial volume (ICV) is 0 mm³. Since the ICV is not significantly different bet
- `BraTS-GLI-00556-100` (conf=0.99) reason: The tumor is located within the Left Hemisphere according to the Structural MRI analysis.
- `BraTS-GLI-02063-103` (conf=0.99) reason: The analysis shows no significant abnormalities detected and the ICV (Intracranial Volume) is 0 mm³, which indicates no tumor mass. However, since FastSurfer co