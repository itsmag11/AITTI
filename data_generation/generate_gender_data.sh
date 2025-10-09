biased_professions=(
    "construction worker"
    "mechanic"
    "firefighter"
    "police officer"
    "engineer"
    "pilot"
    "computer programmer"
    "ceo"
    "surgeon"
    "truck driver"
    "electrician"
    "scientist"
    "nurse"
    "teacher"
    "secretary"
    "librarian"
    "receptionist"
    "administrative assistant"
    "childcare worker"
    "social worker"
    "dental hygienist"
    "therapist"
    "pathologist"
    "nutritionist"
)
genders="a_male a_female"

SEED=666
RUNTIME=100
NUMCOL=10

for profession in "${biased_professions[@]}"; do
    for gender in ${genders}; do
        PROMPT="a high-quality photo of ${gender//_/ } ${profession}"
        OUTDIR="./data/sd15_images/gender-s666/${PROMPT// /_}"
        echo ${PROMPT}
        echo ${OUTDIR}
        
        python generate_data.py \
            --isrun \
            --seed ${SEED} \
            --run_times ${RUNTIME} \
            --num_col ${NUMCOL} \
            --prompts "${PROMPT}" \
            --output_dir ${OUTDIR} \
            --checkface

    done
done
