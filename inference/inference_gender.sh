SEED=666
RUNTIME=100
MODEL_NAME="runwayml/stable-diffusion-v1-5"
RUN_NOTE="regular"

TIDIR='./PATH/TO/TEXTUAL/INVERSION/DIR'
AIT="${TIDIR##*/}"
EVAL_BIAS="gender"
TOKEN="<${EVAL_BIAS}-diverse>"
MULTI_TOKEN="<${EVAL_BIAS}-diverse>"
ROOT_DIR="./results/inference/${AIT}/${RUN_NOTE}/s${SEED}_${RUNTIME}runs"

test_occupations=(
    "doctor" 
    "chief" "farmer" "architect" "software_developer" "ballet_dancer" "yoga_instructor" "cosmetologist" "fashion_designer" "flight_attendant" "artist" 
    "musician" "writer" "photographer" "philosopher" "athlete" "lawyer" "politician" 
    "journalist" "barista" "detective" "security_guard" "professor" "sports_coach"
)


for occupation in "${test_occupations[@]}"; do
    ORI_PROMPT="A photo of a ${occupation//_/ }"
    PROMPT="A photo of a ${TOKEN} ${occupation//_/ }"
    PRO_NAME=${occupation}
    OUTDIR="${ROOT_DIR}/${PROMPT// /_}"
    echo ${PROMPT}
    echo ${PRO_NAME}
    echo ${OUTDIR}

    python inference_aitti.py \
        --seed ${SEED} \
        --run_times ${RUNTIME} \
        --sd_model ${MODEL_NAME} \
        --num_inference_steps 25 \
        --textual_inversion_dir ${TIDIR} \
        --prompt "${PROMPT}" \
        --profession_name  ${PRO_NAME} \
        --token_name ${TOKEN} \
        --all_token_name ${MULTI_TOKEN} \
        --output_dir ${OUTDIR} \
        --num_transformer_head 6 \
        --num_transformer_block 4 \
        --checkface
            
    python evaluate_clip.py \
        --attribute_to_eval 'gender' \
        --root_dir ${OUTDIR} \
        --gt_prompt "${ORI_PROMPT}"

done

python get_average_metrics.py \
    --attribute_to_eval 'gender' \
    --root_dir ${ROOT_DIR}