cd "$(dirname "$0")"

SEED=666
RUNTIME=100
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Released weights on the Hugging Face Hub; set to your training output dir to evaluate your own token instead.
TIDIR='itsmag11/AITTI'
EVAL_BIAS="race"
TOKEN="<${EVAL_BIAS}-inclusive>"
CHANGE_STEP=10
ROOT_DIR="./results/${TIDIR##*/}/${EVAL_BIAS}/s${SEED}_${RUNTIME}runs"

test_occupations=(
    "doctor" "chief" "farmer" "architect" "software_developer" "ballet_dancer" "yoga_instructor" "cosmetologist"
    "fashion_designer" "flight_attendant" "artist" "musician" "writer" "photographer" "philosopher" "athlete"
    "lawyer" "politician" "journalist" "barista" "detective" "security_guard" "professor" "sports_coach"
)

for occupation in "${test_occupations[@]}"; do
    PROMPT="A photo of a ${TOKEN} ${occupation//_/ }"
    echo "${PROMPT}"

    python inference_aitti.py \
        --seed ${SEED} \
        --run_times ${RUNTIME} \
        --sd_model ${MODEL_NAME} \
        --num_inference_steps 25 \
        --change_step ${CHANGE_STEP} \
        --textual_inversion_dir ${TIDIR} \
        --attribute ${EVAL_BIAS} \
        --prompt "${PROMPT}" \
        --profession_name ${occupation} \
        --token_name "${TOKEN}" \
        --output_dir "${ROOT_DIR}/${PROMPT// /_}" \
        --checkface
done

python evaluation.py --attribute ${EVAL_BIAS} --root_dir ${ROOT_DIR}
