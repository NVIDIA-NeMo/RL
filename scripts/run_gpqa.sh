if [ -z "$MODEL_NAME" ] && [ -z "$CHECKPOINT_DIR" ]; then
    echo "Neither MODEL_NAME nor CHECKPOINT_DIR is specified."
    exit 1
elif [ -n "$MODEL_NAME" ] && [ -n "$CHECKPOINT_DIR" ]; then
    echo "Both MODEL_NAME and CHECKPOINT_DIR are specified. Only one of them should be specificed."
    exit 1
fi
# Examples:
# MODEL_NAME: Qwen/Qwen2.5-size-Instruct
# CHECKPOINT_DIR: /home/zhaochengz/lustre/reinforcer/results/Qwen2.5-3B-sft-xxx
if [[ $MODEL_NAME == Qwen/Qwen2.5* ]]; then
    SIZES=${SIZES:-"1.5B 3B 7B 14B"}
elif [[ $MODEL_NAME == Qwen/Qwen3* ]]; then
    SIZES=${SIZES:-"1.7B 4B 8B 14B"}
fi
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-"-1"}
NUM_GENERATION=${NUM_GENERATION:-4}
if [[ $(awk "BEGIN {print ($TEMPERATURE == 0.0) ? 1 : 0}") -eq 1 || "$TOP_K" == "1" ]]; then
    # for greedy decoding, NUM_GENERATION must be 1
    NUM_GENERATION=1
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
if [ -n "$TAG" ]; then
    tag=_${TAG}
fi
if [ -n "$START_STEP" ] && [ -n "$END_STEP" ]; then
    PARALLEL_MODE=true
else
    PARALLEL_MODE=false
fi

set -e
if [ -n "$MODEL_NAME" ]; then
    model_family=$(basename "${MODEL_NAME/-size/}") # Qwen2.5-Instruct
else
    CHECKPOINT_DIR=$(realpath "$CHECKPOINT_DIR")
    model_family=$(basename "$CHECKPOINT_DIR") # Qwen2.5-3B-sft-xxx
fi
summary_file="logs/${model_family}${tag}_summary.txt"
hyperparameters="temperature: $TEMPERATURE, top_p: $TOP_P, top_k: $TOP_K, #generation: $NUM_GENERATION, max_model_len: $MAX_MODEL_LEN"
if [ "$PARALLEL_MODE" = false ]; then
    if [ ! -f "$summary_file" ]; then
        echo "$hyperparameters" >> "$summary_file"
    elif ! grep -Fq "$hyperparameters" "$summary_file"; then
        echo "Found existing evaluation summary $summary_file, but the hyperparameters don't match the current script."
        echo "Please manually delete the summary file to start a new evaluation run."
        exit 1
    else
        echo "Resume from existing summary $summary_file."
    fi
fi

if [ -n "$MODEL_NAME" ]; then
    models=""
    for size in $SIZES; do
        models+="${MODEL_NAME/size/$size} "
    done
else
    models=$(ls -d ${CHECKPOINT_DIR}/hf_step_* | sort -V)
    
    if [ "$PARALLEL_MODE" = true ]; then
        filtered_models=""
        for model in $models; do
            model_name=$(basename "$model")
            step=${model_name/hf_step_/}
            if [ "$step" -gt "$START_STEP" ] && [ "$step" -le "$END_STEP" ]; then
                filtered_models+="$model "
            fi
        done
        models="$filtered_models"
    fi
fi

for model in $models; do
    model_name=$(basename $model)
    record="model_name='$model_name'"
    if [[ $model_name == hf_step_* ]]; then
        # expand hf_step_* to Qwen2.5-3B-sft-xxx_step_*
        model_name=${model_name/hf/$model_family}
    fi
    output_file="logs/${model_name}${tag}.txt"
    if [ -f "$output_file" ]; then
        output_line_num=$(grep -a -Fn "$record" "$output_file" | head -n1 | cut -d: -f1)
        if [ -n "$output_line_num" ]; then
            # output file contains a record
            if [ "$PARALLEL_MODE" = true ]; then
                # in parallel mode, skip if output already has a record
                continue
            else
                # in sequential mode, check summary consistency
                summary_line_num=$(grep -a -Fn "$record" "$summary_file" | head -n1 | cut -d: -f1)
                if [ -n "$summary_line_num" ]; then
                    # if summary also contains a record
                    output_record=$(tail -n +"$((output_line_num + 1))" "$output_file" | head -n6)
                    summary_record=$(tail -n +"$((summary_line_num + 1))" "$summary_file" | head -n6)
                    # if the record is already in the summary, skip
                    if [ "$output_record" == "$summary_record" ]; then
                        continue
                    fi
                fi
                echo "Found unmatched output $output_file and summary $summary_file."
                echo "Please manually rename or delete the old output file."
                exit 1
            fi
        else
            # output doesn't contain a record, proceed to overwrite
            : # proceed to overwrite
        fi
    fi
    uv run python examples/run_eval.py --config examples/configs/eval_gpqa.yaml \
        eval.num_tests_per_prompt=$NUM_GENERATION \
        generation.temperature=$TEMPERATURE generation.top_p=$TOP_P generation.top_k=$TOP_K \
        generation.model_name="$model" generation.vllm_cfg.max_model_len=$MAX_MODEL_LEN \
        | tee "$output_file"

    if [ "$PARALLEL_MODE" = false ]; then
        # add evaluation results to summary
        line_num=$(grep -a -n "============================================================" "$output_file" \
            | awk -F: '{print $1}' | tail -n 2 | head -n 1)
        if [ -n "$line_num" ]; then
            tail -n +$line_num "$output_file" >> "$summary_file"
        else
            echo "Can't find evaluation record in $output_file. Skipping it in summary."
        fi
    fi
done