if [ -z "$CHECKPOINT_DIR" ]; then
    echo "CHECKPOINT_DIR is required but not specified."
    exit 1
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

set -e

CHECKPOINT_DIR=$(realpath "$CHECKPOINT_DIR")
model_family=$(basename "$CHECKPOINT_DIR") # Qwen2.5-3B-sft-xxx

output_files=$(ls "logs/${model_family}_step_*${tag}.txt" | sort -V)
summary_file="logs/${model_family}${tag}_summary.txt"
hyperparameters="temperature: $TEMPERATURE, top_p: $TOP_P, top_k: $TOP_K, #generation: $NUM_GENERATION, max_model_len: $MAX_MODEL_LEN"
if [ ! -f "$summary_file" ]; then
    echo "$hyperparameters" >> "$summary_file"
elif ! grep -Fq "$hyperparameters" "$summary_file"; then
    echo "Found existing evaluation summary $summary_file, but the hyperparameters don't match the current script."
    echo "Please manually delete the summary file to start a new evaluation run."
    exit 1
else
    echo "Resume from existing summary $summary_file."
fi

for output_file in $output_files; do
    model_name=${output_file/*_step_/}
    model_name=${model_name/${tag}.txt/}
    model_name="hf_step_${model_name}" # hf_step_*
    record="model_name='$model_name'"
    output_line_num=$(grep -a -Fn "$record" "$output_file" | head -n1 | cut -d: -f1)
    summary_line_num=$(grep -a -Fn "$record" "$summary_file" | head -n1 | cut -d: -f1)
    if [ -n "$output_line_num" ]; then
        # if output contains a record
        if [ -n "$summary_line_num" ]; then
            # if summary also contains a record
            output_record=$(tail -n +"$((output_line_num + 1))" "$output_file" | head -n6)
            summary_record=$(tail -n +"$((summary_line_num + 1))" "$summary_file" | head -n6)
            # if the record is already in the summary, skip
            if [ "$output_record" == "$summary_record" ]; then
                continue
            else
                echo "Found unmatched output $output_file and summary $summary_file."
                echo "Please manually rename or delete the old output file."
                exit 1
            fi
        else
            # summary doesn't contain this record
            : # proceed to copy from output to summary
        fi
    else
        echo "Can't find evaluation record in $output_file."
        echo "Please manually delete the output file if it is broken."
        exit 1
    fi

    fields=(
        "'temperature': $TEMPERATURE"
        "'top_p': $TOP_P"
        "'top_k': $TOP_K"
        "'num_tests_per_prompt': $NUM_GENERATION"
        "'max_model_len': $MAX_MODEL_LEN"
    )
    for field in "${fields[@]}"; do
        if ! grep -q "$field" "$output_file"; then
            echo "Found unexpected hyperparameters in $output_file."
            echo "Please check the hyperparameters in the output file and this script."
            exit 1
        fi
    done

    # add evaluation results to summary
    line_num=$(grep -a -n "============================================================" "$output_file" \
        | awk -F: '{print $1}' | tail -n 2 | head -n 1)
    if [ -n "$line_num" ]; then
        echo "Collect evaluation record from $output_file."
        tail -n +$line_num "$output_file" >> "$summary_file"
    else
        echo "Can't find evaluation record in $output_file. Skipping it in summary."
    fi
done