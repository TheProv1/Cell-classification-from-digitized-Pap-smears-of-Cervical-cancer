#!/bin/bash

# --- [7/10] Xception ---
echo "--- [7/10] STARTING: mobilenet ---"
python 05-mobilenet-model-code.py > 05-mobilenet-model-code.txt 2>&1 && \
echo "--- [7/10] FINISHED: Mobilenet. Log: 05-mobilenet-model-code.txt ---" && \
echo " " && \

# Check the exit code of the final command to determine overall success or failure
if [ $? -eq 0 ]; then
    echo "--- 🎉 Full Training Pipeline COMPLETED Successfully! ---"
else
    echo "--- 🛑 Full Training Pipeline FAILED on the last-started model. Check its log for errors. ---"
fi

echo "Timestamp: $(date)"
