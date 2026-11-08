#!/bin/bash
# The direction judge (the model served in this job; RUN_NS=judge_<model>) labels the human-validation sheet.
cd ~/stealthy_influence
python -u human_validation.py judge --tag ${RUN_NS#judge_}
