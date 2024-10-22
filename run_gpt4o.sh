#!/bin/bash

for MODE in {0..3}
do
  export CUDA_VISIBLE_DEVICES=2
  python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gpt-4o-mini ${MODE}
done
