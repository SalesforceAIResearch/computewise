# The MLS-Bench KV box (task env `mlsbench-kv`): the b23 box image as it is, by id, plus the package MLS-Bench pins for its
# KV tasks (`transformers-kv-lab` = huggingface/transformers at aad13b87, v5.3.0) and that package's extras, at the versions
# that resolved on 2026-10-22 (constraints.txt, the full set). torch stays 2.11.0+cpu, not upstream's 2.5.1: generation
# never runs under gpu-mock's serving hook, so torch's version cannot reach a served number, and v5.3.0 takes torch >= 2.4
# (the study's ruling, 2026-10-22). Built by build_mlskv_layer.sh. gpu-mock's content is byte-identical to the base.
ARG BASE
FROM ${BASE}
ARG TRANSFORMERS_COMMIT
ARG BUILD_JSON
COPY constraints.txt /tmp/mlskv-constraints.txt
# The extras first under the full resolved set, then transformers from the pinned source with --no-deps, so pip never
# replaces it with a PyPI transformers (hqq names transformers as a dependency). PIP_CONSTRAINT (the base's torch stack)
# is cleared for these two steps only: the base's constraints list no package these add.
RUN PIP_CONSTRAINT= pip install -c /tmp/mlskv-constraints.txt \
      datasets accelerate sentencepiece protobuf hqq optimum-quanto huggingface-hub tokenizers safetensors \
 && PIP_CONSTRAINT= pip install --no-deps "git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}" \
 && python3 -c "import transformers; assert transformers.__version__ == '5.3.0', transformers.__version__; from transformers import DynamicCache" \
 && pip check \
 && cp /tmp/mlskv-constraints.txt /usr/local/share/image-manifest/mlskv-constraints.txt \
 && rm /tmp/mlskv-constraints.txt \
 && pip freeze --all > /usr/local/share/image-manifest/pip-freeze.txt \
 && printf '%s\n' "$BUILD_JSON" > /usr/local/share/image-manifest/mlskv-layer.json
